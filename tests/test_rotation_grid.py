"""Tests for `zdock.rotation_grid` — sampling helpers used by the FFT
docking search and by decoy curation."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from zdock.geom import rotate
from zdock.rotation_grid import (
    euler_quaternions,
    kabsch_quaternion,
    random_quaternions,
    rotation_cone,
)


def _apply_quat(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply a single (4,) quaternion to an (N, 3) tensor via geom.rotate."""
    x = v[:, 0].clone()
    y = v[:, 1].clone()
    z = v[:, 2].clone()
    xn, yn, zn = rotate(x, y, z, q)
    return torch.stack([xn, yn, zn], dim=-1)


def _quat_angle(a: torch.Tensor, b: torch.Tensor) -> float:
    """Angle between two unit quaternions (rad, in [0, π])."""
    dot = float(torch.abs(torch.sum(a * b)).clamp(max=1.0).item())
    return 2.0 * math.acos(dot)


# ---------------------------------------------------------------------------
# random_quaternions / euler_quaternions smoke
# ---------------------------------------------------------------------------


def test_random_quaternions_are_unit():
    q = random_quaternions(20, seed=0)
    norms = q.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_euler_quaternions_first_is_identity():
    q = euler_quaternions(deg=60.0)   # default dtype is float64
    # ZYZ(0,0,0) is the identity, (0,0,0,1) in geom.rotate convention.
    expected = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=q.dtype)
    assert torch.allclose(q[0], expected, atol=1e-12)


# ---------------------------------------------------------------------------
# kabsch_quaternion
# ---------------------------------------------------------------------------


def test_kabsch_recovers_known_rotation():
    """Construct target = scipy.apply(known_rotation, ref); Kabsch should
    return the same known rotation (in geom.rotate convention)."""
    from scipy.spatial.transform import Rotation as R
    rng = np.random.default_rng(0)
    ref_np = rng.standard_normal((50, 3))
    # Target = ref rotated by a known rotation (no translation).
    r_known = R.random(1, random_state=0)
    target_np = r_known.apply(ref_np)

    ref = torch.as_tensor(ref_np, dtype=torch.float64)
    target = torch.as_tensor(target_np, dtype=torch.float64)

    q = kabsch_quaternion(ref, target)
    # Apply via geom.rotate → should recover target.
    recovered = _apply_quat(q, ref)
    diff = (recovered - target).abs().max().item()
    assert diff < 1e-8


def test_kabsch_translation_invariant():
    """Kabsch only cares about the rotation, not the translation."""
    from scipy.spatial.transform import Rotation as R
    rng = np.random.default_rng(1)
    ref_np = rng.standard_normal((30, 3))
    target_np = R.random(1, random_state=42).apply(ref_np) + np.array([7.0, -2.3, 1.5])

    ref = torch.as_tensor(ref_np, dtype=torch.float64)
    target = torch.as_tensor(target_np, dtype=torch.float64)
    shift = torch.tensor([7.0, -2.3, 1.5], dtype=torch.float64)

    q = kabsch_quaternion(ref, target)
    recovered = _apply_quat(q, ref)
    # After applying the Kabsch rotation, the COMs may differ by `shift`.
    residual = (recovered - (target - shift)).abs().max().item()
    assert residual < 1e-8


# ---------------------------------------------------------------------------
# rotation_cone
# ---------------------------------------------------------------------------


def test_rotation_cone_first_sample_is_center():
    """The first sample (θ=0) must equal q_center up to sign.
    (Quaternions q and −q represent the same rotation.)"""
    q_center = torch.tensor([0.1, -0.3, 0.5, 0.8], dtype=torch.float64)
    q_center = q_center / q_center.norm()
    q = rotation_cone(q_center, n=5, cone_deg=20.0, seed=0)
    assert torch.allclose(q[0], q_center, atol=1e-12) or \
           torch.allclose(q[0], -q_center, atol=1e-12)


def test_rotation_cone_stays_within_angle():
    """Every sample should lie within cone_deg of q_center (quaternion
    geodesic distance = 2*acos(|q_center · q|))."""
    q_center = torch.tensor([0.2, 0.5, -0.6, 0.6], dtype=torch.float64)
    q_center = q_center / q_center.norm()
    cone_deg = 25.0
    q = rotation_cone(q_center, n=100, cone_deg=cone_deg, seed=3)
    for i in range(q.shape[0]):
        ang_rad = _quat_angle(q_center, q[i])
        assert math.degrees(ang_rad) <= cone_deg + 1e-6, (
            f"sample {i}: {math.degrees(ang_rad):.2f}° > cone_deg {cone_deg}°"
        )


def test_rotation_cone_zero_cone_gives_identity():
    """cone_deg=0 ⇒ every sample is exactly q_center."""
    q_center = torch.tensor([0.1, 0.2, 0.3, 0.9273], dtype=torch.float64)
    q_center = q_center / q_center.norm()
    q = rotation_cone(q_center, n=8, cone_deg=0.0, seed=0)
    for i in range(q.shape[0]):
        close = torch.allclose(q[i], q_center, atol=1e-12) or \
                torch.allclose(q[i], -q_center, atol=1e-12)
        assert close


def test_rotation_cone_output_is_unit_quaternions():
    q_center = torch.tensor([0.3, -0.4, 0.2, 0.5], dtype=torch.float64)
    q_center = q_center / q_center.norm()
    q = rotation_cone(q_center, n=50, cone_deg=45.0, seed=1)
    norms = q.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-8)
