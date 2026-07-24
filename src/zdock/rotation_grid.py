"""Quaternion grid generators for the FFT docking search.

ZDOCK 3.0.2 uses a specific Euler-angle table at 6° spacing. This
module provides simpler drop-in alternatives that are adequate for
demonstration and experimentation:

  * `random_quaternions` — N uniformly-distributed SO(3) rotations
    via `scipy.spatial.transform.Rotation.random`.
  * `euler_quaternions` — (φ, θ, ψ) ZYZ Euler grid at `deg` spacing.

Exact ZDOCK-table reproduction is a future refinement (see
PORT_PLAN_FFT.md).

**Quaternion convention**: this module returns quaternions in the
format consumed by `geom.rotate` (= `docking.jl::rotate!`), which is
the *inverse* of scipy's active-rotation convention. For a uniform
sampler this is immaterial — the inverted distribution is still
uniform on SO(3). If you need to interoperate with scipy's
`Rotation.apply` outcome-by-outcome, apply the quaternion conjugate
`(x, y, z, w) → (−x, −y, −z, w)`.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.spatial.transform import Rotation as _ScipyRotation


def random_quaternions(
    n: int,
    *,
    seed: int = 0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return `(n, 4)` uniformly-distributed SO(3) quaternions via
    `scipy.spatial.transform.Rotation.random`.

    Output is in `geom.rotate` / Julia `rotate!` convention (see
    module docstring). For uniform sampling the convention choice
    has no effect on the distribution.
    """
    r = _ScipyRotation.random(n, random_state=seed)
    q = r.as_quat()  # (n, 4) scalar-last (x, y, z, w)
    q = torch.as_tensor(q, dtype=dtype)
    if device is not None:
        q = q.to(device)
    return q


def scipy_rotations_to_quaternions(
    r: _ScipyRotation,
    *,
    as_inverse: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Convert a `scipy.spatial.transform.Rotation` (single or batch)
    to `(N, 4)` quaternions for use with `geom.rotate`.

    `as_inverse=True` (default) emits the quaternion conjugate of
    scipy's output so that `geom.rotate(v, q)` produces the same
    rotated vector as `r.apply(v)`. Set to False to pass the raw
    scipy quaternions (equivalent to applying the inverse rotation
    under our convention — fine for uniform samplers but not for
    known oriented rotations).
    """
    q = r.as_quat().reshape(-1, 4)
    if as_inverse:
        q = q.copy()
        q[:, :3] *= -1  # quaternion conjugate = inverse of unit quat
    q_t = torch.as_tensor(np.ascontiguousarray(q), dtype=dtype)
    if device is not None:
        q_t = q_t.to(device)
    return q_t


def kabsch_quaternion(
    ref_xyz: torch.Tensor,              # (N, 3)
    target_xyz: torch.Tensor,           # (N, 3)
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return the quaternion that best rotates `ref_xyz` onto
    `target_xyz` in the least-squares (Kabsch) sense, expressed in
    the `geom.rotate` convention.

    Both point sets are first decentered (COM subtracted) so the
    Kabsch alignment is purely rotational. Equivalent to running
    `scipy.spatial.transform.Rotation.align_vectors(target, ref)` on
    the centered sets and then converting via
    `scipy_rotations_to_quaternions(as_inverse=True)`.

    Returns a `(4,)` quaternion. Caller still needs to separately
    supply the translation `target_COM - ref_COM` to produce the
    full pose.
    """
    if ref_xyz.shape != target_xyz.shape:
        raise ValueError(
            f"ref and target must share shape; got {tuple(ref_xyz.shape)} "
            f"vs {tuple(target_xyz.shape)}"
        )
    if ref_xyz.ndim != 2 or ref_xyz.shape[1] != 3:
        raise ValueError(f"inputs must be (N, 3), got {tuple(ref_xyz.shape)}")

    ref_np = ref_xyz.detach().cpu().numpy().astype(np.float64)
    target_np = target_xyz.detach().cpu().numpy().astype(np.float64)
    ref_c = ref_np - ref_np.mean(axis=0)
    target_c = target_np - target_np.mean(axis=0)

    # Rotation that takes ref onto target (i.e., R · ref ≈ target).
    r_align, _ = _ScipyRotation.align_vectors(target_c, ref_c)
    q = scipy_rotations_to_quaternions(
        r_align, as_inverse=True, device=device, dtype=dtype,
    )
    return q.squeeze(0)  # (4,)


def rotation_cone(
    q_center: torch.Tensor,             # (4,) in geom.rotate convention
    n: int,
    *,
    cone_deg: float = 15.0,
    seed: int = 0,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return `(n, 4)` quaternions sampled within an angular cone of
    radius `cone_deg` degrees around `q_center`.

    Sampling scheme: for each sample, draw a uniform axis on S² and an
    angle θ ~ Uniform(0, cone_deg). Compose δ(axis, θ) · q_center.

    Angle is uniform in [0, cone_deg], which overweights the shell
    near cone_deg slightly relative to a uniform volume sampling — but
    for our purpose (guarantee coverage near q_center, including the
    exact pose at θ = 0) this is fine and predictable.
    """
    if q_center.shape != (4,):
        raise ValueError(f"q_center must be (4,), got {tuple(q_center.shape)}")
    if cone_deg < 0 or cone_deg > 180:
        raise ValueError(f"cone_deg must be in [0, 180], got {cone_deg}")

    if device is None:
        device = q_center.device
    if dtype is None:
        dtype = q_center.dtype

    rng = np.random.default_rng(seed)

    # Random axis uniformly on S² via Gaussian normalization.
    v = rng.standard_normal((n, 3))
    v = v / np.linalg.norm(v, axis=1, keepdims=True)

    # Angle uniform in [0, cone_rad]. θ=0 at i=0 so the exact center
    # is guaranteed to appear — useful for sanity checks.
    cone_rad = math.radians(cone_deg)
    theta = rng.uniform(0.0, cone_rad, size=n)
    theta[0] = 0.0  # first sample is exact center

    half = theta / 2.0
    delta_xyz = v * np.sin(half)[:, None]                # (n, 3)
    delta_w = np.cos(half)                                # (n,)
    delta = np.concatenate([delta_xyz, delta_w[:, None]], axis=1)  # (n, 4)

    # Hamilton quat mul: q = delta · q_center   (both (x, y, z, w)).
    def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
        return np.stack([
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ], axis=-1)

    q_c_np = q_center.detach().cpu().numpy().astype(np.float64)
    q_out = quat_mul(delta, q_c_np)
    q_out = q_out / np.linalg.norm(q_out, axis=1, keepdims=True)

    q_t = torch.as_tensor(q_out, dtype=dtype)
    return q_t.to(device)


def euler_quaternions(
    deg: float = 15.0,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """ZYZ Euler-angle grid at `deg` spacing, converted to
    `geom.rotate`-compatible quaternions.

    Enumerates (φ, θ, ψ) with φ ∈ [0, 360), θ ∈ [0, 180], ψ ∈ [0, 360).
    Does NOT de-duplicate orientations that coincide at θ = 0/180
    (gimbal-lock); callers that care can unique-ify by the resulting
    rotation matrix. Coverage is denser near the poles — a known
    Euler limitation; `random_quaternions` avoids this.

    Returns `(R, 4)` unit quaternions in the convention used by
    `geom.rotate` (q1..q4 → Julia docking.jl rotate!).
    """
    if deg <= 0.0 or deg > 180.0:
        raise ValueError(f"deg must be in (0, 180], got {deg}")
    rad = math.radians(deg)
    # Number of samples per axis.
    n_phi = int(round(360.0 / deg))
    n_theta = int(round(180.0 / deg)) + 1     # endpoints included
    n_psi = int(round(360.0 / deg))
    phi = torch.linspace(0.0, 2.0 * math.pi, n_phi + 1, dtype=dtype)[:-1]
    theta = torch.linspace(0.0, math.pi, n_theta, dtype=dtype)
    psi = torch.linspace(0.0, 2.0 * math.pi, n_psi + 1, dtype=dtype)[:-1]

    # Cartesian product, shape (R, 3)
    ph, th, ps = torch.meshgrid(phi, theta, psi, indexing="ij")
    ph, th, ps = ph.reshape(-1), th.reshape(-1), ps.reshape(-1)

    # Convert ZYZ Euler to quaternion. Use the convention that matches
    # `geom.rotate`'s q1..q4 decomposition.
    #
    # We assemble q = q_z(ψ) · q_y(θ) · q_z(φ) where q_axis(a) is the
    # half-angle rotation around that axis, then map to (q1, q2, q3, q4)
    # = (x, y, z, w). This is standard Hamilton composition.
    def q_z(a):
        z = torch.zeros_like(a)
        return torch.stack([z, z, torch.sin(a / 2), torch.cos(a / 2)], dim=-1)

    def q_y(a):
        z = torch.zeros_like(a)
        return torch.stack([z, torch.sin(a / 2), z, torch.cos(a / 2)], dim=-1)

    def quat_mul(a, b):
        # Hamilton product, both in (x, y, z, w) convention.
        ax, ay, az, aw = a.unbind(-1)
        bx, by, bz, bw = b.unbind(-1)
        return torch.stack([
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ], dim=-1)

    q = quat_mul(q_z(ps), quat_mul(q_y(th), q_z(ph)))
    q = q / q.norm(dim=-1, keepdim=True)
    if device is not None:
        q = q.to(device)
    return q
