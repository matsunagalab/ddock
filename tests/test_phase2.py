"""Phase 2 tests: SASA and grid generation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from zdock.geom import generate_grid
from zdock.sasa import compute_sasa


def _load_xyz(ref: dict, key: str) -> np.ndarray:
    """Julia stores xyz as a (nframe, 3*natom) matrix in column-major.
    HDF5 sees (3*natom, nframe). We want a (natom, 3) view for frame 0."""
    xyz_h5 = np.asarray(ref[key])
    if xyz_h5.ndim == 2:
        # (3*natom, nframe) → (3*natom,) for frame 0
        vec = xyz_h5[:, 0]
    else:
        vec = xyz_h5
    natom3 = vec.size
    assert natom3 % 3 == 0
    natom = natom3 // 3
    # Julia interleaves as [x1, y1, z1, x2, y2, z2, ...]; reshape accordingly.
    return vec.reshape(natom, 3)


# ------------------------------------------------------------------ SASA


def test_sasa_receptor_matches_julia(load_ref, device, dtype):
    ref = load_ref("phase2", "sasa")
    xyz = torch.as_tensor(_load_xyz(ref, "receptor_xyz"), device=device, dtype=dtype)
    radius = torch.as_tensor(np.asarray(ref["receptor_radius"]), device=device, dtype=dtype)
    expected = torch.as_tensor(np.asarray(ref["receptor_sasa"]), device=device, dtype=dtype)

    got = compute_sasa(xyz, radius)  # let device-dependent default pick

    # SASA is computed by ratio-of-960-points; ±1 point gives Δsasa ≈ 4π·r²/960
    # ≈ 0.13 per atom at r=1.5. Allow up to 2 points mismatch (≈ 0.25).
    rtol = 1e-6 if dtype == torch.float64 else 1e-4
    atol = 0.3
    torch.testing.assert_close(got, expected, atol=atol, rtol=rtol)


def test_sasa_ligand_matches_julia(load_ref, device, dtype):
    ref = load_ref("phase2", "sasa")
    xyz = torch.as_tensor(_load_xyz(ref, "ligand_xyz"), device=device, dtype=dtype)
    radius = torch.as_tensor(np.asarray(ref["ligand_radius"]), device=device, dtype=dtype)
    expected = torch.as_tensor(np.asarray(ref["ligand_sasa"]), device=device, dtype=dtype)

    got = compute_sasa(xyz, radius)  # let device-dependent default pick

    rtol = 1e-6 if dtype == torch.float64 else 1e-4
    torch.testing.assert_close(got, expected, atol=0.3, rtol=rtol)


# ---------------------------------------------------------------- grid


def test_generate_grid_matches_julia(load_ref, device, dtype, tol):
    ref = load_ref("phase2", "grid")
    spacing = float(ref["spacing"])

    rec_xyz = torch.as_tensor(
        _load_xyz(ref, "receptor_xyz_prep"), device=device, dtype=dtype
    )
    lig_xyz = torch.as_tensor(
        _load_xyz(ref, "ligand_xyz_prep"), device=device, dtype=dtype
    )

    grid_real, grid_imag, x_grid, y_grid, z_grid = generate_grid(
        rec_xyz, lig_xyz, spacing=spacing
    )

    # Grid coordinate arrays should match Julia's `range(...)` step=spacing.
    torch.testing.assert_close(
        x_grid,
        torch.as_tensor(np.asarray(ref["x_grid"]), device=device, dtype=dtype),
        **tol,
    )
    torch.testing.assert_close(
        y_grid,
        torch.as_tensor(np.asarray(ref["y_grid"]), device=device, dtype=dtype),
        **tol,
    )
    torch.testing.assert_close(
        z_grid,
        torch.as_tensor(np.asarray(ref["z_grid"]), device=device, dtype=dtype),
        **tol,
    )

    expected_shape = tuple(int(s) for s in np.asarray(ref["grid_shape"]))
    assert tuple(grid_real.shape) == expected_shape
    assert tuple(grid_imag.shape) == expected_shape
    # Both grids start zeroed.
    assert torch.all(grid_real == 0)
    assert torch.all(grid_imag == 0)
