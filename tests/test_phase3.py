"""Phase 3 tests: spread_* and calculate_distance against Julia reference
outputs on a small synthetic grid."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from zdock import spread as _spread
from zdock.spread import (
    _nearest_cell_indices,
    calculate_distance,
    spread_nearest_add,
    spread_nearest_substitute,
    spread_neighbors_add,
    spread_neighbors_substitute,
)


@pytest.fixture
def legacy_floor_binning(monkeypatch):
    """Restore the Julia reference's `ceil((x - x_min)/dx) - 1` binning.

    The production code now assigns each atom to its NEAREST grid point; the
    Julia binning was a floor, i.e. a systematic -h/2 offset per axis (see
    `_nearest_cell_indices`). The stored reference grids encode the old
    behaviour, so the two tests below pin the *legacy path* — they verify the
    port is still faithful, not that the current default is correct. The
    current default is covered by `test_nearest_cell_is_actually_nearest`.
    """
    monkeypatch.setattr(_spread, "_LEGACY_FLOOR_BINNING", True)


def test_nearest_cell_is_actually_nearest(device, dtype):
    """Every atom lands on the closest grid point, and the assignment is
    symmetric about each grid point (the old floor binning was not)."""
    h = 1.2
    g = torch.arange(10, device=device, dtype=dtype) * h - 5.0   # -5.0 .. 5.8
    # points at, just below and just above grid node index 3 (x = -1.4)
    node = float(g[3])
    xs = [node - 0.59, node - 0.01, node, node + 0.01, node + 0.59]
    xyz = torch.tensor([[x, node, node] for x in xs], device=device, dtype=dtype)
    ix, iy, iz = _nearest_cell_indices(xyz, g, g, g)
    assert ix.tolist() == [3, 3, 3, 3, 3], ix.tolist()
    assert iy.tolist() == [3] * 5 and iz.tolist() == [3] * 5
    # and one cell over on either side
    xyz2 = torch.tensor([[node - 0.61, node, node],
                         [node + 0.61, node, node]], device=device, dtype=dtype)
    ix2, _, _ = _nearest_cell_indices(xyz2, g, g, g)
    assert ix2.tolist() == [2, 4], ix2.tolist()


def _load_grid(ref: dict, key: str) -> np.ndarray:
    """HDF5 sees Julia's (nx, ny, nz) 3D array as (nz, ny, nx) due to
    column-major → row-major flip. Transpose to (nx, ny, nz)."""
    arr = np.asarray(ref[key])
    if arr.ndim == 3:
        arr = arr.transpose(2, 1, 0)
    return arr


def _as_xyz(ref: dict, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    x = torch.as_tensor(np.asarray(ref["x"]), device=device, dtype=dtype)
    y = torch.as_tensor(np.asarray(ref["y"]), device=device, dtype=dtype)
    z = torch.as_tensor(np.asarray(ref["z"]), device=device, dtype=dtype)
    return torch.stack([x, y, z], dim=1)


def _make_grid(ref: dict, device: torch.device, dtype: torch.dtype):
    xg = torch.as_tensor(np.asarray(ref["x_grid"]), device=device, dtype=dtype)
    yg = torch.as_tensor(np.asarray(ref["y_grid"]), device=device, dtype=dtype)
    zg = torch.as_tensor(np.asarray(ref["z_grid"]), device=device, dtype=dtype)
    nx, ny, nz = xg.numel(), yg.numel(), zg.numel()
    grid = torch.zeros((nx, ny, nz), device=device, dtype=dtype)
    return grid, xg, yg, zg


def test_spread_nearest_add_legacy(load_ref, device, dtype, tol,
                                   legacy_floor_binning):
    ref = load_ref("phase3", "spread")
    xyz = _as_xyz(ref, device, dtype)
    w = torch.as_tensor(np.asarray(ref["weight"]), device=device, dtype=dtype)
    grid, xg, yg, zg = _make_grid(ref, device, dtype)
    spread_nearest_add(grid, xyz, w, xg, yg, zg)

    expected = torch.as_tensor(_load_grid(ref, "nearest_add"), device=device, dtype=dtype)
    torch.testing.assert_close(grid, expected, **tol)


def test_spread_nearest_substitute_legacy(load_ref, device, dtype, tol,
                                          legacy_floor_binning):
    ref = load_ref("phase3", "spread")
    xyz = _as_xyz(ref, device, dtype)
    w = torch.as_tensor(np.asarray(ref["weight"]), device=device, dtype=dtype)
    grid, xg, yg, zg = _make_grid(ref, device, dtype)
    spread_nearest_substitute(grid, xyz, w, xg, yg, zg)

    expected = torch.as_tensor(_load_grid(ref, "nearest_sub"), device=device, dtype=dtype)
    torch.testing.assert_close(grid, expected, **tol)


def test_spread_neighbors_add(load_ref, device, dtype, tol):
    ref = load_ref("phase3", "spread")
    xyz = _as_xyz(ref, device, dtype)
    w = torch.as_tensor(np.asarray(ref["weight"]), device=device, dtype=dtype)
    rcut = torch.as_tensor(np.asarray(ref["rcut"]), device=device, dtype=dtype)
    grid, xg, yg, zg = _make_grid(ref, device, dtype)
    spread_neighbors_add(grid, xyz, w, rcut, xg, yg, zg)

    expected = torch.as_tensor(_load_grid(ref, "neigh_add"), device=device, dtype=dtype)
    torch.testing.assert_close(grid, expected, **tol)


def test_spread_neighbors_substitute(load_ref, device, dtype, tol):
    ref = load_ref("phase3", "spread")
    xyz = _as_xyz(ref, device, dtype)
    # Uniform weight — required for deterministic behaviour with
    # index_put_(accumulate=False) when multiple atoms map to the same
    # cell. Production SC assigns (the only users of this op) always
    # pass uniform weights within a single call.
    uniform = float(np.asarray(ref["neigh_sub_weight"]))
    w = torch.full((xyz.shape[0],), uniform, device=device, dtype=dtype)
    rcut = torch.as_tensor(np.asarray(ref["rcut"]), device=device, dtype=dtype)
    grid, xg, yg, zg = _make_grid(ref, device, dtype)
    spread_neighbors_substitute(grid, xyz, w, rcut, xg, yg, zg)

    expected = torch.as_tensor(_load_grid(ref, "neigh_sub"), device=device, dtype=dtype)
    torch.testing.assert_close(grid, expected, **tol)


def test_calculate_distance(load_ref, device, dtype, tol):
    ref = load_ref("phase3", "spread")
    xyz = _as_xyz(ref, device, dtype)
    w = torch.as_tensor(np.asarray(ref["weight"]), device=device, dtype=dtype)
    rcut = torch.as_tensor(np.asarray(ref["rcut"]), device=device, dtype=dtype)
    grid, xg, yg, zg = _make_grid(ref, device, dtype)
    calculate_distance(grid, xyz, w, rcut, xg, yg, zg)

    expected = torch.as_tensor(_load_grid(ref, "calc_dist"), device=device, dtype=dtype)
    torch.testing.assert_close(grid, expected, **tol)
