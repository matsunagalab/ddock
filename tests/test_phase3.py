"""Phase 3 tests: spread_* and calculate_distance against Julia reference
outputs on a small synthetic grid."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from zdock.spread import (
    calculate_distance,
    spread_nearest_add,
    spread_nearest_substitute,
    spread_neighbors_add,
    spread_neighbors_substitute,
)


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


def test_spread_nearest_add(load_ref, device, dtype, tol):
    ref = load_ref("phase3", "spread")
    xyz = _as_xyz(ref, device, dtype)
    w = torch.as_tensor(np.asarray(ref["weight"]), device=device, dtype=dtype)
    grid, xg, yg, zg = _make_grid(ref, device, dtype)
    spread_nearest_add(grid, xyz, w, xg, yg, zg)

    expected = torch.as_tensor(_load_grid(ref, "nearest_add"), device=device, dtype=dtype)
    torch.testing.assert_close(grid, expected, **tol)


def test_spread_nearest_substitute(load_ref, device, dtype, tol):
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
