"""Atom → grid spreading functions, ported from
`train_param-apart.ipynb` cell 4 (`spread_nearest_*`, `spread_neighbors_*`,
`calculate_distance!`).

All functions run as a single `scatter_add_` / `index_put_` invocation after
a vectorized index build-up — no Python loop over atoms or grid cells. This
keeps the hot path efficient on CUDA / MPS while remaining autograd-friendly
(scatter_add is differentiable w.r.t. the weights).

The five public APIs:

  * `spread_nearest_add(grid, xyz, weights, x_grid, y_grid, z_grid)`
  * `spread_nearest_substitute(grid, xyz, weights, x_grid, y_grid, z_grid)`
  * `spread_neighbors_add(grid, xyz, weights, rcut, x_grid, y_grid, z_grid)`
  * `spread_neighbors_substitute(grid, xyz, weights, rcut, x_grid, y_grid, z_grid)`
  * `calculate_distance(grid, xyz, weights, rcut, x_grid, y_grid, z_grid)`

Grid shape is `(nx, ny, nz)` — the same convention as docking.jl. Index
ordering in `scatter_add` is `flat = ix * ny * nz + iy * nz + iz`.

Index semantics match Julia's `ceil((x - x_min) / dx)` exactly: the *1-based*
Julia cell index is mapped to *0-based* Python by subtracting 1. If an atom
sits exactly on `x_min` that produces index -1; bounds check drops it.
"""

from __future__ import annotations

import math

import torch


def _nearest_cell_indices(
    xyz: torch.Tensor,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    z_grid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-atom nearest (ix, iy, iz) in 0-based Python indexing,
    matching Julia's `ceil((x - x_min)/dx)` after the 1→0 index shift."""
    dx = (x_grid[1] - x_grid[0]).item()
    dy = (y_grid[1] - y_grid[0]).item()
    dz = (z_grid[1] - z_grid[0]).item()
    x_min = x_grid[0].item()
    y_min = y_grid[0].item()
    z_min = z_grid[0].item()
    ix = torch.ceil((xyz[:, 0] - x_min) / dx).long() - 1
    iy = torch.ceil((xyz[:, 1] - y_min) / dy).long() - 1
    iz = torch.ceil((xyz[:, 2] - z_min) / dz).long() - 1
    return ix, iy, iz


def _flat_index(
    ix: torch.Tensor,
    iy: torch.Tensor,
    iz: torch.Tensor,
    shape: tuple[int, int, int],
) -> torch.Tensor:
    nx, ny, nz = shape
    return ix * (ny * nz) + iy * nz + iz


def _in_bounds(
    ix: torch.Tensor,
    iy: torch.Tensor,
    iz: torch.Tensor,
    shape: tuple[int, int, int],
) -> torch.Tensor:
    nx, ny, nz = shape
    return (
        (ix >= 0) & (ix < nx) &
        (iy >= 0) & (iy < ny) &
        (iz >= 0) & (iz < nz)
    )


# ---------------------------------------------------------------------------
# spread_nearest_*
# ---------------------------------------------------------------------------


def spread_nearest_add(
    grid: torch.Tensor,
    xyz: torch.Tensor,
    weights: torch.Tensor,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    z_grid: torch.Tensor,
) -> torch.Tensor:
    """Add `weights[i]` to the nearest grid cell of atom i, for all atoms."""
    shape = tuple(grid.shape)
    ix, iy, iz = _nearest_cell_indices(xyz, x_grid, y_grid, z_grid)
    mask = _in_bounds(ix, iy, iz, shape)
    flat = _flat_index(ix, iy, iz, shape)[mask]
    grid.view(-1).scatter_add_(0, flat, weights[mask])
    return grid


def spread_nearest_substitute(
    grid: torch.Tensor,
    xyz: torch.Tensor,
    weights: torch.Tensor,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    z_grid: torch.Tensor,
) -> torch.Tensor:
    """Overwrite the nearest grid cell of atom i with `weights[i]`.

    When multiple atoms hit the same cell the last-in-atom-order value wins
    (matching Julia's loop over atoms). We emulate this deterministically by
    processing atoms in index order via `scatter` with `reduce="amax"`-free
    fallback: we simply rely on `index_put_` with `accumulate=False`; for
    Julia-compatible ordering callers in the ZDOCK pipeline always supply
    uniform weights within a call, so order is moot.
    """
    shape = tuple(grid.shape)
    ix, iy, iz = _nearest_cell_indices(xyz, x_grid, y_grid, z_grid)
    mask = _in_bounds(ix, iy, iz, shape)
    flat = _flat_index(ix, iy, iz, shape)[mask]
    grid.view(-1).index_put_((flat,), weights[mask], accumulate=False)
    return grid


# ---------------------------------------------------------------------------
# spread_neighbors_* and calculate_distance
# ---------------------------------------------------------------------------


def _neighbors_indices(
    xyz: torch.Tensor,
    rcut: torch.Tensor | float,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    z_grid: torch.Tensor,
    shape: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Enumerate all (atom_idx, flat_grid_idx, distance) triples where the
    grid cell is within `rcut[atom]` of the atom's centre.

    Returns flat index tensor, per-pair distance tensor (for callers that
    need it, e.g. calculate_distance), and per-pair atom index tensor.
    """
    device = xyz.device
    dtype = xyz.dtype
    N = xyz.shape[0]
    nx, ny, nz = shape

    dx = (x_grid[1] - x_grid[0]).item()
    dy = (y_grid[1] - y_grid[0]).item()
    dz = (z_grid[1] - z_grid[0]).item()
    x_min = x_grid[0].item()
    y_min = y_grid[0].item()
    z_min = z_grid[0].item()

    if isinstance(rcut, (int, float)):
        rcut_t = torch.full((N,), float(rcut), device=device, dtype=dtype)
        rcut_max = float(rcut)
    else:
        rcut_t = rcut.to(dtype)
        rcut_max = float(rcut.max().item())

    # Offset box large enough to cover rcut_max along each axis.
    kx = int(math.ceil(rcut_max / dx))
    ky = int(math.ceil(rcut_max / dy))
    kz = int(math.ceil(rcut_max / dz))

    # Center cell per atom (1-based Julia → 0-based Python).
    ix0 = torch.ceil((xyz[:, 0] - x_min) / dx).long() - 1
    iy0 = torch.ceil((xyz[:, 1] - y_min) / dy).long() - 1
    iz0 = torch.ceil((xyz[:, 2] - z_min) / dz).long() - 1

    ox = torch.arange(-kx, kx + 1, device=device)
    oy = torch.arange(-ky, ky + 1, device=device)
    oz = torch.arange(-kz, kz + 1, device=device)
    ox_3d, oy_3d, oz_3d = torch.meshgrid(ox, oy, oz, indexing="ij")
    ox_flat = ox_3d.reshape(-1)
    oy_flat = oy_3d.reshape(-1)
    oz_flat = oz_3d.reshape(-1)
    K = ox_flat.numel()

    # Candidate indices: (N, K)
    ix_cand = ix0.unsqueeze(-1) + ox_flat.unsqueeze(0)
    iy_cand = iy0.unsqueeze(-1) + oy_flat.unsqueeze(0)
    iz_cand = iz0.unsqueeze(-1) + oz_flat.unsqueeze(0)

    in_b = (
        (ix_cand >= 0) & (ix_cand < nx) &
        (iy_cand >= 0) & (iy_cand < ny) &
        (iz_cand >= 0) & (iz_cand < nz)
    )

    # Distances using grid coordinate arrays (safe indexing with clamp+mask).
    ix_safe = ix_cand.clamp(0, nx - 1)
    iy_safe = iy_cand.clamp(0, ny - 1)
    iz_safe = iz_cand.clamp(0, nz - 1)
    gx = x_grid[ix_safe]
    gy = y_grid[iy_safe]
    gz = z_grid[iz_safe]

    dx_v = xyz[:, 0].unsqueeze(-1) - gx
    dy_v = xyz[:, 1].unsqueeze(-1) - gy
    dz_v = xyz[:, 2].unsqueeze(-1) - gz
    d2 = dx_v.pow(2) + dy_v.pow(2) + dz_v.pow(2)

    rcut_sq = (rcut_t.pow(2)).unsqueeze(-1)
    within = (d2 < rcut_sq) & in_b

    # Flatten (N, K) → 1D list of valid (atom, flat_grid) pairs.
    flat = ix_safe * (ny * nz) + iy_safe * nz + iz_safe
    atom_idx = torch.arange(N, device=device).unsqueeze(-1).expand(-1, K)
    d = d2.sqrt()

    return flat[within], d[within], atom_idx[within]


def spread_neighbors_add(
    grid: torch.Tensor,
    xyz: torch.Tensor,
    weights: torch.Tensor,
    rcut: torch.Tensor | float,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    z_grid: torch.Tensor,
) -> torch.Tensor:
    """Add `weights[atom]` to every grid cell whose centre lies strictly
    inside `rcut[atom]` of atom's position."""
    shape = tuple(grid.shape)
    flat, _, atom_idx = _neighbors_indices(xyz, rcut, x_grid, y_grid, z_grid, shape)
    contributions = weights[atom_idx]
    grid.view(-1).scatter_add_(0, flat, contributions)
    return grid


def spread_neighbors_substitute(
    grid: torch.Tensor,
    xyz: torch.Tensor,
    weights: torch.Tensor,
    rcut: torch.Tensor | float,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    z_grid: torch.Tensor,
) -> torch.Tensor:
    """Overwrite grid cells within `rcut[atom]` of any atom with `weights`.

    Caller semantics: the Julia ZDOCK callers always pass uniform weights
    within a single call (e.g. 1.0 in `assign_sc_*_plus!`, 3.5/12.25 in
    `_minus!`), so collisions are harmless. If multiple atoms contribute to
    the same cell with *different* weights, the result is non-deterministic
    (matches PyTorch `index_put_` with `accumulate=False`).
    """
    shape = tuple(grid.shape)
    flat, _, atom_idx = _neighbors_indices(xyz, rcut, x_grid, y_grid, z_grid, shape)
    contributions = weights[atom_idx]
    grid.view(-1).index_put_((flat,), contributions, accumulate=False)
    return grid


def spread_neighbors_coulomb(
    grid: torch.Tensor,
    xyz: torch.Tensor,
    charges: torch.Tensor,
    rcut: torch.Tensor | float,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    z_grid: torch.Tensor,
    *,
    d_floor: float = 1e-6,
) -> torch.Tensor:
    """Physically-correct Coulomb-style potential spread: each cell within
    rcut of atom i receives `charges[i] / d(i, cell)` where d is the
    Euclidean distance.

    This is NOT what docking.jl / the training notebook does — the
    notebook computes `Σ q / Σ d` (see B4 in PORT_PLAN.md). Use this
    function if you want the physically-correct ELEC term instead of the
    thesis-faithful version."""
    shape = tuple(grid.shape)
    flat, d, atom_idx = _neighbors_indices(xyz, rcut, x_grid, y_grid, z_grid, shape)
    contrib = charges[atom_idx] / d.clamp(min=d_floor)
    grid.view(-1).scatter_add_(0, flat, contrib)
    return grid


def calculate_distance(
    grid: torch.Tensor,
    xyz: torch.Tensor,
    weights: torch.Tensor,
    rcut: torch.Tensor | float,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    z_grid: torch.Tensor,
) -> torch.Tensor:
    """Accumulate `sqrt(d_atom_to_cell)` into grid cells within `rcut[atom]`.
    `weights` is accepted (for API parity with Julia `calculate_distance!`)
    but ignored — matching the Julia implementation which never uses it."""
    del weights  # intentionally unused
    shape = tuple(grid.shape)
    flat, d, _ = _neighbors_indices(xyz, rcut, x_grid, y_grid, z_grid, shape)
    grid.view(-1).scatter_add_(0, flat, d)
    return grid
