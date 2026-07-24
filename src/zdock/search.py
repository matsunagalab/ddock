"""FFT-based docking search — pose generation side of the PyTorch ZDOCK
pipeline.

For each quaternion in a caller-supplied rotation grid, we scatter the
rotated ligand onto a complex SC grid, FFT, multiply with the receptor's
precomputed FFT, inverse-FFT, and extract top-N translation peaks. All
translations for a fixed rotation are evaluated in a single FFT pair.

Convention (audited — see `PORT_PLAN_FFT.md`):

    For real grids:
        ifft(fft(R) · conj(fft(L)))[t] = Σ_m R[m] · L[m − t]
    i.e. peak at t = t0 means "the ligand is best placed at translation
    +t0 from its current position." The decoder ADDs t0 to the ligand
    coordinates to produce the docked pose.

    For the complex SC grid Z_R = R_real + i·R_imag, Z_L = L_real +
    i·L_imag, the docking_score_elec's SC term at a fixed pose is
        Σ_cell [real(Z_R · Z_L) − imag(Z_R · Z_L)]
    The FFT generalisation over translations uses the complex
    cross-correlation G = ifft(fft(Z_R) · conj(fft(conj(Z_L)))), which
    for any complex Z_L equals Σ_m Z_R[m] · Z_L[m − t] (no conjugation
    of Z_L inside the sum — the outer conj+conj cancels for this
    purpose). Then score_sc(t) = real(G[t]) − imag(G[t]).

This file implements Phase 1 of `PORT_PLAN_FFT.md`: SC term only. DS,
IFACE, ELEC follow in Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import math

from .atomtypes import iface_ij, partial_charge_per_atom
from .geom import decenter, generate_grid, orient
from .score import (
    _assign_sc_plus,
    _assign_sc_minus,
    _grouped_spread_nearest_add,
    _grouped_spread_neighbors_add,
    docking_score_elec,
)
from .spread import spread_neighbors_coulomb


@dataclass
class DockingResultSC:
    """Phase-1 result: SC-only scores and decoded poses.

    Attributes:
        scores: (ntop,) float, sorted descending (highest SC first).
        quat_indices: (ntop,) int64, index into the caller's quaternion
            grid that produced each top pose.
        translations: (ntop, 3) float, cartesian ligand translation in
            Å to apply on top of the rotation. Decoding: the docked
            ligand coordinates are ``rotate(ref_lig_xyz, q) +
            translations[i]``.
    """
    scores: torch.Tensor
    quat_indices: torch.Tensor
    translations: torch.Tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rotate_batch(lig_xyz: torch.Tensor, quaternions: torch.Tensor) -> torch.Tensor:
    """Rotate an (N, 3) set of ligand coords by each of B quaternions.

    Uses the same rotation-matrix entries as `geom.rotate` (which is in
    turn verbatim from `docking.jl` lines 1469–1490) so poses agree with
    the existing `docking_score_elec` pipeline.

    Returns (B, N, 3).
    """
    q = quaternions
    q1, q2, q3, q4 = q.unbind(-1)

    r1 = 1.0 - 2.0 * q2 * q2 - 2.0 * q3 * q3
    r2 = 2.0 * (q1 * q2 + q3 * q4)
    r3 = 2.0 * (q1 * q3 - q2 * q4)
    r4 = 2.0 * (q1 * q2 - q3 * q4)
    r5 = 1.0 - 2.0 * q1 * q1 - 2.0 * q3 * q3
    r6 = 2.0 * (q2 * q3 + q1 * q4)
    r7 = 2.0 * (q1 * q3 + q2 * q4)
    r8 = 2.0 * (q2 * q3 - q1 * q4)
    r9 = 1.0 - 2.0 * q1 * q1 - 2.0 * q2 * q2

    # (B, 3, 3)
    R_mat = torch.stack(
        [torch.stack([r1, r2, r3], dim=-1),
         torch.stack([r4, r5, r6], dim=-1),
         torch.stack([r7, r8, r9], dim=-1)],
        dim=-2,
    )
    # lig_xyz: (N, 3) → broadcast
    # (B, 3, 3) @ (3, N) → (B, 3, N) → transpose → (B, N, 3)
    return torch.einsum("bij,nj->bni", R_mat, lig_xyz)


def _build_receptor_sc_grids(
    rec_xyz: torch.Tensor,
    rec_radius: torch.Tensor,
    rec_sasa: torch.Tensor,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    z_grid: torch.Tensor,
    *,
    surface_threshold: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter receptor atoms into the SC real/imag grids."""
    nx, ny, nz = x_grid.numel(), y_grid.numel(), z_grid.numel()
    device = rec_xyz.device
    dtype = rec_xyz.dtype
    R_real = torch.zeros((nx, ny, nz), device=device, dtype=dtype)
    R_imag = torch.zeros((nx, ny, nz), device=device, dtype=dtype)
    rec_surf = rec_sasa > surface_threshold
    _assign_sc_plus(R_real, rec_xyz, rec_radius, rec_surf,
                    x_grid, y_grid, z_grid, receptor=True)
    _assign_sc_minus(R_imag, rec_xyz, rec_radius, rec_surf,
                     x_grid, y_grid, z_grid, receptor=True)
    return R_real, R_imag


def _build_ligand_sc_grid_single(
    lig_xyz: torch.Tensor,         # (N, 3)
    lig_radius: torch.Tensor,      # (N,)
    lig_surf: torch.Tensor,        # (N,) bool
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    z_grid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replicate `docking_score_elec`'s ligand SC construction exactly
    (score.py lines 281–313). Produces the real and imag SC grids for
    one ligand pose. Key differences from `_assign_sc_minus(receptor=False)`:

        * Surface atoms contribute 1.0 to the real grid and 3.5 to the
          imag grid (not 12.25).
        * Core atoms contribute 1.0 to real and 12.25 to imag.
        * Core overwrites surface in imag (core takes precedence where
          both cover a cell).
        * Layer boundaries come from (cnt > 0) indicator unions, not
          substitute writes.

    Matching this exactly is required for Phase 1 V-SC parity against
    `docking_score_elec`.
    """
    nx, ny, nz = x_grid.numel(), y_grid.numel(), z_grid.numel()
    device = lig_xyz.device
    dtype = lig_xyz.dtype
    core = ~lig_surf

    def sc_union(xyz_sub, rcut_sub):
        """Binary indicator of cells within rcut_sub of any atom in xyz_sub."""
        cnt = torch.zeros((1, nx, ny, nz), device=device, dtype=dtype)
        if xyz_sub.shape[0] == 0:
            return cnt[0]
        group = torch.zeros(xyz_sub.shape[0], device=device, dtype=torch.long)
        weights = torch.ones(xyz_sub.shape[0], device=device, dtype=dtype)
        _grouped_spread_neighbors_add(
            cnt, xyz_sub, group, weights, rcut_sub, x_grid, y_grid, z_grid,
        )
        return (cnt[0] > 0).to(dtype)

    L_real = torch.zeros((nx, ny, nz), device=device, dtype=dtype)
    L_imag = torch.zeros((nx, ny, nz), device=device, dtype=dtype)

    surf_idx = lig_surf.nonzero(as_tuple=True)[0]
    core_idx = core.nonzero(as_tuple=True)[0]

    if surf_idx.numel() > 0:
        lay1 = sc_union(lig_xyz[surf_idx], lig_radius[surf_idx])
        L_real = torch.maximum(L_real, lay1)
    if core_idx.numel() > 0:
        lay2 = sc_union(
            lig_xyz[core_idx], lig_radius[core_idx] * math.sqrt(1.5),
        )
        L_real = torch.maximum(L_real, lay2)

    if surf_idx.numel() > 0:
        lay1 = sc_union(lig_xyz[surf_idx], lig_radius[surf_idx])
        L_imag = torch.where(lay1 > 0, lay1 * 3.5, L_imag)
    if core_idx.numel() > 0:
        lay2 = sc_union(
            lig_xyz[core_idx], lig_radius[core_idx] * math.sqrt(1.5),
        )
        L_imag = torch.where(lay2 > 0, lay2 * 12.25, L_imag)

    return L_real, L_imag


def _build_ligand_sc_grids_batch(
    lig_xyz_rot: torch.Tensor,   # (B, N, 3)
    lig_radius: torch.Tensor,
    lig_sasa: torch.Tensor,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    z_grid: torch.Tensor,
    *,
    surface_threshold: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched ligand SC grid construction — see
    `_build_ligand_sc_grids_vectorised` for the fast path used by
    `docking_search`. This wrapper is kept as a Phase-1 reference
    (loop over B) for bisection if the vectorised version is
    suspected of introducing numerical differences.

    Returns (L_real, L_imag), each (B, nx, ny, nz).
    """
    B = lig_xyz_rot.shape[0]
    nx, ny, nz = x_grid.numel(), y_grid.numel(), z_grid.numel()
    device = lig_xyz_rot.device
    dtype = lig_xyz_rot.dtype
    L_real = torch.zeros((B, nx, ny, nz), device=device, dtype=dtype)
    L_imag = torch.zeros((B, nx, ny, nz), device=device, dtype=dtype)
    lig_surf = lig_sasa > surface_threshold
    for b in range(B):
        Lr, Li = _build_ligand_sc_grid_single(
            lig_xyz_rot[b], lig_radius, lig_surf, x_grid, y_grid, z_grid,
        )
        L_real[b] = Lr
        L_imag[b] = Li
    return L_real, L_imag


# ---------------------------------------------------------------------------
# Vectorised per-rotation grid construction (GPU perf path for Phase 3b).
#
# Collapses the rotation-chunk Python loop into a single flat-atom scatter
# per physical quantity, using frame-compound group indices. Matches
# `_score_ligand_chunk`'s pattern in score.py.
# ---------------------------------------------------------------------------


def _build_ligand_sc_grids_vectorised(
    lig_xyz_rot: torch.Tensor,   # (B, N, 3)
    lig_radius: torch.Tensor,    # (N,)
    lig_surf: torch.Tensor,      # (N,) bool
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    z_grid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorised SC grid construction across a rotation batch.

    Uses flat atom arrays with per-atom frame index, so the two (surface
    and core) layer scatters each run as a single `_grouped_spread_*`
    call regardless of B — no Python-side iteration over rotations.
    Output is bit-identical to `_build_ligand_sc_grid_single` applied
    per rotation.
    """
    B, N_lig, _ = lig_xyz_rot.shape
    nx, ny, nz = x_grid.numel(), y_grid.numel(), z_grid.numel()
    device = lig_xyz_rot.device
    dtype = lig_xyz_rot.dtype

    xyz_flat = lig_xyz_rot.reshape(-1, 3)
    frame_idx = torch.arange(B, device=device).repeat_interleave(N_lig)
    radius_flat = lig_radius.unsqueeze(0).expand(B, -1).reshape(-1)
    surf_flat = lig_surf.unsqueeze(0).expand(B, -1).reshape(-1)
    core_flat = ~surf_flat

    def sc_union(mask: torch.Tensor, rcut_scale: float) -> torch.Tensor:
        idx = mask.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return torch.zeros((B, nx, ny, nz), device=device, dtype=dtype)
        cnt = torch.zeros((B, nx, ny, nz), device=device, dtype=dtype)
        _grouped_spread_neighbors_add(
            cnt,
            xyz_flat[idx],
            frame_idx[idx],
            torch.ones(idx.numel(), device=device, dtype=dtype),
            radius_flat[idx] * rcut_scale,
            x_grid, y_grid, z_grid,
        )
        return (cnt > 0).to(dtype)

    surf_layer = sc_union(surf_flat, 1.0)
    core_layer = sc_union(core_flat, math.sqrt(1.5))

    L_real = torch.maximum(surf_layer, core_layer)
    # L_imag: core (12.25) overwrites surface (3.5) where both are set
    # — replicates the `torch.where(lay2 > 0, ...)` overwrite pattern in
    # `docking_score_elec`.
    L_imag = torch.where(
        surf_layer > 0, surf_layer * 3.5,
        torch.zeros_like(surf_layer),
    )
    L_imag = torch.where(core_layer > 0, core_layer * 12.25, L_imag)
    return L_real, L_imag


def _build_ligand_iface_grids_vectorised(
    lig_xyz_rot: torch.Tensor,          # (B, N, 3)
    lig_atomtype_id: torch.Tensor,      # (N,) int
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    z_grid: torch.Tensor,
) -> torch.Tensor:                       # (B, 12, nx, ny, nz)
    """Vectorised ligand IFACE binary indicator grids. Compound group
    = ``frame * 12 + atomtype_id−1`` so a single scatter populates the
    (B × 12) slabs simultaneously.
    """
    B, N_lig, _ = lig_xyz_rot.shape
    nx, ny, nz = x_grid.numel(), y_grid.numel(), z_grid.numel()
    device = lig_xyz_rot.device
    dtype = lig_xyz_rot.dtype

    xyz_flat = lig_xyz_rot.reshape(-1, 3)
    frame_idx = torch.arange(B, device=device).repeat_interleave(N_lig)
    type_group = (lig_atomtype_id - 1).to(torch.long).clamp(0, 11)
    group = frame_idx * 12 + type_group.unsqueeze(0).expand(B, -1).reshape(-1)

    count = torch.zeros((B * 12, nx, ny, nz), device=device, dtype=dtype)
    _grouped_spread_nearest_add(
        count, xyz_flat, group,
        torch.ones(xyz_flat.shape[0], device=device, dtype=dtype),
        x_grid, y_grid, z_grid,
    )
    return (count.view(B, 12, nx, ny, nz) > 0).to(dtype)


def _build_ligand_elec_grids_vectorised(
    lig_xyz_rot: torch.Tensor,          # (B, N, 3)
    lig_charge_id: torch.Tensor,        # (N,) int
    charge_score_lut: torch.Tensor,     # (11,)
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    z_grid: torch.Tensor,
) -> torch.Tensor:                       # (B, nx, ny, nz)
    """Vectorised ligand ELEC partial-charge grid. Single scatter with
    frame_idx group, weights = partial charge per atom broadcast over
    the batch.
    """
    B, N_lig, _ = lig_xyz_rot.shape
    nx, ny, nz = x_grid.numel(), y_grid.numel(), z_grid.numel()
    device = lig_xyz_rot.device
    dtype = lig_xyz_rot.dtype

    xyz_flat = lig_xyz_rot.reshape(-1, 3)
    frame_idx = torch.arange(B, device=device).repeat_interleave(N_lig)
    lig_q = partial_charge_per_atom(lig_charge_id, charge_score_lut)
    weights_flat = lig_q.unsqueeze(0).expand(B, -1).reshape(-1)

    grid = torch.zeros((B, nx, ny, nz), device=device, dtype=dtype)
    _grouped_spread_nearest_add(
        grid, xyz_flat, frame_idx, weights_flat,
        x_grid, y_grid, z_grid,
    )
    return grid


def _unshift_cyclic(idx: torch.Tensor, N: int) -> torch.Tensor:
    """Convert 0-indexed DFT bin to signed translation offset.

    DFT output at bin k represents translation:
        +k cells    if k <= N/2
        k − N cells if k >  N/2
    """
    half = N // 2
    return torch.where(idx <= half, idx, idx - N)


# ---------------------------------------------------------------------------
# IFACE term — 12 atom types × 12, precompute weighted receptor, per-rot
# binary ligand scatter.
# ---------------------------------------------------------------------------


def _build_receptor_iface_weighted_grids(
    rec_xyz: torch.Tensor,
    rec_atomtype_id: torch.Tensor,      # (N_rec,) int in [1, 12]
    iface_matrix: torch.Tensor,         # (12, 12)
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    z_grid: torch.Tensor,
    *,
    rcut_iface: float = 6.0,
) -> torch.Tensor:
    """Precompute ``W_i[cell] = Σ_j iface_ij[i, j] · H_j[cell]`` for
    i = 0..11 (= ligand atom type − 1). Returns (12, nx, ny, nz).

    ``H_j[cell]`` is the count of type-j receptor atoms within
    ``rcut_iface`` of that cell — matches the ``H`` tensor built inline
    in ``docking_score_elec``. Applying ``iface_matrix`` on the
    receptor side via linearity lets the per-rotation FFT product be
    just (F_L_i · F_W_i), summed over i after IFFT.
    """
    nx, ny, nz = x_grid.numel(), y_grid.numel(), z_grid.numel()
    device = rec_xyz.device
    dtype = rec_xyz.dtype
    H = torch.zeros((12, nx, ny, nz), device=device, dtype=dtype)
    rec_group = (rec_atomtype_id - 1).to(torch.long).clamp(0, 11)
    ones = torch.ones(rec_xyz.shape[0], device=device, dtype=dtype)
    _grouped_spread_neighbors_add(
        H, rec_xyz, rec_group, ones, rcut_iface,
        x_grid, y_grid, z_grid,
    )
    # W[i] = Σ_j iface_matrix[i, j] · H[j]
    W = torch.einsum("ij,jxyz->ixyz", iface_matrix, H)
    return W


def _build_ligand_iface_grid_single(
    lig_xyz: torch.Tensor,              # (N_lig, 3)
    lig_atomtype_id: torch.Tensor,      # (N_lig,) int in [1, 12]
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    z_grid: torch.Tensor,
) -> torch.Tensor:
    """Replicate ``docking_score_elec``'s ligand IFACE construction
    (score.py lines 319-326): nearest-cell scatter grouped by atom
    type, then ``(count > 0)`` binary indicator. Returns (12, nx, ny, nz).
    """
    nx, ny, nz = x_grid.numel(), y_grid.numel(), z_grid.numel()
    device = lig_xyz.device
    dtype = lig_xyz.dtype
    count = torch.zeros((12, nx, ny, nz), device=device, dtype=dtype)
    lig_group = (lig_atomtype_id - 1).to(torch.long).clamp(0, 11)
    ones = torch.ones(lig_xyz.shape[0], device=device, dtype=dtype)
    _grouped_spread_nearest_add(
        count, lig_xyz, lig_group, ones, x_grid, y_grid, z_grid,
    )
    return (count > 0).to(dtype)


# ---------------------------------------------------------------------------
# ELEC term (Coulomb mode) — receptor V grid (with core zeroing), per-rot
# ligand charge grid.
# ---------------------------------------------------------------------------


def _build_receptor_elec_grid(
    rec_xyz: torch.Tensor,
    rec_charge_id: torch.Tensor,
    charge_score_lut: torch.Tensor,     # (11,) learnable charge LUT
    R_real: torch.Tensor,               # SC real grid for core mask
    R_imag: torch.Tensor,               # SC imag grid for core mask
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    z_grid: torch.Tensor,
    *,
    rcut_elec: float = 8.0,
) -> torch.Tensor:
    """Coulomb-mode receptor potential grid matching
    ``docking_score_elec`` (score.py lines 454-467). V[cell] = Σ_j
    q_j / |cell − r_j|, zeroed inside the receptor SC shape (Chen 2002
    §2.2). Returns (nx, ny, nz) real.
    """
    nx, ny, nz = x_grid.numel(), y_grid.numel(), z_grid.numel()
    device = rec_xyz.device
    dtype = rec_xyz.dtype
    rec_q = partial_charge_per_atom(rec_charge_id, charge_score_lut)
    V = torch.zeros((nx, ny, nz), device=device, dtype=dtype)
    spread_neighbors_coulomb(
        V, rec_xyz, rec_q, rcut_elec, x_grid, y_grid, z_grid,
    )
    open_mask = ((R_real == 0) & (R_imag == 0)).to(dtype)
    return V * open_mask


def _build_ligand_elec_grid_single(
    lig_xyz: torch.Tensor,
    lig_charge_id: torch.Tensor,
    charge_score_lut: torch.Tensor,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    z_grid: torch.Tensor,
) -> torch.Tensor:
    """Ligand partial-charge grid: nearest-cell scatter of each atom's
    partial charge q_i. Matches ``docking_score_elec`` (score.py lines
    330-340). Returns (nx, ny, nz) real.
    """
    nx, ny, nz = x_grid.numel(), y_grid.numel(), z_grid.numel()
    device = lig_xyz.device
    dtype = lig_xyz.dtype
    lig_q = partial_charge_per_atom(lig_charge_id, charge_score_lut)
    # _grouped_spread_nearest_add requires grouping; use a single-group
    # (group=0) view and collect into a 1-slab grid.
    grid = torch.zeros((1, nx, ny, nz), device=device, dtype=dtype)
    group = torch.zeros(lig_xyz.shape[0], device=device, dtype=torch.long)
    _grouped_spread_nearest_add(
        grid, lig_xyz, group, lig_q, x_grid, y_grid, z_grid,
    )
    return grid[0]


# ---------------------------------------------------------------------------
# Direct (non-FFT) per-term scores for tests. O(V²) per translation, so
# only usable on tiny grids. One function per term plus a full-stack
# composer, all producing the same (nx, ny, nz) cyclic-DFT indexed grid
# that docking_search_* returns.
# ---------------------------------------------------------------------------


def docking_score_iface_direct(
    W_i: torch.Tensor,              # (12, nx, ny, nz)
    L_i: torch.Tensor,              # (12, nx, ny, nz)
) -> torch.Tensor:
    """Naive IFACE cross-correlation: `Σ_i Σ_cell W_i[cell] · L_i[cell − t]`
    for every translation t. O(V² · 12) per translation — tiny grids only.
    """
    nx, ny, nz = W_i.shape[-3:]
    out = torch.zeros((nx, ny, nz), device=W_i.device, dtype=W_i.dtype)
    for tx in range(nx):
        for ty in range(ny):
            for tz in range(nz):
                Ls = torch.roll(L_i, shifts=(tx, ty, tz), dims=(-3, -2, -1))
                out[tx, ty, tz] = (W_i * Ls).sum()
    return out


def docking_score_elec_direct(
    V_rec: torch.Tensor,            # (nx, ny, nz)
    Q_L: torch.Tensor,              # (nx, ny, nz)
) -> torch.Tensor:
    """Naive ELEC cross-correlation: `Σ_cell V_rec[cell] · Q_L[cell − t]`."""
    nx, ny, nz = V_rec.shape
    out = torch.zeros_like(V_rec)
    for tx in range(nx):
        for ty in range(ny):
            for tz in range(nz):
                Qs = torch.roll(Q_L, shifts=(tx, ty, tz), dims=(0, 1, 2))
                out[tx, ty, tz] = (V_rec * Qs).sum()
    return out


# ---------------------------------------------------------------------------
# Unified preprocessing — match docking_score_elec's input preconditions.
# ---------------------------------------------------------------------------


def prepare_receptor(
    rec_xyz: torch.Tensor,
    *,
    mass: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decenter receptor coordinates — that is what both
    `docking_score_elec` and `docking_search` expect. No PCA-orient on
    the receptor side (matches `docking.jl::docking()` and
    `docking_score_elec`'s preconditions)."""
    return decenter(rec_xyz, mass=mass)


def prepare_ligand(
    lig_xyz: torch.Tensor,
    lig_atomtype_id: torch.Tensor,
    *,
    iface_matrix: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decenter + PCA-orient ligand coordinates, using the iface-based
    mass weights that `docking_score_elec` applies internally.

    Replicates `docking_score_elec`'s line 425-427:

        iface_matrix_for_mass = iface_ij(device=device, dtype=dtype)
        lig_mass_weights = iface_matrix_for_mass[lig_atomtype_id - 1, 0]
        grid_bounds_lig = orient(lig_xyz[0], mass=lig_mass_weights)

    The returned coords are ready to feed into `docking_search` as
    `lig_xyz_ref` and into `docking_score_elec` as `lig_xyz_for_grid`.
    """
    if lig_xyz.ndim != 2 or lig_xyz.shape[1] != 3:
        raise ValueError(
            f"lig_xyz must be (N_lig, 3), got {tuple(lig_xyz.shape)}"
        )
    if iface_matrix is None:
        iface_matrix = iface_ij(
            device=lig_xyz.device, dtype=lig_xyz.dtype,
        )
    mass_weights = iface_matrix[lig_atomtype_id - 1, 0]
    return orient(lig_xyz, mass=mass_weights)


def prepare_search_inputs(
    rec_xyz: torch.Tensor,
    lig_xyz: torch.Tensor,
    lig_atomtype_id: torch.Tensor,
    *,
    iface_matrix: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-shot preparation: ``(rec_xyz_decentered, lig_xyz_ref_ready)``.

    Input may be raw PDB-extracted coordinates. Output is in the form
    `docking_search` and `docking_score_elec` both expect.
    """
    return (
        prepare_receptor(rec_xyz),
        prepare_ligand(lig_xyz, lig_atomtype_id, iface_matrix=iface_matrix),
    )


# ---------------------------------------------------------------------------
# Top-level search API
# ---------------------------------------------------------------------------


def docking_search(
    rec_xyz: torch.Tensor,
    rec_radius: torch.Tensor,
    rec_sasa: torch.Tensor,
    rec_atomtype_id: torch.Tensor,
    rec_charge_id: torch.Tensor,
    lig_xyz_ref: torch.Tensor,
    lig_radius: torch.Tensor,
    lig_sasa: torch.Tensor,
    lig_atomtype_id: torch.Tensor,
    lig_charge_id: torch.Tensor,
    quaternions: torch.Tensor,
    *,
    alpha: torch.Tensor,
    iface_ij_flat: torch.Tensor,    # (144,) — docking_score_elec convention
    beta: torch.Tensor,
    charge_score_lut: torch.Tensor, # (11,) partial charge LUT
    spacing: float = 3.0,
    surface_threshold: float = 1.0,
    rcut_iface: float = 6.0,
    rcut_elec: float = 8.0,
    ntop: int = 2000,
    rot_chunk_size: int = 16,
) -> DockingResultSC:
    """Full-score FFT docking search matching ``docking_score_elec``.

    For each quaternion, evaluates

        score(t) = α · SC(t) + IFACE(t) + β · ELEC(t)

    at every translation cell in one batched FFT, where SC/IFACE/ELEC
    reproduce ``docking_score_elec`` bit-exactly (modulo float
    roundoff). Keeps top-``ntop`` (quaternion, translation) pairs
    across all rotations.

    Receptor-side precomputes: complex SC grid, 12 weighted IFACE
    grids W_i, 1 Coulomb V_rec grid (with receptor-core zeroing). All
    FFT'd once before the rotation loop.

    Per rotation (batched in chunks of ``rot_chunk_size``):
        * SC: 1 complex FFT pair
        * IFACE: 12 real FFT pairs (sum before IFFT)
        * ELEC: 1 real FFT pair
    Total 14 FFTs per rotation + precompute.
    """
    device = rec_xyz.device
    dtype = rec_xyz.dtype
    n_rot = quaternions.shape[0]

    # Grid axes — match docking_score_elec exactly (same call).
    _, _, x_grid, y_grid, z_grid = generate_grid(
        rec_xyz, lig_xyz_ref, spacing=spacing, device=device, dtype=dtype,
    )
    nx, ny, nz = x_grid.numel(), y_grid.numel(), z_grid.numel()
    V = nx * ny * nz

    # Receptor SC complex grid (+ FFT) — also needed for ELEC core mask.
    R_real, R_imag = _build_receptor_sc_grids(
        rec_xyz, rec_radius, rec_sasa, x_grid, y_grid, z_grid,
        surface_threshold=surface_threshold,
    )
    complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    Z_R = (R_real + 1j * R_imag).to(complex_dtype)
    F_Z_R = torch.fft.fftn(Z_R, dim=(-3, -2, -1))

    # Receptor IFACE (12 W_i grids, pre-weighted by iface_matrix).
    # iface_ij_flat → (12,12) matching docking_score_elec line 413.
    iface_matrix = iface_ij_flat.view(12, 12).T
    W = _build_receptor_iface_weighted_grids(
        rec_xyz, rec_atomtype_id, iface_matrix, x_grid, y_grid, z_grid,
        rcut_iface=rcut_iface,
    )
    F_W = torch.fft.fftn(W, dim=(-3, -2, -1))         # (12, nx, ny, nz) complex

    # Receptor ELEC potential V_rec (Coulomb + core zero).
    V_rec = _build_receptor_elec_grid(
        rec_xyz, rec_charge_id, charge_score_lut, R_real, R_imag,
        x_grid, y_grid, z_grid, rcut_elec=rcut_elec,
    )
    F_V = torch.fft.fftn(V_rec, dim=(-3, -2, -1))

    # Running top-ntop buffer.
    buf_scores = torch.full((ntop,), float("-inf"), device=device, dtype=dtype)
    buf_quat = torch.zeros((ntop,), device=device, dtype=torch.long)
    buf_flat = torch.zeros((ntop,), device=device, dtype=torch.long)

    # Precompute ligand partial charges (scalars per atom — rotation-invariant).
    lig_partial_q = partial_charge_per_atom(lig_charge_id, charge_score_lut)
    lig_surf = lig_sasa > surface_threshold

    for chunk_start in range(0, n_rot, rot_chunk_size):
        chunk_end = min(chunk_start + rot_chunk_size, n_rot)
        B = chunk_end - chunk_start

        lig_rot = _rotate_batch(
            lig_xyz_ref, quaternions[chunk_start:chunk_end],
        )  # (B, N_lig, 3)

        # Batched ligand-side scatters — single scatter per physical
        # quantity using frame-compound group indices (no per-B loop).
        L_sc_real, L_sc_imag = _build_ligand_sc_grids_vectorised(
            lig_rot, lig_radius, lig_surf, x_grid, y_grid, z_grid,
        )
        L_iface = _build_ligand_iface_grids_vectorised(
            lig_rot, lig_atomtype_id, x_grid, y_grid, z_grid,
        )
        L_elec = _build_ligand_elec_grids_vectorised(
            lig_rot, lig_charge_id, charge_score_lut,
            x_grid, y_grid, z_grid,
        )

        # SC: complex FFT, real − imag.
        Z_L = (L_sc_real + 1j * L_sc_imag).to(complex_dtype)
        F_Z_L = torch.fft.fftn(Z_L.conj(), dim=(-3, -2, -1)).conj()
        G_sc = torch.fft.ifftn(F_Z_R.unsqueeze(0) * F_Z_L, dim=(-3, -2, -1))
        score_sc = G_sc.real - G_sc.imag             # (B, nx, ny, nz)

        # IFACE: 12 real FFTs, sum in frequency domain, single IFFT.
        F_L_iface = torch.fft.fftn(L_iface, dim=(-3, -2, -1))  # (B, 12, .)
        # Σ_i F_W[i] · conj(F_L_iface[b, i])
        summed = (F_W.unsqueeze(0) * F_L_iface.conj()).sum(dim=1)
        score_iface = torch.fft.ifftn(summed, dim=(-3, -2, -1)).real

        # ELEC: 1 real FFT pair.
        F_L_elec = torch.fft.fftn(L_elec, dim=(-3, -2, -1))
        score_elec = torch.fft.ifftn(
            F_V.unsqueeze(0) * F_L_elec.conj(), dim=(-3, -2, -1),
        ).real

        # Combine per docking_score_elec: α SC + IFACE + β ELEC.
        score_grid = alpha * score_sc + score_iface + beta * score_elec

        # Top-k per rotation then merge with buffer.
        score_flat = score_grid.reshape(B, -1)
        k_per_rot = min(ntop, score_flat.shape[1])
        top_vals, top_idx = torch.topk(score_flat, k=k_per_rot, dim=1)
        cand_scores = top_vals.reshape(-1)
        cand_quat = (
            chunk_start
            + torch.arange(B, device=device).unsqueeze(-1).expand(-1, k_per_rot)
        ).reshape(-1).to(torch.long)
        cand_flat = top_idx.reshape(-1).to(torch.long)

        all_scores = torch.cat([buf_scores, cand_scores])
        all_quat = torch.cat([buf_quat, cand_quat])
        all_flat = torch.cat([buf_flat, cand_flat])
        new_vals, new_idx = torch.topk(all_scores, k=ntop)
        buf_scores = new_vals
        buf_quat = all_quat[new_idx]
        buf_flat = all_flat[new_idx]

    tz = buf_flat % nz
    ty = (buf_flat // nz) % ny
    tx = (buf_flat // (ny * nz)) % nx
    tx_s = _unshift_cyclic(tx, nx).to(dtype) * spacing
    ty_s = _unshift_cyclic(ty, ny).to(dtype) * spacing
    tz_s = _unshift_cyclic(tz, nz).to(dtype) * spacing

    return DockingResultSC(
        scores=buf_scores,
        quat_indices=buf_quat,
        translations=torch.stack([tx_s, ty_s, tz_s], dim=-1),
    )


# ---------------------------------------------------------------------------
# Gradient-based pose refinement
# ---------------------------------------------------------------------------


def refine_poses_gradient(
    rec_xyz: torch.Tensor, rec_radius: torch.Tensor, rec_sasa: torch.Tensor,
    rec_atomtype_id: torch.Tensor, rec_charge_id: torch.Tensor,
    lig_xyz_ref: torch.Tensor, lig_radius: torch.Tensor, lig_sasa: torch.Tensor,
    lig_atomtype_id: torch.Tensor, lig_charge_id: torch.Tensor,
    q_init: torch.Tensor,                  # (B, 4) starting quaternions
    t_init: torch.Tensor,                  # (B, 3) starting translations (Å)
    *,
    alpha: torch.Tensor,
    iface_ij_flat: torch.Tensor,
    beta: torch.Tensor,
    charge_score_lut: torch.Tensor,
    n_iter: int = 50,
    lr_q: float = 0.02,
    lr_t: float = 0.2,
    spacing: float = 1.2,
    scatter_mode: str = "trilinear",
    frame_chunk_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Adam ascent on (q, t) maximizing ``docking_score_elec`` for a
    batch of B starting poses.

    Inputs are the same 10-field receptor/ligand bundle as
    ``docking_search`` / ``docking_score_elec``, plus:

        q_init : (B, 4)  starting quaternion per pose (same convention
                         as ``geom.rotate`` / ``_rotate_batch``)
        t_init : (B, 3)  starting cartesian translation (Å)

    Returns ``(q_final, t_final, scores_final)`` each leading dim B.
    Rotation is kept on the unit-quaternion manifold by renormalising
    after every Adam step (simple but adequate for local refinement;
    for principled SO(3) optimisation use a Lie-algebra step).

    Refinement uses a single batched ``docking_score_elec`` call per
    iteration, so cost is B-independent in the receptor-precompute
    part and scales per-pose in the ligand-side scatter+IFACE+ELEC.
    """
    if q_init.shape[-1] != 4 or q_init.ndim != 2:
        raise ValueError(f"q_init must be (B, 4), got {tuple(q_init.shape)}")
    if t_init.shape[-1] != 3 or t_init.ndim != 2:
        raise ValueError(f"t_init must be (B, 3), got {tuple(t_init.shape)}")
    if q_init.shape[0] != t_init.shape[0]:
        raise ValueError(
            f"q_init and t_init must share batch size; got "
            f"{q_init.shape[0]} vs {t_init.shape[0]}"
        )

    q = q_init.detach().clone().requires_grad_(True)
    t = t_init.detach().clone().requires_grad_(True)

    opt = torch.optim.Adam([
        {"params": [q], "lr": lr_q},
        {"params": [t], "lr": lr_t},
    ])

    for _step in range(n_iter):
        opt.zero_grad()
        q_norm = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        lig_rot = _rotate_batch(lig_xyz_ref, q_norm)        # (B, N_lig, 3)
        pose = lig_rot + t.unsqueeze(-2)                    # (B, N_lig, 3)
        scores = docking_score_elec(
            rec_xyz, rec_radius, rec_sasa, rec_atomtype_id, rec_charge_id,
            pose, lig_radius, lig_sasa, lig_atomtype_id, lig_charge_id,
            alpha=alpha, iface_ij_flat=iface_ij_flat, beta=beta,
            charge_score=charge_score_lut,
            lig_xyz_for_grid=lig_xyz_ref, spacing=spacing,
            scatter_mode=scatter_mode,
            frame_chunk_size=frame_chunk_size,
        )
        # Maximise scores → minimise −sum(scores).
        (-scores.sum()).backward()
        opt.step()

    with torch.no_grad():
        q_final = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        t_final = t.detach().clone()
        lig_rot = _rotate_batch(lig_xyz_ref, q_final)
        pose = lig_rot + t_final.unsqueeze(-2)
        scores_final = docking_score_elec(
            rec_xyz, rec_radius, rec_sasa, rec_atomtype_id, rec_charge_id,
            pose, lig_radius, lig_sasa, lig_atomtype_id, lig_charge_id,
            alpha=alpha, iface_ij_flat=iface_ij_flat, beta=beta,
            charge_score=charge_score_lut,
            lig_xyz_for_grid=lig_xyz_ref, spacing=spacing,
        )

    return q_final.detach(), t_final, scores_final


def docking_search_sc(
    rec_xyz: torch.Tensor,
    rec_radius: torch.Tensor,
    rec_sasa: torch.Tensor,
    lig_xyz_ref: torch.Tensor,      # (N_lig, 3) — reference ligand, decentered
    lig_radius: torch.Tensor,
    lig_sasa: torch.Tensor,
    quaternions: torch.Tensor,      # (R, 4)
    *,
    spacing: float = 3.0,
    surface_threshold: float = 1.0,
    ntop: int = 100,
    rot_chunk_size: int = 16,
) -> DockingResultSC:
    """FFT-based SC-only docking search.

    For each quaternion in `quaternions`, rotate `lig_xyz_ref` and
    evaluate the SC term at every translation in the receptor grid in
    one FFT per rotation. Keep the top `ntop` (quat, translation) pairs
    across all rotations.

    Inputs:
        rec_xyz: (N_rec, 3) decentered receptor atoms.
        rec_radius, rec_sasa: per-atom receptor features.
        lig_xyz_ref: (N_lig, 3) decentered, orient-aligned ligand atoms.
            The FFT search does not re-apply orient(); pass coords in
            the same frame `docking_score_elec` expects.
        lig_radius, lig_sasa: per-atom ligand features.
        quaternions: (R, 4) caller-supplied rotation grid. Same
            convention as `geom.rotate`.

    Kwargs:
        spacing: grid spacing (Å). Defaults to 3.0 matching
            `docking_score_elec`.
        surface_threshold: SASA cutoff for surface vs core. Default 1.0
            matches `docking_score_elec`.
        ntop: number of top poses to return across all rotations.
        rot_chunk_size: rotations processed per FFT batch (VRAM knob).

    Returns `DockingResultSC`.
    """
    device = rec_xyz.device
    dtype = rec_xyz.dtype
    n_rot = quaternions.shape[0]

    # 1. Build grid axes exactly as `docking_score_elec` does.
    #    generate_grid uses the ligand's bounding box for padding, so we
    #    pass the reference (decentered) ligand — every rotation lives in
    #    this same box because rotation preserves extent on average
    #    (slight variance handled by the padding).
    _, _, x_grid, y_grid, z_grid = generate_grid(
        rec_xyz, lig_xyz_ref, spacing=spacing, device=device, dtype=dtype,
    )
    nx, ny, nz = x_grid.numel(), y_grid.numel(), z_grid.numel()

    # 2. Receptor-side FFT (done once).
    R_real, R_imag = _build_receptor_sc_grids(
        rec_xyz, rec_radius, rec_sasa, x_grid, y_grid, z_grid,
        surface_threshold=surface_threshold,
    )
    complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    Z_R = (R_real + 1j * R_imag).to(complex_dtype)
    F_Z_R = torch.fft.fftn(Z_R, dim=(-3, -2, -1))

    # 3. Running top-ntop buffer across rotations.
    buf_scores = torch.full((ntop,), float("-inf"), device=device, dtype=dtype)
    buf_quat = torch.zeros((ntop,), device=device, dtype=torch.long)
    buf_flat = torch.zeros((ntop,), device=device, dtype=torch.long)

    # 4. Per-rotation FFT loop (chunked for VRAM).
    for chunk_start in range(0, n_rot, rot_chunk_size):
        chunk_end = min(chunk_start + rot_chunk_size, n_rot)
        B = chunk_end - chunk_start

        # Rotate ligand per quaternion in the chunk.
        lig_rot = _rotate_batch(lig_xyz_ref, quaternions[chunk_start:chunk_end])

        # Scatter each rotated ligand into SC grids.
        L_real, L_imag = _build_ligand_sc_grids_batch(
            lig_rot, lig_radius, lig_sasa, x_grid, y_grid, z_grid,
            surface_threshold=surface_threshold,
        )
        Z_L = (L_real + 1j * L_imag).to(complex_dtype)

        # Per-rotation FFT. The identity used here:
        #   G[t] = Σ_m Z_R[m] · Z_L[m − t]
        #        = ifft(fft(Z_R) · conj(fft(conj(Z_L))))[t]
        # which for real Z_R = R_r + i R_i and Z_L = L_r + i L_i gives
        # score_sc(t) = Re(G[t]) − Im(G[t]), matching
        # `docking_score_elec`'s pointwise SC combination at t=0.
        F_Z_L = torch.fft.fftn(Z_L.conj(), dim=(-3, -2, -1)).conj()
        G = torch.fft.ifftn(F_Z_R.unsqueeze(0) * F_Z_L, dim=(-3, -2, -1))
        score_grid = G.real - G.imag           # (B, nx, ny, nz), real dtype

        # 5. Per-rotation top-k, merge with running buffer.
        score_flat = score_grid.reshape(B, -1)  # (B, V)
        k_per_rot = min(ntop, score_flat.shape[1])
        top_vals, top_idx = torch.topk(score_flat, k=k_per_rot, dim=1)

        # Flatten chunk's (B × k_per_rot) candidates.
        cand_scores = top_vals.reshape(-1)
        cand_quat = (
            chunk_start
            + torch.arange(B, device=device).unsqueeze(-1).expand(-1, k_per_rot)
        ).reshape(-1).to(torch.long)
        cand_flat = top_idx.reshape(-1).to(torch.long)

        # Merge and re-prune.
        all_scores = torch.cat([buf_scores, cand_scores])
        all_quat = torch.cat([buf_quat, cand_quat])
        all_flat = torch.cat([buf_flat, cand_flat])
        new_vals, new_idx = torch.topk(all_scores, k=ntop)
        buf_scores = new_vals
        buf_quat = all_quat[new_idx]
        buf_flat = all_flat[new_idx]

    # 6. Decode flat cell indices to signed cartesian translations.
    V = nx * ny * nz
    tz = buf_flat % nz
    ty = (buf_flat // nz) % ny
    tx = (buf_flat // (ny * nz)) % nx
    tx_s = _unshift_cyclic(tx, nx).to(dtype) * spacing
    ty_s = _unshift_cyclic(ty, ny).to(dtype) * spacing
    tz_s = _unshift_cyclic(tz, nz).to(dtype) * spacing
    translations = torch.stack([tx_s, ty_s, tz_s], dim=-1)

    return DockingResultSC(
        scores=buf_scores,
        quat_indices=buf_quat,
        translations=translations,
    )


# ---------------------------------------------------------------------------
# Direct (non-FFT) reference score, for tests. O(N³) per translation, so
# only usable on tiny grids.
# ---------------------------------------------------------------------------


def docking_score_sc_direct(
    R_real: torch.Tensor, R_imag: torch.Tensor,
    L_real: torch.Tensor, L_imag: torch.Tensor,
) -> torch.Tensor:
    """Naive O(V²) cross-correlation for test reference.

    score_sc(t) = Σ_cell [R_r·L_r − R_i·L_i − R_r·L_i − R_i·L_r]
                         evaluated with L shifted by +t.

    Returns a (nx, ny, nz) tensor of scores indexed cyclically (index 0 =
    zero translation, large indices wrap around to negative translations,
    matching `torch.fft.ifftn` output ordering).
    """
    nx, ny, nz = R_real.shape
    out = torch.zeros_like(R_real)
    # Loop over all translations. O(V²) — only for tiny grids.
    for tx in range(nx):
        for ty in range(ny):
            for tz in range(nz):
                # L shifted cyclically by (tx, ty, tz): L_shifted[cell] = L[cell - t]
                Ls_real = torch.roll(L_real, shifts=(tx, ty, tz), dims=(0, 1, 2))
                Ls_imag = torch.roll(L_imag, shifts=(tx, ty, tz), dims=(0, 1, 2))
                s = (R_real * Ls_real - R_imag * Ls_imag
                     - R_real * Ls_imag - R_imag * Ls_real).sum()
                out[tx, ty, tz] = s
    return out
