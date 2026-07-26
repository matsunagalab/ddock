"""Forward scoring functions (`docking_score`, `docking_score_elec`) ported
from `train_param-apart.ipynb` cell 4–5.

Design notes for GPU / MPS performance:

* The inner Julia loops iterate 12×12 = 144 times for the IFACE term and
  11 times for the ELEC term, each spread-ing a single atom type onto a
  fresh grid and doing one elementwise-multiply-and-sum. On the GPU each
  iteration is a tiny kernel launch; 144 launches per frame × nframe adds
  up.

  We replace the nested loops with **batched** spreads on grids of shape
  ``(n_type, nx, ny, nz)`` (one slab per atom type). The dot product
  across type pairs becomes a single ``L @ H.T`` matmul (12, 12) for
  IFACE; ELEC collapses to a single einsum.

* All backbone ops are ``torch`` tensor ops — no Python-level atom loops
  in the hot path. Preprocessing (atom-type ID, SASA, charge ID) is
  one-time and runs off-device through ``atomtypes``.

* ``docking_score_elec`` is fully autograd-safe: α, β, iface_ij, and
  charge_score are leaf tensors; ``score_total = α S_SC + S_IFACE +
  β S_ELEC`` is a linear combination of grid reductions, and every step
  (scatter_add, elementwise ops, reductions, complex multiply) has a
  built-in PyTorch VJP.

The caller is responsible for preparing inputs exactly as the Julia
`docking_score_elec` internally would:

  * ``receptor_xyz`` is already centred (``decenter!``).
  * ``ligand_xyz`` is already PCA-oriented (``orient!``) and centred.

This avoids porting MDToolbox's ``orient!`` at the cost of shifting the
burden to the preprocessing side (Julia `generate_refs.jl` does this).
"""

from __future__ import annotations

import math

import torch
from torch.utils.checkpoint import checkpoint

from typing import Literal

from .geom import generate_grid, orient
from .atomtypes import iface_ij, partial_charge_per_atom
from .spread import (
    _neighbors_indices,
    _flat_index,
    _in_bounds,
    _nearest_cell_indices,
    spread_neighbors_coulomb,
)


ElecMode = Literal["coulomb", "legacy"]


# ---------------------------------------------------------------------------
# Grouped / batched spread helpers — a generalisation of spread.py that
# routes each atom's contribution into a per-type slab of a
# (G, nx, ny, nz) grid in a single scatter call.
# ---------------------------------------------------------------------------


def _grouped_spread_nearest_add(
    grid_batch: torch.Tensor,         # (G, nx, ny, nz) — zero-initialized
    xyz: torch.Tensor,                # (N, 3)
    group: torch.Tensor,              # (N,) int in [0, G)
    weights: torch.Tensor,            # (N,) same dtype as grid_batch
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    z_grid: torch.Tensor,
) -> torch.Tensor:
    G, nx, ny, nz = grid_batch.shape
    ix, iy, iz = _nearest_cell_indices(xyz, x_grid, y_grid, z_grid)
    in_b = _in_bounds(ix, iy, iz, (nx, ny, nz))
    valid = in_b & (group >= 0) & (group < G)
    flat = (
        group[valid] * (nx * ny * nz)
        + ix[valid] * (ny * nz)
        + iy[valid] * nz
        + iz[valid]
    )
    grid_batch.view(-1).scatter_add_(0, flat, weights[valid])
    return grid_batch


def _grouped_spread_trilinear_add(
    grid_batch: torch.Tensor,         # (G, nx, ny, nz)
    xyz: torch.Tensor,                # (N, 3) — may require_grad
    group: torch.Tensor,              # (N,) int in [0, G)
    weights: torch.Tensor,            # (N,)
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    z_grid: torch.Tensor,
) -> torch.Tensor:
    """Trilinear (B-spline order 2) scatter — SPME-style smooth
    particle-to-mesh spreading. Each atom is distributed across the
    8 cells of its containing cube with weights that are smooth
    functions of the atom's fractional position.

    Gradient flows through the corner weights back to ``xyz``:
    ``dx = ix_f - floor(ix_f)`` is continuous in ``xyz`` even though
    ``floor`` is not. Weight sum = 1 is preserved, so the integer
    part being non-differentiable doesn't break the gradient of the
    aggregate score.

    Atoms within one cell of the grid boundary (i.e., any of the 8
    corners would fall outside the grid) are dropped — this is the
    "hard cutoff" at the grid edge. For docking this is fine since
    ligand atoms that escape the receptor-centered grid contribute
    nothing physically meaningful anyway.
    """
    G, nx, ny, nz = grid_batch.shape
    dtype = grid_batch.dtype
    V = nx * ny * nz

    # Uniform spacing per axis — this is how `generate_grid` builds them.
    spacing_x = x_grid[1] - x_grid[0]
    spacing_y = y_grid[1] - y_grid[0]
    spacing_z = z_grid[1] - z_grid[0]

    # Float cell coordinates (continuous, diff wrt xyz).
    ix_f = (xyz[:, 0] - x_grid[0]) / spacing_x
    iy_f = (xyz[:, 1] - y_grid[0]) / spacing_y
    iz_f = (xyz[:, 2] - z_grid[0]) / spacing_z

    # Integer part (non-diff), fractional part (diff).
    ix0 = ix_f.detach().floor().long()
    iy0 = iy_f.detach().floor().long()
    iz0 = iz_f.detach().floor().long()
    dx = ix_f - ix0.to(dtype)          # ∈ [0, 1), diff wrt xyz
    dy = iy_f - iy0.to(dtype)
    dz = iz_f - iz0.to(dtype)

    # In-bounds: need all 8 neighbors (ix0..ix0+1 × ...) within grid.
    in_b = (
        (ix0 >= 0) & (ix0 + 1 < nx)
        & (iy0 >= 0) & (iy0 + 1 < ny)
        & (iz0 >= 0) & (iz0 + 1 < nz)
        & (group >= 0) & (group < G)
    )
    valid = in_b.nonzero(as_tuple=True)[0]
    if valid.numel() == 0:
        return grid_batch

    ix0v = ix0[valid]
    iy0v = iy0[valid]
    iz0v = iz0[valid]
    dxv = dx[valid]
    dyv = dy[valid]
    dzv = dz[valid]
    gv = group[valid]
    wv = weights[valid]

    # 8 corners: (ox, oy, oz) ∈ {0, 1}³. Loop is cheap (8 iterations)
    # and keeps memory bounded — fused corners would use 8× the atom
    # tensors simultaneously.
    for ox in (0, 1):
        wx = dxv if ox == 1 else (1.0 - dxv)
        for oy in (0, 1):
            wy = dyv if oy == 1 else (1.0 - dyv)
            for oz in (0, 1):
                wz = dzv if oz == 1 else (1.0 - dzv)
                corner_w = wx * wy * wz           # (N_valid,)
                contribs = wv * corner_w
                flat = (
                    gv * V
                    + (ix0v + ox) * (ny * nz)
                    + (iy0v + oy) * nz
                    + (iz0v + oz)
                )
                grid_batch.view(-1).scatter_add_(0, flat, contribs)
    return grid_batch


def _grouped_spread_neighbors_add(
    grid_batch: torch.Tensor,
    xyz: torch.Tensor,
    group: torch.Tensor,
    weights: torch.Tensor,
    rcut: torch.Tensor | float,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    z_grid: torch.Tensor,
) -> torch.Tensor:
    G, nx, ny, nz = grid_batch.shape
    flat_cell, _, atom_idx = _neighbors_indices(
        xyz, rcut, x_grid, y_grid, z_grid, (nx, ny, nz)
    )
    g = group[atom_idx]
    valid = (g >= 0) & (g < G)
    flat = g[valid] * (nx * ny * nz) + flat_cell[valid]
    contribs = weights[atom_idx[valid]]
    grid_batch.view(-1).scatter_add_(0, flat, contribs)
    return grid_batch


def _grouped_calculate_distance(
    grid_batch: torch.Tensor,
    xyz: torch.Tensor,
    group: torch.Tensor,
    rcut: torch.Tensor | float,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    z_grid: torch.Tensor,
) -> torch.Tensor:
    G, nx, ny, nz = grid_batch.shape
    flat_cell, d, atom_idx = _neighbors_indices(
        xyz, rcut, x_grid, y_grid, z_grid, (nx, ny, nz)
    )
    g = group[atom_idx]
    valid = (g >= 0) & (g < G)
    flat = g[valid] * (nx * ny * nz) + flat_cell[valid]
    grid_batch.view(-1).scatter_add_(0, flat, d[valid])
    return grid_batch


# ---------------------------------------------------------------------------
# SC (shape complementarity) assign helpers — direct ports of the
# assign_sc_*_plus!/minus! functions in train_param-apart.ipynb cell 4.
# ---------------------------------------------------------------------------


#: ``rho`` of the **pairwise** shape-complementarity grid, Chen et al. 2003
#: Eq. (2) (ZDOCK 2.3, ``PSC+DE+ELEC``):
#:
#:     Re[R] = Re[L] = { rho    solvent-excluding surface layer
#:                       rho^2  protein core
#:                       0      open space }
#:
#: with ``rho = 3.5``, so a grid-point overlap costs ``rho^2 = 12.25``
#: (surface-surface), ``rho^3 = 42.875`` (surface-core) or ``rho^4 = 150.06``
#: (core-core) — a pure clash penalty; the *favourable* signal comes from the
#: pair potential, not from this term. Chen & Weng 2002's earlier grid-based
#: SC used a complex encoding with ``rho = 9`` instead; we follow the newer
#: paper. ``rho`` is exposed as a trainable parameter (see
#: ``docking_score_elec(..., sc_rho=...)``) initialised at the published value.
SC_RHO = 3.5

#: Chen et al. 2003 p.81: "ACE scores can be positive (unfavorable) or negative
#: (favorable)". The pair table (:func:`zdock.atomtypes.iface_ij`, range
#: -1.938..1.884) follows that convention, whereas the SC term above is written
#: so that a clash *lowers* the score — i.e. the two terms disagree about which
#: direction is better. The paper resolves exactly this ("To make these two
#: scores compatible, we flip the signs of the PSC scores"); we flip the pair
#: term instead so that the whole score stays "higher is better" and the FFT
#: search's ``topk`` remains correct.
IFACE_SIGN = -1.0

#: The favourable atom-pair term now lives in ``S_PSC`` itself (Chen & Weng 2003
#: Eq. (3)-(4), ``Re[R_PSC]*Re[L_PSC]``), so the pair table must NOT carry a
#: second copy of it. Kept at 0 and exposed only so the double-counting variant
#: can be reproduced.
IFACE_PAIR_OFFSET = 0.0

#: Grid spacing (Å) the ZDOCK coefficients were derived at — "A grid spacing of
#: 1.2 Å is used throughout this study" (Chen & Weng 2003, Methods).
SC_REFERENCE_SPACING = 1.2


def sc_cell_volume_factor(x_grid, y_grid, z_grid,
                          reference_spacing: float | None) -> float:
    """Rescale a per-cell-counted quantity to a spacing-invariant one.

    ``S_PSC``'s clash channel counts overlapping grid cells, so for a fixed
    physical overlap it grows like ``1/(dx*dy*dz)``. Multiplying by
    ``(dx*dy*dz)/reference_spacing**3`` removes that and is the identity at the
    paper's 1.2 Å. Pass ``None`` to disable.
    """
    if reference_spacing is None:
        return 1.0
    dx = float(x_grid[1] - x_grid[0])
    dy = float(y_grid[1] - y_grid[0])
    dz = float(z_grid[1] - z_grid[0])
    return (dx * dy * dz) / (reference_spacing ** 3)


#: Chen et al. 2003 Eq. (2): ``Im[L_PSC+ELEC] = -1 x (atom charge)``, in a
#: convention where "a more negative score indicates a more favorable
#: interaction energy" (p.81). This repository *maximises* the total score, so
#: the ELEC contribution must be ``-beta * sum(V*q)`` for an attractive
#: (opposite-charge) contact to raise the score. We carry the flip on the
#: ligand deposition, exactly as Eq. (2) writes it. An earlier revision
#: deposited ``+q``, which made the search reward electrostatic *repulsion*;
#: note this could not be absorbed by training, since ``S_ELEC`` is quadratic
#: in the charge LUT (the same LUT builds V_rec and Q_L) and beta is frozen.
ELEC_LIGAND_SIGN = -1.0


def ligand_partial_charge(lig_charge_id: torch.Tensor,
                          charge_score: torch.Tensor) -> torch.Tensor:
    """``Im[L_PSC+ELEC]`` of Chen et al. 2003 Eq. (2) — see
    :data:`ELEC_LIGAND_SIGN`. Every ligand-side ELEC scatter must go through
    this so the FFT and direct paths cannot drift apart."""
    return ELEC_LIGAND_SIGN * partial_charge_per_atom(lig_charge_id,
                                                      charge_score)


#: Chen & Weng 2003, "Optimizing PSC": the favourable component counts receptor
#: atoms within ``D + receptor atom radius`` of each open-space grid point.
#: "The parameters in the penalty term (-9 and -81) have been taken directly
#: from our earlier GSC formulation. Thus, the only adjustable parameter in the
#: PSC scoring function is the distance cutoff D." -> "We have chosen D = 3.6 Å
#: as the default value for subsequent PSC calculations."
PSC_D = 3.6


def sc_encode(shell: torch.Tensor, core: torch.Tensor, *, rho=SC_RHO):
    """Imaginary (clash) channel of Chen & Weng 2003 Eq. (3).

    ``Im[R_PSC] = Im[L_PSC] = rho`` on the solvent-*excluding* surface (cells
    covered by a surface atom), ``rho**2`` in the protein core, ``0`` in open
    space. Overlapping these gives the published penalties ``-rho**2`` /
    ``-rho**3`` / ``-rho**4`` for surface-surface / surface-core / core-core
    once ``-Im[R]*Im[L]`` is taken in Eq. (4).

    **Surface takes precedence over core**, per Chen & Weng 2003 (and repeated
    verbatim in Chen et al. 2003): "The 'solvent excluding surface layer of a
    protein' is defined by the grid points corresponding to surface atoms. *All
    other* grid points corresponding to any core atoms are in the protein
    'core'." The surface layer is defined first; the core is the residual. An
    earlier revision of this function had the precedence inverted, which
    over-penalised exactly the interface-adjacent cells PSC is meant to treat
    leniently (measured on 1KXQ: 17% of occupied receptor cells are covered by
    both a surface and a core atom; the native-pose penalty was 24% too high at
    rho=3.5, i.e. S_PSC 168.75 instead of 291.2).

    Note PSC, unlike GSC, has **no** "solvent accessible surface layer": "any
    grid point that does not correspond to an atom is in the open space".
    """
    shell_b, core_b = sc_layers(shell, core)
    rho_t = rho if torch.is_tensor(rho) else torch.as_tensor(
        rho, dtype=shell.dtype, device=shell.device)
    return shell_b.to(shell.dtype) * rho_t + core_b.to(shell.dtype) * rho_t.pow(2)


def sc_layers(shell: torch.Tensor, core: torch.Tensor):
    """The surface / core occupancy indicators of Eq. (3), with the paper's
    precedence (surface first, core = residual). Returned separately from
    :func:`sc_encode` so the clash channel can be decomposed by overlap class
    without re-deriving it from the encoded values — see
    :func:`psc_clash_counts`.
    """
    shell_b = shell > 0
    core_b = (core > 0) & ~shell_b
    return shell_b, core_b


def psc_clash_counts(rec_surf, rec_core, lig_surf, lig_core):
    """Split ``sum(Im[R] * Im[L])`` into its three rho-independent counts.

    ``Im`` takes only the values ``{0, rho, rho**2}``, so

        sum(Im[R]*Im[L]) = n_ss * rho**2 + n_sc * rho**3 + n_cc * rho**4

    with ``n_ss`` the number of surface-surface cell overlaps, ``n_sc`` the
    surface-core ones (either way round) and ``n_cc`` the core-core ones. The
    counts do not depend on rho, so caching them alongside the favourable pair
    count makes ``S_PSC`` an exact quartic in rho that a trainer can
    differentiate from cached features — see :func:`psc_score_from_terms`.

    Inputs are ``(..., nx, ny, nz)`` boolean-or-0/1 grids; the receptor grids
    broadcast against a leading frame dimension on the ligand side.
    """
    rs, rc = rec_surf.to(lig_surf.dtype), rec_core.to(lig_surf.dtype)
    ls, lc = lig_surf.to(lig_surf.dtype), lig_core.to(lig_surf.dtype)
    n_ss = rs * ls
    n_sc = rs * lc + rc * ls
    n_cc = rc * lc
    return n_ss, n_sc, n_cc


def psc_score_from_terms(terms: torch.Tensor, rho, *,
                         clash_volume_factor: float = 1.0) -> torch.Tensor:
    """Reconstruct ``S_PSC`` from the cached ``(F, 4)`` decomposition.

    ``terms[..., 0]`` is the favourable receptor-ligand atom-pair count
    ``sum(Re[R]*Re[L])`` and ``terms[..., 1:4]`` are ``(n_ss, n_sc, n_cc)``.

        S_PSC = c_pair - k * (n_ss rho^2 + n_sc rho^3 + n_cc rho^4)

    Note ``k`` (:func:`sc_cell_volume_factor`) is applied to the **clash
    channel only**. The clash sum counts overlapping grid *cells* and so grows
    like ``1/(dx dy dz)``, whereas ``c_pair`` is a receptor-ligand *atom-pair*
    count and is spacing-invariant by construction; one scalar on the whole of
    ``S_PSC`` cannot correct both. ``k`` is exactly 1 at the paper's 1.2 A, so
    this only matters when comparing across spacings.
    """
    rho_t = rho if torch.is_tensor(rho) else torch.as_tensor(
        rho, dtype=terms.dtype, device=terms.device)
    clash = (terms[..., 1] * rho_t.pow(2)
             + terms[..., 2] * rho_t.pow(3)
             + terms[..., 3] * rho_t.pow(4))
    return terms[..., 0] - clash_volume_factor * clash


def sc_open_boundary_to_surface(re: torch.Tensor, im: torch.Tensor):
    """Chen & Weng 2002, ``L_SC`` step 3: "if a grid point is assigned ``rho*i``
    and any two of its nearest neighboring grid points have value 0, it is
    changed to 1".

    This converts the outermost shell of the ligand's core into surface cells,
    so a ligand whose interface is formed by buried atoms can still make a
    rewarding surface–surface contact. Works on ``(..., nx, ny, nz)``.
    """
    unoccupied = ((re <= 0) & (im <= 0)).to(re.dtype)
    n_zero = torch.zeros_like(unoccupied)
    for dim in (-3, -2, -1):
        n_zero = (n_zero + torch.roll(unoccupied, 1, dims=dim)
                  + torch.roll(unoccupied, -1, dims=dim))
    flip = (im > 0) & (n_zero >= 2)
    return (torch.where(flip, torch.ones_like(re), re),
            torch.where(flip, torch.zeros_like(im), im))


def _sc_indicator(shape, xyz, rcut, x_grid, y_grid, z_grid):
    """Indicator grid of "within ``rcut[atom]`` of some atom"."""
    from .spread import spread_neighbors_add

    grid = torch.zeros(shape, device=xyz.device, dtype=x_grid.dtype)
    if xyz.shape[0]:
        spread_neighbors_add(
            grid, xyz, torch.ones(xyz.shape[0], device=xyz.device, dtype=x_grid.dtype),
            rcut, x_grid, y_grid, z_grid,
        )
    return grid


def psc_grids(
    xyz: torch.Tensor,
    radius: torch.Tensor,
    id_surface: torch.Tensor,
    x_grid, y_grid, z_grid,
    *,
    receptor: bool,
    rho=SC_RHO,
    psc_d: float = PSC_D,
    return_layers: bool = False,
):
    """``(Re, Im)`` of Chen & Weng 2003 Eq. (3) for one molecule.

    ``Im`` is the clash channel (see :func:`sc_encode`). ``Re`` differs between
    the two partners and is what supplies PSC's *favourable* component:

    * receptor — ``Re[R_PSC]`` is "the number of receptor atoms within
      ``(D + receptor atom radius)``" of the grid point, and is non-zero **only
      in open space**;
    * ligand — ``Re[L_PSC]`` is "1 if this grid is the nearest grid of a ligand
      atom". We accumulate the count instead of clamping to 1 so that two
      ligand atoms sharing a cell still contribute two pairs; at the paper's
      1.2 Å spacing the two are almost always identical.

    Eq. (4) then reads ``Re[R.L] = Re[R]Re[L] - Im[R]Im[L]``: the first product
    is the total number of receptor-ligand atom pairs within the cutoff, the
    second is the clash penalty, "with a higher score indicating better shape
    complementarity".
    """
    shape = (x_grid.numel(), y_grid.numel(), z_grid.numel())
    surf, core = id_surface, ~id_surface
    surf_ind = _sc_indicator(shape, xyz[surf], radius[surf], x_grid, y_grid, z_grid)
    core_ind = _sc_indicator(shape, xyz[core], radius[core], x_grid, y_grid, z_grid)
    im = sc_encode(surf_ind, core_ind, rho=rho)
    layers = sc_layers(surf_ind, core_ind) if return_layers else None

    if receptor:
        counts = _sc_indicator(shape, xyz, radius + psc_d, x_grid, y_grid, z_grid)
        re = counts * (im <= 0).to(counts.dtype)          # open space only
    else:
        from .spread import spread_nearest_add
        re = torch.zeros(shape, device=xyz.device, dtype=x_grid.dtype)
        spread_nearest_add(
            re, xyz, torch.ones(xyz.shape[0], device=xyz.device, dtype=x_grid.dtype),
            x_grid, y_grid, z_grid)
    if return_layers:
        return re, im, layers[0], layers[1]
    return re, im


def iface_score_matrix(iface_ij_flat: torch.Tensor) -> torch.Tensor:
    """The (12, 12) matrix ``docking_score_elec`` actually contracts ``T`` with.

    Anything that reconstructs a score from cached ``(S_SC, T, S_ELEC)``
    features must use this, not the raw table: the score applies the ACE/PSC
    sign reconciliation (:data:`IFACE_SIGN`) and, when enabled, the pair-count
    offset. Reconstructing with the raw table silently optimises a different
    objective from the one the FFT search ranks by.
    """
    return IFACE_PAIR_OFFSET + IFACE_SIGN * iface_ij_flat.view(12, 12).T


def _score_ligand_chunk(
    lig_xyz: torch.Tensor,                     # (F_c, N_lig, 3)
    alpha: torch.Tensor,
    iface_matrix: torch.Tensor,                # (12, 12)
    beta: torch.Tensor,
    charge_score: torch.Tensor,                # (11,)
    H: torch.Tensor,                           # (12, nx, ny, nz)
    rec_sc_real: torch.Tensor,                 # (nx, ny, nz)
    rec_sc_imag: torch.Tensor,                 # (nx, ny, nz)
    V_rec_or_U: torch.Tensor,                  # (nx, ny, nz) if coulomb; (11, nx, ny, nz) if legacy
    *,
    lig_radius: torch.Tensor,
    lig_sasa: torch.Tensor,
    lig_atomtype_id: torch.Tensor,
    lig_charge_id: torch.Tensor,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    z_grid: torch.Tensor,
    surface_threshold: float,
    elec_mode: ElecMode,
    scatter_mode: str = "nearest",
    sc_reference_spacing: float | None = SC_REFERENCE_SPACING,
    sc_rho=SC_RHO,
    psc_d: float = PSC_D,
    return_components: bool = False,
    psc_decompose: bool = False,
    rec_sc_surf: torch.Tensor | None = None,
    rec_sc_core: torch.Tensor | None = None,
):
    """Per-frame total scores for a single ligand frame-chunk, re-using
    precomputed receptor grids.

    If ``return_components`` is True, return the tuple ``(score_sc, T,
    score_elec)`` instead of the combined score, where ``T`` is the
    ``(F, 12, 12)`` IFACE contraction ``Σ_cell L_i H_j`` (so that
    ``score_iface = (iface_matrix * T).sum((-2, -1))``). This exposes the
    per-pose features that make the score linear in ``(alpha, iface)`` and
    lets a caller cache them once and train the parameters cheaply.

    Split out of `docking_score_elec` so that callers can loop over
    chunks of F and optionally wrap each call in
    `torch.utils.checkpoint.checkpoint` to keep peak VRAM from scaling
    with F (instead it scales with F_chunk). Receptor grids — which do
    not depend on F — are computed once in the parent and passed in.
    """
    device = lig_xyz.device
    dtype = lig_xyz.dtype
    F = lig_xyz.shape[0]
    N_lig = lig_xyz.shape[1]
    nx, ny, nz = rec_sc_real.shape
    V = nx * ny * nz

    lig_group_iface = (lig_atomtype_id - 1).to(torch.long).clamp(0, 11)
    lig_in_charge = (lig_atomtype_id >= 1) & (lig_atomtype_id <= 11)
    lig_group_charge = torch.where(
        lig_in_charge, lig_atomtype_id - 1, torch.full_like(lig_atomtype_id, -1)
    ).to(torch.long)
    lig_surf = lig_sasa > surface_threshold

    frame_arange = torch.arange(F, device=device)
    lig_surf_expanded = lig_surf.unsqueeze(0).expand(F, -1)
    lig_radius_expanded = lig_radius.unsqueeze(0).expand(F, -1)

    frame_idx_per_atom = frame_arange.unsqueeze(-1).expand(-1, N_lig).reshape(-1)
    lxyz_flat = lig_xyz.reshape(-1, 3)
    lig_radius_flat = lig_radius_expanded.reshape(-1)
    lig_group_iface_flat = lig_group_iface.unsqueeze(0).expand(F, -1).reshape(-1)
    lig_surf_flat = lig_surf_expanded.reshape(-1)
    lig_group_charge_flat = lig_group_charge.unsqueeze(0).expand(F, -1).reshape(-1)

    def sc_union(xyz_f, group_frame, rcut_f, grid_shape):
        cnt = torch.zeros(grid_shape, device=device, dtype=dtype)
        _grouped_spread_neighbors_add(
            cnt, xyz_f, group_frame,
            torch.ones(xyz_f.shape[0], device=device, dtype=dtype),
            rcut_f, x_grid, y_grid, z_grid,
        )
        return (cnt > 0).to(dtype)

    surf_mask_flat = lig_surf_flat
    core_mask_flat = ~lig_surf_flat
    surf_idx = surf_mask_flat.nonzero(as_tuple=True)[0]
    core_idx = core_mask_flat.nonzero(as_tuple=True)[0]

    # Chen & Weng 2003 Eq. (3), ligand side: `Im[L_PSC]` is the clash channel
    # built by `sc_encode` (surface layer first, core = residual) and
    # `Re[L_PSC]` is one count at each atom's nearest grid point.
    zeros_g = torch.zeros((F, nx, ny, nz), device=device, dtype=dtype)
    surf_ind = (sc_union(lxyz_flat[surf_idx], frame_idx_per_atom[surf_idx],
                         lig_radius_flat[surf_idx], (F, nx, ny, nz))
                if surf_idx.numel() > 0 else zeros_g)
    # Plain vdW radii for both partners: Chen & Weng 2003 Fig. 1(b) draws PSC's
    # occupancy as the atom circles themselves. The sqrt(1.5) / sqrt(0.8)
    # scalings belong to the older GSC formulation.
    core_ind = (sc_union(lxyz_flat[core_idx], frame_idx_per_atom[core_idx],
                         lig_radius_flat[core_idx], (F, nx, ny, nz))
                if core_idx.numel() > 0 else zeros_g)
    lig_surf_b, lig_core_b = sc_layers(surf_ind, core_ind)
    rho_t = sc_rho if torch.is_tensor(sc_rho) else torch.as_tensor(
        sc_rho, dtype=dtype, device=device)
    lig_sc_imag = (lig_surf_b.to(dtype) * rho_t
                   + lig_core_b.to(dtype) * rho_t.pow(2))
    # Re[L_PSC]: one count at each ligand atom's nearest grid point (Eq. (3)).
    lig_sc_real = torch.zeros((F, nx, ny, nz), device=device, dtype=dtype)
    _grouped_spread_nearest_add(
        lig_sc_real, lxyz_flat, frame_idx_per_atom,
        torch.ones(lxyz_flat.shape[0], device=device, dtype=dtype),
        x_grid, y_grid, z_grid)

    vol_k = sc_cell_volume_factor(x_grid, y_grid, z_grid, sc_reference_spacing)
    c_pair = (rec_sc_real.unsqueeze(0) * lig_sc_real).reshape(F, -1).sum(-1)
    clash = (rec_sc_imag.unsqueeze(0) * lig_sc_imag).reshape(F, -1).sum(-1)
    # Chen & Weng 2003 Eq. (4): S_PSC = Re[R_PSC . L_PSC]
    #   = Re[R]Re[L] (favourable atom-pair count) - Im[R]Im[L] (clash penalty),
    # "with a higher score indicating better shape complementarity".
    score_sc = (c_pair - clash) * vol_k

    psc_terms = None
    if psc_decompose:
        if rec_sc_surf is None or rec_sc_core is None:
            raise ValueError("psc_decompose=True needs rec_sc_surf/rec_sc_core "
                             "(psc_grids(..., return_layers=True))")
        n_ss, n_sc, n_cc = psc_clash_counts(
            rec_sc_surf.unsqueeze(0), rec_sc_core.unsqueeze(0),
            lig_surf_b, lig_core_b)
        psc_terms = torch.stack([
            c_pair,
            n_ss.reshape(F, -1).sum(-1),
            n_sc.reshape(F, -1).sum(-1),
            n_cc.reshape(F, -1).sum(-1),
        ], dim=-1)                                    # (F, 4)

    L_count = torch.zeros((F * 12, nx, ny, nz), device=device, dtype=dtype)
    group_f12 = frame_idx_per_atom * 12 + lig_group_iface_flat
    if scatter_mode == "trilinear":
        _grouped_spread_trilinear_add(
            L_count, lxyz_flat, group_f12,
            torch.ones(lxyz_flat.shape[0], device=device, dtype=dtype),
            x_grid, y_grid, z_grid,
        )
        # `clamp(max=1)` replaces the non-diff `(count > 0)` indicator
        # with a differentiable "is any atom nearby?" surrogate. Gradient
        # saturates at 1 per cell which is fine for docking ranking.
        L = L_count.view(F, 12, nx, ny, nz).clamp(max=1.0)
    else:
        _grouped_spread_nearest_add(
            L_count, lxyz_flat, group_f12,
            torch.ones(lxyz_flat.shape[0], device=device, dtype=dtype),
            x_grid, y_grid, z_grid,
        )
        L = (L_count.view(F, 12, nx, ny, nz) > 0).to(dtype)
    T = torch.einsum("fiv,jv->fij", L.reshape(F, 12, V), H.reshape(12, V))
    score_iface = (iface_matrix.unsqueeze(0) * T).reshape(F, -1).sum(-1)

    if elec_mode == "coulomb":
        V_rec = V_rec_or_U
        lig_partial_q = ligand_partial_charge(lig_charge_id, charge_score)
        lig_partial_q_flat = lig_partial_q.unsqueeze(0).expand(F, -1).reshape(-1)
        Q_L = torch.zeros((F, nx, ny, nz), device=device, dtype=dtype)
        if scatter_mode == "trilinear":
            _grouped_spread_trilinear_add(
                Q_L, lxyz_flat, frame_idx_per_atom, lig_partial_q_flat,
                x_grid, y_grid, z_grid,
            )
        else:
            _grouped_spread_nearest_add(
                Q_L, lxyz_flat, frame_idx_per_atom, lig_partial_q_flat,
                x_grid, y_grid, z_grid,
            )
        score_elec = (V_rec.unsqueeze(0) * Q_L).reshape(F, -1).sum(-1)
    else:  # elec_mode == "legacy"
        U = V_rec_or_U
        valid_flat = lig_group_charge_flat >= 0
        grp_f11 = frame_idx_per_atom * 11 + lig_group_charge_flat.clamp(min=0)
        grp_f11 = grp_f11[valid_flat]
        xyz_c_flat = lxyz_flat[valid_flat]
        V_count = torch.zeros((F * 11, nx, ny, nz), device=device, dtype=dtype)
        _grouped_spread_nearest_add(
            V_count, xyz_c_flat, grp_f11,
            torch.ones(xyz_c_flat.shape[0], device=device, dtype=dtype),
            x_grid, y_grid, z_grid,
        )
        V_grid = V_count.view(F, 11, nx, ny, nz)
        c = (V_grid.reshape(F, 11, V) * U.reshape(11, V).unsqueeze(0)).sum(-1)
        score_elec = (charge_score.pow(2).unsqueeze(0) * c).sum(-1)

    if return_components:
        return (psc_terms if psc_decompose else score_sc), T, score_elec
    return alpha * score_sc + score_iface + beta * score_elec


def docking_score_elec(
    rec_xyz: torch.Tensor,              # (N_rec, 3) — already decentered
    rec_radius: torch.Tensor,           # (N_rec,)
    rec_sasa: torch.Tensor,             # (N_rec,)
    rec_atomtype_id: torch.Tensor,      # (N_rec,) int in [1, 12]
    rec_charge_id: torch.Tensor,        # (N_rec,) int in [1, 11]
    lig_xyz: torch.Tensor,              # (F, N_lig, 3) — each frame already oriented+decentered
    lig_radius: torch.Tensor,           # (N_lig,)
    lig_sasa: torch.Tensor,             # (N_lig,)
    lig_atomtype_id: torch.Tensor,      # (N_lig,)
    lig_charge_id: torch.Tensor,        # (N_lig,)
    alpha: torch.Tensor,                # scalar
    iface_ij_flat: torch.Tensor,        # (144,) — column-major of 12x12
    beta: torch.Tensor,                 # scalar
    charge_score: torch.Tensor,         # (11,)
    *,
    lig_xyz_for_grid: torch.Tensor | None = None,  # (N_lig, 3) post-orient
    spacing: float = SC_REFERENCE_SPACING,
    rcut_iface: float = 6.0,
    rcut_elec: float = 8.0,
    surface_threshold: float = 1.0,
    elec_mode: ElecMode = "coulomb",
    frame_chunk_size: int | None = None,
    scatter_mode: str = "nearest",
    sc_reference_spacing: float | None = SC_REFERENCE_SPACING,
    sc_rho=SC_RHO,
    psc_d: float = PSC_D,
    return_components: bool = False,
    psc_decompose: bool = False,
):
    """Return a (F,) tensor of docking scores.

    If ``return_components`` is True, return ``(score_sc, T, score_elec)``
    where ``T`` is ``(F, 12, 12)`` — the per-pose geometric features that
    the score is linear in for ``(alpha, iface)``.

    With ``psc_decompose=True`` the first element becomes ``(F, 4)`` instead:
    ``(c_pair, n_ss, n_sc, n_cc)``, from which
    ``psc_score_from_terms(terms, rho)`` reconstructs ``score_sc`` exactly for
    **any** rho. Caching that instead of the collapsed scalar is what makes rho
    trainable from cached features (``S_PSC`` is an exact quartic in rho;
    ``T`` and ``score_elec`` do not depend on rho at all). See
    ``_score_ligand_chunk`` for the exact definition. ``score_elec`` is
    evaluated with the supplied ``charge_score`` (so freeze it at the
    ZDOCK default to treat ELEC as a fixed per-pose feature).

    Default `elec_mode="coulomb"` implements the physically-correct Chen 2002 /
    Chen 2003 ELEC: receptor generates a Coulombic potential V(r) = Σⱼ qⱼ / |r−rⱼ|
    (zeroed inside the receptor SC shape), ligand stores -q at the nearest grid
    cell of each atom, and `score_elec` accumulates V × (-q) ≡ Coulomb energy
    across all (lig-atom × rec-atom) pairs. β scales this sum in `score_total`.

    `elec_mode="legacy"` preserves the notebook's original (buggy) formulation
    that groups ELEC by atom-type and computes `Σq / Σr` instead of Σq/r. This
    matches the Julia reference before the B10/B11/B12/B13 fixes and exists for
    bit-exact reproduction of the master thesis numbers.

    SC + IFACE are unchanged between modes (they have no ELEC-specific bugs).

    `frame_chunk_size`: if set to a positive int smaller than F, the
    ligand-side forward is split into chunks of that size and each chunk
    is wrapped in `torch.utils.checkpoint.checkpoint` when gradients are
    required. Peak VRAM then scales with F_chunk instead of F, at the
    cost of one extra forward per chunk during backward. `None` (default)
    or `<= 0` disables chunking (original behaviour).

    `scatter_mode`: controls how ligand atoms are distributed onto the
    grid.
      - ``"nearest"`` (default) — each atom assigned to a single cell
        (integer index, non-differentiable wrt ligand position but
        matches the ZDOCK / Julia reference convention).
      - ``"trilinear"`` — each atom spread across 8 surrounding cells
        via SPME-style trilinear (B-spline order 2) weights.
        Differentiable wrt ligand position so gradients can flow back
        to ``lig_xyz`` for pose refinement. Score values agree with
        nearest mode to within a few percent on typical complexes.
        Only the IFACE and ELEC ligand-side scatters switch; SC uses
        a separate neighbor-rcut path that remains nearest.
    """
    device = rec_xyz.device
    dtype = rec_xyz.dtype

    # Reshape iface_ij_flat (column-major 12×12) → (12, 12) matrix where
    # M[i, j] = iface_ij_flat[12*j + i]. Julia's k = 12*(j-1)+i maps to
    # Python index 12*j+i after 1-based → 0-based. The fortran-order view:
    # IFACE_SIGN reconciles the pair table's "favourable = negative" convention
    # with the clash-penalty sign of S_SC (Chen et al. 2003, p.81).
    iface_matrix = iface_score_matrix(iface_ij_flat)  # (12, 12), M[i, j]

    # Julia's generate_grid applies `orient!` (PCA rotation) to the
    # ligand internally before computing grid bounds. We compute the same
    # rotation in Python via `orient()`, using the ligand IFACE values
    # as inertia weights (matching Julia's notebook preprocessing that
    # sets `ligands.mass = iface_score[atomtype_id]`). If the caller
    # overrides with `lig_xyz_for_grid`, use that directly (useful for
    # tests that pin to Julia's exact SVD sign choice).
    if lig_xyz_for_grid is not None:
        grid_bounds_lig = lig_xyz_for_grid
    else:
        iface_matrix_for_mass = iface_ij(device=device, dtype=dtype)
        lig_mass_weights = iface_matrix_for_mass[lig_atomtype_id - 1, 0]
        grid_bounds_lig = orient(lig_xyz[0], mass=lig_mass_weights)
    grid_real, grid_imag, x_grid, y_grid, z_grid = generate_grid(
        rec_xyz, grid_bounds_lig, spacing=spacing
    )
    nx, ny, nz = grid_real.shape

    # Precompute receptor SC slabs (real + imag parts of SC filter).
    rec_surf = rec_sasa > surface_threshold
    if psc_decompose:
        rec_sc_real, rec_sc_imag, rec_sc_surf, rec_sc_core = psc_grids(
            rec_xyz, rec_radius, rec_surf, x_grid, y_grid, z_grid, receptor=True,
            rho=sc_rho, psc_d=psc_d, return_layers=True)
    else:
        rec_sc_real, rec_sc_imag = psc_grids(
            rec_xyz, rec_radius, rec_surf, x_grid, y_grid, z_grid, receptor=True,
            rho=sc_rho, psc_d=psc_d)
        rec_sc_surf = rec_sc_core = None

    # Precompute receptor IFACE contribution slabs H[j] for j in 1..12.
    # H[j] = Σ_atoms_of_type_j (within rcut=6 of cell) indicator. Weight 1.
    H = torch.zeros((12, nx, ny, nz), device=device, dtype=dtype)
    rec_group_iface = (rec_atomtype_id - 1).to(torch.long).clamp(0, 11)
    rec_weights_ones = torch.ones(rec_xyz.shape[0], device=device, dtype=dtype)
    _grouped_spread_neighbors_add(
        H, rec_xyz, rec_group_iface, rec_weights_ones, rcut_iface,
        x_grid, y_grid, z_grid,
    )

    # --- Receptor ELEC (mode-dependent) ---------------------------------

    if elec_mode == "coulomb":
        # Chen 2002 p284: V(r) = Σⱼ qⱼ / |r − rⱼ|. Zero out cells that fall
        # inside the receptor SC shape (Chen 2002 p284: "grid points in the
        # core of the receptor are assigned a value of 0 for the electric
        # potential, to avoid the contributions from non-physical
        # receptor-core/ligand contacts").
        #
        # The occupancy channel is `rec_sc_imag`: it is `rho` on the
        # solvent-excluding surface layer, `rho**2` in the core and exactly 0
        # in open space ("any grid point that does not correspond to an atom is
        # in the open space", Chen & Weng 2003). `rec_sc_real` must NOT be used
        # here: after the PSC rewrite it is the *count of receptor atoms within
        # `radius + D`*, so `rec_sc_real == 0` selects cells further than ~5.4 Å
        # from every receptor atom, i.e. bulk solvent. Conjoining it deleted the
        # whole contact band (measured on 1KXQ: 80% of Σ|V|; nearest surviving
        # cell 5.12 Å from any receptor atom, so every interface ligand atom saw
        # V = 0) and left beta and the charge LUT with identically zero gradient
        # for any contacting pose.
        rec_partial_q = partial_charge_per_atom(rec_charge_id, charge_score)
        V_rec = torch.zeros((nx, ny, nz), device=device, dtype=dtype)
        spread_neighbors_coulomb(
            V_rec, rec_xyz, rec_partial_q, rcut_elec,
            x_grid, y_grid, z_grid,
        )
        open_space_mask = rec_sc_imag <= 0
        V_rec = V_rec * open_space_mask.to(dtype)
    else:  # elec_mode == "legacy"
        # Original notebook behaviour: group by atomtype_id (B9), compute
        # per-type `count / Σ√d` pseudo-potential (B10). Preserved for
        # reproducing thesis numbers bit-for-bit against the original Julia
        # reference.
        rec_in_charge = (rec_atomtype_id >= 1) & (rec_atomtype_id <= 11)
        rec_group_charge = torch.where(
            rec_in_charge, rec_atomtype_id - 1, torch.full_like(rec_atomtype_id, -1)
        ).to(torch.long)
        rec_xyz_c = rec_xyz[rec_in_charge]
        rec_group_c = rec_group_charge[rec_in_charge]
        U_num = torch.zeros((11, nx, ny, nz), device=device, dtype=dtype)
        U_den = torch.zeros((11, nx, ny, nz), device=device, dtype=dtype)
        _grouped_spread_nearest_add(
            U_num, rec_xyz_c, rec_group_c,
            torch.ones(rec_xyz_c.shape[0], device=device, dtype=dtype),
            x_grid, y_grid, z_grid,
        )
        _grouped_calculate_distance(
            U_den, rec_xyz_c, rec_group_c, rcut_elec,
            x_grid, y_grid, z_grid,
        )
        eps = torch.finfo(dtype).eps
        U = torch.where(U_den > 0, U_num / U_den.clamp(min=eps), torch.zeros_like(U_num))

    # Ligand-side processing is F-linear in memory; extracted into
    # `_score_ligand_chunk` so that we can loop over frame chunks and
    # optionally checkpoint each chunk. Receptor grids above are reused.
    V_rec_or_U = V_rec if elec_mode == "coulomb" else U

    F_total = lig_xyz.shape[0]
    chunk_kwargs = dict(
        lig_radius=lig_radius, lig_sasa=lig_sasa,
        lig_atomtype_id=lig_atomtype_id, lig_charge_id=lig_charge_id,
        x_grid=x_grid, y_grid=y_grid, z_grid=z_grid,
        surface_threshold=surface_threshold, elec_mode=elec_mode,
        scatter_mode=scatter_mode, sc_reference_spacing=sc_reference_spacing,
        sc_rho=sc_rho, psc_d=psc_d,
        psc_decompose=psc_decompose,
        rec_sc_surf=rec_sc_surf, rec_sc_core=rec_sc_core,
    )
    use_chunks = (
        frame_chunk_size is not None
        and frame_chunk_size > 0
        and frame_chunk_size < F_total
    )

    if return_components:
        # Feature-extraction path: return (score_sc, T, score_elec),
        # concatenated over frame chunks. No checkpointing (used under
        # no_grad for one-time caching).
        step = frame_chunk_size if use_chunks else F_total
        sc_parts, T_parts, elec_parts = [], [], []
        for s in range(0, F_total, step):
            e = min(s + step, F_total)
            sc_c, T_c, elec_c = _score_ligand_chunk(
                lig_xyz[s:e], alpha, iface_matrix, beta, charge_score,
                H, rec_sc_real, rec_sc_imag, V_rec_or_U,
                return_components=True, **chunk_kwargs,
            )
            sc_parts.append(sc_c)
            T_parts.append(T_c)
            elec_parts.append(elec_c)
        return (
            torch.cat(sc_parts, dim=0),
            torch.cat(T_parts, dim=0),
            torch.cat(elec_parts, dim=0),
        )

    if not use_chunks:
        return _score_ligand_chunk(
            lig_xyz, alpha, iface_matrix, beta, charge_score,
            H, rec_sc_real, rec_sc_imag, V_rec_or_U,
            **chunk_kwargs,
        )

    # Checkpoint only pays off when something downstream is collecting
    # autograd; under `torch.no_grad()` we just loop to cap peak memory.
    use_checkpoint = torch.is_grad_enabled() and any(
        t.requires_grad for t in (alpha, iface_matrix, beta, charge_score, V_rec_or_U)
    )

    def _run_chunk(
        lxc, a, im, b, cs, Ht, rsr, rsi, vru,
    ):
        return _score_ligand_chunk(
            lxc, a, im, b, cs, Ht, rsr, rsi, vru, **chunk_kwargs,
        )

    parts: list[torch.Tensor] = []
    for s in range(0, F_total, frame_chunk_size):
        e = min(s + frame_chunk_size, F_total)
        lxc = lig_xyz[s:e]
        if use_checkpoint:
            scores_chunk = checkpoint(
                _run_chunk,
                lxc, alpha, iface_matrix, beta, charge_score,
                H, rec_sc_real, rec_sc_imag, V_rec_or_U,
                use_reentrant=False,
            )
        else:
            scores_chunk = _run_chunk(
                lxc, alpha, iface_matrix, beta, charge_score,
                H, rec_sc_real, rec_sc_imag, V_rec_or_U,
            )
        parts.append(scores_chunk)
    return torch.cat(parts, dim=0)
