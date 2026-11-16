"""Exact compact-support deposition of the smooth physical fields.

Only grid nodes outside every atom's analytic C2 support are omitted.  The
potential, energy model, charge normalization, and global FFT grid are not
approximated or cropped.  A conservative fixed local cube contains each
atom's support; scatter accumulation can differ from dense reduction by
floating-point roundoff (and may be nondeterministic on GPU).
"""

from __future__ import annotations

import math

import torch
from torch.utils.checkpoint import checkpoint

from .fields import FieldParameters, MolecularFields, _gaussian, _validate_atoms, support_radii
from .types import GridSpec


def _atom_stencil_values(
    xyz: torch.Tensor,
    radii: torch.Tensor,
    charges: torch.Tensor,
    offsets: torch.Tensor,
    grid: GridSpec,
    params: FieldParameters,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return flat grid indices and three atom contributions for one chunk."""
    origin = xyz.new_tensor(grid.origin)
    shape = torch.tensor(grid.shape, dtype=torch.long, device=xyz.device)
    # Integer topology is piecewise constant; no derivative through floor is
    # required.  The extra stencil margin ensures changing cells only adds or
    # removes identically zero contributions, including their first two
    # derivatives at the analytic support boundary.
    base = torch.floor((xyz - origin) / grid.spacing).to(torch.long)
    indices = base[:, None, :] + offsets[None, :, :]
    points = origin + indices.to(xyz.dtype) * grid.spacing
    valid = ((indices >= 0) & (indices < shape)).all(dim=-1)
    distance2 = (points - xyz[:, None, :]).square().sum(-1)
    sigma = radii[:, None] / math.sqrt(2.0 * math.log(2.0))
    expanded_sigma = (radii[:, None] + params.surface_width) / math.sqrt(2.0 * math.log(2.0))
    density = _gaussian(distance2, sigma, params.occupancy_switch_start, params.occupancy_cutoff_sigma) * valid
    expanded = _gaussian(distance2, expanded_sigma, params.occupancy_switch_start, params.occupancy_cutoff_sigma) * valid
    charge_weights = _gaussian(distance2, params.charge_sigma,
                               params.charge_switch_start, params.charge_cutoff_sigma) * valid
    normalizers = charge_weights.sum(dim=-1) * grid.voxel_volume
    if bool((normalizers <= torch.finfo(xyz.dtype).tiny).any()):
        raise ValueError("Charge Gaussian has no resolvable support on this grid")
    charge_density = charge_weights * (charges / normalizers)[:, None]
    # Outside-grid stencil points are analytically zero after the complete-
    # support validation.  Explicit masking keeps clamped scatter indices from
    # accidentally depositing their weights onto a boundary node.
    safe = torch.minimum(indices.clamp_min(0), shape - 1)
    flat = (safe[..., 0] * grid.shape[1] + safe[..., 1]) * grid.shape[2] + safe[..., 2]
    return flat.reshape(-1), density.reshape(-1), expanded.reshape(-1), charge_density.reshape(-1)


def build_sparse_fields(
    xyz: torch.Tensor,
    radii: torch.Tensor,
    charges: torch.Tensor,
    grid: GridSpec,
    params: FieldParameters | None = None,
    *,
    checkpoint_chunks: bool = False,
) -> MolecularFields:
    """Deposit identical finite-support fields using local atom stencils.

    The stencil half-width is ``ceil(max_support / spacing) + 1`` and uses
    the same Gaussian/C2 kernels as :func:`build_fields`.  Charge is normalized
    per atom over the retained grid nodes, with normalization differentiated.
    XYZ JVPs and ordinary reverse gradients of XYZ, radii, and charges are
    supported.  Discrete stencil selection has zero tangent; conservative
    padding preserves the continuous field when an atom crosses a grid cell.

    ``checkpoint_chunks=True`` checkpoints one atom chunk for reverse-mode
    autograd only.  It is optional; the default forward/JVP path has no
    checkpoint.  Grid-wide output arrays and FFT buffers remain dense.
    """
    if not isinstance(checkpoint_chunks, bool):
        raise TypeError("checkpoint_chunks must be a bool")
    if checkpoint_chunks and any(torch.autograd.forward_ad.unpack_dual(tensor).tangent is not None
                                 for tensor in (xyz, radii, charges)):
        raise ValueError("checkpoint_chunks is reverse-mode only; use False for forward AD/JVP")
    params = params or FieldParameters()
    _validate_atoms(xyz, radii, charges, grid, params)
    support = support_radii(radii, params)
    # Only the conservative integer work-list size is detached.  Radii and
    # charge normalization remain differentiable inside the physical kernel.
    half_width = math.ceil(float(support.detach().max()) / grid.spacing) + 1
    axis = torch.arange(-half_width, half_width + 1, dtype=torch.long, device=xyz.device)
    offsets = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(-1, 3)
    density = xyz.new_zeros(grid.n_points)
    expanded = torch.zeros_like(density)
    charge_density = torch.zeros_like(density)
    use_checkpoint = checkpoint_chunks and torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (xyz, radii, charges)
    )
    for start in range(0, len(xyz), params.atom_chunk_size):
        sl = slice(start, start + params.atom_chunk_size)
        args = (xyz[sl], radii[sl], charges[sl], offsets, grid, params)
        if use_checkpoint:
            indices, atom_density, atom_expanded, atom_charge = checkpoint(
                _atom_stencil_values, *args, use_reentrant=False, preserve_rng_state=False,
            )
        else:
            indices, atom_density, atom_expanded, atom_charge = _atom_stencil_values(*args)
        density = density.scatter_add(0, indices, atom_density)
        expanded = expanded.scatter_add(0, indices, atom_expanded)
        charge_density = charge_density.scatter_add(0, indices, atom_charge)
    occupancy = -torch.expm1(-density)
    surface = -torch.expm1(-expanded) - occupancy
    return MolecularFields(
        occupancy.reshape(grid.shape), surface.reshape(grid.shape), charge_density.reshape(grid.shape),
        (xyz - support[:, None]).amin(0), (xyz + support[:, None]).amax(0),
    )
