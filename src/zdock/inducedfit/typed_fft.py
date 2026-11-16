"""Type-specific potential grids and C2 mass assignment for FFT proposals.

Grid correlation is an approximation to the continuous pair potential.
It is exact only relative to these sampled channels. Recheck shortlisted
poses with direct_pair_energy; coarse-grid scores are not direct energies.
"""
from __future__ import annotations
from dataclasses import dataclass
import math

import torch

from .typed_potential import (TypedParameters, TypedPotentialConfig, _validate_xyz,
                              lj_potential, charge_kernel)
from .types import GridSpec
from .fields import signed_lag_axes


@dataclass(frozen=True, eq=False)
class ProbeTypes:
    sigma: torch.Tensor
    epsilon: torch.Tensor
    atom_type: torch.Tensor

    @property
    def n_types(self):
        return len(self.sigma)


def build_probe_types(params: TypedParameters) -> ProbeTypes:
    params.validate()
    active = params.epsilon > 0
    rows = torch.stack((params.sigma[active], params.epsilon[active]), dim=-1)
    unique, inverse = torch.unique(rows, dim=0, sorted=True, return_inverse=True)
    mapping = torch.full((len(params.sigma),), -1, dtype=torch.long)
    mapping[active] = inverse
    return ProbeTypes(unique[:, 0], unique[:, 1], mapping)


def _check_probes(probes):
    if (probes.sigma.ndim != 1 or probes.epsilon.shape != probes.sigma.shape
            or probes.sigma.dtype != torch.float64 or probes.epsilon.dtype != torch.float64
            or probes.sigma.device.type != 'cpu' or probes.epsilon.device.type != 'cpu'
            or not bool(torch.isfinite(probes.sigma).all() and torch.isfinite(probes.epsilon).all())
            or not bool((probes.sigma > 0).all() and (probes.epsilon > 0).all())):
        raise ValueError('invalid nonzero LJ probe types')


def cardinal_prefilter(channels: torch.Tensor) -> torch.Tensor:
    """Cubic interpolation coefficients with ZERO ghost-coefficient BC.

    Invert the separable node matrix diag(2/3), offdiag(1/6) using Thomas
    sweeps. This is not a periodic FFT deconvolution and cannot wrap a field
    from one boundary onto the opposite boundary. Inputs should have their
    complete physical support within the grid; receptor_channels enforces it.
    The linear operation supports reverse AD and JVP. Interpolation may still
    overshoot on coarse grids: validation against the continuous potential is
    mandatory even though integer-node interpolation is exact.
    """
    if channels.ndim != 4 or any(n < 2 for n in channels.shape[1:]):
        raise ValueError('cardinal prefilter expects (C,nx,ny,nz), axes>=2')
    result = channels
    off = 1/6
    for axis in (1, 2, 3):
        moved = result.movedim(axis, 0)
        denominators = [2/3]
        forward = [moved[0]/denominators[0]]
        for i in range(1, len(moved)):
            denominators.append(2/3-off*off/denominators[-1])
            forward.append((moved[i]-off*forward[-1])/denominators[-1])
        backward = [forward[-1]]
        for i in range(len(moved)-2, -1, -1):
            backward.append(forward[i]-off/denominators[i]*backward[-1])
        result = torch.stack(backward[::-1], dim=0).movedim(0, axis)
    return result


def receptor_channels(rx: torch.Tensor, rparams: TypedParameters, probes: ProbeTypes,
                      grid: GridSpec, config: TypedPotentialConfig | None = None) -> torch.Tensor:
    """Return (n_nonzero_ligand_types+1, *shape) potentials in kJ/mol.

    Receptor radial potentials are evaluated directly on nodes. No Gaussian
    smoothing, voxel factor, or charge-density convolution is applied.
    Node lists are piecewise constant with padding beyond the C2 cutoff.
    Cardinal interpolation uses a nonperiodic coefficient prefilter; complete
    radial support must then fit to justify the zero exterior coefficients.
    The explicit 'mass' ablation omits that numerical correction.
    """
    config = config or TypedPotentialConfig()
    config.validate()
    _validate_xyz(rx, rparams)
    _check_probes(probes)
    if config.grid_interpolation == 'cardinal' and not grid.contains_support(rx, config.cutoff):
        raise ValueError('Complete receptor potential support leaves cardinal interpolation grid')
    origin = rx.new_tensor(grid.origin)
    shape = torch.tensor(grid.shape, dtype=torch.long)
    half = math.ceil(config.cutoff/grid.spacing)+1
    axis = torch.arange(-half, half+1, dtype=torch.long)
    offsets = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3)
    result = rx.new_zeros((probes.n_types+1, grid.n_points))
    for start in range(0, len(rx), config.atom_chunk_size):
        end = start+config.atom_chunk_size
        xyz = rx[start:end]
        base = torch.floor((xyz-origin)/grid.spacing).to(torch.long)
        indices = base[:, None, :] + offsets[None, :, :]
        valid = ((indices >= 0) & (indices < shape)).all(-1)
        points = origin + indices.to(rx.dtype)*grid.spacing
        d2 = (points-xyz[:, None, :]).square().sum(-1)
        safe = torch.minimum(indices.clamp_min(0), shape-1)
        flat = ((safe[..., 0]*grid.shape[1]+safe[..., 1])*grid.shape[2]+safe[..., 2]).reshape(-1)
        sigma = .5*(rparams.sigma[None, start:end, None]+probes.sigma[:, None, None])
        epsilon = (rparams.epsilon[None, start:end, None]*probes.epsilon[:, None, None]).sqrt()
        lj = lj_potential(d2[None, :, :], sigma, epsilon, config)*valid[None, :, :]
        q = charge_kernel(d2, config)*rparams.charges[start:end, None]*valid
        values = torch.cat((lj, q[None, :, :]), dim=0).reshape(probes.n_types+1, -1)
        result = result.scatter_add(1, flat[None, :].expand(probes.n_types+1, -1), values)
    fields = result.reshape(probes.n_types+1, *grid.shape)
    return cardinal_prefilter(fields) if config.grid_interpolation == 'cardinal' else fields


def _bspline(distance: torch.Tensor) -> torch.Tensor:
    s = distance.abs()
    return torch.where(s < 1, 2/3-s.square()+.5*s.pow(3),
                       torch.where(s < 2, (2-s).pow(3)/6, torch.zeros_like(s)))


def assignment_stencil(lx: torch.Tensor, grid: GridSpec):
    """Return flat (N,64) indices and differentiable normalized cubic weights."""
    origin = lx.new_tensor(grid.origin)
    scaled = (lx-origin)/grid.spacing
    base = torch.floor(scaled).to(torch.long)
    axis = torch.arange(-1, 3, dtype=torch.long)
    offsets = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3)
    indices = base[:, None, :] + offsets[None, :, :]
    shape = torch.tensor(grid.shape, dtype=torch.long)
    if not bool(((indices >= 0) & (indices < shape)).all()):
        raise ValueError('Complete ligand B-spline support leaves grid')
    weights = _bspline(scaled[:, None, :]-indices.to(lx.dtype)).prod(-1)
    weights = weights / weights.sum(-1, keepdim=True)
    flat = (indices[..., 0]*grid.shape[1]+indices[..., 1])*grid.shape[2]+indices[..., 2]
    return flat, weights


def ligand_channels(lx: torch.Tensor, lparams: TypedParameters, probes: ProbeTypes,
                    grid: GridSpec) -> torch.Tensor:
    """C2 mass assignment: each atom has mass one, charge mass q; no h³."""
    _validate_xyz(lx, lparams)
    _check_probes(probes)
    if probes.atom_type.shape != (len(lx),) or probes.atom_type.dtype != torch.long:
        raise ValueError('ProbeTypes must come from these ligand parameters')
    active = lparams.epsilon > 0
    if not torch.equal(probes.atom_type < 0, ~active):
        raise ValueError('zero-epsilon ligand type mapping differs')
    if bool(active.any()):
        mapping = probes.atom_type[active]
        if (bool((mapping < 0).any()) or bool((mapping >= probes.n_types).any())
                or not torch.equal(probes.sigma[mapping], lparams.sigma[active])
                or not torch.equal(probes.epsilon[mapping], lparams.epsilon[active])):
            raise ValueError('ligand type parameters differ from probe grid')
    flat, weights = assignment_stencil(lx, grid)
    count = (probes.n_types+1)*grid.n_points
    result = lx.new_zeros(count)
    if probes.n_types:
        combined = probes.atom_type.clamp_min(0)[:, None]*grid.n_points+flat
        result = result.scatter_add(0, combined.reshape(-1), (weights*active[:, None]).reshape(-1))
    charge_indices = probes.n_types*grid.n_points+flat
    result = result.scatter_add(0, charge_indices.reshape(-1), (weights*lparams.charges[:, None]).reshape(-1))
    return result.reshape(probes.n_types+1, *grid.shape)


def valid_translation_mask(lx: torch.Tensor, grid: GridSpec) -> torch.Tensor:
    """Conservative full continuous ±2h assignment support at every shift."""
    lower, upper = grid.bounds(dtype=lx.dtype, device=lx.device)
    support_lower, support_upper = (lx-2*grid.spacing).amin(0), (lx+2*grid.spacing).amax(0)
    allowed = []
    for axis, lag in enumerate(signed_lag_axes(grid.shape, device=lx.device)):
        shift = lag.to(lx.dtype)*grid.spacing
        allowed.append((support_lower[axis]+shift >= lower[axis]) & (support_upper[axis]+shift <= upper[axis]))
    return allowed[0][:, None, None] & allowed[1][None, :, None] & allowed[2][None, None, :]


def jvp_receptor_channels(rx: torch.Tensor, rparams: TypedParameters, probes: ProbeTypes,
                         grid: GridSpec, config: TypedPotentialConfig, direction: torch.Tensor):
    if direction.shape != rx.shape or direction.dtype != rx.dtype or direction.device != rx.device:
        raise ValueError('direction must match receptor coordinates')
    return torch.func.jvp(lambda xyz: receptor_channels(xyz, rparams, probes, grid, config), (rx,), (direction,))


def grid_pair_energy(rx: torch.Tensor, lx: torch.Tensor, rparams: TypedParameters,
                     lparams: TypedParameters, grid: GridSpec,
                     config: TypedPotentialConfig | None = None):
    probes = build_probe_types(lparams)
    return (receptor_channels(rx, rparams, probes, grid, config)
            *ligand_channels(lx, lparams, probes, grid)).sum()
