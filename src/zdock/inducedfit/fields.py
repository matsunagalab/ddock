"""Smooth coarse physical fields and nonperiodic FFT correlations.

All energies are in the effective unit epsilon (returned as E / epsilon).
The definition is a fixed-grid quadrature, not an atomistic force field.
FFT translations are signed integer lattice shifts of the ligand: positive
``t`` evaluates ``sum_x receptor[x] * ligand[x - t]``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.utils.checkpoint import checkpoint

from .types import GridSpec


@dataclass(frozen=True, slots=True)
class FieldParameters:
    surface_width: float = 1.0
    occupancy_switch_start: float = 4.0
    occupancy_cutoff_sigma: float = 5.0
    charge_sigma: float = 0.6
    charge_switch_start: float = 4.0
    charge_cutoff_sigma: float = 5.0
    screening_kappa: float = 0.1
    screening_softening: float = 1.0
    screening_switch_start: float = 16.0
    screening_cutoff: float = 20.0
    repulsion_weight: float = 1.0
    charge_weight: float = 1.0
    reference_volume: float = 1.0
    reference_length: float = 1.0
    atom_chunk_size: int = 32
    point_chunk_size: int = 2048
    backend: str = "dense"

    def validate(self) -> None:
        if self.backend not in ("dense", "sparse"):
            raise ValueError("field backend must be dense or sparse")
        values = (
            self.surface_width, self.charge_sigma, self.screening_softening,
            self.reference_volume, self.reference_length,
        )
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("Widths, softening and reference units must be positive finite values")
        pairs = (
            (self.occupancy_switch_start, self.occupancy_cutoff_sigma),
            (self.charge_switch_start, self.charge_cutoff_sigma),
            (self.screening_switch_start, self.screening_cutoff),
        )
        if not all(math.isfinite(a) and math.isfinite(b) and 0 <= a < b for a, b in pairs):
            raise ValueError("Each taper must satisfy 0 <= switch_start < cutoff")
        if not all(math.isfinite(v) and v >= 0 for v in (
            self.screening_kappa, self.repulsion_weight, self.charge_weight,
        )):
            raise ValueError("Kappa and energy weights must be finite and nonnegative")
        if self.atom_chunk_size < 1 or self.point_chunk_size < 1:
            raise ValueError("Chunk sizes must be positive")


@dataclass(frozen=True, slots=True)
class MolecularFields:
    occupancy: torch.Tensor
    surface: torch.Tensor
    charge_density: torch.Tensor
    support_lower: torch.Tensor
    support_upper: torch.Tensor


def _c2_switch(distance: torch.Tensor, start: float, cutoff: float) -> torch.Tensor:
    u = ((distance - start) / (cutoff - start)).clamp(0.0, 1.0)
    # Factored quintic: avoids subtractive cancellation and negative tails
    # close to the cutoff in float32.
    return (1.0 - u).pow(3) * (1.0 + 3.0 * u + 6.0 * u.square())


def _gaussian(squared_distance: torch.Tensor, sigma: torch.Tensor | float,
              start: float, cutoff: float) -> torch.Tensor:
    scaled2 = squared_distance / (sigma * sigma)
    # The floor avoids a sqrt derivative singularity at a grid-aligned atom.
    # It lies entirely inside the constant-one part of the taper.
    radial = scaled2.clamp_min(1e-24).sqrt()
    return torch.exp(-0.5 * scaled2) * _c2_switch(radial, start, cutoff)


def support_radii(radii: torch.Tensor, params: FieldParameters) -> torch.Tensor:
    """Conservative compact support of all three molecular fields (angstrom)."""
    shell = (radii + params.surface_width) / math.sqrt(2.0 * math.log(2.0))
    return torch.maximum(shell * params.occupancy_cutoff_sigma,
                         torch.full_like(radii, params.charge_sigma * params.charge_cutoff_sigma))


def _validate_atoms(xyz: torch.Tensor, radii: torch.Tensor, charges: torch.Tensor,
                    grid: GridSpec, params: FieldParameters) -> None:
    params.validate()
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] == 0:
        raise ValueError("xyz must have shape (N, 3) with N > 0")
    if radii.shape != xyz.shape[:1] or charges.shape != xyz.shape[:1]:
        raise ValueError("radii and charges must have shape (N,)")
    if not xyz.is_floating_point() or any(t.dtype != xyz.dtype or t.device != xyz.device
                                          for t in (radii, charges)):
        raise ValueError("All atom tensors must share floating dtype and device")
    if not all(bool(torch.isfinite(t).all()) for t in (xyz, radii, charges)):
        raise ValueError("Atom tensors must be finite")
    if not bool((radii > 0).all()):
        raise ValueError("Atom radii must be positive")
    lower, upper = grid.bounds(device=xyz.device, dtype=xyz.dtype)
    support = support_radii(radii, params)
    if bool(((xyz - support[:, None]) < lower).any()) or bool(((xyz + support[:, None]) > upper).any()):
        raise ValueError("Molecular compact support leaves the grid; enlarge GridSpec (no truncation allowed)")


def _charge_normalizer_atoms(atoms: torch.Tensor, points: torch.Tensor,
                             params: FieldParameters, voxel_volume: float) -> torch.Tensor:
    """One atom chunk, with explicit inputs for safe backward recomputation."""
    normalizer = atoms.new_zeros(atoms.shape[0])
    for start in range(0, points.shape[0], params.point_chunk_size):
        block = points[start:start + params.point_chunk_size]
        d2 = (block[:, None, :] - atoms[None, :, :]).square().sum(-1)
        weights = _gaussian(d2, params.charge_sigma,
                            params.charge_switch_start, params.charge_cutoff_sigma)
        normalizer = normalizer + weights.sum(0) * voxel_volume
    return normalizer


def _fields_at_points(block: torch.Tensor, xyz: torch.Tensor, sigmas: torch.Tensor,
                      expanded_sigmas: torch.Tensor, charges: torch.Tensor,
                      normalizers: torch.Tensor, params: FieldParameters
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One point chunk; never close over a changing atom/point loop index."""
    density = xyz.new_zeros(block.shape[0])
    expanded = torch.zeros_like(density)
    charge_density = torch.zeros_like(density)
    for start in range(0, xyz.shape[0], params.atom_chunk_size):
        sl = slice(start, start + params.atom_chunk_size)
        d2 = (block[:, None, :] - xyz[None, sl, :]).square().sum(-1)
        density = density + _gaussian(d2, sigmas[None, sl],
                                     params.occupancy_switch_start, params.occupancy_cutoff_sigma).sum(1)
        expanded = expanded + _gaussian(d2, expanded_sigmas[None, sl],
                                       params.occupancy_switch_start, params.occupancy_cutoff_sigma).sum(1)
        charge_weights = _gaussian(d2, params.charge_sigma,
                                   params.charge_switch_start, params.charge_cutoff_sigma)
        charge_density = charge_density + (charge_weights * (charges[sl] / normalizers[sl])[None, :]).sum(1)
    occupancy = -torch.expm1(-density)
    surface = -torch.expm1(-expanded) - occupancy
    return occupancy, surface, charge_density


def build_fields(xyz: torch.Tensor, radii: torch.Tensor, charges: torch.Tensor,
                 grid: GridSpec, params: FieldParameters | None = None, *,
                 checkpoint_chunks: bool = False) -> MolecularFields:
    """Build smooth fields without allocating an atom-by-entire-grid tensor.

    Occupancy Gaussians have unit peak, never integral normalization. Charge
    Gaussians are normalized separately for EACH atom on this discrete grid.
    Their normalization remains in the autograd graph.

    ``checkpoint_chunks=True`` uses non-reentrant activation checkpointing
    for ordinary reverse-mode ``torch.autograd.grad``/``backward``.  It
    recomputes one field point chunk or one normalizer atom chunk during
    backward, trading extra compute for fewer retained atom-by-grid tensors.
    It does NOT checkpoint the whole builder.  RNG preservation is disabled
    because these pure tensor functions contain no random operations.

    Use the default ``False`` for torch.func transforms/JVPs.  Forward AD
    explicitly rejects checkpoint mode; other torch.func transforms are not
    supported by the checkpoint path.  No-grad calls bypass checkpointing.
    See https://docs.pytorch.org/docs/stable/checkpoint.html .
    """
    params = params or FieldParameters()
    if params.backend == "sparse":
        from .sparse_fields import build_sparse_fields
        return build_sparse_fields(xyz, radii, charges, grid, params,
                                   checkpoint_chunks=checkpoint_chunks)
    if not isinstance(checkpoint_chunks, bool):
        raise TypeError("checkpoint_chunks must be a bool")
    if checkpoint_chunks and any(torch.autograd.forward_ad.unpack_dual(tensor).tangent is not None
                                 for tensor in (xyz, radii, charges)):
        raise ValueError("checkpoint_chunks is reverse-mode only; use False for forward AD/JVP")
    _validate_atoms(xyz, radii, charges, grid, params)
    points = grid.points(device=xyz.device, dtype=xyz.dtype)
    h3 = grid.voxel_volume
    atom_chunk, point_chunk = params.atom_chunk_size, params.point_chunk_size
    sigmas = radii / math.sqrt(2.0 * math.log(2.0))
    expanded_sigmas = (radii + params.surface_width) / math.sqrt(2.0 * math.log(2.0))
    use_checkpoint = checkpoint_chunks and torch.is_grad_enabled()

    # First pass obtains per-atom charge normalizers, retaining their JVP.
    normalizer_chunks = []
    for ia in range(0, xyz.shape[0], atom_chunk):
        atoms = xyz[ia:ia + atom_chunk]
        if use_checkpoint and atoms.requires_grad:
            normalizer = checkpoint(_charge_normalizer_atoms, atoms, points, params, h3,
                                    use_reentrant=False, preserve_rng_state=False)
        else:
            normalizer = _charge_normalizer_atoms(atoms, points, params, h3)
        normalizer_chunks.append(normalizer)
    normalizers = torch.cat(normalizer_chunks)
    if bool((normalizers <= torch.finfo(xyz.dtype).tiny).any()):
        raise ValueError("Charge Gaussian has no resolvable support on this grid")

    occupancy_chunks, surface_chunks, charge_chunks = [], [], []
    for ip in range(0, points.shape[0], point_chunk):
        block = points[ip:ip + point_chunk]
        args = (block, xyz, sigmas, expanded_sigmas, charges, normalizers, params)
        if use_checkpoint and any(t.requires_grad for t in (xyz, radii, charges)):
            occupancy, surface, charge_density = checkpoint(
                _fields_at_points, *args, use_reentrant=False, preserve_rng_state=False,
            )
        else:
            occupancy, surface, charge_density = _fields_at_points(*args)
        occupancy_chunks.append(occupancy)
        surface_chunks.append(surface)
        charge_chunks.append(charge_density)
    support = support_radii(radii, params)
    return MolecularFields(
        torch.cat(occupancy_chunks).reshape(grid.shape),
        torch.cat(surface_chunks).reshape(grid.shape),
        torch.cat(charge_chunks).reshape(grid.shape),
        (xyz - support[:, None]).amin(0),
        (xyz + support[:, None]).amax(0),
    )


def screened_kernel(distance: torch.Tensor, params: FieldParameters) -> torch.Tensor:
    """Regularized screened 1/r kernel, with its explicit C2 cutoff."""
    regularized = (distance.square() + params.screening_softening ** 2).sqrt()
    return (torch.exp(-params.screening_kappa * regularized) / regularized
            * _c2_switch(distance, params.screening_switch_start, params.screening_cutoff))


def kernel_grid(grid: GridSpec, params: FieldParameters, *, device=None, dtype=None) -> torch.Tensor:
    """Centered kernel on integer offsets including its complete compact support."""
    params.validate()
    extent = math.ceil(params.screening_cutoff / grid.spacing)
    offsets = torch.arange(-extent, extent + 1, device=device, dtype=dtype) * grid.spacing
    coords = torch.stack(torch.meshgrid(offsets, offsets, offsets, indexing="ij"), -1)
    return screened_kernel(coords.square().sum(-1).sqrt(), params)


def linear_convolve_same(density: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Linear convolution sampled on the density grid, never circular.

    ``kernel`` is centered at floor(shape/2) and must have odd dimensions.
    The full linear convolution is padded before the central region is read.
    Only the OUTPUT potential is cropped: all source density and all kernel
    contributions to every retained point remain present.
    """
    if density.ndim != 3 or kernel.ndim != 3 or any(n % 2 != 1 for n in kernel.shape):
        raise ValueError("Expected 3D density and odd-sized centered kernel")
    full_shape = tuple(n + k - 1 for n, k in zip(density.shape, kernel.shape))
    transformed = torch.fft.rfftn(density, s=full_shape) * torch.fft.rfftn(kernel, s=full_shape)
    full = torch.fft.irfftn(transformed, s=full_shape)
    crop = tuple(slice(k // 2, k // 2 + n) for n, k in zip(density.shape, kernel.shape))
    return full[crop]


def direct_linear_convolve_same(density: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Independent spatial convolution oracle (small numerical tests)."""
    if density.ndim != 3 or kernel.ndim != 3 or any(n % 2 != 1 for n in kernel.shape):
        raise ValueError("Expected 3D density and odd-sized centered kernel")
    # conv3d is cross-correlation; flipping its kernel yields convolution.
    padding = tuple(k // 2 for k in kernel.shape)
    return torch.nn.functional.conv3d(density[None, None], kernel.flip((0, 1, 2))[None, None],
                                      padding=padding)[0, 0]


def receptor_channels(fields: MolecularFields, grid: GridSpec,
                       params: FieldParameters | None = None, *, direct: bool = False) -> torch.Tensor:
    """Three energy-weighted channels, including the charge potential.

    Dotting these against ligand [occupancy, surface, charge density] gives
    E/epsilon. Potential = h^3 * (kernel * receptor charge density).
    """
    params = params or FieldParameters()
    params.validate()
    kernel = kernel_grid(grid, params, device=fields.charge_density.device, dtype=fields.charge_density.dtype)
    convolution = direct_linear_convolve_same if direct else linear_convolve_same
    potential = grid.voxel_volume * convolution(fields.charge_density, kernel)
    return torch.stack((
        params.repulsion_weight * grid.voxel_volume / params.reference_volume * fields.occupancy,
        -grid.voxel_volume / params.reference_volume * fields.surface,
        params.charge_weight * params.reference_length * grid.voxel_volume * potential,
    ))


def ligand_channels(fields: MolecularFields) -> torch.Tensor:
    return torch.stack((fields.occupancy, fields.surface, fields.charge_density))


def physical_energy(receptor: MolecularFields, ligand: MolecularFields,
                    grid: GridSpec, params: FieldParameters | None = None,
                    *, direct: bool = False) -> torch.Tensor:
    """Actual-coordinate fixed-grid energy; ``direct=True`` bypasses all FFTs."""
    return (receptor_channels(receptor, grid, params, direct=direct) * ligand_channels(ligand)).sum()


def correlation_spectra(channels: torch.Tensor) -> torch.Tensor:
    """Cache each channel's spectrum on the full linear-correlation grid."""
    if channels.ndim != 4:
        raise ValueError("channels must have shape (C, nx, ny, nz)")
    fft_shape = tuple(2 * n - 1 for n in channels.shape[-3:])
    return torch.fft.rfftn(channels, s=fft_shape, dim=(-3, -2, -1))


def correlate_from_spectra(receptor_spectra: torch.Tensor, ligand_spectra: torch.Tensor,
                           shape: tuple[int, int, int]) -> torch.Tensor:
    """Return signed-lag order, with zero shift at index (nx-1,ny-1,nz-1)."""
    if receptor_spectra.shape != ligand_spectra.shape:
        raise ValueError("Receptor and ligand spectra must have identical shape")
    fft_shape = tuple(2 * n - 1 for n in shape)
    products = (receptor_spectra * ligand_spectra.conj()).sum(0)
    cyclic_order = torch.fft.irfftn(products, s=fft_shape, dim=(-3, -2, -1))
    return torch.fft.fftshift(cyclic_order, dim=(-3, -2, -1))


def fft_correlate_channels(receptor: torch.Tensor, ligand: torch.Tensor) -> torch.Tensor:
    if receptor.shape != ligand.shape:
        raise ValueError("Receptor and ligand channels must have identical shape")
    return correlate_from_spectra(correlation_spectra(receptor), correlation_spectra(ligand),
                                  tuple(receptor.shape[-3:]))


def signed_lag_axes(shape: tuple[int, int, int], *, device=None) -> tuple[torch.Tensor, ...]:
    return tuple(torch.arange(-(n - 1), n, device=device) for n in shape)


def direct_correlation_at_shift(receptor: torch.Tensor, ligand: torch.Tensor,
                                shift: tuple[int, int, int]) -> torch.Tensor:
    """Independent nonperiodic overlap sum for one signed lattice shift."""
    if receptor.shape != ligand.shape or receptor.ndim != 4:
        raise ValueError("Expected equal (C,nx,ny,nz) channel tensors")
    rec_slices, lig_slices = [], []
    for n, t in zip(receptor.shape[-3:], shift):
        if abs(t) >= n:
            return receptor.sum() * 0.0
        rec_slices.append(slice(max(t, 0), min(n, n + t)))
        lig_slices.append(slice(max(-t, 0), min(n, n - t)))
    return (receptor[(slice(None), *rec_slices)] * ligand[(slice(None), *lig_slices)]).sum()


def valid_translation_mask(fields: MolecularFields, grid: GridSpec) -> torch.Tensor:
    """Only translations retaining the COMPLETE ligand field support are valid.

    Correlation values outside this mask are mathematical overlap sums and
    MUST NOT be interpreted as physical poses (part of the ligand left grid).
    """
    lower, upper = grid.bounds(device=fields.support_lower.device, dtype=fields.support_lower.dtype)
    shifts = signed_lag_axes(grid.shape, device=fields.support_lower.device)
    allowed = []
    for axis, lag in enumerate(shifts):
        shift = lag.to(fields.support_lower.dtype) * grid.spacing
        allowed.append((fields.support_lower[axis] + shift >= lower[axis])
                       & (fields.support_upper[axis] + shift <= upper[axis]))
    return allowed[0][:, None, None] & allowed[1][None, :, None] & allowed[2][None, None, :]


def jvp_receptor_fields(xyz: torch.Tensor, radii: torch.Tensor, charges: torch.Tensor,
                        direction: torch.Tensor, grid: GridSpec,
                        params: FieldParameters | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Directional derivative of all weighted fixed-side fields, including charge normalization."""
    if direction.shape != xyz.shape or direction.device != xyz.device or direction.dtype != xyz.dtype:
        raise ValueError("direction must have the same shape, dtype and device as xyz")
    params = params or FieldParameters()

    def field_function(coordinates: torch.Tensor) -> torch.Tensor:
        return receptor_channels(build_fields(coordinates, radii, charges, grid, params), grid, params)

    return torch.func.jvp(field_function, (xyz,), (direction,))
