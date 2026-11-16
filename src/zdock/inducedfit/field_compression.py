"""Opt-in compression of prior-whitened receptor field responses.

For mode amplitudes a, let D be the derivative of the *weighted scoring
channels* and K the original positive prior Hessian.  A randomized range
sketch of A = D K**(-1/2) selects right singular directions.  The retained
coordinates are mapped back to the original nonlinear model and whitened
in its actual CA displacement metric.  This changes the allowed deformation
subspace; it is not an exact acceleration unless the full subspace survives.

This CPU float64 preprocessing prototype streams composite-direction JVPs.
It stores q=min(M, rank+oversampling) sketch fields, then streams M more JVPs
to form Q.T @ A.  It never stacks the M individual derivative fields.  QR
workspaces and field/FFT intermediates still contribute to peak memory; when
q=M, even the sketch occupies M fields.  Only the *subsequent* derivative
FFT cache has rank instead of M channel sets.  No end-to-end memory or speed
improvement is promised by this construction alone.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import torch

from .fields import FieldParameters, jvp_receptor_fields
from .types import GridSpec, MonomerInput


class CompressibleModeModel(Protocol):
    reference: torch.Tensor
    ca_indices: torch.Tensor
    n_modes: int
    basis: torch.Tensor
    stiffness: torch.Tensor

    def coordinates(self, amplitudes: torch.Tensor) -> torch.Tensor: ...
    def deformation_energy(self, amplitudes: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class CompressedModeModel:
    """CA-normalized subspace of an unchanged base nonlinear model.

    ``transform`` has shape (base M, retained r), so base amplitudes are T b.
    ``stiffness`` is T.T K T.  The original deformation energy, including its
    original regularizer, is composed with T without refitting coefficients.
    The CA metric is infinitesimal; finite displacement checks remain needed.
    """

    base_model: CompressibleModeModel
    transform: torch.Tensor
    basis: torch.Tensor
    stiffness: torch.Tensor
    diagnostics: dict

    @property
    def reference(self) -> torch.Tensor:
        return self.base_model.reference

    @property
    def ca_indices(self) -> torch.Tensor:
        return self.base_model.ca_indices

    @property
    def n_modes(self) -> int:
        return self.transform.shape[1]

    def _base_amplitudes(self, amplitudes: torch.Tensor) -> torch.Tensor:
        # Metadata checks only: keep coordinates/energy compatible with AD.
        if amplitudes.shape != (self.n_modes,):
            raise ValueError('compressed amplitudes have the wrong shape')
        if amplitudes.dtype != self.transform.dtype or amplitudes.device != self.transform.device:
            raise ValueError('compressed amplitudes must share transform dtype and device')
        return self.transform @ amplitudes

    def coordinates(self, amplitudes: torch.Tensor) -> torch.Tensor:
        return self.base_model.coordinates(self._base_amplitudes(amplitudes))

    def deformation_energy(self, amplitudes: torch.Tensor) -> torch.Tensor:
        return self.base_model.deformation_energy(self._base_amplitudes(amplitudes))


def _validate_inputs(receptor, model, parameters, rank, oversampling, seed, rank_rtol):
    if receptor.xyz.device.type != 'cpu' or receptor.xyz.dtype != torch.float64:
        raise TypeError('field compression prototype requires CPU float64 inputs')
    if type(model.n_modes) is not int or model.n_modes < 1:
        raise ValueError('base model must contain a positive number of modes')
    if type(rank) is not int or not 1 <= rank <= model.n_modes:
        raise ValueError('rank must be an integer in [1, base n_modes]')
    if type(oversampling) is not int or oversampling < 0:
        raise ValueError('oversampling must be a nonnegative integer')
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('seed must be an integer in [0, 2**63)')
    if isinstance(rank_rtol, bool) or not math.isfinite(rank_rtol) or not 0 < rank_rtol < 1:
        raise ValueError('rank_rtol must be finite and in (0, 1)')
    parameters.validate()
    for name in ('reference', 'basis', 'stiffness'):
        value = getattr(model, name)
        if not isinstance(value, torch.Tensor) or value.dtype != receptor.xyz.dtype or value.device != receptor.xyz.device:
            raise ValueError(f'model {name} must share receptor dtype and device')
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f'model {name} must be finite')
    if not torch.equal(model.reference, receptor.xyz):
        raise ValueError('model reference must match receptor coordinates')
    if model.basis.shape != (model.n_modes, *receptor.xyz.shape):
        raise ValueError('model basis must have shape (M, N_atoms, 3)')
    if model.stiffness.shape != (model.n_modes, model.n_modes):
        raise ValueError('model stiffness must have shape (M, M)')
    if not torch.allclose(model.stiffness, model.stiffness.T, atol=1e-12, rtol=1e-10):
        raise ValueError('model stiffness must be symmetric')
    ca = model.ca_indices
    if (not isinstance(ca, torch.Tensor) or ca.dtype != torch.long or ca.device != receptor.xyz.device
            or ca.ndim != 1 or ca.numel() == 0 or bool((ca < 0).any())
            or bool((ca >= receptor.n_atoms).any()) or ca.unique().numel() != ca.numel()):
        raise ValueError('model CA indices must be unique nonempty in-range CPU long indices')
    ca_basis = model.basis[:, ca].reshape(model.n_modes, -1)
    base_metric = ca_basis @ ca_basis.T / ca.numel()
    # Existing amplitude-ball solvers assume this normalization.  A different
    # base metric would change the full-rank feasible set, so reject it.
    if not torch.allclose(base_metric, torch.eye(model.n_modes, dtype=receptor.xyz.dtype), atol=1e-7, rtol=1e-7):
        raise ValueError('base model must have an identity CA displacement metric')
    zero = receptor.xyz.new_zeros(model.n_modes)
    if not torch.allclose(model.coordinates(zero), receptor.xyz, atol=1e-12, rtol=1e-12):
        raise ValueError('base coordinates at zero must reproduce the receptor')
    origin_energy = model.deformation_energy(zero)
    if origin_energy.numel() != 1 or not bool(torch.isfinite(origin_energy)) or abs(float(origin_energy)) > 1e-12:
        raise ValueError('base deformation energy at zero must be finite and zero')
    return base_metric


@torch.no_grad()
def compress_response_fields(
    receptor: MonomerInput,
    model: CompressibleModeModel,
    grid: GridSpec,
    parameters: FieldParameters,
    *,
    rank: int,
    oversampling: int = 4,
    seed: int = 0,
    rank_rtol: float = 1e-10,
) -> CompressedModeModel:
    """Sketch D K**(-1/2), stream its projected matrix, return r CA modes.

    D is defined using the supplied model's coordinate tangents ``basis``.
    The error norm is ordinary Euclidean/Frobenius norm of flattened
    ``receptor_channels`` on this fixed grid, including their existing energy
    weights.  No extra channel normalization or voxel factor is introduced.
    This metric does not guarantee pose rankings, bounded-response errors or
    finite-coordinate energies.  Seed and dimensions must be fixed without
    reference-complex labels.  No native coordinates are accepted.

    Rank deficiency is an explicit error; requested rank is never silently
    changed.  The random generator is local and leaves global RNG untouched.
    Differentiation through sketch/rank selection is not supported; the
    returned coordinates and deformation energy remain differentiable.
    """
    base_metric = _validate_inputs(receptor, model, parameters, rank, oversampling, seed, rank_rtol)
    stiffness = .5 * (model.stiffness + model.stiffness.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(stiffness)
    if not bool((eigenvalues > 0).all()):
        raise ValueError('model stiffness must be positive definite')
    inverse_root = (eigenvectors * eigenvalues.rsqrt()[None, :]) @ eigenvectors.T
    if not bool(torch.isfinite(inverse_root).all()):
        raise FloatingPointError('inverse prior square root is not finite')
    m = model.n_modes
    q = min(m, rank + oversampling)
    field_size = 3 * grid.n_points
    if q > field_size:
        raise ValueError('sketch dimension exceeds the field dimension')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    omega = torch.randn(m, q, generator=generator, dtype=receptor.xyz.dtype)
    jvp_calls = 0

    def apply_a(vector):
        nonlocal jvp_calls
        direction = torch.einsum('m,mnc->nc', inverse_root @ vector, model.basis)
        _, tangent = jvp_receptor_fields(receptor.xyz, receptor.radii, receptor.charges,
                                         direction, grid, parameters)
        flat = tangent.detach().reshape(-1)
        if flat.numel() != field_size or not bool(torch.isfinite(flat).all()):
            raise FloatingPointError('field response has invalid shape or nonfinite values')
        jvp_calls += 1
        return flat

    sketch = receptor.xyz.new_empty(field_size, q)
    for column in range(q):
        sketch[:, column] = apply_a(omega[:, column])
    orthogonal, triangular = torch.linalg.qr(sketch, mode='reduced')
    del sketch
    sketch_left, sketch_values, _ = torch.linalg.svd(triangular, full_matrices=False)
    retained_sketch = sketch_values > rank_rtol * sketch_values.max()
    sketch_rank = int(retained_sketch.sum())
    if sketch_rank < rank:
        raise ValueError(f'field sketch has numerical rank {sketch_rank}, below requested {rank}')
    if sketch_rank < q:
        orthogonal = orthogonal @ sketch_left[:, retained_sketch]

    projected = receptor.xyz.new_empty(sketch_rank, m)
    frobenius_squared = receptor.xyz.new_zeros(())
    # Stream the second pass: the M individual field columns never coexist.
    for column in range(m):
        unit = receptor.xyz.new_zeros(m)
        unit[column] = 1.
        field_column = apply_a(unit)
        projected[:, column] = orthogonal.T @ field_column
        frobenius_squared += torch.dot(field_column, field_column)
    del orthogonal, field_column
    if not bool(torch.isfinite(projected).all()) or not bool(torch.isfinite(frobenius_squared)):
        raise FloatingPointError('projected response or squared field norm is not representable')
    _, values, right_h = torch.linalg.svd(projected, full_matrices=False)
    field_rank = int((values > rank_rtol * values.max()).sum())
    if field_rank < rank:
        raise ValueError(f'projected field has numerical rank {field_rank}, below requested {rank}')
    prior_whitened_transform = inverse_root @ right_h[:rank].T
    metric = prior_whitened_transform.T @ base_metric @ prior_whitened_transform
    metric = .5 * (metric + metric.T)
    metric_values, metric_vectors = torch.linalg.eigh(metric)
    if not bool(torch.isfinite(metric_values).all()) or not bool((metric_values > rank_rtol * metric_values.abs().max()).all()):
        raise ValueError('retained directions have a singular or ill-conditioned CA metric')
    metric_inverse_root = (metric_vectors * metric_values.rsqrt()[None, :]) @ metric_vectors.T
    transform = prior_whitened_transform @ metric_inverse_root
    basis = torch.einsum('mr,mnc->rnc', transform, model.basis)
    reduced_stiffness = transform.T @ stiffness @ transform
    reduced_stiffness = .5 * (reduced_stiffness + reduced_stiffness.T)
    if (not bool(torch.isfinite(transform).all()) or not bool(torch.isfinite(basis).all())
            or not bool(torch.isfinite(reduced_stiffness).all())
            or not bool((torch.linalg.eigvalsh(reduced_stiffness) > 0).all())):
        raise FloatingPointError('compressed coordinates or stiffness are invalid')
    actual_ca = basis[:, model.ca_indices].reshape(rank, -1)
    actual_metric = actual_ca @ actual_ca.T / len(model.ca_indices)
    metric_error = float((actual_metric - torch.eye(rank, dtype=basis.dtype)).abs().max())
    if metric_error > 1e-7:
        raise FloatingPointError('compressed CA metric failed normalization')

    total_squared = float(frobenius_squared)
    range_residual_squared = max(0., total_squared - float(projected.square().sum()))
    sketch_residual_squared = max(0., total_squared - float(values[:rank].square().sum()))
    diagnostics = {
        'method': 'randomized_prior_whitened_field_range_sketch_streamed_second_pass',
        'opt_in_model_change': True,
        'base_n_modes': m, 'actual_modes': rank, 'requested_modes': rank,
        'oversampling': oversampling, 'sketch_columns': q, 'sketch_numerical_rank': sketch_rank,
        'projected_field_numerical_rank': field_rank, 'seed': seed, 'rank_rtol': rank_rtol,
        'field_metric': 'Euclidean norm of flattened weighted receptor_channels on supplied grid; no additional channel rescaling or quadrature weight',
        'field_vector_size': field_size,
        'prior_condition_number': float(eigenvalues[-1] / eigenvalues[0]),
        'projected_singular_values': values.tolist(),
        'field_frobenius_norm': math.sqrt(total_squared),
        'range_projection_residual_frobenius': math.sqrt(range_residual_squared),
        'sketched_rank_r_reconstruction_residual_frobenius': math.sqrt(sketch_residual_squared),
        'residual_note': 'The sketched reconstruction residual bounds the right-subspace field residual from above; it is not a pose-response error bound.',
        'CA_metric_max_abs_error': metric_error,
        'amplitude_metric_max_abs_error': float((transform.T @ transform - torch.eye(rank, dtype=transform.dtype)).abs().max()),
        'prior': 'original nonlinear base deformation_energy(transform @ b); original regularization unchanged',
        'jvp_calls': jvp_calls,
        'theoretical_sketch_bytes': field_size * q * receptor.xyz.element_size(),
        'theoretical_projected_matrix_bytes': sketch_rank * m * receptor.xyz.element_size(),
        'subsequent_derivative_fft_channel_sets_before': m,
        'subsequent_derivative_fft_channel_sets_after': rank,
        'memory_note': 'Sketch bytes are not peak RSS: QR/Q, field/JVP/FFT intermediates and allocator storage are additional. q=M retains M sketch fields. No peak-memory or speed reduction measured here.',
        'native_labels_used': False,
    }
    return CompressedModeModel(model, transform.detach(), basis.detach(), reduced_stiffness.detach(), diagnostics)
