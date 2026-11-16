"""Small, correctness-first FFT response search with supplied mode tangents.

The receptor and its directional derivative spectra are built once per pass.
Each ligand rotation uses one set of ligand spectra for both rigid energy and
all mode gradients.  Full linear correlations avoid periodic wrap-around;
only translations retaining the ligand's complete support are selectable.

This is a numerical prototype, not the production protein protocol: callers
supply charges, a fixed grid, modes and a matching positive stiffness. ANM,
internal-coordinate maps and nonlinear refinement are separate modules; this
function does not silently prepare or repair a protein.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from zdock.geom import rotate

from .fields import (
    FieldParameters,
    build_fields,
    correlate_from_spectra,
    correlation_spectra,
    jvp_receptor_fields,
    ligand_channels,
    receptor_channels,
    valid_translation_mask,
)
from .response import QuadraticResponse
from .types import GridSpec, MonomerInput


@dataclass(frozen=True)
class PreparedResponseReceptor:
    grid: GridSpec
    parameters: FieldParameters
    rigid_spectrum: torch.Tensor
    derivative_spectra: tuple[torch.Tensor, ...]
    response: QuadraticResponse
    maximum_atom_motion_per_unit_amplitude: float

    def bounded_radius(self, radius: float, max_atom_displacement: float | None = None) -> float:
        """Conservative *linear* per-atom bound for the Euclidean mode ball.

        Uses max_i ||B_i||_2, not the norm of each mode separately. Finite
        nonlinear coordinates must still be checked after applying a step.
        """
        if not math.isfinite(radius) or radius < 0:
            raise ValueError("radius must be finite and nonnegative")
        if max_atom_displacement is None:
            return radius
        if not math.isfinite(max_atom_displacement) or max_atom_displacement < 0:
            raise ValueError("max_atom_displacement must be finite and nonnegative")
        coefficient = self.maximum_atom_motion_per_unit_amplitude
        return radius if coefficient == 0 else min(radius, max_atom_displacement / coefficient)


@dataclass(frozen=True)
class ResponseCandidate:
    rotation_index: int
    translation: torch.Tensor
    amplitudes: torch.Tensor
    score: float
    rigid_energy: float
    predicted_gain: float
    strain: float


def prepare_response_receptor(
    receptor: MonomerInput,
    grid: GridSpec,
    parameters: FieldParameters,
    basis: torch.Tensor,
    stiffness: torch.Tensor,
) -> PreparedResponseReceptor:
    """Cache fixed-side spectra, with ``basis.shape == (M, N_atoms, 3)``.

    The radius passed to search is the Euclidean *amplitude* norm.  It is a
    small-displacement RMSD only if the caller orthonormalizes the supplied
    basis in the atomic/CA metric (B.T @ B / N = I, including cross terms).
    Stiffness must use that same
    basis.  This API does not infer a physical stiffness from optimizer state.
    """
    if basis.ndim != 3 or basis.shape[1:] != receptor.xyz.shape:
        raise ValueError("basis must have shape (M, N_receptor_atoms, 3)")
    if basis.dtype != receptor.xyz.dtype or basis.device != receptor.xyz.device:
        raise ValueError("basis and receptor must share dtype and device")
    if not torch.isfinite(basis).all():
        raise ValueError("basis must be finite")
    m = basis.shape[0]
    if stiffness.shape != (m, m):
        raise ValueError("stiffness must have shape (M, M)")
    if stiffness.dtype != basis.dtype or stiffness.device != basis.device:
        raise ValueError("stiffness and basis must share dtype and device")
    response = QuadraticResponse(stiffness.detach().clone())
    with torch.no_grad():
        fields = build_fields(receptor.xyz, receptor.radii, receptor.charges, grid, parameters)
        channels = receptor_channels(fields, grid, parameters)
        rigid_spectrum = correlation_spectra(channels).detach()
    derivatives = []
    for direction in basis:
        # Only this preparation step needs a differentiable field builder.
        with torch.enable_grad():
            _, tangent = jvp_receptor_fields(
                receptor.xyz, receptor.radii, receptor.charges, direction, grid, parameters
            )
        derivatives.append(correlation_spectra(tangent.detach()).detach())
    atom_coefficient = float(torch.linalg.matrix_norm(basis.permute(1, 2, 0), ord=2).max()) if m else 0.
    if not math.isfinite(atom_coefficient):
        raise FloatingPointError("per-atom response norm is not representable")
    return PreparedResponseReceptor(
        grid, parameters, rigid_spectrum, tuple(derivatives), response, atom_coefficient
    )


def _lag_at(index: int, shape: tuple[int, int, int], ref: torch.Tensor) -> torch.Tensor:
    sizes = tuple(2 * n - 1 for n in shape)
    ix, rest = divmod(index, sizes[1] * sizes[2])
    iy, iz = divmod(rest, sizes[2])
    return ref.new_tensor((ix - shape[0] + 1, iy - shape[1] + 1, iz - shape[2] + 1))


@torch.no_grad()
def search_response(
    prepared: PreparedResponseReceptor,
    ligand: MonomerInput,
    quaternions: torch.Tensor,
    *,
    radius: float,
    top_k: int = 50,
    peaks_per_rotation: int = 4,
    min_peak_distance: float = 3.0,
    response_tile_size: int = 4096,
    selection: str = "response",
    max_atom_displacement: float | None = None,
) -> list[ResponseCandidate]:
    """Select low-energy candidates; no coordinates are silently deformed.

    Quaternion convention is the legacy ``zdock.geom.rotate`` convention,
    with identity ``[0, 0, 0, 1]``.  Rotations are about the supplied origin;
    a caller preparing proteins must retain the inverse centering transform.
    ``selection='rigid'`` omits derivative correlations and response solves.
    Both arms use identical peak selection and non-wrapping validity rules.
    When supplied, max_atom_displacement conservatively bounds the linear
    prediction; callers must use prepared.bounded_radius() for the matching
    refinement amplitude constraint and check the true nonlinear displacement.
    """
    if selection not in ("rigid", "response"):
        raise ValueError("selection must be 'rigid' or 'response'")
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("radius must be finite and nonnegative")
    if any(type(value) is not int or value < 1 for value in (top_k, peaks_per_rotation, response_tile_size)):
        raise ValueError("candidate counts and tile size must be positive")
    if not math.isfinite(min_peak_distance) or min_peak_distance < 0:
        raise ValueError("min_peak_distance must be finite and nonnegative")
    if quaternions.ndim != 2 or quaternions.shape[1] != 4 or len(quaternions) == 0:
        raise ValueError("quaternions must have nonempty shape (N, 4)")
    if quaternions.dtype != ligand.xyz.dtype or quaternions.device != ligand.xyz.device:
        raise ValueError("quaternions and ligand must share dtype and device")
    if not torch.isfinite(quaternions).all() or not torch.allclose(
        torch.linalg.vector_norm(quaternions, dim=-1),
        quaternions.new_ones(len(quaternions)), atol=1e-6, rtol=1e-6,
    ):
        raise ValueError("quaternions must be finite and normalized")
    expected_dtype = prepared.rigid_spectrum.real.dtype
    if ligand.xyz.dtype != expected_dtype or ligand.xyz.device != prepared.rigid_spectrum.device:
        raise ValueError("prepared receptor and ligand must share dtype and device")

    grid, parameters = prepared.grid, prepared.parameters
    radius = prepared.bounded_radius(radius, max_atom_displacement)
    m = len(prepared.derivative_spectra)
    candidates: list[ResponseCandidate] = []
    for rotation_index, quaternion in enumerate(quaternions):
        xyz = torch.stack(rotate(*ligand.xyz.unbind(-1), quaternion), dim=-1)
        fields = build_fields(xyz, ligand.radii, ligand.charges, grid, parameters)
        ligand_spectrum = correlation_spectra(ligand_channels(fields))
        rigid = correlate_from_spectra(prepared.rigid_spectrum, ligand_spectrum, grid.shape)
        valid = valid_translation_mask(fields, grid)
        if not torch.isfinite(rigid[valid]).all():
            raise FloatingPointError("nonfinite rigid energy at a valid translation")
        score = rigid.clone().reshape(-1)
        gradients = None
        if selection == "response" and m:
            gradients = torch.stack([
                correlate_from_spectra(spectrum, ligand_spectrum, grid.shape)
                for spectrum in prepared.derivative_spectra
            ], dim=-1).reshape(-1, m)
            # Keep M gradient maps; tile the solve rather than retaining an
            # amplitude tensor for every translation and all iterations.
            valid_indices = valid.reshape(-1).nonzero().flatten()
            for indices in valid_indices.split(response_tile_size):
                solved = prepared.response.solve(gradients[indices], radius)
                score[indices] += solved.gain
        if not torch.isfinite(score[valid.reshape(-1)]).all():
            raise FloatingPointError("nonfinite corrected energy at a valid translation")
        score[~valid.reshape(-1)] = torch.inf
        for _ in range(peaks_per_rotation):
            value, index = score.min(dim=0)
            if not torch.isfinite(value):
                break
            flat_index = int(index)
            translation = _lag_at(flat_index, grid.shape, ligand.xyz) * grid.spacing
            if gradients is not None:
                solved = prepared.response.solve(gradients[flat_index], radius)
                amplitudes = solved.amplitudes.clone()
                gain, strain = float(solved.gain), float(solved.strain)
            else:
                amplitudes = ligand.xyz.new_zeros(m)
                gain, strain = 0.0, 0.0
            candidates.append(ResponseCandidate(
                rotation_index=rotation_index, translation=translation,
                amplitudes=amplitudes, score=float(value),
                rigid_energy=float(rigid.reshape(-1)[flat_index]),
                predicted_gain=gain, strain=strain,
            ))
            # Small NMS radius: suppress a bounded local box without storing
            # another full V×3 coordinate tensor for each rotation.
            sizes = score.reshape(rigid.shape).shape
            center_x, remainder = divmod(flat_index, sizes[1] * sizes[2])
            center_y, center_z = divmod(remainder, sizes[2])
            center = [center_x, center_y, center_z]
            cells = math.ceil(min_peak_distance / grid.spacing)
            lower = [max(0, center[d] - cells) for d in range(3)]
            upper = [min(sizes[d], center[d] + cells + 1) for d in range(3)]
            axes = [torch.arange(lower[d], upper[d], device=score.device) - center[d] for d in range(3)]
            distances = sum(axis.square() for axis in torch.meshgrid(*axes, indexing="ij"))
            suppress = distances * grid.spacing**2 <= min_peak_distance**2
            slices = tuple(slice(lower[d], upper[d]) for d in range(3))
            score.reshape(rigid.shape)[slices].masked_fill_(suppress, torch.inf)

    # Stable ordering makes equal-energy ties reproducible in the prototype.
    candidates.sort(key=lambda c: (c.score, c.rotation_index, tuple(c.translation.tolist())))
    return candidates[:top_k]
