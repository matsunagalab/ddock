"""Constrained local relaxation using the same smooth physical field energy.

This correctness-first projected-gradient/backtracking solver couples a rigid
pose to a supplied differentiable internal-coordinate model. It is not an
atomistic force field: stereochemical topology belongs to the deformation
model; Ramachandran statistics and nonbonded intramolecular energy are not
silently added. No labels or native structure are accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import torch

from zdock.geom import rotate
from .fields import FieldParameters, build_fields, physical_energy
from .types import GridSpec, MonomerInput


class DeformationModel(Protocol):
    reference: torch.Tensor
    ca_indices: torch.Tensor
    n_modes: int

    def coordinates(self, amplitudes: torch.Tensor) -> torch.Tensor: ...
    def deformation_energy(self, amplitudes: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class RelaxationResult:
    receptor_xyz: torch.Tensor
    ligand_xyz: torch.Tensor
    amplitudes: torch.Tensor
    rotation_increment: torch.Tensor
    translation_increment: torch.Tensor
    energy: float
    strain: float
    ca_rmsd: float
    maximum_atom_displacement: float
    status: str
    accepted_steps: int
    energy_evaluations: int
    energy_history: tuple[float, ...]


def _rotate_vector(xyz: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """Rodrigues with sinc coefficients, including a finite derivative at zero."""
    angle = torch.linalg.vector_norm(omega)
    first = torch.linalg.cross(omega.expand_as(xyz), xyz, dim=-1)
    second = torch.linalg.cross(omega.expand_as(xyz), first, dim=-1)
    return xyz + torch.sinc(angle / math.pi) * first + .5 * torch.sinc(angle / (2 * math.pi)).square() * second


def _project_ball(value: torch.Tensor, radius: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(value)
    scale = torch.clamp(radius / norm.clamp_min(torch.finfo(value.dtype).tiny), max=1.)
    return value * scale


def relax_pose(
    receptor: MonomerInput,
    ligand: MonomerInput,
    model: DeformationModel,
    grid: GridSpec,
    parameters: FieldParameters,
    quaternion: torch.Tensor,
    translation: torch.Tensor,
    *,
    radius: float,
    flexible: bool = True,
    initial_amplitudes: torch.Tensor | None = None,
    max_iterations: int = 50,
    max_evaluations: int = 250,
    max_translation: float = 2.,
    max_rotation: float = .35,
    max_atom_displacement: float = 3.,
    rotation_scale: float = .05,
    translation_scale: float = .5,
    amplitude_scale: float = .1,
    gradient_tolerance: float = 1e-6,
    checkpoint_fields: bool = False,
) -> RelaxationResult:
    """Jointly relax pose and deformation, starting at a=0 by default.

    Rotation increments are axis-angle radians about the initially rotated
    ligand centroid. Translation increments are angstrom. ``radius`` bounds
    BOTH amplitude norm and actual CA RMSD relative to the input. A separate
    atom-displacement bound is also checked. All trial energies rebuild the
    fields on ``grid``; invalid compact support is rejected by backtracking.
    """
    bounds = (radius, max_translation, max_rotation)
    scales = (max_atom_displacement, rotation_scale, translation_scale, amplitude_scale, gradient_tolerance)
    if not all(math.isfinite(v) and v >= 0 for v in bounds):
        raise ValueError("displacement/rotation bounds must be finite and nonnegative")
    if not all(math.isfinite(v) and v > 0 for v in scales):
        raise ValueError("scales and tolerances must be finite and positive")
    if any(type(n) is not int or n < 1 for n in (max_iterations, max_evaluations)):
        raise ValueError("iteration and evaluation budgets must be positive integers")
    ref = receptor.xyz
    if any(t.dtype != ref.dtype or t.device != ref.device for t in (ligand.xyz, quaternion, translation, model.reference)):
        raise ValueError("all inputs must share dtype and device")
    if model.reference.shape != ref.shape or not torch.equal(model.reference, ref):
        raise ValueError("deformation reference must match the receptor coordinates")
    if quaternion.shape != (4,) or translation.shape != (3,):
        raise ValueError("quaternion and translation require shapes (4,) and (3,)")
    if not torch.isfinite(quaternion).all() or not torch.isfinite(translation).all():
        raise ValueError("pose must be finite")
    if not torch.allclose(quaternion.norm(), quaternion.new_tensor(1.), atol=1e-6, rtol=1e-6):
        raise ValueError("quaternion must be normalized")

    m = model.n_modes
    initial_a = ref.new_zeros(m) if initial_amplitudes is None else initial_amplitudes.detach().clone()
    if initial_a.shape != (m,) or initial_a.dtype != ref.dtype or initial_a.device != ref.device:
        raise ValueError("initial amplitudes must have matching shape, dtype and device")
    if not torch.isfinite(initial_a).all() or initial_a.norm() > radius:
        raise ValueError("initial amplitudes must be finite and within the radius")
    if not flexible and bool((initial_a != 0).any()):
        raise ValueError("rigid relaxation requires zero amplitudes")
    y = torch.cat((ref.new_zeros(6), initial_a / amplitude_scale if flexible else ref.new_zeros(0)))
    rotated = torch.stack(rotate(*ligand.xyz.unbind(-1), quaternion), dim=-1).detach()
    center = rotated.mean(0)
    evaluations = 0

    def coordinates(variables):
        a = variables[6:] * amplitude_scale if flexible else initial_a
        rec_xyz = model.coordinates(a)
        lig_xyz = _rotate_vector(rotated - center, variables[:3] * rotation_scale) + center
        lig_xyz = lig_xyz + translation + variables[3:6] * translation_scale
        return rec_xyz, lig_xyz, a

    def geometrically_valid(rec_xyz):
        displacement = rec_xyz - ref
        ca_rmsd = displacement[model.ca_indices].square().sum(-1).mean().sqrt()
        max_atom = torch.linalg.vector_norm(displacement, dim=-1).max()
        return bool(ca_rmsd <= radius + 1e-10) and bool(max_atom <= max_atom_displacement + 1e-10)

    def objective(variables):
        nonlocal evaluations
        if evaluations >= max_evaluations:
            raise RuntimeError("evaluation_budget")
        evaluations += 1
        rec_xyz, lig_xyz, a = coordinates(variables)
        if not geometrically_valid(rec_xyz):
            raise ValueError("deformation exceeds actual-coordinate displacement bound")
        rec_fields = build_fields(rec_xyz, receptor.radii, receptor.charges, grid, parameters,
                                  checkpoint_chunks=checkpoint_fields)
        lig_fields = build_fields(lig_xyz, ligand.radii, ligand.charges, grid, parameters,
                                  checkpoint_chunks=checkpoint_fields)
        strain = model.deformation_energy(a)
        energy = physical_energy(rec_fields, lig_fields, grid, parameters) + strain
        if not torch.isfinite(energy):
            raise FloatingPointError("nonfinite nonlinear energy")
        return energy, strain

    def value_gradient(variables):
        with torch.enable_grad():
            differentiable = variables.detach().requires_grad_(True)
            energy, strain = objective(differentiable)
            gradient, = torch.autograd.grad(energy, differentiable)
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("nonfinite nonlinear energy gradient")
        return float(energy.detach()), float(strain.detach()), gradient.detach()

    def project(variables):
        parts = [
            _project_ball(variables[:3] * rotation_scale, max_rotation) / rotation_scale,
            _project_ball(variables[3:6] * translation_scale, max_translation) / translation_scale,
        ]
        if flexible:
            parts.append(_project_ball(variables[6:] * amplitude_scale, radius) / amplitude_scale)
        return torch.cat(parts)

    energy, strain, gradient = value_gradient(y)
    history = [energy]
    status = "max_iterations"
    for _ in range(max_iterations):
        if gradient.norm() <= gradient_tolerance:
            status = "converged_gradient"
            break
        if (project(y - gradient) - y).norm() < 1e-12:
            status = "converged_projected_step"
            break
        accepted = False
        step = 1.
        for _ in range(24):
            if evaluations >= max_evaluations:
                status = "evaluation_budget"
                break
            proposal = project(y - step * gradient)
            increment = proposal - y
            if increment.norm() < 1e-12:
                status = "line_search_step_too_small"
                break
            try:
                with torch.no_grad():
                    trial_energy, trial_strain = objective(proposal)
                trial_energy, trial_strain = float(trial_energy), float(trial_strain)
            except ValueError:
                # The objective's support/displacement checks are hard
                # feasibility constraints; do not crop or clamp fields.
                step *= .5
                continue
            if trial_energy <= energy + 1e-4 * float(gradient @ increment):
                y, energy, strain = proposal.detach(), trial_energy, trial_strain
                history.append(energy)
                accepted = True
                break
            step *= .5
        if not accepted:
            if status not in ("evaluation_budget", "line_search_step_too_small"):
                status = "line_search_stalled"
            break
        if evaluations >= max_evaluations:
            status = "evaluation_budget"
            break
        energy, strain, gradient = value_gradient(y)
    with torch.no_grad():
        rec_xyz, lig_xyz, amplitudes = coordinates(y)
        displacement = rec_xyz - ref
        ca_rmsd = float(displacement[model.ca_indices].square().sum(-1).mean().sqrt())
        max_atom = float(torch.linalg.vector_norm(displacement, dim=-1).max())
    return RelaxationResult(
        rec_xyz.detach(), lig_xyz.detach(), amplitudes.detach(),
        (y[:3] * rotation_scale).detach(), (y[3:6] * translation_scale).detach(),
        energy, strain, ca_rmsd, max_atom, status, len(history) - 1,
        evaluations, tuple(history),
    )
