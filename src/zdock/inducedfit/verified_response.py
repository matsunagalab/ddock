"""Verify frozen response proposals using the actual nonlinear energy.

This is a reference-free selection operation.  A predicted response bonus is
never used as the returned score: accepted proposals use their actual energy,
and rejected proposals fall back to the recomputed rigid energy at a=0.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import torch

from zdock.geom import rotate
from .fields import FieldParameters, build_fields, ligand_channels, receptor_channels
from .search import ResponseCandidate
from .types import GridSpec, MonomerInput


class ResponseModel(Protocol):
    reference: torch.Tensor
    ca_indices: torch.Tensor
    stiffness: torch.Tensor
    n_modes: int

    def coordinates(self, amplitudes: torch.Tensor) -> torch.Tensor: ...
    def deformation_energy(self, amplitudes: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class VerifiedResponse:
    original_candidate: ResponseCandidate
    score: float
    rigid_energy: float
    verified_gain: float
    predicted_gain: float
    amplitudes: torch.Tensor
    strain: float
    accepted_scale: float | None
    fallback: bool
    rigid_energy_error: float
    trials: tuple[dict, ...]

    @property
    def rotation_index(self) -> int:
        return self.original_candidate.rotation_index

    @property
    def translation(self) -> torch.Tensor:
        return self.original_candidate.translation


class NonlinearResponseVerifier:
    """Cache rigid receptor channels and backtrack each candidate direction.

    Default scales are 1, 1/2, ..., 1/32.  Acceptance requires actual geometry
    feasibility, complete field support, a strict energy decrease and
    ``actual_gain / predicted_gain >= min_agreement`` with both gains negative.
    Geometry/support infeasibility is an ordinary rejected trial.  Nonfinite
    values, an invalid model, or failure to reproduce rigid energy raise an
    exception; they must not be silently converted to a successful fallback.
    """

    @torch.no_grad()
    def __init__(
        self,
        receptor: MonomerInput,
        ligand: MonomerInput,
        model: ResponseModel,
        grid: GridSpec,
        parameters: FieldParameters,
        *,
        radius: float,
        max_atom_displacement: float,
        min_agreement: float = 0.75,
        max_trials: int = 6,
        rigid_atol: float | None = None,
        rigid_rtol: float | None = None,
    ):
        if not all(math.isfinite(value) and value >= 0 for value in (radius, max_atom_displacement)):
            raise ValueError("displacement bounds must be finite and nonnegative")
        if not math.isfinite(min_agreement) or not 0 < min_agreement <= 1:
            raise ValueError("min_agreement must lie in (0, 1]")
        if type(max_trials) is not int or max_trials < 1:
            raise ValueError("max_trials must be a positive integer")
        if receptor.xyz.dtype not in (torch.float32, torch.float64):
            raise TypeError("verification requires float32 or float64")
        if any(value.dtype != receptor.xyz.dtype or value.device != receptor.xyz.device
               for value in (ligand.xyz, model.reference, model.stiffness)):
            raise ValueError("monomers and model must share dtype and device")
        if not torch.equal(model.reference, receptor.xyz):
            raise ValueError("model reference must equal receptor coordinates")
        if type(model.n_modes) is not int or model.n_modes < 0 or model.stiffness.shape != (model.n_modes, model.n_modes):
            raise ValueError("model mode count/stiffness shape is invalid")
        if not bool(torch.isfinite(model.stiffness).all()):
            raise ValueError("model stiffness must be finite")
        if model.ca_indices.dtype != torch.long or model.ca_indices.ndim != 1 or len(model.ca_indices) == 0:
            raise ValueError("model CA indices must be a nonempty long vector")
        if model.ca_indices.device != receptor.xyz.device or bool((model.ca_indices < 0).any()) or bool((model.ca_indices >= receptor.n_atoms).any()):
            raise ValueError("model CA indices must share device and lie in range")
        self.rigid_atol = (1e-7 if receptor.xyz.dtype == torch.float64 else 1e-4) if rigid_atol is None else rigid_atol
        self.rigid_rtol = (1e-9 if receptor.xyz.dtype == torch.float64 else 1e-5) if rigid_rtol is None else rigid_rtol
        if not all(math.isfinite(value) and value >= 0 for value in (self.rigid_atol, self.rigid_rtol)):
            raise ValueError("rigid energy tolerances must be finite and nonnegative")
        self.receptor, self.ligand, self.model = receptor, ligand, model
        self.grid, self.parameters = grid, parameters
        self.radius, self.max_atom_displacement = float(radius), float(max_atom_displacement)
        self.min_agreement, self.max_trials = float(min_agreement), max_trials
        zero = receptor.xyz.new_zeros(model.n_modes)
        zero_coordinates, zero_strain = model.coordinates(zero), model.deformation_energy(zero)
        if not bool(torch.isfinite(zero_coordinates).all()) or not bool(torch.isfinite(zero_strain)):
            raise FloatingPointError("nonfinite model at zero amplitude")
        tolerance = 64 * torch.finfo(receptor.xyz.dtype).eps
        if not torch.allclose(zero_coordinates, receptor.xyz, atol=tolerance, rtol=tolerance) or abs(float(zero_strain)) > tolerance:
            raise ValueError("zero amplitude must reproduce the receptor with zero deformation cost")
        fields = build_fields(receptor.xyz, receptor.radii, receptor.charges, grid, parameters)
        self._rigid_channels = receptor_channels(fields, grid, parameters)
        if not bool(torch.isfinite(self._rigid_channels).all()):
            raise FloatingPointError("nonfinite rigid receptor channels")

    @torch.no_grad()
    def verify(self, candidate: ResponseCandidate, quaternion: torch.Tensor) -> VerifiedResponse:
        """Verify one pose; ligand fields are reused across its trial scales."""
        ref, model = self.receptor.xyz, self.model
        for value, shape, label in ((quaternion, (4,), "quaternion"),
                                    (candidate.translation, (3,), "translation"),
                                    (candidate.amplitudes, (model.n_modes,), "amplitudes")):
            if value.shape != shape or value.dtype != ref.dtype or value.device != ref.device:
                raise ValueError(f"{label} has an incompatible shape, dtype or device")
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"nonfinite {label}")
        if not torch.allclose(quaternion.norm(), quaternion.new_tensor(1.), atol=1e-6, rtol=1e-6):
            raise ValueError("quaternion must be normalized")
        if not all(math.isfinite(value) for value in (candidate.score, candidate.rigid_energy, candidate.predicted_gain, candidate.strain)):
            raise FloatingPointError("nonfinite input candidate energy")
        ligand_xyz = torch.stack(rotate(*self.ligand.xyz.unbind(-1), quaternion), dim=-1) + candidate.translation
        ligand_fields = build_fields(ligand_xyz, self.ligand.radii, self.ligand.charges, self.grid, self.parameters)
        ligand_values = ligand_channels(ligand_fields)
        rigid_energy = float((self._rigid_channels * ligand_values).sum())
        if not math.isfinite(rigid_energy):
            raise FloatingPointError("nonfinite recomputed rigid energy")
        rigid_error = rigid_energy - candidate.rigid_energy
        if not math.isclose(rigid_energy, candidate.rigid_energy, abs_tol=self.rigid_atol, rel_tol=self.rigid_rtol):
            raise ValueError(f"rigid energy mismatch: recomputed minus candidate = {rigid_error}")
        amplitudes = candidate.amplitudes
        quadratic_strain = float(0.5 * amplitudes @ model.stiffness @ amplitudes)
        if not math.isfinite(quadratic_strain) or quadratic_strain < 0:
            raise FloatingPointError("invalid quadratic strain")
        linear_term = candidate.predicted_gain - quadratic_strain
        trials = []
        geometry_tolerance = 1e-10 if ref.dtype == torch.float64 else 1e-5
        for index in range(self.max_trials):
            scale = 0.5 ** index
            a = amplitudes * scale
            prediction = scale * linear_term + scale * scale * quadratic_strain
            coordinates = model.coordinates(a)
            if coordinates.shape != ref.shape or not bool(torch.isfinite(coordinates).all()):
                raise FloatingPointError("nonfinite or invalid nonlinear coordinates")
            displacement = coordinates - ref
            ca_rmsd = float(displacement[model.ca_indices].square().sum(-1).mean().sqrt())
            atom_max = float(torch.linalg.vector_norm(displacement, dim=-1).max())
            amplitude_norm = float(a.norm())
            if not all(math.isfinite(value) for value in (prediction, ca_rmsd, atom_max, amplitude_norm)):
                raise FloatingPointError("nonfinite predicted gain or geometry")
            trial = {
                "scale": scale, "predicted_gain": prediction, "amplitude_norm": amplitude_norm,
                "ca_rmsd": ca_rmsd, "maximum_atom_displacement": atom_max,
                "amplitude_bound_ok": amplitude_norm <= self.radius + geometry_tolerance,
                "actual_ca_bound_ok": ca_rmsd <= self.radius + geometry_tolerance,
                "actual_atom_bound_ok": atom_max <= self.max_atom_displacement + geometry_tolerance,
            }
            trials.append(trial)
            if not all(trial[key] for key in ("amplitude_bound_ok", "actual_ca_bound_ok", "actual_atom_bound_ok")):
                trial["status"] = "geometry_infeasible"
                continue
            try:
                fields = build_fields(coordinates, self.receptor.radii, self.receptor.charges, self.grid, self.parameters)
            except ValueError as error:
                if "compact support" not in str(error):
                    raise
                trial.update(status="support_infeasible", error=str(error))
                continue
            strain = float(model.deformation_energy(a))
            energy = float((receptor_channels(fields, self.grid, self.parameters) * ligand_values).sum()) + strain
            if not all(math.isfinite(value) for value in (energy, strain)):
                raise FloatingPointError("nonfinite actual nonlinear energy or strain")
            if strain < 0:
                raise ValueError("negative deformation strain")
            gain = energy - rigid_energy
            trial.update(actual_energy=energy, actual_gain=gain, actual_strain=strain)
            if prediction >= 0 or gain >= 0:
                trial["status"] = "no_predicted_or_actual_decrease"
                continue
            agreement = gain / prediction
            if not math.isfinite(agreement):
                raise FloatingPointError("nonfinite actual/predicted gain ratio")
            trial["agreement"] = agreement
            if agreement < self.min_agreement:
                trial["status"] = "poor_agreement"
                continue
            trial["status"] = "accepted"
            return VerifiedResponse(candidate, energy, rigid_energy, gain, prediction, a.clone(), strain,
                                    scale, False, rigid_error, tuple(trials))
        return VerifiedResponse(candidate, rigid_energy, rigid_energy, 0., 0., torch.zeros_like(amplitudes),
                                0., None, True, rigid_error, tuple(trials))
