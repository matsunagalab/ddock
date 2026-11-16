"""Apo ANM directions projected into an anchored torsion coordinate map.

The ANM is used to choose directions, not to supply their final stiffness.
The actual torsion-map CA Jacobian defines the RMS displacement metric, and
the same nonlinear CA spring energy defines both refinement cost and response
stiffness.  Construction is preprocessing; differentiating through mode or
rank selection is not part of this module's contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import torch


class TorsionMap(Protocol):
    reference: torch.Tensor
    ca_indices: torch.Tensor
    n_torsions: int

    def coordinates(self, delta_angles: torch.Tensor) -> torch.Tensor: ...


def _positive_scalar(value: float, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite scalar")
    if value < 0 or (not allow_zero and value == 0):
        qualifier = "nonnegative" if allow_zero else "strictly positive"
        raise ValueError(f"{name} must be {qualifier}")
    return float(value)


def _check_xyz(xyz: torch.Tensor, name: str) -> None:
    if not isinstance(xyz, torch.Tensor) or xyz.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must be a float32 or float64 tensor")
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] < 2:
        raise ValueError(f"{name} must have shape (N, 3) with N >= 2")
    if not bool(torch.isfinite(xyz).all()):
        raise ValueError(f"{name} must be finite")


def _component_count(n_nodes: int, pairs: torch.Tensor) -> int:
    parents = list(range(n_nodes))

    def root(node: int) -> int:
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    for first, second in pairs.cpu().tolist():
        parents[root(first)] = root(second)
    return len({root(node) for node in range(n_nodes)})


@dataclass(frozen=True)
class ANMModel:
    """Central-force CA network with all numerical zero modes removed.

    ``modes`` has shape (M, N_CA, 3), with unit Euclidean norm per mode;
    ``eigenvalues`` contains their ascending positive Hessian eigenvalues.
    More than six zero modes can occur even in a connected underconstrained
    network.  Their actual number is recorded, never assumed to be six.
    """

    reference: torch.Tensor
    pairs: torch.Tensor
    reference_distances: torch.Tensor
    hessian: torch.Tensor
    eigenvalues: torch.Tensor
    modes: torch.Tensor
    cutoff: float
    gamma: float
    diagnostics: dict

    @property
    def n_modes(self) -> int:
        return self.modes.shape[0]


@torch.no_grad()
def build_anm(
    ca_xyz: torch.Tensor,
    cutoff: float = 12.0,
    gamma: float = 1.0,
    *,
    reject_disconnected: bool = True,
    zero_eigenvalue_rtol: float | None = None,
) -> ANMModel:
    """Build an apo-only uniform central-spring ANM, using fixed CA edges.

    The Hessian is that of ``gamma/2 * sum((d_ij - d_ij_ref)**2)`` at the
    input structure.  Its units are energy / Angstrom squared.  No native
    interface, target conformation, or reference-complex geometry is used.
    Disconnected networks are rejected by default; opting out retains the
    explicit component count and removes *all* numerical zero modes.
    """
    _check_xyz(ca_xyz, "ca_xyz")
    cutoff = _positive_scalar(cutoff, "cutoff")
    gamma = _positive_scalar(gamma, "gamma")
    if not isinstance(reject_disconnected, bool):
        raise TypeError("reject_disconnected must be a bool")
    rtol = 64 * torch.finfo(ca_xyz.dtype).eps if zero_eigenvalue_rtol is None else _positive_scalar(
        zero_eigenvalue_rtol, "zero_eigenvalue_rtol",
    )
    reference = ca_xyz.detach().clone()
    n_ca = len(reference)
    all_pairs = torch.triu_indices(n_ca, n_ca, offset=1, device=reference.device).T
    all_vectors = reference[all_pairs[:, 1]] - reference[all_pairs[:, 0]]
    all_distances = torch.linalg.vector_norm(all_vectors, dim=-1)
    if bool((all_distances == 0).any()):
        raise ValueError("coincident CA atoms do not define a central-spring direction")
    if not bool(torch.isfinite(all_distances).all()):
        raise ValueError("CA distances must be representable in the input dtype")
    within = all_distances <= cutoff
    pairs = all_pairs[within]
    distances = all_distances[within]
    n_components = _component_count(n_ca, pairs)
    if reject_disconnected and n_components != 1:
        raise ValueError(f"ANM network is disconnected ({n_components} components at cutoff={cutoff})")

    directions = all_vectors[within] / distances[:, None]
    blocks = gamma * directions[:, :, None] * directions[:, None, :]
    diagonal = reference.new_zeros((n_ca, 3, 3))
    diagonal.index_add_(0, pairs[:, 0], blocks)
    diagonal.index_add_(0, pairs[:, 1], blocks)
    hessian_blocks = reference.new_zeros((n_ca, n_ca, 3, 3))
    hessian_blocks[pairs[:, 0], pairs[:, 1]] = -blocks
    hessian_blocks[pairs[:, 1], pairs[:, 0]] = -blocks
    index = torch.arange(n_ca, device=reference.device)
    hessian_blocks[index, index] = diagonal
    hessian = hessian_blocks.permute(0, 2, 1, 3).reshape(3 * n_ca, 3 * n_ca)
    eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
    threshold = rtol * eigenvalues.abs().amax()
    if bool((eigenvalues < -threshold).any()):
        raise FloatingPointError("ANM Hessian has negative curvature beyond the numerical zero tolerance")
    positive = eigenvalues > threshold
    modes = eigenvectors[:, positive].T.reshape(-1, n_ca, 3)
    diagnostics = {
        "n_ca": n_ca,
        "n_edges": len(pairs),
        "n_components": n_components,
        "n_zero_modes": int((~positive).sum()),
        "n_positive_modes": int(positive.sum()),
        "zero_eigenvalue_rtol": float(rtol),
        "zero_eigenvalue_threshold": float(threshold),
    }
    return ANMModel(reference, pairs, distances, hessian, eigenvalues[positive], modes, cutoff, gamma, diagnostics)


@dataclass(frozen=True)
class ModeModel:
    """Anchored nonlinear coordinates and their consistent spring prior.

    ``angle_basis`` is (N_torsions, M), in radians / Angstrom; ``basis`` is
    (M, N_atoms, 3), the actual coordinate derivative at zero.  CA vectors
    satisfy ``B_CA.T @ B_CA / N_CA = I`` after flattening atoms/components.
    Amplitudes thus have units of infinitesimal CA RMS displacement; a finite
    amplitude norm does not by itself guarantee the same finite Cartesian
    displacement.  No frame alignment or hidden rigid-motion subtraction is
    applied to coordinates, their metric, or their energy.
    """

    kinematics: TorsionMap
    angle_basis: torch.Tensor
    basis: torch.Tensor
    stiffness: torch.Tensor
    anm: ANMModel
    regularization_strength: float
    diagnostics: dict

    @property
    def n_modes(self) -> int:
        return self.angle_basis.shape[1]

    @property
    def reference(self) -> torch.Tensor:
        return self.kinematics.reference

    @property
    def ca_indices(self) -> torch.Tensor:
        return self.kinematics.ca_indices

    def coordinates(self, amplitudes: torch.Tensor) -> torch.Tensor:
        if amplitudes.shape != (self.n_modes,):
            raise ValueError(f"amplitudes must have shape ({self.n_modes},)")
        if amplitudes.dtype != self.angle_basis.dtype or amplitudes.device != self.angle_basis.device:
            raise ValueError("amplitudes must share the mode basis dtype and device")
        return self.kinematics.coordinates(self.angle_basis @ amplitudes)

    def deformation_energy(self, amplitudes: torch.Tensor) -> torch.Tensor:
        ca_xyz = self.coordinates(amplitudes)[self.kinematics.ca_indices]
        vectors = ca_xyz[self.anm.pairs[:, 1]] - ca_xyz[self.anm.pairs[:, 0]]
        distances = torch.linalg.vector_norm(vectors, dim=-1)
        springs = 0.5 * self.anm.gamma * ((distances - self.anm.reference_distances) ** 2).sum()
        return springs + 0.5 * self.regularization_strength * amplitudes.square().sum()


def build_torsion_modes(
    kinematics: TorsionMap,
    n_modes: int = 4,
    *,
    cutoff: float = 12.0,
    gamma: float = 1.0,
    fit_relative_damping: float = 1e-3,
    relative_regularization: float = 1e-4,
    allow_rank_reduction: bool = False,
    rank_rtol: float | None = None,
) -> ModeModel:
    """Fit the lowest ANM directions into an anchored torsion Jacobian.

    Damped least squares uses the *actual anchored* CA Jacobian, without alignment
    or projection of its rigid component.  The resulting displacement metric
    is whitened using that same Jacobian.  A smaller achievable mode count
    raises by default; explicit ``allow_rank_reduction=True`` permits and
    records a reduction.  Poor fits are reported, not silently called exact.

    The angular-fit Tikhonov damping is ``lambda_fit = fit_relative_damping
    * sigma_max(J_CA)``.  Its spectral filter is ``sigma/(sigma**2 +
    lambda_fit**2)``; zero damping recovers the truncated least-squares fit.
    Damping discourages enormous angle coefficients along weak-leverage
    directions, but is not a finite-displacement or maximum-angle guarantee.

    The regularizer eta is ``relative_regularization * median(positive
    eigenvalues of K_spring)``.  It is frozen in the nonlinear prior as well
    as its Hessian.  This is an explicit extra physical-model assumption,
    not a solver-only stabilizer.  Full ANM and Jacobian eigensystems are
    materialized in this initial implementation.
    """
    reference = kinematics.reference
    _check_xyz(reference, "kinematics.reference")
    if isinstance(n_modes, bool) or not isinstance(n_modes, int) or n_modes < 0:
        raise ValueError("n_modes must be a nonnegative integer")
    if isinstance(kinematics.n_torsions, bool) or not isinstance(kinematics.n_torsions, int) or kinematics.n_torsions < 0:
        raise ValueError("kinematics.n_torsions must be a nonnegative integer")
    if not isinstance(allow_rank_reduction, bool):
        raise TypeError("allow_rank_reduction must be a bool")
    fit_relative_damping = _positive_scalar(fit_relative_damping, "fit_relative_damping", allow_zero=True)
    relative_regularization = _positive_scalar(relative_regularization, "relative_regularization", allow_zero=True)
    rtol = max(64 * torch.finfo(reference.dtype).eps, 1e-10) if rank_rtol is None else _positive_scalar(rank_rtol, "rank_rtol")
    ca_indices = kinematics.ca_indices
    if not isinstance(ca_indices, torch.Tensor) or ca_indices.dtype != torch.long or ca_indices.ndim != 1:
        raise ValueError("kinematics.ca_indices must be a one-dimensional long tensor")
    if ca_indices.device != reference.device or len(ca_indices) < 2:
        raise ValueError("CA indices must share the reference device and include at least two atoms")
    if bool((ca_indices < 0).any()) or bool((ca_indices >= len(reference)).any()) or len(torch.unique(ca_indices)) != len(ca_indices):
        raise ValueError("CA indices must be unique and in range")
    origin = reference.new_zeros(kinematics.n_torsions)
    at_origin = kinematics.coordinates(origin)
    if at_origin.shape != reference.shape or not bool(torch.isfinite(at_origin).all()):
        raise ValueError("kinematics at zero must return finite reference-shaped coordinates")
    coordinate_tolerance = 64 * torch.finfo(reference.dtype).eps
    if not torch.allclose(at_origin, reference, atol=coordinate_tolerance, rtol=coordinate_tolerance):
        raise ValueError("kinematics.coordinates(0) must reproduce its reference")
    anm = build_anm(reference[ca_indices], cutoff=cutoff, gamma=gamma)
    diagnostics = {
        "requested_modes": n_modes,
        "rank_reduction_allowed": allow_rank_reduction,
        "rank_rtol": float(rtol),
        "gauge": "anchored_actual_CA_Jacobian_no_alignment",
        "anm": anm.diagnostics,
        "fit_relative_damping": fit_relative_damping,
        "relative_regularization": relative_regularization,
    }
    if n_modes == 0:
        diagnostics.update({"actual_modes": 0, "rank_reduced": False, "regularization_strength": 0.0})
        return ModeModel(kinematics, reference.new_empty((kinematics.n_torsions, 0)),
                         reference.new_empty((0, len(reference), 3)), reference.new_empty((0, 0)), anm, 0.0, diagnostics)

    target_count = min(n_modes, anm.n_modes)
    if target_count < n_modes and not allow_rank_reduction:
        raise ValueError(f"requested {n_modes} modes but ANM provides only {anm.n_modes} positive modes")
    if target_count == 0 or kinematics.n_torsions == 0:
        raise ValueError("no nonzero ANM directions or torsion degrees of freedom are available")

    # The torsion input is much smaller than the all-atom Cartesian output;
    # forward AD avoids batching a reverse pass over every atom component.
    full_jacobian = torch.func.jacfwd(kinematics.coordinates)(origin).detach()
    if full_jacobian.shape != (*reference.shape, kinematics.n_torsions) or not bool(torch.isfinite(full_jacobian).all()):
        raise ValueError("kinematics coordinate Jacobian has an invalid shape or nonfinite values")
    ca_jacobian = full_jacobian[ca_indices].reshape(-1, kinematics.n_torsions)
    left, singular_values, right_h = torch.linalg.svd(ca_jacobian, full_matrices=False)
    retained = singular_values > rtol * singular_values.amax()
    if not bool(retained.any()):
        raise ValueError("torsion CA Jacobian has zero numerical rank")
    targets = anm.modes[:target_count].reshape(target_count, -1).T * math.sqrt(len(ca_indices))
    fit_damping = fit_relative_damping * singular_values.amax()
    spectral_filter = singular_values[retained] / (singular_values[retained].square() + fit_damping.square())
    fitted_angles = right_h[retained].T @ ((left[:, retained].T @ targets) * spectral_filter[:, None])
    fitted_ca = ca_jacobian @ fitted_angles
    fit_relative_residual = torch.linalg.vector_norm(fitted_ca - targets, dim=0) / torch.linalg.vector_norm(targets, dim=0)
    gram = fitted_ca.T @ fitted_ca / len(ca_indices)
    metric_values, metric_vectors = torch.linalg.eigh(0.5 * (gram + gram.T))
    metric_retained = metric_values > rtol * metric_values.abs().amax()
    actual_count = int(metric_retained.sum())
    if actual_count == 0:
        raise ValueError("ANM directions have zero-rank projection into the torsion CA Jacobian")
    if actual_count < n_modes and not allow_rank_reduction:
        raise ValueError(f"requested {n_modes} modes but projected CA directions have numerical rank {actual_count}")
    angle_basis = (fitted_angles @ metric_vectors[:, metric_retained]) / torch.sqrt(metric_values[metric_retained])[None, :]
    basis = torch.einsum("nct,tm->mnc", full_jacobian, angle_basis)
    amplitudes_zero = reference.new_zeros(actual_count)

    def spring_energy(amplitudes: torch.Tensor) -> torch.Tensor:
        xyz = kinematics.coordinates(angle_basis @ amplitudes)[ca_indices]
        distance = torch.linalg.vector_norm(xyz[anm.pairs[:, 1]] - xyz[anm.pairs[:, 0]], dim=-1)
        return 0.5 * anm.gamma * ((distance - anm.reference_distances) ** 2).sum()

    spring_stiffness = torch.func.hessian(spring_energy)(amplitudes_zero).detach()
    spring_stiffness = 0.5 * (spring_stiffness + spring_stiffness.T)
    spring_eigenvalues = torch.linalg.eigvalsh(spring_stiffness)
    curvature_threshold = rtol * spring_eigenvalues.abs().amax()
    if bool((spring_eigenvalues < -curvature_threshold).any()):
        raise FloatingPointError("reference spring prior has negative curvature beyond numerical tolerance")
    positive_curvatures = spring_eigenvalues[spring_eigenvalues > curvature_threshold]
    if len(positive_curvatures) == 0:
        raise ValueError("projected directions have no positive spring curvature")
    eta = relative_regularization * float(positive_curvatures.median())
    stiffness = spring_stiffness + eta * torch.eye(actual_count, dtype=reference.dtype, device=reference.device)
    if not bool(torch.isfinite(stiffness).all()) or not bool((torch.linalg.eigvalsh(stiffness) > 0).all()):
        raise ValueError("mode stiffness is not finite positive definite; use an explicit positive prior regularizer")
    actual_ca = basis[:, ca_indices].reshape(actual_count, -1)
    gram_error = (actual_ca @ actual_ca.T / len(ca_indices) - torch.eye(actual_count, dtype=reference.dtype, device=reference.device)).abs().amax()
    diagnostics.update({
        "actual_modes": actual_count,
        "rank_reduced": actual_count < n_modes,
        "torsion_CA_rank": int(retained.sum()),
        "fitted_CA_rank": actual_count,
        "source_ANM_eigenvalues": anm.eigenvalues[:target_count].cpu().tolist(),
        "fit_relative_residuals": fit_relative_residual.cpu().tolist(),
        "fit_absolute_damping": float(fit_damping),
        "torsion_CA_sigma_max": float(singular_values.amax()),
        "torsion_CA_sigma_min_retained": float(singular_values[retained].amin()),
        "angle_basis_max_abs_rad_per_A": float(angle_basis.abs().amax()),
        "max_angle_per_unit_amplitude_rad_per_A": float(torch.linalg.vector_norm(angle_basis, dim=1).amax()),
        "CA_metric_max_abs_error": float(gram_error),
        "spring_stiffness_eigenvalues": spring_eigenvalues.cpu().tolist(),
        "regularization_strength": eta,
    })
    return ModeModel(kinematics, angle_basis.detach(), basis.detach(), stiffness, anm, eta, diagnostics)
