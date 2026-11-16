"""Candidate-local convex response with individual linear atom bounds.

The constraints are on tangents, not on finite nonlinear coordinates. Callers
must still verify actual geometry and energy. This CPU float64 prototype is
intended for a retained candidate pool, not every FFT translation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import minimize, nnls
import torch

from .response import QuadraticResponse


class AtomwiseConvergenceError(RuntimeError):
    """A numerical failure with diagnostics; no unverified answer is returned."""

    def __init__(self, message: str, diagnostics: dict):
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class AtomwiseResponseResult:
    amplitudes: torch.Tensor
    gain: torch.Tensor
    strain: torch.Tensor
    diagnostics: dict


class AtomwiseResponse:
    """Solve g.a + 1/2 a.K.a in a CA ball intersected with atom balls.

    ``basis`` is (M,N,3), CA-normalized by its caller. ``radius`` is the
    requested infinitesimal CA displacement radius, and each atom obeys
    ||sum_m a_m B_mi|| <= max_atom_displacement. SLSQP convergence alone is
    insufficient: primal feasibility, nonnegative KKT multipliers,
    stationarity, complementarity and the conservative-ball reference are
    checked independently. No differentiation through this solver is claimed.
    Line-search status 8 may be accepted only by that independent certificate
    and is retained as an optimizer warning; iteration limits remain failures.
    """

    def __init__(self, stiffness: torch.Tensor, basis: torch.Tensor):
        if (not isinstance(stiffness, torch.Tensor) or not isinstance(basis, torch.Tensor)
                or stiffness.dtype != torch.float64 or basis.dtype != torch.float64
                or stiffness.device.type != 'cpu' or basis.device.type != 'cpu'):
            raise TypeError('atomwise response requires CPU float64 tensors')
        if stiffness.ndim != 2 or stiffness.shape[0] != stiffness.shape[1]:
            raise ValueError('stiffness must be square')
        m = len(stiffness)
        if basis.ndim != 3 or basis.shape[0] != m or basis.shape[2] != 3 or basis.shape[1] < 1:
            raise ValueError('basis must have shape (M,N>=1,3)')
        if not bool(torch.isfinite(basis).all()):
            raise ValueError('basis must be finite')
        self.reference_solver = QuadraticResponse(stiffness)
        self.stiffness = stiffness.detach().clone()
        self.basis = basis.detach().clone()
        self.n_modes = m
        self.atom_grams = torch.einsum('mnc,knc->nmk', self.basis, self.basis).numpy()
        self.maximum_atom_coefficient = (float(torch.linalg.matrix_norm(self.basis.permute(1, 2, 0), ord=2).max())
                                         if m else 0.)

    @torch.no_grad()
    def solve(self, gradient: torch.Tensor, radius: float, max_atom_displacement: float,
              *, max_iterations: int = 500, tolerance: float = 1e-11,
              feasibility_tolerance: float = 1e-8, kkt_tolerance: float = 1e-6) -> AtomwiseResponseResult:
        if (not isinstance(gradient, torch.Tensor) or gradient.shape != (self.n_modes,)
                or gradient.dtype != torch.float64 or gradient.device.type != 'cpu'):
            raise ValueError('gradient must be a CPU float64 mode vector')
        if not bool(torch.isfinite(gradient).all()):
            raise ValueError('gradient must be finite')
        for value, name in ((radius, 'radius'), (max_atom_displacement, 'max_atom_displacement')):
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        for value, name in ((tolerance, 'tolerance'), (feasibility_tolerance, 'feasibility_tolerance'),
                            (kkt_tolerance, 'kkt_tolerance')):
            if isinstance(value, bool) or not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f'{name} must be finite in (0,1)')
        if type(max_iterations) is not int or max_iterations < 1:
            raise ValueError('max_iterations must be a positive integer')
        radius, atom_limit = float(radius), float(max_atom_displacement)
        diagnostics = {'method': 'candidate_local_atomwise_convex_response',
            'finite_geometry_verified': False, 'requested_radius': radius,
            'max_atom_displacement': atom_limit, 'n_modes': self.n_modes,
            'n_atoms': self.basis.shape[1], 'max_iterations': max_iterations,
            'tolerance': tolerance, 'feasibility_tolerance': feasibility_tolerance,
            'kkt_tolerance': kkt_tolerance}
        if self.n_modes == 0 or radius == 0:
            diagnostics.update(status='exact_zero_feasible_set' if radius == 0 else 'no_modes',
                               iterations=0, maximum_linear_atom_displacement=0., amplitude_norm=0.)
            return self._result(torch.zeros_like(gradient), gradient, diagnostics)

        # A zero atomic radius is a linear nullspace problem; squared-norm
        # constraints violate regularity at zero and cannot give a finite KKT
        # certificate in the original SLSQP formulation.
        if atom_limit == 0:
            flat = self.basis.permute(1, 2, 0).reshape(-1, self.n_modes)
            # Keep a complete right basis without allocating a (3N)^2 left
            # matrix for the usual tall protein tangent matrix.
            _, singular, vh = torch.linalg.svd(flat, full_matrices=flat.shape[0] < flat.shape[1])
            threshold = 64 * torch.finfo(flat.dtype).eps * max(flat.shape) * (float(singular.max()) if len(singular) else 0.)
            rank = int((singular > threshold).sum())
            null = vh[rank:].T
            if null.shape[1]:
                reduced = QuadraticResponse(null.T @ self.stiffness @ null).solve(null.T @ gradient, radius)
                amplitude = null @ reduced.amplitudes
            else:
                amplitude = torch.zeros_like(gradient)
            displacement = torch.einsum('m,mnc->nc', amplitude, self.basis)
            residual = float(displacement.abs().max())
            if residual > feasibility_tolerance:
                raise AtomwiseConvergenceError('zero-bound nullspace verification failed', {**diagnostics, 'residual': residual})
            diagnostics.update(status='zero_atom_bound_nullspace', atom_nullity=null.shape[1],
                               nullspace_threshold=threshold, iterations=0,
                               maximum_linear_atom_displacement=float(displacement.norm(dim=-1).max()),
                               amplitude_norm=float(amplitude.norm()))
            return self._result(amplitude, gradient, diagnostics)

        conservative_radius = (radius if self.maximum_atom_coefficient == 0 else
                               min(radius, atom_limit / self.maximum_atom_coefficient))
        conservative = self.reference_solver.solve(gradient, conservative_radius)
        diagnostics.update(conservative_ball_radius=conservative_radius,
                           conservative_ball_gain=float(conservative.gain))
        matrix = self.stiffness.numpy()
        vector = gradient.detach().numpy()
        # y=a/radius; all squared constraints then have unit right hand side.
        grams = np.concatenate((np.eye(self.n_modes)[None], (radius / atom_limit)**2 * self.atom_grams))
        scale = max(1., float(np.linalg.norm(radius * vector)), float(np.linalg.norm(radius**2 * matrix, ord=2)))
        scaled_k, scaled_g = radius**2 * matrix / scale, radius * vector / scale
        objective = lambda y: float(scaled_g @ y + .5 * y @ scaled_k @ y)
        jacobian = lambda y: scaled_g + scaled_k @ y
        constraints = lambda y: 1. - np.einsum('i,nij,j->n', y, grams, y)
        constraint_jacobian = lambda y: -2 * np.einsum('nij,j->ni', grams, y)
        try:
            optimum = minimize(objective, conservative.amplitudes.numpy() / radius,
                jac=jacobian, constraints={'type': 'ineq', 'fun': constraints, 'jac': constraint_jacobian},
                method='SLSQP', options={'ftol': tolerance, 'maxiter': max_iterations, 'disp': False})
        except Exception as error:
            raise AtomwiseConvergenceError('atomwise optimizer raised an exception',
                {**diagnostics, 'error_type': type(error).__name__, 'error': str(error)}) from error
        y = np.asarray(optimum.x)
        diagnostics.update(optimizer_success=bool(optimum.success), optimizer_status=int(optimum.status),
                           optimizer_message=str(optimum.message), iterations=int(optimum.nit), objective_scale=scale)
        if not np.isfinite(y).all() or (not bool(optimum.success) and int(optimum.status) != 8):
            raise AtomwiseConvergenceError('atomwise optimizer did not converge', diagnostics)
        # SLSQP can report line-search status 8 at an accurate boundary point.
        # Only the independent convex KKT certificate below can rehabilitate
        # that particular warning. Iteration limits and other failures cannot.
        provisional_slack = constraints(y)
        projection_scale = 1.
        preliminary_violation = max(0., -float(provisional_slack.min()))
        if 0 < preliminary_violation <= feasibility_tolerance:
            projection_scale = (1 - 4 * np.finfo(float).eps) / math.sqrt(1 + preliminary_violation)
            y = y * projection_scale
        diagnostics.update(pre_certificate_squared_violation=preliminary_violation,
                           feasibility_projection_scale=projection_scale)
        slack = constraints(y)
        active_tolerance = max(1e-7, 100 * tolerance)
        active = np.flatnonzero(slack <= active_tolerance)
        lagrange = np.zeros(len(grams))
        grad = jacobian(y)
        if len(active):
            normals = -constraint_jacobian(y)[active].T
            try:
                multipliers, _ = nnls(normals, -grad, maxiter=max(1000, 3 * len(active)))
            except Exception as error:
                raise AtomwiseConvergenceError('KKT multiplier recovery failed',
                    {**diagnostics, 'error_type': type(error).__name__, 'error': str(error)}) from error
            lagrange[active] = multipliers
        stationarity = grad - constraint_jacobian(y).T @ lagrange
        stationarity_relative = float(np.linalg.norm(stationarity) / max(1., np.linalg.norm(scaled_g)))
        complementarity = float(np.max(np.abs(lagrange * slack)))
        violation = max(0., -float(slack.min()))
        amplitude = torch.from_numpy(y.copy()) * radius
        displacement = torch.einsum('m,mnc->nc', amplitude, self.basis)
        gain = float(gradient @ amplitude + .5 * amplitude @ self.stiffness @ amplitude)
        reference_error = gain - float(conservative.gain)
        diagnostics.update(squared_constraint_violation=violation,
            stationarity_relative=stationarity_relative, complementarity=complementarity,
            active_constraint_indices=active.tolist(), active_atom_indices=(active[active > 0] - 1).tolist(),
            active_multipliers=lagrange[active].tolist(), active_tolerance=active_tolerance,
            maximum_linear_atom_displacement=float(displacement.norm(dim=-1).max()),
            amplitude_norm=float(amplitude.norm()), gain_minus_conservative_reference=reference_error)
        if (violation > feasibility_tolerance or stationarity_relative > kkt_tolerance
                or complementarity > kkt_tolerance
                or reference_error > kkt_tolerance * max(1., abs(float(conservative.gain)))):
            raise AtomwiseConvergenceError('atomwise response failed its independent certificate', diagnostics)
        diagnostics['status'] = 'certified' if bool(optimum.success) else 'certified_optimizer_warning'
        diagnostics['success_basis'] = 'independent_certificate'
        return self._result(amplitude, gradient, diagnostics)

    def _result(self, amplitude, gradient, diagnostics):
        amplitude = amplitude.detach().clone()
        strain = .5 * amplitude @ self.stiffness @ amplitude
        gain = gradient @ amplitude + strain
        if not bool(torch.isfinite(strain) and torch.isfinite(gain)):
            raise AtomwiseConvergenceError('nonfinite response energy', diagnostics)
        return AtomwiseResponseResult(amplitude, gain.detach(), strain.detach(), diagnostics)
