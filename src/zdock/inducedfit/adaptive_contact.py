"""Contact-driven torsion directions outside an existing deformation space.

Local means selected *torsion variables*, not a promise that downstream atom
motion remains spatially local. The present open-chain phi/psi map has no chi
variables, antibody disulfide support or finite loop closure. Those require a
different kinematic map before this machinery can safely act on a closed loop.

Only HVPs are used; a Cartesian or full torsion Hessian is never constructed.
Directions remain in the local torsion mask while satisfying A.T H p = 0.
The output basis must be passed to the physical model builder to recompute
its stiffness and nonlinear energy; a regularized proposal Hessian must never
silently replace the physical prior.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch


class AdaptiveDirectionError(RuntimeError):
    def __init__(self, message: str, diagnostics: dict):
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class AdaptiveDirections:
    angle_basis: torch.Tensor
    basis: torch.Tensor
    diagnostics: dict


def _cpu_double(value, shape, name):
    if not isinstance(value, torch.Tensor) or value.dtype != torch.float64 or value.device.type != 'cpu':
        raise TypeError(f'{name} must be a CPU float64 tensor')
    if tuple(value.shape) != tuple(shape) or not bool(torch.isfinite(value).all()):
        raise ValueError(f'{name} must be finite with shape {tuple(shape)}')


def contact_torsion_mask(kinematics, receptor_xyz: torch.Tensor, ligand_xyz: torch.Tensor,
                         *, contact_radius: float = 8., atom_chunk_size: int = 128) -> torch.Tensor:
    """Select joints whose residue has an atom within a predicted contact.

    This does not read a native interface and does not select an upstream
    torsion merely because its moving subtree contains a contact atom. Joint
    residue identity is taken from axis_start, supporting the current full-H
    and heavy-only kinematic maps consistently.
    """
    _cpu_double(receptor_xyz, kinematics.reference.shape, 'receptor_xyz')
    if not isinstance(ligand_xyz, torch.Tensor) or ligand_xyz.ndim != 2 or ligand_xyz.shape[1] != 3 or len(ligand_xyz) < 1:
        raise ValueError('ligand_xyz must have nonempty shape (N,3)')
    _cpu_double(ligand_xyz, ligand_xyz.shape, 'ligand_xyz')
    if isinstance(contact_radius, bool) or not math.isfinite(contact_radius) or contact_radius <= 0:
        raise ValueError('contact_radius must be positive finite')
    if type(atom_chunk_size) is not int or atom_chunk_size < 1:
        raise ValueError('atom_chunk_size must be a positive integer')
    if len(kinematics.atom_keys) != len(receptor_xyz) or len(kinematics.joints) != kinematics.n_torsions:
        raise ValueError('kinematic atom/joint metadata are inconsistent')
    identity = lambda key: (key.chain, key.residue_number, key.insertion_code)
    contacted = set()
    with torch.no_grad():
        for start in range(0, len(receptor_xyz), atom_chunk_size):
            stop = min(start + atom_chunk_size, len(receptor_xyz))
            hit = (torch.cdist(receptor_xyz[start:stop], ligand_xyz) <= contact_radius).any(1)
            for offset in hit.nonzero().flatten().tolist():
                contacted.add(identity(kinematics.atom_keys[start + offset]))
    return torch.tensor([identity(kinematics.atom_keys[joint.axis_start]) in contacted
                         for joint in kinematics.joints], dtype=torch.bool)


def _constrained_metric_inverse(metric, rows):
    """Inverse of a numerical metric restricted to ker(rows), not of H.

    For G=LL.T, Y=G^-1 R.T and W=R Y, the inverse action is
    G^-1 r - Y W^-1 R G^-1 r. R already has the constraint rank determined
    from the unpreconditioned physical problem; no constraints are dropped.
    """
    factor, info = torch.linalg.cholesky_ex(metric)
    if int(info) != 0:
        raise AdaptiveDirectionError('local numerical preconditioner is not positive definite',
            {'cholesky_info': int(info), 'local_dimension': len(metric)})
    solve = lambda rhs: torch.cholesky_solve(rhs[:, None], factor)[:, 0]
    if len(rows):
        y = torch.cholesky_solve(rows.T, factor)
        schur = rows @ y
        # This is numerical metric roundoff, never physical-Hessian editing.
        asymmetry = float((schur - schur.T).abs().max())
        scale = max(1., float(schur.abs().max()))
        if asymmetry > 1e-10 * scale:
            raise AdaptiveDirectionError('preconditioner constraint Schur matrix is not symmetric',
                {'maximum_asymmetry': asymmetry})
        schur_factor, info = torch.linalg.cholesky_ex(.5 * (schur + schur.T))
        if int(info) != 0:
            raise AdaptiveDirectionError('preconditioner constraint Schur matrix is not positive definite',
                {'cholesky_info': int(info), 'constraint_rank': len(rows)})

        def inverse(rhs):
            solved = solve(rhs)
            multiplier = torch.cholesky_solve((rows @ solved)[:, None], schur_factor)[:, 0]
            return solved - y @ multiplier
    else:
        inverse, asymmetry = solve, 0.
    return inverse, {'method': 'constrained_metric_inverse', 'local_dimension': len(metric),
                     'constraint_rank': len(rows), 'schur_maximum_asymmetry': asymmetry,
                     'additional_metric_ridge': 0., 'physical_hessian_modified': False}


def _projected_cg(apply, rhs, projector, *, tolerance, max_iterations, curvature_tolerance,
                  preconditioner=None):
    """CG/PCG with the same physical residual and curvature certificates."""
    x = torch.zeros_like(rhs)
    residual = projector(rhs)
    rhs_norm = float(residual.norm())
    if rhs_norm == 0:
        return x, {'iterations': 0, 'relative_residual': 0., 'minimum_observed_rayleigh': None}
    def precondition(residual):
        value = projector(preconditioner(residual))
        product = residual @ value
        if not bool(torch.isfinite(value).all()) or not math.isfinite(float(product)) or float(product) <= 0:
            raise AdaptiveDirectionError('preconditioner has nonpositive residual product',
                {'residual_preconditioned_product': float(product),
                 'relative_residual': float(residual.norm()) / rhs_norm})
        return value, product

    if preconditioner is None:
        direction = residual.clone()
        squared = residual @ residual
    else:
        direction, squared = precondition(residual)
    minimum = math.inf
    for iteration in range(1, max_iterations + 1):
        hd = projector(apply(direction))
        denominator = direction @ hd
        norm_squared = direction @ direction
        rayleigh = float(denominator / norm_squared)
        if not math.isfinite(rayleigh) or rayleigh <= curvature_tolerance:
            raise AdaptiveDirectionError('nonpositive or unresolved curvature in local response solve',
                {'iterations': iteration, 'observed_rayleigh': rayleigh,
                 'relative_residual': float(residual.norm()) / rhs_norm})
        minimum = min(minimum, rayleigh)
        step = squared / denominator
        x = projector(x + step * direction)
        residual = projector(residual - step * hd)
        new_squared = residual @ residual
        relative = float(new_squared.sqrt()) / rhs_norm
        if relative <= tolerance:
            # Recompute, rather than certify only the recursive CG residual.
            true_relative = float((projector(apply(x)) - projector(rhs)).norm()) / rhs_norm
            if true_relative <= max(5 * tolerance, 1e-12):
                return x, {'iterations': iteration, 'relative_residual': true_relative,
                           'minimum_observed_rayleigh': minimum}
        if preconditioner is None:
            direction = projector(residual + (new_squared / squared) * direction)
            squared = new_squared
        else:
            preconditioned, new_product = precondition(residual)
            direction = projector(preconditioned + (new_product / squared) * direction)
            squared = new_product
    true_relative = float((projector(apply(x)) - projector(rhs)).norm()) / rhs_norm
    raise AdaptiveDirectionError('local response solve did not converge',
        {'iterations': max_iterations, 'relative_residual': true_relative,
         'minimum_observed_rayleigh': minimum})


def build_contact_directions(kinematics, theta0: torch.Tensor, existing_angle_basis: torch.Tensor,
                             torsion_gradients: torch.Tensor, local_masks: torch.Tensor,
                             hessian_vector: Callable[[torch.Tensor], torch.Tensor], *,
                             max_directions: int = 4, regularization: float = 0., seed: int = 0,
                             method: str = 'response', cg_tolerance: float = 1e-5,
                             max_cg_iterations: int = 128, rank_rtol: float = 1e-9,
                             curvature_tolerance: float = 1e-10,
                             preconditioner_metric: torch.Tensor | None = None) -> AdaptiveDirections:
    """Build one new local response per supplied pose, up to max_directions.

    The callback applies the physical torsion Hessian at theta0. For response
    proposals, projected CG solves the mask-restricted response while imposing
    H-orthogonality to existing and previously accepted directions. With an
    explicitly nonzero regularization, H+lambda*I defines the proposal and its
    orthogonality metric; the final physical stiffness must still be rebuilt.

    An optional full-torsion SPD preconditioner_metric changes only the response
    solver, not H or its constraints. Its principal local submatrix is used
    inside the exact constraint nullspace. For example, the caller can provide
    the input-apo anchor metric G=anchor_k J_input.T J_input+I. None retains raw
    CG; random-control draws and projection are unchanged when G is supplied.

    Random controls use local seeded Gaussian torsion vectors with the same
    masks and orthogonality constraints. No failed or missing response is
    silently replaced by a random direction. Missing contact, exhausted local
    rank or invisible CA motion is recorded as shortfall. Numerical HVP/solve
    failures are explicit exceptions, not a successful shortened output.
    """
    t = kinematics.n_torsions
    _cpu_double(theta0, (t,), 'theta0')
    if existing_angle_basis.ndim != 2 or existing_angle_basis.shape[0] != t:
        raise ValueError('existing_angle_basis must have shape (T,M)')
    _cpu_double(existing_angle_basis, existing_angle_basis.shape, 'existing_angle_basis')
    if torsion_gradients.ndim != 2 or torsion_gradients.shape[1] != t:
        raise ValueError('torsion_gradients must have shape (P,T)')
    _cpu_double(torsion_gradients, torsion_gradients.shape, 'torsion_gradients')
    if (not isinstance(local_masks, torch.Tensor) or local_masks.shape != torsion_gradients.shape
            or local_masks.dtype != torch.bool or local_masks.device.type != 'cpu'):
        raise ValueError('local_masks must be a CPU bool tensor with shape (P,T)')
    if method not in ('response', 'random'):
        raise ValueError('method must be response or random')
    # Basis selection is preprocessing; keep no caller autograd graph alive.
    theta0 = theta0.detach().clone()
    torsion_gradients = torsion_gradients.detach().clone()
    if type(max_directions) is not int or max_directions < 0:
        raise ValueError('max_directions must be a nonnegative integer')
    if type(max_cg_iterations) is not int or max_cg_iterations < 1:
        raise ValueError('max_cg_iterations must be a positive integer')
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('seed must be in [0,2**63)')
    for value, name in ((regularization, 'regularization'), (curvature_tolerance, 'curvature_tolerance')):
        if isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be nonnegative finite')
    for value, name in ((rank_rtol, 'rank_rtol'), (cg_tolerance, 'cg_tolerance')):
        if isinstance(value, bool) or not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(f'{name} must be finite in (0,1)')
    metric_asymmetry = 0.
    if preconditioner_metric is not None:
        _cpu_double(preconditioner_metric, (t, t), 'preconditioner_metric')
        preconditioner_metric = preconditioner_metric.detach().clone()
        metric_asymmetry = float((preconditioner_metric - preconditioner_metric.T).abs().max()) if t else 0.
        metric_scale = max(1., float(preconditioner_metric.abs().max())) if t else 1.
        if metric_asymmetry > 1e-10 * metric_scale:
            raise ValueError('preconditioner_metric must be symmetric')
        preconditioner_metric = .5 * (preconditioner_metric + preconditioner_metric.T)
        if int(torch.linalg.cholesky_ex(preconditioner_metric)[1]) != 0:
            raise ValueError('preconditioner_metric must be positive definite')
    ca = kinematics.ca_indices
    if (not isinstance(ca, torch.Tensor) or ca.dtype != torch.long or ca.ndim != 1 or len(ca) == 0
            or ca.device.type != 'cpu' or bool((ca < 0).any()) or bool((ca >= len(kinematics.reference)).any())):
        raise ValueError('CA indices must be nonempty and in range')
    origin = kinematics.coordinates(theta0).detach()
    _cpu_double(origin, kinematics.reference.shape, 'origin')
    calls = 0

    def apply(vector):
        nonlocal calls
        hv = hessian_vector(vector.detach().clone())
        _cpu_double(hv, (t,), 'HVP result')
        calls += 1
        return hv.detach() + regularization * vector

    generator = torch.Generator(device='cpu').manual_seed(seed)
    original = existing_angle_basis.detach().clone()
    h_original = torch.stack([apply(a) for a in original.T], 1) if original.shape[1] else theta0.new_empty(t, 0)
    accepted, h_accepted, cartesian, records = [], [], [], []
    active_union = torch.zeros(t, dtype=torch.bool)
    for pose_index, (gradient, mask) in enumerate(zip(torsion_gradients, local_masks)):
        if len(accepted) >= max_directions:
            break
        indices = mask.nonzero().flatten()
        record = {'pose_index': pose_index, 'local_torsion_count': len(indices),
                  'local_torsion_indices': indices.tolist()}
        records.append(record)
        if len(indices) == 0:
            record['status'] = 'no_contact_torsions'
            continue
        h_basis = torch.cat([h_original, torch.stack(h_accepted, 1)], 1) if h_accepted else h_original
        constraints = h_basis[indices].T
        if len(constraints):
            _, singular, vh = torch.linalg.svd(constraints, full_matrices=False)
            threshold = rank_rtol * float(singular.max()) if len(singular) else 0.
            rank = int((singular > threshold).sum())
            rows = vh[:rank]
        else:
            rank, rows = 0, theta0.new_empty(0, len(indices))
        record['orthogonality_constraint_rank'] = rank
        if rank == len(indices):
            record['status'] = 'local_nullspace_exhausted'
            continue
        projector = lambda value: value - rows.T @ (rows @ value)

        def local_apply(value):
            full = torch.zeros_like(theta0)
            full[indices] = value
            return apply(full)[indices]

        rhs = -projector(gradient[indices])
        if method == 'response':
            if float(rhs.norm()) <= rank_rtol * max(1., float(gradient[indices].norm())):
                record['status'] = 'no_residual_contact_force'
                continue
            try:
                preconditioner = None
                if preconditioner_metric is not None:
                    local_metric = preconditioner_metric[indices[:, None], indices[None, :]]
                    preconditioner, record['preconditioner'] = _constrained_metric_inverse(local_metric, rows)
                local, cg = _projected_cg(local_apply, rhs, projector, tolerance=cg_tolerance,
                    max_iterations=max_cg_iterations, curvature_tolerance=curvature_tolerance,
                    preconditioner=preconditioner)
            except AdaptiveDirectionError as error:
                raise AdaptiveDirectionError(str(error), {'pose_records': records,
                    'failed_pose': pose_index, 'cg': error.diagnostics, 'hvp_calls': calls}) from error
            record['cg'] = cg
        else:
            local = projector(torch.randn(len(indices), generator=generator, dtype=theta0.dtype))
            record['cg'] = None
        if float(local.norm()) <= rank_rtol:
            record['status'] = 'numerically_empty_direction'
            continue
        vector = torch.zeros_like(theta0)
        vector[indices] = local
        hv = apply(vector)
        curvature = float(vector @ hv)
        if curvature <= curvature_tolerance * float(vector @ vector):
            raise AdaptiveDirectionError('accepted proposal has nonpositive curvature',
                {'pose_records': records, 'failed_pose': pose_index, 'curvature': curvature})
        tangent = torch.func.jvp(kinematics.coordinates, (theta0,), (vector,))[1].detach()
        ca_flat = tangent[ca].reshape(-1)
        ca_norm = float((ca_flat.square().sum() / len(ca)).sqrt())
        if ca_norm <= rank_rtol:
            record['status'] = 'no_ca_motion'
            continue
        vector, hv, tangent = vector / ca_norm, hv / ca_norm, tangent / ca_norm
        candidate_ca = torch.stack([*cartesian, tangent])[:, ca].reshape(len(accepted) + 1, -1)
        metric_values = torch.linalg.eigvalsh(candidate_ca @ candidate_ca.T / len(ca))
        if float(metric_values.min()) <= rank_rtol * float(metric_values.max()):
            record['status'] = 'ca_metric_rank_loss'
            continue
        orth_error = float((h_basis.T @ vector).abs().max()) if h_basis.shape[1] else 0.
        denominator = max(1., float(h_basis.norm()) * float(vector.norm()))
        if orth_error / denominator > 100 * rank_rtol:
            raise AdaptiveDirectionError('local H-orthogonality verification failed',
                {'pose_records': records, 'failed_pose': pose_index, 'relative_error': orth_error / denominator})
        accepted.append(vector)
        h_accepted.append(hv)
        cartesian.append(tangent)
        active_union |= mask
        record.update(status='accepted', proposal_ca_norm_before_normalizing=ca_norm,
                      h_orthogonality_relative_error=orth_error / denominator,
                      proposal_curvature_before_normalizing=curvature,
                      maximum_atom_motion_per_unit_CA=float(tangent.norm(dim=-1).max()))

    if accepted:
        angle_basis = torch.stack(accepted, 1)
        basis = torch.stack(cartesian)
        b_ca = basis[:, ca].reshape(len(accepted), -1)
        metric = b_ca @ b_ca.T / len(ca)
        values, vectors = torch.linalg.eigh(.5 * (metric + metric.T))
        transform = (vectors * values.rsqrt()[None]) @ vectors.T
        angle_basis = angle_basis @ transform
        basis = torch.einsum('mr,mnc->rnc', transform, basis)
        final_ca = basis[:, ca].reshape(len(accepted), -1)
        metric_error = float((final_ca @ final_ca.T / len(ca) - torch.eye(len(accepted), dtype=theta0.dtype)).abs().max())
        if metric_error > 1e-7 or not bool(torch.isfinite(angle_basis).all() and torch.isfinite(basis).all()):
            raise AdaptiveDirectionError('final adaptive CA metric verification failed', {'metric_error': metric_error})
        outside = float(angle_basis[~active_union].abs().max()) if bool((~active_union).any()) else 0.
    else:
        angle_basis, basis = theta0.new_empty(t, 0), theta0.new_empty(0, *origin.shape)
        metric_error, outside = 0., 0.
    diagnostics = {'method': method, 'source': 'predicted partner contact forces; no native coordinates',
        'requested_max_directions': max_directions, 'actual_directions': len(accepted),
        'shortfall': max_directions - len(accepted), 'n_candidate_poses': len(torsion_gradients),
        'hvp_calls': calls, 'regularization': regularization,
        'regularization_policy': 'proposal H+lambda I only; physical prior and stiffness must be rebuilt unchanged',
        'seed': seed, 'cg_tolerance': cg_tolerance, 'max_cg_iterations': max_cg_iterations,
        'response_solver': 'projected_pcg' if method == 'response' and preconditioner_metric is not None else 'projected_cg' if method == 'response' else 'none_random_control',
        'preconditioner_metric_supplied': preconditioner_metric is not None,
        'preconditioner_metric_maximum_asymmetry': metric_asymmetry,
        'preconditioning_policy': 'numerical metric only; same physical residual and curvature gates; random distribution unchanged',
        'rank_rtol': rank_rtol, 'curvature_tolerance': curvature_tolerance,
        'ca_metric_max_abs_error': metric_error, 'outside_contact_mask_union_max_abs': outside,
        'locality': 'torsion mask; final CA whitening mixes accepted masks within their union; downstream atom motion can be nonlocal',
        'kinematic_scope': 'existing open-chain phi/psi map; no chi, disulfide or finite loop closure support added',
        'pose_records': records, 'status': 'complete' if len(accepted) == max_directions else 'rank_shortfall'}
    return AdaptiveDirections(angle_basis.detach(), basis.detach(), diagnostics)
