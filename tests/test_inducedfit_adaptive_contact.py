from types import SimpleNamespace

import pytest
import torch

from zdock.inducedfit.adaptive_contact import (
    AdaptiveDirectionError, build_contact_directions, contact_torsion_mask,
)
from zdock.inducedfit.kinematics import TorsionJoint
from zdock.inducedfit.types import AtomKey


DTYPE = torch.float64
pytestmark = pytest.mark.filterwarnings('ignore:`torch.jit.script` is deprecated.*:DeprecationWarning')


class LinearMap:
    def __init__(self, count):
        self.n_torsions = count
        self.reference = torch.zeros(count, 3, dtype=DTYPE)
        self.ca_indices = torch.arange(count)

    def coordinates(self, theta):
        return self.reference + theta[:, None] * theta.new_tensor([1., 0., 0.])


def test_response_directions_remove_existing_motion_and_normalize_ca_metric():
    kin = LinearMap(5)
    existing = torch.eye(5, dtype=DTYPE)[:, :1]
    h = torch.diag(torch.arange(1., 6., dtype=DTYPE))
    gradients = torch.tensor([[1., 2., 0., 0., 0.], [1., 1., 3., 0., 0.]], dtype=DTYPE)
    masks = torch.tensor([[True, True, False, False, False], [True, True, True, False, False]])
    result = build_contact_directions(kin, torch.zeros(5, dtype=DTYPE), existing, gradients, masks,
                                      lambda v: h @ v, max_directions=2)
    assert result.angle_basis.shape == (5, 2)
    torch.testing.assert_close(existing.T @ h @ result.angle_basis, torch.zeros(1, 2, dtype=DTYPE), atol=1e-12, rtol=0)
    ca = result.basis.reshape(2, -1)
    torch.testing.assert_close(ca @ ca.T / 5, torch.eye(2, dtype=DTYPE), atol=1e-12, rtol=0)
    assert not bool(result.angle_basis[3:].any())
    assert result.diagnostics['status'] == 'complete'
    assert result.diagnostics['hvp_calls'] > 0


def test_local_H_projection_never_introduces_outside_mask_torsions():
    kin = LinearMap(4)
    factor = torch.tensor([[2., .3, .2, .1], [.2, 1.4, .4, 0.], [0., .3, 1.2, .2], [.1, 0., .2, 1.1]], dtype=DTYPE)
    h = factor.T @ factor
    existing = torch.tensor([[1.], [1.], [0.], [1.]], dtype=DTYPE)
    gradient = torch.tensor([[.2, 2., -1., .7]], dtype=DTYPE)
    mask = torch.tensor([[False, True, True, False]])
    result = build_contact_directions(kin, torch.zeros(4, dtype=DTYPE), existing, gradient, mask,
                                      lambda v: h @ v, max_directions=1)
    assert not bool(result.angle_basis[[0, 3]].any())
    torch.testing.assert_close(existing.T @ h @ result.angle_basis, torch.zeros(1, 1, dtype=DTYPE), atol=1e-11, rtol=0)
    assert float(gradient @ result.angle_basis) < 0


def test_contact_mask_uses_joint_residue_not_downstream_subtree():
    keys = tuple(AtomKey('A', i+1, '', 'ALA', 'CA') for i in range(3))
    xyz = torch.tensor([[0., 0., 0.], [10., 0., 0.], [20., 0., 0.]], dtype=DTYPE)
    joints = tuple(TorsionJoint('phi', i, i, min(i+1, 2), tuple(range(i, 3))) for i in range(3))
    kin = SimpleNamespace(reference=xyz, atom_keys=keys, joints=joints, n_torsions=3)
    mask = contact_torsion_mask(kin, xyz, torch.tensor([[20.2, 0., 0.]], dtype=DTYPE), contact_radius=1.)
    assert mask.tolist() == [False, False, True]


def test_no_contact_or_no_force_is_reported_as_shortfall():
    kin = LinearMap(3)
    zero = torch.zeros(3, dtype=DTYPE)
    none = torch.empty(3, 0, dtype=DTYPE)
    result = build_contact_directions(kin, zero, none, torch.ones(1, 3, dtype=DTYPE), torch.zeros(1, 3, dtype=torch.bool),
                                      lambda v: v, max_directions=1)
    assert result.angle_basis.shape == (3, 0)
    assert result.diagnostics['pose_records'][0]['status'] == 'no_contact_torsions'
    result = build_contact_directions(kin, zero, none, torch.zeros(1, 3, dtype=DTYPE), torch.ones(1, 3, dtype=torch.bool),
                                      lambda v: v, max_directions=1)
    assert result.diagnostics['pose_records'][0]['status'] == 'no_residual_contact_force'


def test_exhausted_local_nullspace_does_not_add_a_duplicate_mode():
    kin = LinearMap(3)
    result = build_contact_directions(kin, torch.zeros(3, dtype=DTYPE), torch.eye(3, dtype=DTYPE),
        torch.ones(1, 3, dtype=DTYPE), torch.ones(1, 3, dtype=torch.bool), lambda v: v, max_directions=1)
    assert result.diagnostics['actual_directions'] == 0
    assert result.diagnostics['pose_records'][0]['status'] == 'local_nullspace_exhausted'


def test_random_control_is_seeded_and_does_not_change_global_rng():
    kin = LinearMap(4)
    inputs = (kin, torch.zeros(4, dtype=DTYPE), torch.empty(4, 0, dtype=DTYPE),
              torch.ones(2, 4, dtype=DTYPE), torch.ones(2, 4, dtype=torch.bool), lambda v: v)
    before = torch.random.get_rng_state().clone()
    first = build_contact_directions(*inputs, max_directions=2, method='random', seed=19)
    second = build_contact_directions(*inputs, max_directions=2, method='random', seed=19)
    third = build_contact_directions(*inputs, max_directions=2, method='random', seed=20)
    torch.testing.assert_close(torch.random.get_rng_state(), before)
    torch.testing.assert_close(first.angle_basis, second.angle_basis, atol=0, rtol=0)
    assert not torch.allclose(first.angle_basis, third.angle_basis)


def test_negative_curvature_is_rejected_without_silent_regularization():
    kin = LinearMap(2)
    args = (kin, torch.zeros(2, dtype=DTYPE), torch.empty(2, 0, dtype=DTYPE),
            torch.ones(1, 2, dtype=DTYPE), torch.ones(1, 2, dtype=torch.bool), lambda v: -.5 * v)
    with pytest.raises(AdaptiveDirectionError, match='curvature'):
        build_contact_directions(*args, max_directions=1)
    result = build_contact_directions(*args, max_directions=1, regularization=1.)
    assert result.diagnostics['regularization'] == 1.
    assert 'physical prior' in result.diagnostics['regularization_policy']


def test_CG_iteration_limit_is_a_failure_not_an_adaptive_shortfall():
    kin = LinearMap(3)
    h = torch.diag(torch.tensor([1., 3., 5.], dtype=DTYPE))
    with pytest.raises(AdaptiveDirectionError, match='did not converge'):
        build_contact_directions(kin, torch.zeros(3, dtype=DTYPE), torch.empty(3, 0, dtype=DTYPE),
            torch.ones(1, 3, dtype=DTYPE), torch.ones(1, 3, dtype=torch.bool), lambda v: h @ v,
            max_directions=1, max_cg_iterations=1)


def test_nonfinite_HVP_is_rejected():
    kin = LinearMap(2)
    with pytest.raises(ValueError, match='HVP result'):
        build_contact_directions(kin, torch.zeros(2, dtype=DTYPE), torch.empty(2, 0, dtype=DTYPE),
            torch.ones(1, 2, dtype=DTYPE), torch.ones(1, 2, dtype=torch.bool), lambda v: v * float('nan'), max_directions=1)


def test_PCG_matches_independent_constrained_oracle_and_reduces_iterations():
    generator = torch.Generator().manual_seed(41)
    count = 14
    kin = LinearMap(count)
    rotation, _ = torch.linalg.qr(torch.randn(count, count, dtype=DTYPE, generator=generator))
    h = (rotation * torch.logspace(0., 6., count, dtype=DTYPE)) @ rotation.T
    existing = torch.randn(count, 2, dtype=DTYPE, generator=generator)
    gradients = torch.randn(1, count, dtype=DTYPE, generator=generator)
    mask = torch.zeros((1, count), dtype=torch.bool)
    mask[0, [1, 2, 3, 5, 7, 8, 10, 12]] = True
    indices = mask[0].nonzero().flatten()

    # Independent dense nullspace oracle for the ORIGINAL physical problem.
    constraints = existing.T @ h[:, indices]
    _, singular, vh = torch.linalg.svd(constraints, full_matrices=True)
    rank = int((singular > 1e-9 * singular.max()).sum())
    nullspace = vh[rank:].T
    local_h = h[indices[:, None], indices[None, :]]
    oracle_local = -nullspace @ torch.linalg.solve(nullspace.T @ local_h @ nullspace,
                                                   nullspace.T @ gradients[0, indices])
    oracle = torch.zeros(count, dtype=DTYPE)
    oracle[indices] = oracle_local
    oracle_ca_norm = oracle.norm() / count**.5

    args = (kin, torch.zeros(count, dtype=DTYPE), existing, gradients, mask, lambda v: h @ v)
    raw = build_contact_directions(*args, max_directions=1, cg_tolerance=1e-9, max_cg_iterations=64)
    # G=H makes the constrained numerical inverse exact in this oracle test.
    # The operator supplied to CG remains H, including all off-mask coupling.
    pcg = build_contact_directions(*args, max_directions=1, cg_tolerance=1e-9,
                                   max_cg_iterations=64, preconditioner_metric=h)
    for result in (raw, pcg):
        record = result.diagnostics['pose_records'][0]
        solution = result.angle_basis[:, 0] * record['proposal_ca_norm_before_normalizing']
        torch.testing.assert_close(solution, oracle, atol=1e-11, rtol=1e-8)
        torch.testing.assert_close(result.angle_basis[:, 0], oracle / oracle_ca_norm, atol=1e-8, rtol=1e-8)
        assert not bool(solution[~mask[0]].any())
        projected_residual = nullspace.T @ (local_h @ solution[indices] + gradients[0, indices])
        assert float(projected_residual.norm() / (nullspace.T @ gradients[0, indices]).norm()) < 5e-9
        assert float((constraints @ solution[indices]).abs().max()) < 1e-9
    assert pcg.diagnostics['pose_records'][0]['cg']['iterations'] == 1
    assert raw.diagnostics['pose_records'][0]['cg']['iterations'] > 1
    assert pcg.diagnostics['hvp_calls'] < raw.diagnostics['hvp_calls']
    assert pcg.diagnostics['response_solver'] == 'projected_pcg'
    assert raw.diagnostics['response_solver'] == 'projected_cg'
    assert pcg.diagnostics['pose_records'][0]['preconditioner']['physical_hessian_modified'] is False


def test_PCG_never_substitutes_preconditioned_norm_for_physical_residual_gate():
    kin = LinearMap(4)
    h = torch.diag(torch.tensor([1., 3., 7., 11.], dtype=DTYPE))
    # This makes the preconditioned residual arbitrarily tiny at every iterate.
    # One iteration still cannot solve the original four-curvature problem.
    with pytest.raises(AdaptiveDirectionError, match='did not converge') as caught:
        build_contact_directions(kin, torch.zeros(4, dtype=DTYPE), torch.empty(4, 0, dtype=DTYPE),
            torch.ones(1, 4, dtype=DTYPE), torch.ones(1, 4, dtype=torch.bool), lambda v: h @ v,
            max_directions=1, max_cg_iterations=1, cg_tolerance=1e-9,
            preconditioner_metric=1e16 * torch.eye(4, dtype=DTYPE))
    assert caught.value.diagnostics['cg']['relative_residual'] > .1


def test_PCG_metric_does_not_regularize_negative_physical_curvature():
    kin = LinearMap(2)
    with pytest.raises(AdaptiveDirectionError, match='curvature'):
        build_contact_directions(kin, torch.zeros(2, dtype=DTYPE), torch.empty(2, 0, dtype=DTYPE),
            torch.ones(1, 2, dtype=DTYPE), torch.ones(1, 2, dtype=torch.bool), lambda v: -.5 * v,
            max_directions=1, preconditioner_metric=.001 * torch.eye(2, dtype=DTYPE))


def test_supplied_metric_preserves_random_controls_bit_for_bit():
    kin = LinearMap(5)
    h = torch.diag(torch.arange(1., 6., dtype=DTYPE))
    existing = torch.ones(5, 1, dtype=DTYPE)
    gradients = torch.ones(2, 5, dtype=DTYPE)
    masks = torch.tensor([[True, True, True, False, False], [False, True, True, True, True]])
    args = (kin, torch.zeros(5, dtype=DTYPE), existing, gradients, masks, lambda v: h @ v)
    before = torch.random.get_rng_state().clone()
    raw = build_contact_directions(*args, max_directions=2, method='random', seed=19)
    with_metric = build_contact_directions(*args, max_directions=2, method='random', seed=19,
        preconditioner_metric=h + torch.ones(5, 5, dtype=DTYPE))
    torch.testing.assert_close(torch.random.get_rng_state(), before, atol=0, rtol=0)
    torch.testing.assert_close(raw.angle_basis, with_metric.angle_basis, atol=0, rtol=0)
    torch.testing.assert_close(raw.basis, with_metric.basis, atol=0, rtol=0)
    assert raw.diagnostics['hvp_calls'] == with_metric.diagnostics['hvp_calls']
    assert with_metric.diagnostics['response_solver'] == 'none_random_control'
    assert all('preconditioner' not in record for record in with_metric.diagnostics['pose_records'])


@pytest.mark.parametrize('metric,match', [
    (torch.tensor([[1., .1], [0., 1.]], dtype=DTYPE), 'symmetric'),
    (-torch.eye(2, dtype=DTYPE), 'positive definite'),
    (torch.ones(2, 2, dtype=DTYPE), 'positive definite'),
    (torch.eye(3, dtype=DTYPE), 'shape'),
    (torch.full((2, 2), float('nan'), dtype=DTYPE), 'finite'),
])
def test_invalid_preconditioning_metric_is_explicitly_rejected(metric, match):
    kin = LinearMap(2)
    with pytest.raises(ValueError, match=match):
        build_contact_directions(kin, torch.zeros(2, dtype=DTYPE), torch.empty(2, 0, dtype=DTYPE),
            torch.ones(1, 2, dtype=DTYPE), torch.ones(1, 2, dtype=torch.bool), lambda v: v,
            max_directions=1, preconditioner_metric=metric)
