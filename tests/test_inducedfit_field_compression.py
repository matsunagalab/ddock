"""Response-field subspace compression, not docking-accuracy evaluation."""
from copy import deepcopy
from dataclasses import replace
import math

import pytest
import torch

from zdock.inducedfit.field_compression import compress_response_fields
from zdock.inducedfit.fields import FieldParameters, build_fields, physical_energy, receptor_channels
from zdock.inducedfit.response import QuadraticResponse
from zdock.inducedfit.types import AtomKey, GridSpec, MonomerInput


pytestmark = pytest.mark.filterwarnings('ignore:`torch.jit.script` is deprecated.*:DeprecationWarning')


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class NonlinearModel:
    """Independent small coordinate/prior map with a non-diagonal SPD prior."""

    def __init__(self, reference):
        self.reference = reference.clone()
        self.ca_indices = torch.arange(3)
        self.n_modes = 4
        raw = reference.new_tensor([
            [1., 0., .2, 0.], [0., 1., 0., .3], [.1, .2, 1., 0.],
            [-.5, .2, 0., 1.], [.2, -.6, .1, 0.], [0., .3, -.8, .3],
            [.4, 0., .3, -.4], [-.1, .2, .4, .2], [.2, .1, .3, .4],
        ])
        q, _ = torch.linalg.qr(raw, mode='reduced')
        self.basis = q.T.reshape(4, 3, 3) * math.sqrt(3.)
        self.original_regularization = .17
        self.stiffness = reference.new_tensor([
            [2., .3, 0., .1], [.3, 1.4, .2, 0.],
            [0., .2, 3., .4], [.1, 0., .4, .9],
        ]) + self.original_regularization * torch.eye(4, dtype=reference.dtype)
        self.curvature = reference.new_tensor([[.1, .2, 0.], [.3, -.1, .1], [-.2, 0., .2]])

    def coordinates(self, a):
        return self.reference + torch.einsum('m,mnc->nc', a, self.basis) + a.square().sum() * self.curvature

    def deformation_energy(self, a):
        return .5 * a @ self.stiffness @ a + .31 * a.square().sum().square() + .07 * (a[0] * a[1]).square()


@pytest.fixture
def system():
    xyz = torch.tensor([[-.65, 0., 0.], [.55, .1, 0.], [0., .7, .2]], dtype=torch.float64)
    receptor = MonomerInput(xyz, xyz.new_tensor([.45, .42, .48]), xyz.new_tensor([.3, -.2, .1]),
                            tuple(AtomKey('A', i + 1, '', 'ALA', 'CA') for i in range(3)))
    model = NonlinearModel(xyz)
    grid = GridSpec((13, 13, 13), .45, (-2.7, -2.7, -2.7))
    parameters = FieldParameters(surface_width=.15, occupancy_switch_start=2., occupancy_cutoff_sigma=3.,
                                 charge_sigma=.3, charge_switch_start=2., charge_cutoff_sigma=3.,
                                 screening_switch_start=.9, screening_cutoff=1.2,
                                 repulsion_weight=3., charge_weight=.7,
                                 atom_chunk_size=3, point_chunk_size=2197)
    return receptor, model, grid, parameters


def compress(system, **kwargs):
    return compress_response_fields(*system, **kwargs)


def explicit_field_derivative(system):
    """Small full Jacobian oracle differentiates coordinates, not model.basis."""
    receptor, model, grid, parameters = system

    def channels(a):
        fields = build_fields(model.coordinates(a), receptor.radii, receptor.charges, grid, parameters)
        return receptor_channels(fields, grid, parameters).reshape(-1)

    return torch.func.jacfwd(channels)(receptor.xyz.new_zeros(model.n_modes))


def test_full_rank_preserves_nonlinear_coordinates_prior_energy_and_constrained_gain(system):
    receptor, model, grid, parameters = system
    reduced = compress(system, rank=4, oversampling=0, seed=7)
    transform = reduced.transform
    torch.testing.assert_close(transform.T @ transform, torch.eye(4, dtype=transform.dtype), atol=2e-12, rtol=2e-12)
    a = receptor.xyz.new_tensor([.04, -.03, .01, .025])
    b = transform.T @ a
    torch.testing.assert_close(reduced.coordinates(b), model.coordinates(a), atol=2e-13, rtol=2e-13)
    torch.testing.assert_close(reduced.deformation_energy(b), model.deformation_energy(a), atol=2e-13, rtol=2e-13)
    assert reduced.base_model is model
    assert reduced.reference is model.reference and reduced.ca_indices is model.ca_indices
    assert model.original_regularization == .17
    ligand = build_fields(receptor.xyz + .2, receptor.radii, -receptor.charges, grid, parameters)
    original_field = build_fields(model.coordinates(a), receptor.radii, receptor.charges, grid, parameters)
    compressed_field = build_fields(reduced.coordinates(b), receptor.radii, receptor.charges, grid, parameters)
    original_energy = physical_energy(original_field, ligand, grid, parameters, direct=True) + model.deformation_energy(a)
    reduced_energy = physical_energy(compressed_field, ligand, grid, parameters, direct=True) + reduced.deformation_energy(b)
    torch.testing.assert_close(reduced_energy, original_energy, atol=3e-12, rtol=3e-12)
    gradient = receptor.xyz.new_tensor([1., -.6, .4, .2])
    for radius in (0., .08, 2.):
        full = QuadraticResponse(model.stiffness).solve(gradient, radius)
        subspace = QuadraticResponse(reduced.stiffness).solve(transform.T @ gradient, radius)
        torch.testing.assert_close(subspace.gain, full.gain, atol=2e-12, rtol=2e-12)
        torch.testing.assert_close(transform @ subspace.amplitudes, full.amplitudes, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize('rank', [1, 2])
def test_reduced_subspace_cannot_improve_more_than_full_constrained_problem(system, rank):
    receptor, model, _, _ = system
    reduced = compress(system, rank=rank, oversampling=1, seed=3)
    gradients = receptor.xyz.new_tensor([[1., -.6, .4, .2], [-.1, .7, -1., .6], [0., 0., 0., 0.]])
    for radius in (0., .05, .5, 3.):
        full = QuadraticResponse(model.stiffness).solve(gradients, radius)
        subspace = QuadraticResponse(reduced.stiffness).solve(gradients @ reduced.transform, radius)
        # gain is the negative minimized objective; a subspace cannot do better.
        assert bool((subspace.gain >= full.gain - 2e-12).all())
        base_a = subspace.amplitudes @ reduced.transform.T
        assert bool((base_a.norm(dim=-1) <= radius + 2e-12).all())
        objective = (gradients * base_a).sum(-1) + .5 * torch.einsum('bi,ij,bj->b', base_a, model.stiffness, base_a)
        torch.testing.assert_close(subspace.gain, objective, atol=2e-12, rtol=2e-12)


def test_reduced_metric_basis_and_prior_hessian_match_actual_nonlinear_map(system):
    receptor, model, _, _ = system
    reduced = compress(system, rank=2, oversampling=1, seed=4)
    zero = receptor.xyz.new_zeros(reduced.n_modes)
    jacobian = torch.func.jacfwd(reduced.coordinates)(zero).permute(2, 0, 1)
    torch.testing.assert_close(reduced.basis, jacobian, atol=2e-13, rtol=2e-13)
    ca_basis = jacobian[:, reduced.ca_indices].reshape(2, -1)
    torch.testing.assert_close(ca_basis @ ca_basis.T / len(reduced.ca_indices), torch.eye(2, dtype=zero.dtype), atol=2e-12, rtol=2e-12)
    hessian = torch.func.hessian(reduced.deformation_energy)(zero)
    torch.testing.assert_close(hessian, reduced.stiffness, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(reduced.stiffness, reduced.transform.T @ model.stiffness @ reduced.transform, atol=2e-12, rtol=2e-12)
    b = receptor.xyz.new_tensor([.13, -.08])
    # Nonlinear terms survive composition; replacing the prior by .5*bKb fails.
    assert float(reduced.deformation_energy(b)) > float(.5 * b @ reduced.stiffness @ b)
    torch.testing.assert_close(reduced.deformation_energy(b), model.deformation_energy(reduced.transform @ b))


def test_sketch_matches_full_field_svd_when_oversampling_spans_all_modes(system):
    _, model, _, _ = system
    reduced = compress(system, rank=2, oversampling=2, seed=12)
    derivative = explicit_field_derivative(system)
    values, vectors = torch.linalg.eigh(model.stiffness)
    inverse_root = (vectors * values.rsqrt()) @ vectors.T
    root = (vectors * values.sqrt()) @ vectors.T
    actual_a = derivative @ inverse_root
    _, singular_values, right_h = torch.linalg.svd(actual_a, full_matrices=False)
    selected, _ = torch.linalg.qr(root @ reduced.transform, mode='reduced')
    torch.testing.assert_close(selected @ selected.T, right_h[:2].T @ right_h[:2], atol=2e-10, rtol=2e-10)
    diagnostics = reduced.diagnostics
    torch.testing.assert_close(torch.tensor(diagnostics['projected_singular_values']), singular_values, check_dtype=False, atol=2e-7, rtol=2e-7)
    assert diagnostics['field_frobenius_norm'] == pytest.approx(float(actual_a.norm()), rel=2e-12)
    expected_error = float((actual_a - actual_a @ selected @ selected.T).norm())
    assert diagnostics['sketched_rank_r_reconstruction_residual_frobenius'] == pytest.approx(expected_error, rel=3e-11)


def test_approximate_sketch_reports_field_error_bound_and_construction_storage(system):
    _, model, grid, _ = system
    reduced = compress(system, rank=1, oversampling=0, seed=5)
    derivative = explicit_field_derivative(system)
    values, vectors = torch.linalg.eigh(model.stiffness)
    root = (vectors * values.sqrt()) @ vectors.T
    inverse_root = (vectors * values.rsqrt()) @ vectors.T
    actual_a = derivative @ inverse_root
    selected, _ = torch.linalg.qr(root @ reduced.transform, mode='reduced')
    actual_error = float((actual_a - actual_a @ selected @ selected.T).norm())
    diagnostics = reduced.diagnostics
    assert actual_error <= diagnostics['sketched_rank_r_reconstruction_residual_frobenius'] + 2e-12
    assert diagnostics['jvp_calls'] == 1 + model.n_modes
    assert diagnostics['theoretical_sketch_bytes'] == 3 * grid.n_points * 8
    assert diagnostics['subsequent_derivative_fft_channel_sets_after'] == 1
    assert diagnostics['subsequent_derivative_fft_channel_sets_before'] == 4
    assert 'not peak RSS' in diagnostics['memory_note']
    assert 'weighted receptor_channels' in diagnostics['field_metric']
    assert not diagnostics['native_labels_used']


def test_seed_is_local_reproducible_and_can_change_approximate_subspace(system):
    state = torch.random.get_rng_state()
    first = compress(system, rank=1, oversampling=0, seed=17)
    assert torch.equal(torch.random.get_rng_state(), state)
    second = compress(system, rank=1, oversampling=0, seed=17)
    third = compress(system, rank=1, oversampling=0, seed=18)
    assert torch.equal(first.transform, second.transform)
    assert first.diagnostics == second.diagnostics
    assert not torch.allclose(first.transform @ first.transform.T, third.transform @ third.transform.T)
    assert torch.equal(torch.random.get_rng_state(), state)


def test_dense_and_sparse_field_compression_choose_same_subspace(system):
    receptor, model, grid, parameters = system
    dense = compress(system, rank=2, oversampling=1, seed=8)
    sparse = compress((receptor, model, grid, replace(parameters, backend='sparse')),
                      rank=2, oversampling=1, seed=8)
    torch.testing.assert_close(dense.transform @ dense.transform.T, sparse.transform @ sparse.transform.T,
                               atol=2e-10, rtol=2e-10)


def test_field_rank_failure_is_explicit_and_extra_sketch_nulls_are_dropped(system):
    receptor, _, grid, parameters = system
    xyz = receptor.xyz.new_zeros(2, 3)
    atoms = MonomerInput(xyz, xyz.new_full((2,), .45), xyz.new_zeros(2),
                        (AtomKey('A', 1, '', 'ALA', 'CA'), AtomKey('A', 2, '', 'ALA', 'CA')))
    model = NonlinearModel(receptor.xyz)
    model.reference = xyz
    model.ca_indices = torch.arange(2)
    model.n_modes = 2
    model.basis = xyz.new_tensor([[[1., 0., 0.], [1., 0., 0.]], [[0., 1., 0.], [0., -1., 0.]]])
    model.stiffness = torch.eye(2, dtype=xyz.dtype)
    model.curvature = torch.zeros_like(xyz)
    with pytest.raises(ValueError, match='numerical rank 1'):
        compress_response_fields(atoms, model, grid, parameters, rank=2, oversampling=0)
    reduced = compress_response_fields(atoms, model, grid, parameters, rank=1, oversampling=1)
    assert reduced.n_modes == 1 and reduced.diagnostics['sketch_numerical_rank'] == 1
    assert abs(float(reduced.transform[1, 0])) < 1e-12


@pytest.mark.parametrize('kwargs,match', [
    ({'rank': 0}, 'rank'), ({'rank': 5}, 'rank'), ({'rank': True}, 'rank'),
    ({'rank': 1, 'oversampling': -1}, 'oversampling'),
    ({'rank': 1, 'seed': -1}, 'seed'),
    ({'rank': 1, 'rank_rtol': float('nan')}, 'rank_rtol'),
])
def test_invalid_options_fail_before_field_work(system, kwargs, match):
    with pytest.raises(ValueError, match=match):
        compress(system, **kwargs)


@pytest.mark.parametrize('failure,match', [
    ('reference', 'reference'), ('metric', 'CA displacement metric'),
    ('indefinite', 'positive definite'), ('nonsymmetric', 'symmetric'), ('nonfinite', 'finite'),
])
def test_invalid_model_contract_is_not_silently_repaired(system, failure, match):
    receptor, original, grid, parameters = system
    model = deepcopy(original)
    if failure == 'reference':
        model.reference[0, 0] += .01
    elif failure == 'metric':
        model.basis *= 2.
    elif failure == 'indefinite':
        model.stiffness[0, 0] = -1.
    elif failure == 'nonsymmetric':
        model.stiffness[0, 1] += .1
    else:
        model.basis[0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match=match):
        compress_response_fields(receptor, model, grid, parameters, rank=1)


def test_cpu_float64_contract_and_returned_argument_checks(system):
    receptor, model, grid, parameters = system
    single = MonomerInput(receptor.xyz.float(), receptor.radii.float(), receptor.charges.float(), receptor.atom_keys)
    with pytest.raises(TypeError, match='CPU float64'):
        compress_response_fields(single, model, grid, parameters, rank=1)
    reduced = compress(system, rank=1, oversampling=0)
    with pytest.raises(ValueError, match='shape'):
        reduced.coordinates(torch.zeros(2, dtype=torch.float64))
    with pytest.raises(ValueError, match='dtype'):
        reduced.deformation_energy(torch.zeros(1, dtype=torch.float32))
