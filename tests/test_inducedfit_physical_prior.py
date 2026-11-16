"""The physical prior retains forces, a stationary baseline and true curvature."""
from types import SimpleNamespace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from zdock.inducedfit import physical_prior as module
from zdock.inducedfit.physical_prior import (PhysicalPrior, PhysicalPreparationError,
    build_physical_modes, load_prepared_physical_prior)
from zdock.inducedfit.chemistry import prepare_amber_charges
from zdock.inducedfit.pdb_input import read_pdb_monomer
from zdock.inducedfit.types import AtomKey
from test_inducedfit_chemistry import make_dipeptide_pdb

pytestmark = pytest.mark.filterwarnings("ignore:`torch.jit.script` is deprecated.*:DeprecationWarning")


class LinearMap:
    def __init__(self):
        self.reference = torch.tensor([[0., 0., 0.], [2., 0., 0.], [0., 2., 1.]], dtype=torch.float64)
        self.ca_indices = torch.arange(3)
        self.n_torsions = 2
        self.jacobian = torch.tensor([[[1., 0., 0.], [0., 0., 0.], [0., 0., 1.]],
                                      [[0., 0., 0.], [0., 1., 0.], [0., 0., 1.]]], dtype=torch.float64)

    def coordinates(self, theta):
        return self.reference + torch.einsum('m,mnc->nc', theta, self.jacobian)


@pytest.fixture
def toy_prior(monkeypatch):
    kin = LinearMap()
    target = kin.reference + kin.jacobian[0]*.08 - kin.jacobian[1]*.05

    class HarmonicContext:
        def __init__(self, protein):
            self.provenance = {'test_only': True}
            self.evaluations = 0
            self.closed = False

        def evaluate(self, xyz, *, components=False):
            if self.closed:
                raise RuntimeError('closed')
            delta = xyz.detach()-target
            self.evaluations += 1
            value = float(3. * delta.square().sum())
            return value, 6.*delta, dict(bonded=0., vdw=value, coulomb=0., gbsa=0.) if components else None

        def close(self):
            self.closed = True

    monkeypatch.setattr(module, '_ForceContext', HarmonicContext)
    keys = tuple(AtomKey('A', i+1, '', 'GLY', 'CA') for i in range(3))
    protein = SimpleNamespace(monomer=SimpleNamespace(xyz=kin.reference,
        charges=torch.zeros(3, dtype=torch.float64), atom_keys=keys),
        bonds=((0, 1), (1, 2)), heavy_indices=torch.arange(3))
    prior = PhysicalPrior(protein, kinematics=kin)
    yield prior
    prior.close()


def test_common_stationary_reference_is_true_minimum_without_linear_shift(toy_prior):
    prior = toy_prior.prepare()
    expected = torch.tensor([.08, -.05], dtype=torch.float64)*6./16.
    torch.testing.assert_close(prior.theta0, expected, atol=2e-10, rtol=0.)
    assert prior.diagnostics['maximum_abs_torque'] < 1e-8
    assert prior.diagnostics['status'] == 'prepared'
    assert torch.equal(prior.anchor_reference, prior.kinematics.reference)
    torch.testing.assert_close(prior.prepared_kinematics.coordinates(torch.zeros(2, dtype=torch.float64)), prior.reference)
    assert len(prior.diagnostics['evaluations']) > 1
    with pytest.raises(ValueError, match='already prepared'):
        prior.prepare()


def test_energy_bridge_and_full_torque_hvp_match_independent_quadratic(toy_prior):
    prior = toy_prior
    theta = torch.tensor([.03, -.02], dtype=torch.float64)
    direction = torch.tensor([.7, -.4], dtype=torch.float64)
    jac = prior.kinematics.jacobian.reshape(2, -1)
    expected_hessian = 16. * jac @ jac.T
    hv = prior.hessian_vector(theta, direction)
    torch.testing.assert_close(hv, expected_hessian @ direction, atol=1e-9, rtol=1e-10)
    energy, gradient = prior.energy_and_gradient(theta)
    h = 1e-6
    numeric = (float(prior.energy(theta+h*direction))-float(prior.energy(theta-h*direction)))/(2*h)
    assert float(gradient @ direction) == pytest.approx(numeric, abs=1e-8)
    torch.testing.assert_close(prior.hessian_vector(theta, torch.zeros_like(theta)), torch.zeros_like(theta))
    assert np.isfinite(energy)


def test_hvp_includes_curvature_of_the_coordinate_map(toy_prior):
    original = toy_prior.kinematics
    curvature = torch.tensor([[.4, .3, 0.], [0., -.1, .2], [.1, 0., -.2]], dtype=torch.float64)

    class CurvedMap(LinearMap):
        def coordinates(self, theta):
            return super().coordinates(theta) + .5 * theta[0].square() * curvature

    toy_prior.kinematics = CurvedMap()
    theta = torch.tensor([.04, -.03], dtype=torch.float64)
    direction = torch.tensor([.7, -.4], dtype=torch.float64)
    tangents = original.jacobian.clone()
    tangents[0] += theta[0] * curvature
    displacement = toy_prior.kinematics.coordinates(theta)-original.reference
    target_displacement = original.jacobian[0]*.08-original.jacobian[1]*.05
    cartesian_gradient = 16.*displacement - 6.*target_displacement
    expected = 16.*tangents.reshape(2,-1) @ tangents.reshape(2,-1).T
    expected[0,0] += (cartesian_gradient*curvature).sum()
    actual = toy_prior.hessian_vector(theta, direction, step=1e-5)
    torch.testing.assert_close(actual, expected @ direction, atol=1e-8, rtol=1e-9)


@pytest.mark.parametrize('method', ['diagonal', 'anchor_metric'])
def test_preconditioner_changes_variables_without_changing_physical_minimum(toy_prior, method):
    transform, _, diagnostics = toy_prior._variable_transform(method)
    jacobian = toy_prior.kinematics.jacobian.reshape(2, -1)
    metric = toy_prior.anchor_k * jacobian @ jacobian.T + torch.eye(2, dtype=torch.float64)
    if method == 'diagonal':
        metric = torch.diag(metric.diag())
    torch.testing.assert_close(transform.T @ metric @ transform, torch.eye(2, dtype=torch.float64),
                               atol=1e-12, rtol=0.)
    assert diagnostics['materialized_physical_hessian'] is False
    assert len(diagnostics['transform_sha256']) == 64
    toy_prior.prepare(preconditioner=method)
    expected = torch.tensor([.08, -.05], dtype=torch.float64)*6./16.
    torch.testing.assert_close(toy_prior.theta0, expected, atol=2e-10, rtol=0.)


def test_physical_modes_have_ca_metric_and_stiffness_from_same_energy(toy_prior):
    prior = toy_prior.prepare()
    model = build_physical_modes(prior, torch.eye(2, dtype=torch.float64))
    ca = model.basis[:, model.ca_indices].reshape(2, -1)
    torch.testing.assert_close(ca @ ca.T/3., torch.eye(2, dtype=torch.float64), atol=1e-12, rtol=0.)
    # This fixture has atomwise Hessian 16 I and all atoms are CA, hence K=48 I.
    torch.testing.assert_close(model.stiffness, 48.*torch.eye(2, dtype=torch.float64), atol=1e-9, rtol=1e-10)
    a = torch.tensor([.03, -.01], dtype=torch.float64, requires_grad=True)
    strain = model.deformation_energy(a)
    assert float(strain.detach()) == pytest.approx(float(.5*a.detach() @ model.stiffness @ a.detach()), abs=1e-10)
    gradient, = torch.autograd.grad(strain, a)
    torch.testing.assert_close(gradient, model.stiffness @ a.detach(), atol=1e-9, rtol=1e-9)
    assert model.deformation_energy(torch.zeros_like(a)).item() == pytest.approx(0., abs=1e-12)


def test_preparation_failures_cannot_create_a_usable_model(toy_prior):
    with pytest.raises(PhysicalPreparationError, match='moved a heavy atom') as caught:
        toy_prior.prepare(max_heavy_displacement=1e-5)
    assert caught.value.diagnostics['status'] == 'failed'
    assert toy_prior.theta0 is None
    assert caught.value.diagnostics['evaluations']
    with pytest.raises(ValueError, match='successfully prepared'):
        build_physical_modes(toy_prior, torch.eye(2, dtype=torch.float64))


def test_budget_failure_retains_history(toy_prior):
    with pytest.raises(PhysicalPreparationError, match='budget') as caught:
        toy_prior.prepare(max_evaluations=1)
    assert len(caught.value.diagnostics['evaluations']) <= 1
    assert caught.value.diagnostics['openmm_evaluations'] <= 1
    assert toy_prior.theta0 is None


def test_newton_polishing_resolves_residual_without_hiding_lbfgs_failure(toy_prior):
    toy_prior.prepare(max_iterations=1)
    assert toy_prior.diagnostics['lbfgs_result']['success'] is False
    assert toy_prior.diagnostics['polishing']['success'] is True
    assert toy_prior.diagnostics['polishing']['iterations']
    assert toy_prior.diagnostics['maximum_abs_torque'] < 1e-3
    assert toy_prior.diagnostics['polishing']['physical_hessian_materialized'] is False


def test_newton_nonpositive_curvature_is_retained_as_failure(toy_prior):
    def concave(theta, phase):
        return float(-.5*theta.square().sum()), -theta
    with pytest.raises(RuntimeError, match='nonpositive curvature'):
        toy_prior._polish_torque(torch.ones(2, dtype=torch.float64),
            torch.eye(2, dtype=torch.float64), concave, .001)
    assert toy_prior.diagnostics['polishing']['iterations'][0]['cg_curvatures'][0] < 0


def test_singular_metric_and_nonpositive_stiffness_are_errors(toy_prior, monkeypatch):
    prior = toy_prior.prepare()
    with pytest.raises(ValueError, match='rank deficient'):
        build_physical_modes(prior, torch.ones((2, 2), dtype=torch.float64))
    monkeypatch.setattr(prior, 'hessian_vector', lambda theta, vector, step: -vector)
    with pytest.raises(ValueError, match='not positive definite'):
        build_physical_modes(prior, torch.eye(2, dtype=torch.float64))


def test_unresolved_hessian_is_not_silently_symmetrized(toy_prior, monkeypatch):
    prior = toy_prior.prepare()
    matrix = torch.tensor([[3., 2.], [0., 3.]], dtype=torch.float64)
    monkeypatch.setattr(prior, 'hessian_vector', lambda theta, vector, step: matrix @ vector)
    with pytest.raises(ValueError, match='unresolved'):
        build_physical_modes(prior, torch.eye(2, dtype=torch.float64))


@pytest.mark.parametrize('name,value', [('anchor_k', 0.), ('anchor_k', float('nan'))])
def test_invalid_model_parameters_fail_before_openmm(toy_prior, name, value):
    with pytest.raises(ValueError, match=name):
        PhysicalPrior(toy_prior.protein, kinematics=toy_prior.kinematics, **{name: value})


def test_openmm_cartesian_and_torsional_forces_include_all_physical_terms(tmp_path):
    pytest.importorskip('openmm')
    protein = prepare_amber_charges(read_pdb_monomer(make_dipeptide_pdb(tmp_path/'pair.pdb'),
        chain='A', charge_policy='zero_diagnostic'), seed=0)
    with PhysicalPrior(protein) as prior:
        theta = torch.linspace(-.01, .015, prior.kinematics.n_torsions, dtype=torch.float64)
        energy, gradient = prior.energy_and_gradient(theta)
        direction = torch.arange(1, len(theta)+1, dtype=torch.float64)
        direction /= direction.norm()
        h = 1e-6
        numeric = (float(prior.energy(theta+h*direction))-float(prior.energy(theta-h*direction)))/(2*h)
        assert float(gradient @ direction) == pytest.approx(numeric, rel=2e-6, abs=1e-5)
        components = prior.components(theta)
        assert components['total'] == pytest.approx(energy, abs=1e-8)
        assert energy == pytest.approx(sum(components[k] for k in ['bonded','vdw','coulomb','gbsa','anchor']), abs=1e-8)
        # Cartesian energy/force bridge has explicit Angstrom-to-nanometer conversion.
        xyz = prior.kinematics.coordinates(theta).detach().requires_grad_(True)
        cart_grad, = torch.autograd.grad(prior.energy_xyz(xyz), xyz)
        offset = torch.zeros_like(xyz); offset[-1, 0] = h
        numeric_cart = (float(prior.energy_xyz(xyz.detach()+offset))-float(prior.energy_xyz(xyz.detach()-offset)))/(2*h)
        assert float(cart_grad[-1,0]) == pytest.approx(numeric_cart, rel=2e-6, abs=1e-5)
        prior.prepare(max_iterations=300, max_evaluations=1200)
        assert prior.diagnostics['maximum_abs_torque'] <= 1e-3
        model = build_physical_modes(prior, torch.eye(prior.kinematics.n_torsions, dtype=torch.float64)[:, :1])
        assert model.stiffness[0,0] > 0
        assert float(model.deformation_energy(torch.zeros(1, dtype=torch.float64))) == pytest.approx(0., abs=1e-9)
        # Replay requires identical sources/input and fresh full-energy/torque checks.
        repo = Path(module.__file__).resolve().parents[3]
        paths = [repo/'src/zdock/inducedfit'/name for name in
                 ['physical_prior.py', 'atomistic_score.py', 'hydrogen_kinematics.py',
                  'kinematics.py', 'chemistry.py', 'pdb_input.py', 'types.py']]
        archive = tmp_path/'arrays.npz'
        np.savez(archive, theta0=prior.theta0.numpy(), reference=prior.reference.numpy())
        report = dict(status='passed', preparation=prior.diagnostics,
            arrays_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            registration=dict(configuration=dict(anchor_k=10., stationarity_tolerance=1e-3,
                max_heavy_displacement=1.), source_sha256={str(p.relative_to(repo)):
                hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
        report_path = tmp_path/'report.json'
        report_path.write_text(json.dumps(report))
        with load_prepared_physical_prior(protein, report_path) as restored:
            torch.testing.assert_close(restored.theta0, prior.theta0, atol=0., rtol=0.)
            assert restored.diagnostics['restore_validation']['maximum_abs_torque'] <= 1e-3
        with pytest.raises(ValueError, match='different anchor_k'):
            load_prepared_physical_prior(protein, report_path, anchor_k=11.)
        altered = prior.theta0 + .02
        altered_xyz = prior.kinematics.coordinates(altered).detach()
        np.savez(archive, theta0=altered.numpy(), reference=altered_xyz.numpy())
        report['arrays_sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
        report['preparation']['final_energy'] = float(prior.energy(altered))
        report_path.write_text(json.dumps(report))
        with pytest.raises(ValueError, match='recomputed stationarity'):
            load_prepared_physical_prior(protein, report_path)
