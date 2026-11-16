"""Actual-energy acceptance, nonlinear trust bounds, and rigid fallback."""

from dataclasses import replace

import pytest
import torch

from zdock.inducedfit.fields import FieldParameters, build_fields, physical_energy, support_radii
from zdock.inducedfit.response import QuadraticResponse
from zdock.inducedfit.search import ResponseCandidate
from zdock.inducedfit.types import AtomKey, GridSpec, MonomerInput
from zdock.inducedfit.verified_response import NonlinearResponseVerifier


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class ToyModel:
    """One-mode numerical model with controllable nonlinear prior/geometry."""

    def __init__(self, reference, quartic=0., coordinate_cubic=0.):
        self.reference = reference.clone()
        self.ca_indices = torch.tensor([0])
        self.n_modes = 1
        self.stiffness = reference.new_tensor([[2.]])
        self.quartic = quartic
        self.coordinate_cubic = coordinate_cubic

    def coordinates(self, amplitudes):
        amount = amplitudes[0] + self.coordinate_cubic * amplitudes[0] ** 3
        displacement = torch.stack((amount, amount * 0, amount * 0))
        return self.reference + displacement

    def deformation_energy(self, amplitudes):
        return 0.5 * amplitudes @ self.stiffness @ amplitudes + self.quartic * amplitudes[0] ** 4


@pytest.fixture
def system():
    grid = GridSpec((17, 17, 17), .6, (-4.8, -4.8, -4.8))
    parameters = FieldParameters(surface_width=.2, occupancy_switch_start=2., occupancy_cutoff_sigma=3.,
                                 charge_sigma=.4, charge_switch_start=2., charge_cutoff_sigma=3.,
                                 screening_switch_start=1.8, screening_cutoff=2.4,
                                 atom_chunk_size=2, point_chunk_size=211)
    rec_xyz = torch.tensor([[0., 0., 0.]], dtype=torch.float64)
    lig_xyz = torch.tensor([[1.1, .2, .1]], dtype=torch.float64)
    receptor = MonomerInput(rec_xyz, rec_xyz.new_tensor([.55]), rec_xyz.new_tensor([.7]),
                            (AtomKey('R', 1, '', 'ALA', 'CA'),))
    ligand = MonomerInput(lig_xyz, lig_xyz.new_tensor([.55]), lig_xyz.new_tensor([-.6]),
                          (AtomKey('L', 1, '', 'ALA', 'CA'),))
    return receptor, ligand, grid, parameters


def make_candidate(receptor, ligand, model, grid, parameters, radius=.3):
    a = receptor.xyz.new_zeros(1).requires_grad_()
    rec_fields = build_fields(model.coordinates(a), receptor.radii, receptor.charges, grid, parameters)
    lig_fields = build_fields(ligand.xyz, ligand.radii, ligand.charges, grid, parameters)
    energy = physical_energy(rec_fields, lig_fields, grid, parameters)
    gradient, = torch.autograd.grad(energy, a)
    solution = QuadraticResponse(model.stiffness).solve(gradient, radius)
    assert solution.gain < 0
    return ResponseCandidate(0, receptor.xyz.new_zeros(3), solution.amplitudes,
                             float(energy.detach()) + float(solution.gain), float(energy.detach()),
                             float(solution.gain), float(solution.strain))


def verifier(system, model, **kwargs):
    receptor, ligand, grid, parameters = system
    return NonlinearResponseVerifier(receptor, ligand, model, grid, parameters,
                                      radius=.3, max_atom_displacement=.3, **kwargs)


def identity(reference):
    return reference.new_tensor([0., 0., 0., 1.])


def test_quartic_misprediction_backtracks_and_scores_actual_energy(system):
    receptor, ligand, grid, parameters = system
    model = ToyModel(receptor.xyz)
    candidate = make_candidate(receptor, ligand, model, grid, parameters)
    model.quartic = 10 * abs(candidate.predicted_gain) / float(candidate.amplitudes[0] ** 4)
    result = verifier(system, model).verify(candidate, identity(receptor.xyz))
    assert not result.fallback
    assert 0 < result.accepted_scale < 1
    assert len(result.trials) > 1
    assert result.trials[-1]['status'] == 'accepted'
    assert result.trials[-1]['agreement'] >= .75
    assert result.score < result.rigid_energy
    assert result.score == pytest.approx(result.rigid_energy + result.verified_gain)
    assert result.score != pytest.approx(candidate.score)
    assert result.original_candidate is candidate
    # Independent spatial convolution oracle at the actual accepted structure.
    rec_fields = build_fields(model.coordinates(result.amplitudes), receptor.radii, receptor.charges, grid, parameters)
    lig_fields = build_fields(ligand.xyz, ligand.radii, ligand.charges, grid, parameters)
    expected = physical_energy(rec_fields, lig_fields, grid, parameters, direct=True) + model.deformation_energy(result.amplitudes)
    assert result.score == pytest.approx(float(expected), abs=1e-12)
    quadratic_strain = .5 * candidate.amplitudes @ model.stiffness @ candidate.amplitudes
    for trial in result.trials:
        scale = trial['scale']
        expected_prediction = scale * (candidate.predicted_gain - float(quadratic_strain)) + scale**2 * float(quadratic_strain)
        assert trial['predicted_gain'] == pytest.approx(expected_prediction)


def test_actual_ca_displacement_is_checked_separately_from_amplitude_norm(system):
    receptor, ligand, grid, parameters = system
    model = ToyModel(receptor.xyz)
    candidate = make_candidate(receptor, ligand, model, grid, parameters)
    amplitude = abs(float(candidate.amplitudes[0]))
    model.coordinate_cubic = (0.6 / amplitude - 1) / amplitude**2
    result = verifier(system, model).verify(candidate, identity(receptor.xyz))
    assert result.trials[0]['amplitude_bound_ok']
    assert not result.trials[0]['actual_ca_bound_ok']
    assert result.trials[0]['status'] == 'geometry_infeasible'
    assert all(trial['status'] != 'accepted' or trial['actual_ca_bound_ok'] for trial in result.trials)
    assert result.score <= result.rigid_energy


def test_all_six_bad_energy_trials_fall_back_to_zero_without_predicted_bonus(system):
    receptor, ligand, grid, parameters = system
    model = ToyModel(receptor.xyz)
    candidate = make_candidate(receptor, ligand, model, grid, parameters)
    model.quartic = 1e8 * abs(candidate.predicted_gain) / float(candidate.amplitudes[0] ** 4)
    result = verifier(system, model).verify(candidate, identity(receptor.xyz))
    assert result.fallback
    assert result.accepted_scale is None
    assert len(result.trials) == 6
    assert result.trials[-1]['scale'] == 1 / 32
    assert all(trial['status'] == 'no_predicted_or_actual_decrease' for trial in result.trials)
    assert result.score == result.rigid_energy
    assert result.verified_gain == result.predicted_gain == result.strain == 0
    assert not bool(result.amplitudes.any())


def test_all_geometry_trials_fail_safely(system):
    receptor, ligand, grid, parameters = system
    model = ToyModel(receptor.xyz)
    candidate = make_candidate(receptor, ligand, model, grid, parameters)
    smallest_amplitude = abs(float(candidate.amplitudes[0])) / 32
    model.coordinate_cubic = (0.6 / smallest_amplitude - 1) / smallest_amplitude**2
    result = verifier(system, model).verify(candidate, identity(receptor.xyz))
    assert result.fallback
    assert len(result.trials) == 6
    assert all(trial['status'] == 'geometry_infeasible' for trial in result.trials)
    assert result.score == pytest.approx(candidate.rigid_energy, abs=1e-12)


def test_support_violation_is_logged_as_trial_rejection(system):
    receptor, ligand, grid, parameters = system
    xyz = receptor.xyz.clone()
    xyz[0, 0] = grid.bounds(dtype=xyz.dtype)[1][0] - support_radii(receptor.radii, parameters)[0]
    receptor = replace(receptor, xyz=xyz)
    model = ToyModel(xyz)
    rec_fields = build_fields(xyz, receptor.radii, receptor.charges, grid, parameters)
    lig_fields = build_fields(ligand.xyz, ligand.radii, ligand.charges, grid, parameters)
    rigid = float(physical_energy(rec_fields, lig_fields, grid, parameters))
    candidate = ResponseCandidate(0, xyz.new_zeros(3), xyz.new_tensor([.2]), rigid - 1., rigid, -1., .04)
    result = verifier((receptor, ligand, grid, parameters), model).verify(candidate, identity(xyz))
    assert result.fallback
    assert all(trial['status'] == 'support_infeasible' for trial in result.trials)
    assert result.score == rigid


@pytest.mark.parametrize("backend", ['dense', 'sparse'])
def test_verified_score_never_deteriorates_for_physical_fixture(system, backend):
    receptor, ligand, grid, parameters = system
    parameters = replace(parameters, backend=backend)
    model = ToyModel(receptor.xyz)
    candidate = make_candidate(receptor, ligand, model, grid, parameters)
    result = verifier((receptor, ligand, grid, parameters), model).verify(candidate, identity(receptor.xyz))
    assert result.score <= result.rigid_energy
    assert abs(result.rigid_energy_error) < 1e-12
    if not result.fallback:
        assert result.verified_gain < 0 and result.predicted_gain < 0
        assert result.verified_gain / result.predicted_gain >= .75
        assert result.amplitudes.norm() <= .3 + 1e-12


def test_rigid_mismatch_and_nonfinite_energy_are_hard_failures(system):
    receptor, ligand, grid, parameters = system
    model = ToyModel(receptor.xyz)
    candidate = make_candidate(receptor, ligand, model, grid, parameters)
    checker = verifier(system, model)
    with pytest.raises(ValueError, match='rigid energy mismatch'):
        checker.verify(replace(candidate, rigid_energy=candidate.rigid_energy + 1.), identity(receptor.xyz))
    with pytest.raises(FloatingPointError, match='nonfinite input candidate'):
        checker.verify(replace(candidate, predicted_gain=float('nan')), identity(receptor.xyz))
    model.quartic = float('inf')
    with pytest.raises(FloatingPointError, match='actual nonlinear energy'):
        checker.verify(candidate, identity(receptor.xyz))


def test_invalid_verification_policy_and_reference_are_rejected(system):
    receptor, _, _, _ = system
    model = ToyModel(receptor.xyz)
    for threshold in (0., 1.1, float('nan')):
        with pytest.raises(ValueError, match='min_agreement'):
            verifier(system, model, min_agreement=threshold)
    with pytest.raises(ValueError, match='max_trials'):
        verifier(system, model, max_trials=0)
    model.reference += 1
    with pytest.raises(ValueError, match='reference'):
        verifier(system, model)
