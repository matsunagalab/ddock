from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def module(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('mechanism_summary_under_test', scripts / 'summarize_physical_mechanisms.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def verified_example():
    model = dict(basis=np.array([[[1., 0., 0.], [10., 0., 0.]]]), stiffness=np.array([[2.]]))
    row = dict(rotation_index=0, translation=[0., 4., 0.], proposed_amplitudes=[.15],
        gradient=[-.3],
        verified_amplitudes=[.075], fallback=False, accepted_scale=.5, constraint='ball',
        trials=[dict(scale=.5, accepted=True, geometry_feasible=True,
                     ca_rmsd=.07, maximum_atom_displacement=.7, strain=.005625)],
        solver={}, quadratic_gain=-.2, verified_gain=-.1, rigid_energy=-2., verified_energy=-2.1)
    return row, model


def test_summary_distinguishes_tangent_from_finite_deformation_and_active_bounds(module, verified_example):
    row, model = verified_example
    result = module.verification_details(row, model, .5, 1.5, .15, flexible=True)
    assert result['proposed_CA_tangent_norm'] == pytest.approx(.15)
    assert result['proposed_maximum_atom_tangent'] == pytest.approx(1.5)
    assert result['verified_CA_tangent_norm'] == pytest.approx(.075)
    assert result['verified_finite_CA_rmsd'] == pytest.approx(.07)
    assert result['verified_actual_strain'] == pytest.approx(.005625)
    assert result['proposed_quadratic_strain'] == pytest.approx(.0225)
    assert result['amplitude_bound_active'] and result['linear_atom_bound_active']
    assert not result['finite_CA_bound_active'] and not result['finite_atom_bound_active']
    assert result['proposed_usage_conservative_ball'] == pytest.approx(1.)
    assert result['unconstrained_amplitude_norm'] == pytest.approx(.15)
    assert result['proposal_minus_unconstrained_norm'] == 0.


def test_fallback_reports_zero_actual_motion_not_rejected_trial_strain(module, verified_example):
    row, model = verified_example
    row = deepcopy(row)
    row.update(fallback=True, verified_amplitudes=[0.], verified_gain=0., verified_energy=-2.)
    row.pop('accepted_scale')
    row['trials'][0]['accepted'] = False
    result = module.verification_details(row, model, .5, 1.5, .15, flexible=True)
    assert result['verified_finite_CA_rmsd'] == result['verified_actual_strain'] == 0.
    assert result['energy_rejected_trials'] == 1
    aggregated = module.summarize_verification([result])
    assert aggregated['fallback_count'] == 1 and aggregated['accepted_scale_counts'] == {}


@pytest.mark.parametrize('difference', [0., 1e-12])
def test_fullcoords_are_matched_by_pose_key_across_different_selection_ranks(module, tmp_path, difference):
    pose = dict(rotation_index=0, translation=[0., 4., 0.], export_complete=True)
    report = {'arms': {'ball8': {'candidates': [dict(pose, selection_rank=1)]},
                       'atom8': {'candidates': [dict(pose, selection_rank=2)]}}}
    for arm, rank in (('ball8', 1), ('atom8', 2)):
        directory = tmp_path / arm / f'final_{rank:03d}'
        directory.mkdir(parents=True)
        ligand = np.zeros((2, 3))
        if arm == 'atom8':
            ligand[0, 0] = difference
        np.savez(directory / 'full_coordinates.npz', receptor=np.zeros((2, 3)), ligand=ligand)
    result = module.compare_ball_atom(report, tmp_path)
    assert result['common_exported_pose_count'] == 1
    assert result['all_common_full_coordinates_bitwise_equal'] is (difference == 0.)
    assert result['full_coordinates'][0]['partners']['ligand']['maximum_absolute_difference'] == pytest.approx(difference)


def test_incomplete_run_is_rejected_before_reading_artifacts(module, tmp_path):
    (tmp_path / 'report.json').write_text(json.dumps(dict(status='running', native_access=False)))
    with pytest.raises(ValueError, match='completed'):
        module.summarize_run(tmp_path)
