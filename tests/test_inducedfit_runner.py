"""Selection integration tests; physical verifier accuracy is tested separately.

Mock expensive preparation/search/energy operations at their API boundaries so
these tests isolate pool sharing, truncation, ranking and refinement starts.
"""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from zdock.inducedfit.search import ResponseCandidate
from zdock.inducedfit.types import AtomKey, GridSpec, MonomerInput
from zdock.inducedfit.verified_response import VerifiedResponse


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[1] / 'scripts/run_induced_fit_pair.py'
    spec = spec_from_file_location('inducedfit_runner_under_test', path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def proposal(index):
    return ResponseCandidate(
        rotation_index=index,
        translation=torch.tensor([float(index), 0., 0.], dtype=torch.float64),
        amplitudes=torch.tensor([.1], dtype=torch.float64),
        score=-100. + index, rigid_energy=100., predicted_gain=-200. + index,
        strain=.01,
    )


def verified(candidate, score, *, fallback=False):
    return VerifiedResponse(
        original_candidate=candidate, score=score, rigid_energy=100.,
        verified_gain=score - 100., predicted_gain=0. if fallback else -250.,
        amplitudes=torch.zeros_like(candidate.amplitudes) if fallback else .5 * candidate.amplitudes,
        strain=0. if fallback else .005, accepted_scale=None if fallback else .5,
        fallback=fallback, rigid_energy_error=0.,
        trials=({'scale': 1., 'status': 'poor_agreement'},),
    )


def test_verifies_entire_pool_before_topk_and_uses_actual_scores(runner):
    pool = [proposal(i) for i in range(5)]
    quaternions = torch.arange(20, dtype=torch.float64).reshape(5, 4)
    # The last two proposals win after verification, contrary to raw scores.
    # An actual rigid fallback is still ranked by its measured score.
    scores = [100., -10., 0., -40., -40.]
    events, records = [], []

    def verify(candidate, quaternion):
        index = next(i for i, item in enumerate(pool) if item is candidate)
        torch.testing.assert_close(quaternion, quaternions[index])
        events.append(('verify', index))
        return verified(candidate, scores[index], fallback=index == 0)

    def record(row):
        records.append(row)
        events.append(('record', row['pool_rank'] - 1))

    selected = runner.verify_and_rank_candidates(
        pool, quaternions, SimpleNamespace(verify=verify), 2, record,
    )
    assert events == [event for i in range(5) for event in [('verify', i), ('record', i)]]
    assert [item.original_candidate.rotation_index for item in selected] == [3, 4]
    assert [item.score for item in selected] == [-40., -40.]
    assert [row['pool_rank'] for row in records] == [1, 2, 3, 4, 5]
    assert records[0]['fallback'] and records[0]['score'] == 100.
    assert records[4]['original_score'] == pool[4].score
    assert records[4]['original_amplitudes'] == [.1]
    assert [item.rotation_index for item in pool] == list(range(5))


@pytest.fixture
def stubbed_run(runner, monkeypatch, tmp_path):
    """Real runner control flow with deterministic recorded component results."""
    reference = torch.tensor([[0., 0., 0.]], dtype=torch.float64)
    monomer = MonomerInput(reference, reference.new_ones(1), reference.new_zeros(1),
                           (AtomKey('A', 1, '', 'GLY', 'CA'),))
    protein = SimpleNamespace(monomer=monomer, heavy_indices=torch.tensor([0]), provenance={})
    model = SimpleNamespace(reference=monomer.xyz, stiffness=reference.new_ones(1, 1),
                            basis=reference.new_zeros(1, 1, 3), n_modes=1, diagnostics={},
                            coordinates=lambda a: monomer.xyz + a[0])
    grid = GridSpec((3, 3, 3), 2., (-2., -2., -2.))
    pool = [proposal(i) for i in range(50)]
    quaternions = reference.new_tensor([0., 0., 0., 1.]).repeat(50, 1)
    events, exports, verifier_options, mode_options = [], [], [], []
    fail_at = {'index': None}
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text('{}')
    inputs = {side: SimpleNamespace(provenance={}) for side in ('receptor', 'ligand')}
    monkeypatch.setattr(runner, 'load_pair_manifest', lambda path: ({'split': 'development'}, inputs))
    monkeypatch.setattr(runner, 'prepare_amber_charges', lambda *a, **kw: protein)
    monkeypatch.setattr(runner, 'centered_protein', lambda value: (value, reference.new_zeros(3)))
    monkeypatch.setattr(runner.HydrogenatedKinematics, 'from_prepared', lambda value: value)
    def build_modes(*args, **kwargs):
        mode_options.append(kwargs)
        return model
    monkeypatch.setattr(runner, 'build_torsion_modes', build_modes)
    monkeypatch.setattr(runner, 'pair_grid', lambda *a, **kw: grid)
    cached = SimpleNamespace(bounded_radius=lambda radius, displacement: radius,
                             maximum_atom_motion_per_unit_amplitude=1.)
    monkeypatch.setattr(runner, 'prepare_response_receptor', lambda *a, **kw: cached)
    monkeypatch.setattr(runner, 'rotation_sample', lambda *a, **kw: quaternions)

    def search(*args, **kwargs):
        events.append(('search', kwargs.copy()))
        return pool[:kwargs['top_k']]

    class Verifier:
        def __init__(self, *args, **kwargs):
            verifier_options.append(kwargs)

        def verify(self, candidate, quaternion):
            index = candidate.rotation_index
            assert candidate is pool[index]
            torch.testing.assert_close(quaternion, quaternions[index])
            events.append(('verify', index))
            if index == fail_at['index']:
                raise FloatingPointError('controlled nonfinite verification')
            # Candidates 48/49 are outside the raw TopK; stable ties retain 48.
            return verified(candidate, -1000. if index >= 48 else float(index))

    def export(path, rec_input, lig_input, rec_xyz, lig_xyz, center):
        exports.append((path, rec_xyz.clone(), lig_xyz.clone()))
        return {side: path / f'{side}.pdb' for side in ('receptor', 'ligand', 'complex')}

    active_report = {}

    def relax(rec, lig, deformation, spec, parameters, quaternion, translation, *,
              initial_amplitudes=None, **kwargs):
        # This is the actual relax_pose zero-default contract, tested at its
        # caller boundary; a nonzero verified warm start must fail this test.
        assert rec is monomer and lig is monomer and deformation is model
        assert initial_amplitudes is None or torch.equal(initial_amplitudes, reference.new_zeros(1))
        stage = active_report['value']['stage']
        events.append(('relax', stage, int(translation[0]), kwargs.copy()))
        return SimpleNamespace(
            receptor_xyz=monomer.xyz, ligand_xyz=monomer.xyz + translation,
            amplitudes=reference.new_zeros(1), rotation_increment=reference.new_tensor([.01, .02, .03]),
            translation_increment=reference.new_tensor([.1, .2, .3]),
            energy=float(translation[0]), strain=0., ca_rmsd=0., maximum_atom_displacement=0.,
            status='converged', accepted_steps=0, energy_evaluations=1,
            energy_history=(float(translation[0]),),
        )

    monkeypatch.setattr(runner, 'search_response', search)
    monkeypatch.setattr(runner, 'NonlinearResponseVerifier', Verifier)
    monkeypatch.setattr(runner, 'export_heavy_pair', export)
    monkeypatch.setattr(runner, 'relax_pose', relax)

    def execute(arms, **options):
        args = SimpleNamespace(
            manifest=manifest_path, out=tmp_path / 'out', ph=7., seed=0, modes=1,
            fit_damping=.001, field_backend='sparse', spacing=2.,
            max_atom_displacement=1.5, max_translation=.5, prepare_only=False,
            max_grid_points=1000, radius=.5, rotations=50, arms=arms,
            response_pool_size=50, top_k=2, peaks_per_rotation=4,
            iterations=8, max_evaluations=64, checkpoint_fields=True,
        )
        for key, value in options.items():
            setattr(args, key, value)
        report = {'arms': {arm: {'status': 'not_attempted'} for arm in arms}}
        active_report['value'] = report
        runner.run(args, report, lambda: None)
        return report

    return SimpleNamespace(execute=execute, events=events, exports=exports,
                           verifier_options=verifier_options, mode_options=mode_options,
                           fail_at=fail_at, pool=pool)


@pytest.mark.parametrize('arms', [('R1', 'R2', 'R2v'), ('R2v', 'R2', 'R1')])
def test_shared_pool_real_ranking_and_zero_refinement_are_arm_order_independent(stubbed_run, arms):
    state = stubbed_run
    report = state.execute(arms)
    assert report['status'] == 'completed'
    searches = [event[1] for event in state.events if event[0] == 'search']
    assert len(searches) == 2  # one rigid cache and one shared response cache
    assert {row['selection']: row['top_k'] for row in searches} == {'rigid': 2, 'response': 50}
    assert [event[1] for event in state.events if event[0] == 'verify'] == list(range(50))
    r2, r2v = report['arms']['R2'], report['arms']['R2v']
    assert [row['rotation_index'] for row in r2['candidates']] == [0, 1]
    assert [row['selected_score'] for row in r2['candidates']] == [-100., -99.]
    assert [row['rotation_index'] for row in r2v['candidates']] == [48, 49]
    assert [row['selected_score'] for row in r2v['candidates']] == [-1000., -1000.]
    assert [row['verification']['pool_rank'] for row in r2v['candidates']] == [49, 50]
    assert len(r2v['verification']['proposals']) == 50
    assert state.verifier_options == [{'radius': .5, 'max_atom_displacement': 1.5,
                                       'min_agreement': .75, 'max_trials': 6}]
    last_verify = max(i for i, event in enumerate(state.events) if event[0] == 'verify')
    r2v_relax = [i for i, event in enumerate(state.events)
                 if event[0] == 'relax' and event[1].startswith('R2v_')]
    assert last_verify < min(r2v_relax)
    relax_calls = [event for event in state.events if event[0] == 'relax']
    assert len(relax_calls) == 6
    assert all(event[3] == relax_calls[0][3] for event in relax_calls)
    assert relax_calls[0][3]['flexible'] is True
    for arm in arms:
        for row in report['arms'][arm]['candidates']:
            assert row['amplitudes'] == [0.]
            assert row['rotation_increment'] == [.01, .02, .03]
            assert row['translation_increment'] == [.1, .2, .3]
    # Verified deformations are exported separately, never substituted for the
    # selected rigid starting coordinates or final local-refinement coordinates.
    for path, rec_xyz, _ in state.exports:
        expected = .05 if path.name.startswith('verified_') else 0.
        torch.testing.assert_close(rec_xyz, torch.full_like(rec_xyz, expected))
    assert all(row['predicted_amplitudes'] == [.05] for row in r2v['candidates'])


def test_verification_hard_failure_retains_partial_diagnostics_and_runs_other_arm(stubbed_run):
    state = stubbed_run
    state.fail_at['index'] = 1
    report = state.execute(('R2v', 'R2'))
    assert report['status'] == 'partial_failure'
    failed = report['arms']['R2v']
    assert failed['status'] == 'verification_failed'
    assert failed['error_type'] == 'FloatingPointError'
    assert len(failed['verification']['proposals']) == 1
    assert report['arms']['R2']['status'] == 'completed'
    assert len([event for event in state.events if event[0] == 'search']) == 1
    assert not any(event[0] == 'relax' and event[1].startswith('R2v_') for event in state.events)
    assert not any(path.parent.name == 'R2v' for path, _, _ in state.exports)


def test_r2_alone_does_not_expand_or_verify_response_pool(stubbed_run):
    state = stubbed_run
    report = state.execute(('R2',))
    assert report['status'] == 'completed'
    searches = [event[1] for event in state.events if event[0] == 'search']
    assert len(searches) == 1 and searches[0]['top_k'] == 2
    assert not state.verifier_options
    assert not any(event[0] == 'verify' for event in state.events)


def test_runner_passes_explicit_mode_backends_without_changing_defaults(stubbed_run):
    state = stubbed_run
    state.execute(('R1',))
    assert state.mode_options[-1] == dict(n_modes=1, fit_relative_damping=.001,
        jacobian_backend='dense', jvp_chunk_size=8, anm_backend='dense', anm_seed=0, anm_tolerance=1e-9)
    state.execute(('R1',), jacobian_backend='chunked', jvp_chunk_size=4,
                  anm_backend='iterative', anm_seed=17, anm_tolerance=1e-8)
    assert state.mode_options[-1] == dict(n_modes=1, fit_relative_damping=.001,
        jacobian_backend='chunked', jvp_chunk_size=4, anm_backend='iterative', anm_seed=17, anm_tolerance=1e-8)
