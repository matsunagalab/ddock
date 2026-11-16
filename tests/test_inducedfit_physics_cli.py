"""CLI failure-accounting checks with tiny inputs and no FFT/OpenMM work."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from zdock.inducedfit import physics_pipeline
from zdock.inducedfit.typed_potential import TypedParameters
from zdock.inducedfit.types import GridSpec


@pytest.fixture
def cli():
    script = Path(__file__).resolve().parents[1] / 'scripts/run_physics_induced_fit_pair.py'
    spec = importlib.util.spec_from_file_location('physics_cli_under_test', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tiny_run(cli, monkeypatch, tmp_path):
    dtype = torch.float64
    xyz = torch.zeros((1, 3), dtype=dtype)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}')
    output = tmp_path / 'output'
    output.mkdir()
    args = SimpleNamespace(manifest=manifest, out=output, seed=3,
        prepared_prior=None, anchor_k=10., preparation_iterations=1,
        preparation_evaluations=2, stationarity_tolerance=.001,
        spacing=2., max_atom=.2, max_translation=4., max_grid_points=1000,
        rotations=1, max_cache_bytes=100000, radius=.5, pool_size=2,
        arms=['rigid'], top_k=2, iterations=1, max_evaluations=4,
        adaptive_preconditioner='anchor', refinement_preconditioner='physical')
    inputs = {'receptor': object(), 'ligand': object()}
    proteins = {key: SimpleNamespace(monomer=SimpleNamespace(xyz=xyz.clone()),
        provenance={}, heavy_indices=torch.tensor([0]), side=key) for key in inputs}
    prior = SimpleNamespace(reference=xyz, theta0=torch.zeros(8, dtype=dtype),
        prepared_kinematics=object(), diagnostics={'status': 'prepared'},
        _context=SimpleNamespace(evaluations=0), closed=False)
    prior.close = lambda: setattr(prior, 'closed', True)
    modes = SimpleNamespace(reference=xyz, angle_basis=torch.eye(8, dtype=dtype),
        basis=torch.zeros((8, 1, 3), dtype=dtype), stiffness=torch.eye(8, dtype=dtype),
        ca_indices=torch.tensor([0]), n_modes=8,
        diagnostics={'baseline_projected_gradient': [0.] * 8})
    raw = {'rigid': [dict(rotation_index=0, translation=[0., y, 0.],
                         coarse_score=y, coarse_rigid_energy=y) for y in (4., 8.)],
           'response': []}
    monkeypatch.setattr(cli, 'load_pair_manifest', lambda path: ({'input_kind': 'toy_apo'}, inputs))
    monkeypatch.setattr(cli, 'prepare_amber_charges', lambda item, seed:
        proteins[next(side for side, value in inputs.items() if value is item)])
    monkeypatch.setattr(cli, 'centered_protein', lambda p: (p, torch.zeros(3, dtype=dtype)))
    monkeypatch.setattr(cli, 'extract_typed_parameters', lambda p: TypedParameters(
        torch.zeros(1, dtype=dtype), torch.zeros(1, dtype=dtype),
        torch.tensor([1. if p.side == 'receptor' else -1.], dtype=dtype)))
    monkeypatch.setattr(cli, 'prepare_physical_prior', lambda *a, **k: prior)
    monkeypatch.setattr(cli, 'build_torsion_modes', lambda *a, **k:
        SimpleNamespace(angle_basis=torch.eye(8, dtype=dtype), diagnostics={}))
    monkeypatch.setattr(cli, 'build_physical_modes', lambda *a, **k: modes)
    monkeypatch.setattr(cli, 'typed_pair_grid', lambda *a, **k: GridSpec((3, 3, 3), 2., (-2.,) * 3))
    monkeypatch.setattr(cli, 'prepare_typed_cache', lambda *a, **k: SimpleNamespace(diagnostics={}))
    monkeypatch.setattr(cli, 'typed_search', lambda *a, **k: deepcopy(raw))

    def export(directory, *unused):
        directory.mkdir(parents=True, exist_ok=False)
        paths = {name: directory / f'{name}.pdb' for name in ('receptor', 'ligand', 'complex')}
        for path in paths.values():
            path.write_text('END\n')
        return {name: str(path) for name, path in paths.items()}

    def refine(pose, *unused, **kwargs):
        ligand = xyz + xyz.new_tensor(pose['translation'])
        return dict(status='max_iterations', energy=-pose['translation'][1],
                    strain=0., amplitudes=[0.] * 8), xyz, ligand

    monkeypatch.setattr(cli, 'export_heavy_pair', export)
    monkeypatch.setattr(cli, 'refine_physical_pose', refine)
    report = {'arms': {'rigid': {'status': 'pending', 'requested_candidate_count': 2}}}
    snapshots = []
    save = lambda: snapshots.append(deepcopy(report))
    return args, report, save, prior, snapshots


def test_archive_write_failure_with_energy_is_excluded_from_ranks(cli, tiny_run, monkeypatch):
    args, report, save, prior, snapshots = tiny_run
    original_savez = cli.np.savez

    def fail_first_archive(path, *args, **kwargs):
        if Path(path).name == 'full_coordinates.npz' and Path(path).parent.name == 'final_001':
            raise OSError('injected full coordinate archive failure')
        return original_savez(path, *args, **kwargs)

    monkeypatch.setattr(cli.np, 'savez', fail_first_archive)
    cli.run(args, report, save)
    arm = report['arms']['rigid']
    failed, successful = arm['candidates']
    assert 'energy' in failed  # This was the path that previously leaked into ranks.
    assert failed['status'] == 'exception' and not failed['export_complete']
    assert 'final_rank' not in failed
    assert successful['export_complete'] and successful['final_rank'] == 1
    assert arm['exported_count'] == arm['failure_count'] == arm['missing_requested_count'] == 1
    assert arm['not_attempted_count'] == 0
    assert arm['requested_candidate_count'] == 2
    assert arm['status'] == report['status'] == 'partial_failure'
    assert prior.closed
    assert any(s['arms']['rigid'].get('candidates', [{}])[0].get('status') == 'exception'
               for s in snapshots if s['arms']['rigid'].get('candidates'))


def test_verification_failure_keeps_pose_and_requested_denominator(cli, tiny_run, monkeypatch):
    args, report, save, prior, snapshots = tiny_run
    actual_pair_energy = physics_pipeline.direct_pair_energy

    def fail_second_pose(rx, lx, *args, **kwargs):
        if float(lx[0, 1]) == 8.:
            raise RuntimeError('injected verification failure')
        return actual_pair_energy(rx, lx, *args, **kwargs)

    monkeypatch.setattr(physics_pipeline, 'direct_pair_energy', fail_second_pose)
    cli.run(args, report, save)
    arm = report['arms']['rigid']
    assert arm['status'] == 'failed' and report['status'] == 'partial_failure'
    assert len(arm['verification']) == 1 and arm['pool_count'] == 2
    assert arm['failed_verification']['pool_index'] == 2
    assert arm['failed_verification']['pose']['translation'] == [0., 8., 0.]
    assert arm['error_type'] == 'RuntimeError'
    assert arm['requested_candidate_count'] == 2
    assert arm['missing_requested_count'] == arm['not_attempted_count'] == 2
    assert arm['exported_count'] == arm['failure_count'] == 0
    assert 'current_verification' not in arm
    assert prior.closed
    assert any(s['arms']['rigid'].get('current_verification', {}).get('pool_index') == 2
               for s in snapshots)


def test_reranking_discards_stale_ranks_and_never_shrinks_requested_denominator(cli):
    state = {'candidates': [
        dict(selection_rank=1, energy=-4., export_complete=False, final_rank=1, status='exception'),
        dict(selection_rank=2, energy=-2., export_complete=True, final_rank=2, status='max_iterations'),
        dict(selection_rank=3, energy=-3., export_complete=True, status='stationary'),
        dict(selection_rank=4, energy=float('nan'), export_complete=True, status='max_iterations'),
    ]}
    assert not cli.finalize_arm_counts(state, 5)
    assert state['exported_count'] == 2 and state['failure_count'] == 2
    assert state['not_attempted_count'] == 1 and state['missing_requested_count'] == 3
    assert 'final_rank' not in state['candidates'][0]
    assert state['candidates'][1]['final_rank'] == 2
    assert state['candidates'][2]['final_rank'] == 1
    assert 'final_rank' not in state['candidates'][3]


def test_cli_rejects_native_manifest_field_before_chemical_preparation(cli, tmp_path, monkeypatch):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(format_version=1, system_id='toy', input_kind='apo',
        split='pilot', receptor={}, ligand={}, native='native_must_not_be_opened.pdb')))
    monkeypatch.setattr(cli, 'prepare_amber_charges', lambda *a, **k: pytest.fail('chemistry must not run'))
    with pytest.raises(ValueError, match='unrecognized fields'):
        cli.run(SimpleNamespace(manifest=manifest), {}, lambda: None)


@pytest.mark.parametrize('failure_stage', ['manifest', 'prior_cache'])
def test_shared_failure_retains_denominator_for_every_unattempted_arm(cli, tmp_path, monkeypatch, failure_stage):
    manifest = tmp_path / 'invalid_manifest.json'
    manifest.write_text('{}')
    output = tmp_path / 'result'
    monkeypatch.setattr(cli.sys, 'argv', ['physics_cli', '--manifest', str(manifest),
        '--out', str(output), '--arms', 'rigid', 'atom8', '--top-k', '2', '--pool-size', '2'])
    relative = 'scripts/run_physics_induced_fit_pair.py'
    snapshot_hashes = {relative: cli.digest(Path(cli.__file__))}
    monkeypatch.setattr(cli, 'source_hashes', lambda: dict(snapshot_hashes))
    if failure_stage == 'prior_cache':
        def fail_cached_prior(args, report, save):
            report['stage'] = 'physical_apo_preparation'
            raise ValueError('injected cached-prior hash validation failure')
        monkeypatch.setattr(cli, 'run', fail_cached_prior)
    assert cli.main() == 1
    report = json.loads((output / 'report.json').read_text())
    assert report['status'] == 'exception'
    for state in report['arms'].values():
        assert state['status'] == 'not_attempted'
        assert state['requested_candidate_count'] == 2
        assert state['missing_requested_count'] == state['not_attempted_count'] == 2
        assert state['failure_count'] == state['exported_count'] == 0
