"""Lightweight saved-array checks; never run a mode calculation or subprocess."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/benchmark_inducedfit_modes.py'
SPEC = importlib.util.spec_from_file_location('benchmark_inducedfit_modes_checks', SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def write_arrays(path, *, synthetic=False, gain_offset=0.):
    path.parent.mkdir(parents=True, exist_ok=True)
    basis = np.zeros((2, 2, 3))
    basis[0, 0, 0] = basis[1, 1, 1] = 1.
    arrays = {'basis': basis, 'stiffness': np.diag([1., 2.])}
    if synthetic:
        arrays['eigenvalues'] = np.array([1., 2.])
    else:
        arrays.update(ca_basis=basis, probe_coordinates=np.zeros((4, 2, 3)),
                      probe_energies=np.zeros(4), probe_gains=np.arange(4.) + gain_offset)
    np.savez(path, **arrays)


def metadata(variant, *, source='a' * 64, input_hash='b' * 64):
    return {'status': 'completed', 'variant': variant, 'basis_metric_denominator': 1,
            'source_sha256': {'src/model.py': source}, 'input_xyz_sha256': input_hash,
            'mode_seconds': 1., 'preparation_seconds': .2, 'wall_seconds': 1.3,
            'peak_rss_bytes': 1024}


def trial(repeat, variant):
    return {'repeat': repeat, 'variant': variant, 'status': 'completed', 'returncode': 0,
            'mode_seconds': 1., 'preparation_seconds': .2, 'wall_seconds': 1.3,
            'peak_rss_bytes': 1024}


def test_gain_difference_is_a_required_failure_even_when_geometry_and_strain_agree(tmp_path):
    reference, candidate = tmp_path / 'a.npz', tmp_path / 'b.npz'
    write_arrays(reference)
    write_arrays(candidate, gain_offset=.01)
    metrics = benchmark.compare(reference, candidate, 1)
    assert metrics['probe_maximum_atom_error'] == 0
    assert metrics['probe_maximum_strain_error'] == 0
    assert metrics['probe_maximum_quadratic_gain_error'] == pytest.approx(.01)
    assert not benchmark.comparison_passes(metrics, synthetic=False)
    metrics['probe_maximum_quadratic_gain_error'] = 1e-8
    assert benchmark.comparison_passes(metrics, synthetic=False)
    del metrics['probe_maximum_quadratic_gain_error']
    assert not benchmark.comparison_passes(metrics, synthetic=False)


def test_input_hash_difference_fails_global_integrity():
    references = {(0, 'dense'): (Path('.'), metadata('dense')),
                  (0, 'iterative'): (Path('.'), metadata('iterative', input_hash='c' * 64))}
    gate = benchmark.integrity_gate(references)
    assert not gate['passed'] and not gate['input_consistent_across_all_trials']
    assert gate['source_consistent_across_all_trials']


def test_source_drift_between_repeats_fails_even_when_each_repeat_matches():
    references = {}
    for repeat, source in ((0, 'a' * 64), (1, 'c' * 64)):
        for variant in ('dense', 'iterative'):
            references[repeat, variant] = (Path('.'), metadata(variant, source=source))
    gate = benchmark.integrity_gate(references)
    assert not gate['passed'] and not gate['source_consistent_across_all_trials']
    assert gate['input_consistent_across_all_trials']


@pytest.mark.parametrize('missing', ['source_sha256', 'input_xyz_sha256'])
def test_missing_fingerprint_is_an_explicit_integrity_failure(missing):
    record = metadata('dense')
    del record[missing]
    result = benchmark.integrity_gate({(0, 'dense'): (Path('.'), record)})
    assert not result['passed']
    assert len(result['errors']) == 1


def test_comparison_exception_is_saved_as_failed_not_left_running(tmp_path):
    references = {(0, v): (tmp_path / v, metadata(v)) for v in ('dense', 'iterative')}
    result = {'status': 'running', 'trials': [trial(0, v) for v in ('dense', 'iterative')]}
    benchmark.aggregate_results(result, references, repeats=1, variants=['dense', 'iterative'])
    assert result['status'] == 'failed'
    assert result['comparisons'][0]['passed'] is False
    assert result['comparisons'][0]['error_type'] == 'FileNotFoundError'
    benchmark.save(tmp_path / 'comparison.json', result)
    assert json.loads((tmp_path / 'comparison.json').read_text())['status'] == 'failed'


def test_nonfinite_arrays_are_rejected_before_json_serialization(tmp_path):
    reference, candidate = tmp_path / 'a.npz', tmp_path / 'b.npz'
    write_arrays(reference, synthetic=True)
    write_arrays(candidate, synthetic=True)
    with np.load(candidate) as data:
        arrays = {name: data[name].copy() for name in data.files}
    arrays['basis'][0, 0, 0] = np.nan
    np.savez(candidate, **arrays)
    with pytest.raises(ValueError, match='nonfinite'):
        benchmark.compare(reference, candidate, 1)


def saved_run(source):
    source.mkdir()
    rows = [trial(0, variant) for variant in ('dense', 'iterative')]
    for row in rows:
        directory = source / f"repeat_0_{row['variant']}"
        write_arrays(directory / 'arrays.npz', synthetic=True)
        (directory / 'report.json').write_text(json.dumps(metadata(row['variant'])))
    (source / 'comparison.json').write_text(json.dumps({'status': 'passed', 'trials': rows,
                                                       'timing_policy': 'saved synthetic fixture'}))
    return rows


def test_reaggregation_only_reads_frozen_artifacts_and_retains_measurements(tmp_path, monkeypatch):
    source, out = tmp_path / 'original', tmp_path / 'rechecked'
    saved_run(source)
    before = {path.relative_to(source): path.read_bytes() for path in source.rglob('*') if path.is_file()}
    def forbidden(*args, **kwargs):
        raise AssertionError('reaggregation must not launch workers')
    monkeypatch.setattr(benchmark.subprocess, 'Popen', forbidden)
    assert benchmark.reaggregate_saved(source, out) == 0
    report = json.loads((out / 'comparison.json').read_text())
    assert report['status'] == 'passed'
    assert report['integrity']['passed']
    assert report['medians']['dense']['mode_seconds'] == 1.
    assert len(report['saved_artifact_sha256']) == 4
    assert len(report['aggregation_source_sha256']) == 64
    after = {path.relative_to(source): path.read_bytes() for path in source.rglob('*') if path.is_file()}
    assert before == after


def test_reaggregation_does_not_rehabilitate_failed_worker_trial(tmp_path):
    source, out = tmp_path / 'original', tmp_path / 'rechecked'
    rows = saved_run(source)
    rows[1].update(status='failed', returncode=-9, error='prior timeout')
    (source / 'comparison.json').write_text(json.dumps({'status': 'failed', 'trials': rows}))
    assert benchmark.reaggregate_saved(source, out) == 1
    result = json.loads((out / 'comparison.json').read_text())
    assert result['status'] == 'failed'
    assert result['trials'][1]['status'] == 'failed'
    assert 'original worker trial was unsuccessful' in result['trials'][1]['error']
