#!/usr/bin/env python3
"""Compare existing ANM computations in isolated, interleaved CPU processes.

No native structures or accuracy labels. Large arrays/logs are written under
the requested output, not committed. RSS includes preparation and dependencies.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import statistics
import subprocess
import sys
import time


COMPARISON_LIMITS = {
    'subspace_residual': 1e-6,
    'metric_error': 1e-6,
    'stiffness_relative_error': 1e-6,
    'probe_maximum_atom_error': 1e-5,
    'probe_maximum_strain_error': 1e-6,
    'probe_maximum_quadratic_gain_error': 1e-6,
    'eigenvalue_relative_error': 1e-6,
}


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def worker(args):
    import numpy as np
    import scipy
    import torch
    from zdock.inducedfit.modes import build_anm, build_torsion_modes
    from zdock.inducedfit.anm_iterative import build_anm_iterative
    from zdock.inducedfit.response import QuadraticResponse

    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    names = ('modes', 'anm_iterative', 'kinematics', 'hydrogen_kinematics',
             'chemistry', 'pdb_input', 'pipeline', 'types', 'fields', 'response')
    paths = [root / f'src/zdock/inducedfit/{name}.py' for name in names] + [Path(__file__).resolve(), root / 'src/zdock/geom.py']
    report = {'status': 'running', 'command': sys.argv, 'variant': args.variant,
              'synthetic_nodes': args.synthetic_nodes, 'modes': args.modes,
              'platform': platform.platform(), 'python': platform.python_version(),
              'torch': torch.__version__, 'scipy': scipy.__version__,
              'device': 'cpu', 'dtype': 'float64', 'threads': 1,
              'native_access': False, 'seed': 0, 'source_sha256': {
                  str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    start = time.monotonic()
    try:
        if args.synthetic_nodes:
            n = args.synthetic_nodes
            side = math.ceil(n ** (1 / 3))
            lattice = torch.cartesian_prod(*[torch.arange(side, dtype=torch.float64)] * 3)[:n] * 3.8
            generator = torch.Generator().manual_seed(101)
            xyz = lattice + .3 * (torch.rand(lattice.shape, generator=generator, dtype=torch.float64) - .5)
            report['input_xyz_sha256'] = hashlib.sha256(xyz.numpy().tobytes()).hexdigest()
            report['preparation_seconds'] = time.monotonic() - start
            started = time.monotonic()
            model = (build_anm_iterative(xyz, n_modes=args.modes, seed=0, tolerance=1e-9)
                     if args.variant == 'iterative' else build_anm(xyz))
            report['mode_seconds'] = time.monotonic() - started
            arrays = {'basis': model.modes[:args.modes].numpy(),
                      'stiffness': np.diag(model.eigenvalues[:args.modes].numpy()),
                      'eigenvalues': model.eigenvalues[:args.modes].numpy()}
            report['diagnostics'] = model.diagnostics
            report['n_atoms'] = n
            report['basis_metric_denominator'] = 1
        else:
            from zdock.inducedfit.chemistry import prepare_amber_charges
            from zdock.inducedfit.pipeline import centered_protein, load_pair_manifest
            from zdock.inducedfit.hydrogen_kinematics import HydrogenatedKinematics
            manifest, inputs = load_pair_manifest(args.manifest)
            charged, _ = centered_protein(prepare_amber_charges(inputs['receptor'], pH=7., seed=0))
            kin = HydrogenatedKinematics.from_prepared(charged)
            report.update(input=manifest, input_manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                          n_atoms=len(kin.reference), n_torsions=kin.n_torsions, n_ca=len(kin.ca_indices),
                          chemistry=charged.provenance)
            report['preparation_seconds'] = time.monotonic() - start
            started = time.monotonic()
            model = build_torsion_modes(kin, n_modes=args.modes,
                jacobian_backend='dense' if args.variant == 'dense' else 'chunked',
                jvp_chunk_size=8, anm_backend='iterative' if args.variant == 'iterative' else 'dense',
                anm_seed=0, anm_tolerance=1e-9)
            report['mode_seconds'] = time.monotonic() - started
            report['diagnostics'] = model.diagnostics
            generator = torch.Generator().manual_seed(29)
            forces = torch.randn((4, *model.reference.shape), generator=generator, dtype=torch.float64)
            gradients = torch.einsum('pnc,mnc->pm', forces, model.basis)
            solved = QuadraticResponse(model.stiffness).solve(gradients, radius=.05)
            probes = torch.stack([model.coordinates(a) for a in solved.amplitudes])
            energies = torch.stack([model.deformation_energy(a) for a in solved.amplitudes])
            arrays = {'basis': model.basis.numpy(), 'ca_basis': model.basis[:, model.ca_indices].numpy(),
                      'angle_basis': model.angle_basis.numpy(), 'stiffness': model.stiffness.numpy(),
                      'probe_coordinates': probes.numpy(), 'probe_energies': energies.numpy(),
                      'probe_gains': solved.gain.numpy()}
            report['basis_metric_denominator'] = len(kin.ca_indices)
        np.savez_compressed(args.out / 'arrays.npz', **arrays)
        report['array_bytes'] = {k: value.nbytes for k, value in arrays.items()}
        report['status'] = 'completed'
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
    report['wall_seconds'] = time.monotonic() - start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['peak_rss_bytes'] = rss if sys.platform == 'darwin' else rss * 1024
    save(args.out / 'report.json', report)
    return 0 if report['status'] == 'completed' else 1


def compare(reference, candidate, denominator):
    import numpy as np
    with np.load(reference) as a, np.load(candidate) as b:
        if not math.isfinite(denominator) or denominator <= 0:
            raise ValueError('basis metric denominator must be positive and finite')
        if a['basis'].shape != b['basis'].shape:
            raise ValueError('full basis shapes differ')
        for label, arrays in (('reference', a), ('candidate', b)):
            for name in arrays.files:
                if not np.isfinite(arrays[name]).all():
                    raise ValueError(f'{label} array {name} contains nonfinite values')
        metric = 'ca_basis' if 'ca_basis' in a else 'basis'
        left, right = a[metric].reshape(len(a['basis']), -1), b[metric].reshape(len(b['basis']), -1)
        if left.shape != right.shape or len(left) == 0:
            raise ValueError('metric bases must have the same nonempty shape')
        if a['stiffness'].shape != (len(left), len(left)) or b['stiffness'].shape != a['stiffness'].shape:
            raise ValueError('stiffness shapes differ from the mode count')
        if np.linalg.norm(right) == 0 or np.linalg.norm(a['stiffness']) == 0:
            raise ValueError('zero basis or reference stiffness cannot define relative errors')
        rotation = left @ right.T / denominator
        result = {'subspace_residual': float(np.linalg.norm(right - rotation.T @ left) / np.linalg.norm(right)),
                  'metric_error': float(np.max(np.abs(rotation.T @ rotation - np.eye(len(rotation))))),
                  'stiffness_relative_error': float(np.linalg.norm(b['stiffness'] - rotation.T @ a['stiffness'] @ rotation)
                                                     / np.linalg.norm(a['stiffness']))}
        if 'probe_coordinates' in a:
            result.update(probe_maximum_atom_error=float(np.linalg.norm(a['probe_coordinates'] - b['probe_coordinates'], axis=-1).max()),
                          probe_maximum_strain_error=float(np.max(np.abs(a['probe_energies'] - b['probe_energies']))),
                          probe_maximum_quadratic_gain_error=float(np.max(np.abs(a['probe_gains'] - b['probe_gains']))))
        else:
            if a['eigenvalues'].shape != b['eigenvalues'].shape or np.any(a['eigenvalues'] <= 0):
                raise ValueError('eigenvalue arrays must match and reference values must be positive')
            result['eigenvalue_relative_error'] = float(np.max(np.abs((a['eigenvalues'] - b['eigenvalues']) / a['eigenvalues'])))
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError('comparison produced nonfinite errors')
    return result


def comparison_passes(metrics, *, synthetic):
    """Every applicable metric is mandatory, including the response gain."""
    required = ['subspace_residual', 'metric_error', 'stiffness_relative_error']
    required += (['eigenvalue_relative_error'] if synthetic else
                 ['probe_maximum_atom_error', 'probe_maximum_strain_error', 'probe_maximum_quadratic_gain_error'])
    return all(name in metrics and math.isfinite(metrics[name])
               and 0 <= metrics[name] <= COMPARISON_LIMITS[name] for name in required)


def _valid_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(char in '0123456789abcdef' for char in value)


def integrity_gate(references):
    """Require the same recorded input and worker sources across ALL trials.

    The current aggregation script is deliberately not compared with worker
    source hashes: reaggregation uses a newer checker on frozen measurements.
    """
    source_fingerprints, input_fingerprints, errors = set(), set(), []
    for (repeat, variant), (_, record) in sorted(references.items()):
        try:
            sources = record.get('source_sha256')
            if not isinstance(sources, dict) or not sources or not all(
                    isinstance(name, str) and _valid_sha256(digest) for name, digest in sources.items()):
                raise ValueError('missing or invalid worker source SHA256 map')
            source_fingerprints.add(hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest())
            keys = [key for key in ('input_manifest_sha256', 'input_xyz_sha256') if key in record]
            if len(keys) != 1 or not _valid_sha256(record[keys[0]]):
                raise ValueError('exactly one valid input manifest/coordinate SHA256 is required')
            input_fingerprints.add((keys[0], record[keys[0]]))
        except Exception as error:
            errors.append({'repeat': repeat, 'variant': variant,
                           'error_type': type(error).__name__, 'error': str(error)})
    same_sources = len(source_fingerprints) == 1
    same_inputs = len(input_fingerprints) == 1
    return {'passed': bool(references) and not errors and same_sources and same_inputs,
            'source_consistent_across_all_trials': same_sources,
            'input_consistent_across_all_trials': same_inputs,
            'source_fingerprints': sorted(source_fingerprints),
            'input_fingerprints': [{'kind': key, 'sha256': digest} for key, digest in sorted(input_fingerprints)],
            'errors': errors}


def aggregate_results(result, references, *, repeats, variants):
    """Analyze completed saved arrays; comparison exceptions become failures."""
    result['comparison_limits'] = dict(COMPARISON_LIMITS)
    result['integrity'] = integrity_gate(references)
    result['comparisons'] = []
    synthetic = variants == ['dense', 'iterative']
    for repeat in range(repeats):
        if (repeat, 'dense') not in references:
            continue
        ref_dir, ref = references[repeat, 'dense']
        for variant in variants[1:]:
            if (repeat, variant) not in references:
                continue
            directory, report = references[repeat, variant]
            row = {'repeat': repeat, 'variant': variant}
            try:
                metrics = compare(ref_dir / 'arrays.npz', directory / 'arrays.npz', ref['basis_metric_denominator'])
                passed = comparison_passes(metrics, synthetic=synthetic)
                row.update(passed=passed and result['integrity']['passed'], metrics_passed=passed, **metrics)
            except Exception as error:
                row.update(passed=False, metrics_passed=False,
                           error_type=type(error).__name__, error=str(error))
            result['comparisons'].append(row)
    result['medians'] = {}
    for variant in variants:
        rows = [row for row in result['trials'] if row['variant'] == variant and row['status'] == 'completed']
        if rows:
            result['medians'][variant] = {key: statistics.median(row[key] for row in rows)
                for key in ('mode_seconds', 'preparation_seconds', 'wall_seconds', 'peak_rss_bytes')}
    result['status'] = 'passed' if (
        len(result['trials']) == repeats * len(variants)
        and all(row['status'] == 'completed' for row in result['trials'])
        and result['integrity']['passed']
        and len(result['comparisons']) == repeats * (len(variants) - 1)
        and all(row['passed'] for row in result['comparisons'])) else 'failed'
    return result


def write_summary(path, result):
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = ['repeat', 'variant', 'status', 'mode_seconds', 'preparation_seconds', 'wall_seconds', 'peak_rss_bytes', 'error']
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n', extrasaction='ignore')
            writer.writeheader()
            writer.writerows(result['trials'])


def reaggregate_saved(source, out, summary=None):
    """Read reports/NPZ only; preserve the original benchmark and its timings."""
    source = source.resolve()
    original_path = source / 'comparison.json'
    original = json.loads(original_path.read_text())
    trials = original['trials']
    repeats = 1 + max(row['repeat'] for row in trials)
    variants = ['dense', 'chunked', 'iterative'] if any(row['variant'] == 'chunked' for row in trials) else ['dense', 'iterative']
    out.mkdir(parents=True, exist_ok=False)
    result = {'status': 'reaggregating', 'command': sys.argv, 'trials': [], 'comparisons': [],
        'scope': 'saved reports/arrays only; no new measurements or accuracy/native evaluation',
        'timing_policy': original.get('timing_policy'),
        'measurement_source': str(source),
        'original_comparison_sha256': hashlib.sha256(original_path.read_bytes()).hexdigest(),
        'aggregation_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'saved_artifact_sha256': {}}
    save(out / 'comparison.json', result)
    references = {}
    for trial in trials:
        row = {key: trial[key] for key in ('repeat', 'variant')}
        directory = source / f"repeat_{row['repeat']}_{row['variant']}"
        result['trials'].append(row)
        try:
            path = directory / 'report.json'
            record = json.loads(path.read_text())
            result['saved_artifact_sha256'][str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
            array_path = directory / 'arrays.npz'
            if array_path.is_file():
                result['saved_artifact_sha256'][str(array_path.relative_to(source))] = hashlib.sha256(array_path.read_bytes()).hexdigest()
            row.update(status=record['status'], **{key: record[key] for key in
                ('mode_seconds', 'preparation_seconds', 'wall_seconds', 'peak_rss_bytes', 'error') if key in record})
            # Do not rehabilitate a timed-out/nonzero-exit measurement merely
            # because a completed child report happens to have survived.
            if trial['status'] != 'completed' or trial.get('returncode', 1) != 0:
                raise ValueError(f"original worker trial was unsuccessful: {trial.get('error', trial['status'])}")
            if record['status'] != 'completed' or record['variant'] != row['variant']:
                raise ValueError('saved worker report is incomplete or has a mismatched variant')
            references[row['repeat'], row['variant']] = (directory, record)
        except Exception as error:
            row.update(status='failed', error_type=type(error).__name__, error=str(error))
        save(out / 'comparison.json', result)
    aggregate_results(result, references, repeats=repeats, variants=variants)
    save(out / 'comparison.json', result)
    write_summary(summary, result)
    print(json.dumps({'status': result['status'], 'out': str(out), 'new_measurements': False}), flush=True)
    return 0 if result['status'] == 'passed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--synthetic-nodes', type=int, default=0)
    parser.add_argument('--modes', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=180.)
    parser.add_argument('--summary', type=Path)
    parser.add_argument('--reaggregate', type=Path,
                        help='recheck saved report/NPZ files from this prior benchmark; --out must be new')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--variant', choices=['dense', 'chunked', 'iterative'], default='dense', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.modes < 1 or args.repeats < 1 or args.synthetic_nodes < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('counts and timeout must be positive')
    if args.summary and args.summary.exists():
        parser.error('summary already exists')
    if args.reaggregate is not None:
        if args.worker or args.manifest is not None or args.synthetic_nodes:
            parser.error('--reaggregate does not accept workers or new input structures')
        return reaggregate_saved(args.reaggregate, args.out, args.summary)
    if (args.manifest is None) == (args.synthetic_nodes == 0):
        parser.error('provide exactly one of --manifest or --synthetic-nodes')
    if args.worker:
        return worker(args)
    args.out.mkdir(parents=True, exist_ok=False)
    variants = ['dense', 'iterative'] if args.synthetic_nodes else ['dense', 'chunked', 'iterative']
    result = {'status': 'running', 'command': sys.argv, 'trials': [], 'comparisons': [],
              'scope': 'same-model computation; no accuracy/native evaluation',
              'timing_policy': 'isolated sequential subprocesses, interleaved variants; process RSS includes preparation'}
    save(args.out / 'comparison.json', result)
    references = {}
    for repeat in range(args.repeats):
        order = variants[repeat % len(variants):] + variants[:repeat % len(variants)]
        for variant in order:
            directory = args.out / f'repeat_{repeat}_{variant}'
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--variant', variant,
                       '--out', str(directory), '--modes', str(args.modes)]
            command += ['--synthetic-nodes', str(args.synthetic_nodes)] if args.synthetic_nodes else ['--manifest', str(args.manifest)]
            row = {'repeat': repeat, 'variant': variant, 'command': command}
            result['trials'].append(row)
            save(args.out / 'comparison.json', result)
            try:
                with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                      start_new_session=True) as process:
                    try:
                        stdout, stderr = process.communicate(timeout=args.timeout)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        stdout, stderr = process.communicate()
                        (args.out / f'{directory.name}.stdout.log').write_text(stdout)
                        (args.out / f'{directory.name}.stderr.log').write_text(stderr)
                        raise TimeoutError(f'worker exceeded {args.timeout} seconds')
                (args.out / f'{directory.name}.stdout.log').write_text(stdout)
                (args.out / f'{directory.name}.stderr.log').write_text(stderr)
                record = json.loads((directory / 'report.json').read_text())
                row.update(returncode=process.returncode, status=record['status'],
                           **{k: record[k] for k in ('mode_seconds', 'preparation_seconds', 'wall_seconds', 'peak_rss_bytes', 'error') if k in record})
                if process.returncode != 0 or record['status'] != 'completed':
                    raise RuntimeError(f'worker failed: {record.get("error", record["status"])}')
                references[repeat, variant] = (directory, record)
            except Exception as error:
                row.update(status='failed', error_type=type(error).__name__, error=str(error))
            save(args.out / 'comparison.json', result)
            print(json.dumps(row), flush=True)
    aggregate_results(result, references, repeats=args.repeats, variants=variants)
    save(args.out / 'comparison.json', result)
    write_summary(args.summary, result)
    print(json.dumps({'status': result['status'], 'medians': result['medians']}), flush=True)
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
