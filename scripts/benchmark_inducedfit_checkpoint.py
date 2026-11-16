#!/usr/bin/env python3
"""Interleave identical synthetic smoke runs with field checkpointing off/on.

Reports all trials, checks hashes/energies/export equivalence, and records CPU
wall time and process peak RSS. This is not protein accuracy or GPU throughput.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import signal
import subprocess
import sys


def fingerprint_outputs(report, directory):
    result = {}
    for arm, values in report['arms'].items():
        for row in values['candidates']:
            for kind, name in row['files'].items():
                result[f'{arm}/{row["selection_rank"]}/{kind}'] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--repeats', type=int, default=3)
    args = ap.parse_args()
    if args.repeats < 1:
        ap.error('repeats must be positive')
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    result = {'experiment': 'synthetic_local_field_checkpoint_cpu_v1', 'command': sys.argv,
              'scope': 'identical synthetic41atom partners; CPU smoke; no accuracy comparison',
              'trials': [], 'status': 'running'}
    reference = None
    def save():
        (args.out / 'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    save()
    for repeat in range(args.repeats):
        # Alternate which variant goes first, so warm filesystem effects are
        # not assigned exclusively to the checkpoint variant.
        for enabled in ([False, True] if repeat % 2 == 0 else [True, False]):
            directory = args.out / f'repeat_{repeat}_checkpoint_{int(enabled)}'
            command = [sys.executable, str(root / 'scripts/verify_amber_pair.py'),
                       '--out', str(directory), '--modes', '1', '--timeout', '120']
            if enabled:
                command.append('--checkpoint-fields')
            row = {'repeat': repeat, 'checkpoint_fields': enabled, 'command': command}
            result['trials'].append(row)
            save()
            try:
                # Own a separate process group, so a timed-out wrapper cannot
                # leave a docking grandchild running into the next trial.
                with subprocess.Popen(command, text=True, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, start_new_session=True) as child:
                    try:
                        row['stdout'], row['stderr'] = child.communicate(timeout=180)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        row['stdout'], row['stderr'] = child.communicate()
                        row['returncode'] = child.returncode
                        raise TimeoutError('smoke process group exceeded 180 seconds')
                    row['returncode'] = child.returncode
                    if child.returncode:
                        raise RuntimeError(f'smoke exited {child.returncode}')
                r = json.loads((directory / 'run/report.json').read_text())
                v = json.loads((directory / 'verification.json').read_text())
                if r['status'] != 'completed' or v['status'] != 'passed':
                    raise RuntimeError(f'incomplete smoke: {r["status"]}, {v["status"]}')
                observed = {
                    'input_sha256': {s: r['input'][s]['sha256'] for s in ('receptor', 'ligand')},
                    'source_sha256': r['source_sha256'],
                    'outputs': fingerprint_outputs(r, directory / 'run'),
                    'energies': {a: [c['energy'] for c in x['candidates']] for a, x in r['arms'].items()},
                    'amplitudes': {a: [c['amplitudes'] for c in x['candidates']] for a, x in r['arms'].items()},
                }
                if reference is None:
                    reference = observed
                error = max(abs(e - e0) for a, values in observed['energies'].items()
                            for e, e0 in zip(values, reference['energies'][a], strict=True))
                amplitude_error = max(abs(x - y) for a, values in observed['amplitudes'].items()
                                      for xs, ys in zip(values, reference['amplitudes'][a], strict=True)
                                      for x, y in zip(xs, ys, strict=True))
                equivalent = (error < 1e-10 and amplitude_error < 1e-10
                              and all(observed[k] == reference[k] for k in ('input_sha256', 'source_sha256', 'outputs')))
                row.update(status='passed' if equivalent else 'equivalence_failed',
                           maximum_energy_error=error, maximum_amplitude_error=amplitude_error,
                           wall_seconds=r['wall_seconds'], peak_rss_bytes=r['process_peak_rss_bytes'], **observed)
            except Exception as error:
                row.update(status='exception', error_type=type(error).__name__, error=str(error))
            save()
            print(json.dumps({k: row[k] for k in ('repeat', 'checkpoint_fields', 'status', 'wall_seconds', 'peak_rss_bytes', 'error') if k in row}), flush=True)
    result['status'] = 'passed' if all(r['status'] == 'passed' for r in result['trials']) else 'failed'
    if result['status'] == 'passed':
        result['medians'] = {}
        for enabled in (False, True):
            rows = [r for r in result['trials'] if r['checkpoint_fields'] == enabled]
            result['medians'][str(int(enabled))] = {
                key: statistics.median(r[key] for r in rows) for key in ('wall_seconds', 'peak_rss_bytes')
            }
    save()
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
