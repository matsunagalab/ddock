#!/usr/bin/env python3
"""Run reference-free Amber-charge induced-fit smoke/pilot on a PDB pair.

Output energies are effective smooth-field units, NOT Amber kcal/mol.
R0: rigid selection + rigid refinement; R1: rigid selection + flexible
refinement; R2: response selection + the same flexible refinement.
R2v: nonlinear verification of an oversampled response pool before TopK.
Native structures are neither accepted nor opened by this process.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import torch

from zdock.inducedfit.chemistry import prepare_amber_charges
from zdock.inducedfit.fields import FieldParameters
from zdock.inducedfit.hydrogen_kinematics import HydrogenatedKinematics
from zdock.inducedfit.modes import build_torsion_modes
from zdock.inducedfit.pipeline import centered_protein, export_heavy_pair, load_pair_manifest, pair_grid, rotation_sample
from zdock.inducedfit.relax import relax_pose
from zdock.inducedfit.search import prepare_response_receptor, search_response
from zdock.inducedfit.verified_response import NonlinearResponseVerifier
from zdock.geom import rotate


def verification_record(verified, pool_rank):
    original = verified.original_candidate
    return {
        'pool_rank': pool_rank, 'rotation_index': verified.rotation_index,
        'translation': verified.translation.tolist(),
        'original_score': original.score, 'original_predicted_gain': original.predicted_gain,
        'original_amplitudes': original.amplitudes.tolist(),
        'score': verified.score, 'rigid_energy': verified.rigid_energy,
        'verified_gain': verified.verified_gain, 'predicted_gain': verified.predicted_gain,
        'amplitudes': verified.amplitudes.tolist(), 'strain': verified.strain,
        'accepted_scale': verified.accepted_scale, 'fallback': verified.fallback,
        'rigid_energy_error': verified.rigid_energy_error, 'trials': list(verified.trials),
    }


def verify_and_rank_candidates(pool, quaternions, verifier, top_k, record):
    """Verify the entire pool before truncation; no labels or warm starts."""
    verified = []
    for pool_rank, candidate in enumerate(pool, 1):
        result = verifier.verify(candidate, quaternions[candidate.rotation_index])
        verified.append(result)
        record(verification_record(result, pool_rank))
    # Stable ties retain the original, deterministic search order.
    return sorted(verified, key=lambda candidate: candidate.score)[:top_k]


def run(args, report, save):
    preparation_started = time.monotonic()
    manifest, inputs = load_pair_manifest(args.manifest)
    report['input'] = manifest
    report['input_manifest_sha256'] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    proteins, centers = {}, {}
    report['preparation'] = {}
    for side in ('receptor', 'ligand'):
        report['stage'] = f'prepare_{side}'
        save()
        charged = prepare_amber_charges(inputs[side], pH=args.ph, seed=args.seed)
        proteins[side], centers[side] = centered_protein(charged)
        report['preparation'][side] = {
            'chemistry': charged.provenance, 'pdb': inputs[side].provenance,
            'heavy_atoms': len(charged.heavy_indices), 'all_atoms': charged.monomer.n_atoms,
            'center': centers[side].tolist(), 'heavy_to_all_atom_indices': charged.heavy_indices.tolist(),
        }
        save()
    receptor, ligand = proteins['receptor'].monomer, proteins['ligand'].monomer
    report['stage'] = 'modes'
    save()
    model = build_torsion_modes(HydrogenatedKinematics.from_prepared(proteins['receptor']),
                                n_modes=args.modes, fit_relative_damping=args.fit_damping,
                                jacobian_backend=getattr(args, 'jacobian_backend', 'dense'),
                                jvp_chunk_size=getattr(args, 'jvp_chunk_size', 8),
                                anm_backend=getattr(args, 'anm_backend', 'dense'),
                                anm_seed=getattr(args, 'anm_seed', 0),
                                anm_tolerance=getattr(args, 'anm_tolerance', 1e-9))
    report['mode_diagnostics'] = model.diagnostics
    parameters = FieldParameters(surface_width=.5, repulsion_weight=3., charge_weight=1.,
                                 atom_chunk_size=32, point_chunk_size=2048, backend=args.field_backend)
    grid = pair_grid(receptor, ligand, parameters, spacing=args.spacing,
                     max_atom_displacement=args.max_atom_displacement,
                     max_translation=args.max_translation)
    correlation_shape = tuple(2 * n - 1 for n in grid.shape)
    spectral_points = correlation_shape[0] * correlation_shape[1] * (correlation_shape[2] // 2 + 1)
    report.update(parameters=asdict(parameters), grid=asdict(grid), stiffness=model.stiffness.tolist(),
                  correlation_shape=correlation_shape,
                  cached_receptor_spectra_bytes=(1 + model.n_modes) * 3 * spectral_points * 16,
                  memory_note='spectra bytes exclude fields, autograd, temporary FFT/maps and solver; grid cap is not a memory limit')
    if args.prepare_only:
        report.update(status='prepared', stage='complete')
        return
    if grid.n_points > args.max_grid_points:
        raise ValueError(f'grid has {grid.n_points} points, exceeds explicit limit {args.max_grid_points}')
    report['stage'] = 'receptor_spectra'
    save()
    cached = prepare_response_receptor(receptor, grid, parameters, model.basis, model.stiffness)
    radius = cached.bounded_radius(args.radius, args.max_atom_displacement)
    report.update(effective_radius=radius,
                  maximum_atom_motion_per_unit_amplitude=cached.maximum_atom_motion_per_unit_amplitude)
    quaternions = rotation_sample(args.rotations, args.seed, dtype=receptor.xyz.dtype)
    report['quaternions'] = quaternions.tolist()
    report['shared_preparation_seconds'] = time.monotonic() - preparation_started
    report['search_seconds_by_selection'] = {}
    candidates_by_selection = {}
    for arm in args.arms:
        selection = 'response' if arm in ('R2', 'R2v') else 'rigid'
        report['stage'] = f'{arm}_search'
        report['arms'][arm]['status'] = 'searching'
        save()
        started = time.monotonic()
        try:
            if selection not in candidates_by_selection:
                candidates_by_selection[selection] = search_response(
                    cached, ligand, quaternions, radius=radius,
                    top_k=(args.response_pool_size if selection == 'response' and 'R2v' in args.arms else args.top_k),
                    peaks_per_rotation=args.peaks_per_rotation, selection=selection,
                )
                report['search_seconds_by_selection'][selection] = time.monotonic() - started
        except Exception as error:
            report['arms'][arm].update(status='search_failed', error_type=type(error).__name__, error=str(error))
            save()
            continue
        pool = candidates_by_selection[selection]
        search_seconds_this_arm = time.monotonic() - started
        candidates = pool[:args.top_k]
        verification = None
        if arm == 'R2v':
            report['stage'] = 'R2v_nonlinear_verification'
            verification = {'pool_count': len(pool), 'requested_pool_count': args.response_pool_size,
                            'min_agreement': .75, 'max_trials': 6, 'proposals': [],
                            'refinement_initial_amplitudes': 'zero; verified deformation used only for selection'}
            report['arms'][arm]['verification'] = verification
            save()
            verification_start = time.monotonic()
            def record_verification(row):
                verification['proposals'].append(row)
                save()
            try:
                verifier = NonlinearResponseVerifier(receptor, ligand, model, grid, parameters,
                    radius=radius, max_atom_displacement=args.max_atom_displacement,
                    min_agreement=.75, max_trials=6)
                candidates = verify_and_rank_candidates(pool, quaternions, verifier, args.top_k, record_verification)
            except Exception as error:
                report['arms'][arm].update(status='verification_failed', error_type=type(error).__name__, error=str(error))
                verification['seconds'] = time.monotonic() - verification_start
                save()
                continue
            verification['seconds'] = time.monotonic() - verification_start
            verification['fallback_count'] = sum(p['fallback'] for p in verification['proposals'])
            del verifier
        result = {'status': 'refining', 'selection': selection, 'flexible': arm != 'R0', 'candidates': [],
                  'search_seconds_this_arm': search_seconds_this_arm,
                  'candidate_count': len(candidates), 'requested_candidate_count': args.top_k,
                  'candidate_shortfall': max(0, args.top_k - len(candidates))}
        if verification is not None:
            result['verification'] = verification
        report['arms'][arm] = result
        if not candidates:
            result.update(status='no_candidates', exported_count=0, failure_count=0)
            save()
            continue
        for index, candidate in enumerate(candidates, 1):
            report['stage'] = f'{arm}_refine_{index}'
            save()
            start = time.monotonic()
            row = {'selection_rank': index, 'rotation_index': candidate.rotation_index,
                   'translation': candidate.translation.tolist(), 'selected_score': candidate.score,
                   'selected_rigid_energy': candidate.rigid_energy,
                   'predicted_gain': candidate.predicted_gain, 'predicted_amplitudes': candidate.amplitudes.tolist()}
            result['candidates'].append(row)
            if arm == 'R2v':
                row['verification'] = verification_record(candidate, next(
                    i for i, c in enumerate(pool, 1) if c is candidate.original_candidate))
            try:
                initial_ligand = torch.stack(rotate(*ligand.xyz.unbind(-1),
                                                    quaternions[candidate.rotation_index]), dim=-1) + candidate.translation
                initial_paths = export_heavy_pair(args.out / arm / f'selected_{index:03d}',
                    inputs['receptor'], inputs['ligand'],
                    receptor.xyz[proteins['receptor'].heavy_indices],
                    initial_ligand[proteins['ligand'].heavy_indices], centers['receptor'])
                row['initial_files'] = {k: str(Path(v).relative_to(args.out)) for k, v in initial_paths.items()}
                if arm == 'R2v':
                    verified_paths = export_heavy_pair(args.out / arm / f'verified_{index:03d}',
                        inputs['receptor'], inputs['ligand'],
                        model.coordinates(candidate.amplitudes)[proteins['receptor'].heavy_indices],
                        initial_ligand[proteins['ligand'].heavy_indices], centers['receptor'])
                    row['verified_files'] = {k: str(Path(v).relative_to(args.out)) for k, v in verified_paths.items()}
                refined = relax_pose(receptor, ligand, model, grid, parameters,
                                     quaternions[candidate.rotation_index], candidate.translation,
                                     radius=radius, flexible=arm != 'R0',
                                     max_iterations=args.iterations, max_evaluations=args.max_evaluations,
                                     max_translation=args.max_translation, max_rotation=.15,
                                     max_atom_displacement=args.max_atom_displacement,
                                     checkpoint_fields=args.checkpoint_fields)
                paths = export_heavy_pair(args.out / arm / f'candidate_{index:03d}',
                    inputs['receptor'], inputs['ligand'],
                    refined.receptor_xyz[proteins['receptor'].heavy_indices],
                    refined.ligand_xyz[proteins['ligand'].heavy_indices], centers['receptor'])
                row.update(status=refined.status, energy=refined.energy, strain=refined.strain,
                           ca_rmsd=refined.ca_rmsd, maximum_atom_displacement=refined.maximum_atom_displacement,
                           amplitudes=refined.amplitudes.tolist(), accepted_steps=refined.accepted_steps,
                           rotation_increment=refined.rotation_increment.tolist(),
                           translation_increment=refined.translation_increment.tolist(),
                           energy_evaluations=refined.energy_evaluations, energy_history=refined.energy_history,
                           files={k: str(Path(v).relative_to(args.out)) for k, v in paths.items()})
            except Exception as error:
                row.update(status='exception', error_type=type(error).__name__, error=str(error))
            row['refinement_seconds'] = time.monotonic() - start
            save()
        ranked = sorted((r for r in result['candidates'] if 'energy' in r), key=lambda r: (r['energy'], r['selection_rank']))
        for rank, row in enumerate(ranked, 1):
            row['final_rank'] = rank
        result['exported_count'] = len(ranked)
        result['failure_count'] = len(candidates) - len(ranked)
        result['status'] = 'completed' if result['failure_count'] == 0 and result['candidate_shortfall'] == 0 else 'partial_failure'
        save()
    report['status'] = 'completed' if all(a['status'] == 'completed' for a in report['arms'].values()) else 'partial_failure'
    report['stage'] = 'complete'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True, help='new directory; existing outputs are never overwritten')
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--checkpoint-fields', action='store_true',
                    help='recompute field chunks in local reverse-mode gradients to reduce retained memory')
    ap.add_argument('--field-backend', choices=['dense', 'sparse'], default='dense',
                    help='sparse evaluates the same compact fields only on atom-local grid stencils')
    ap.add_argument('--arms', choices=['R0', 'R1', 'R2', 'R2v'], nargs='+', default=['R1', 'R2'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ph', type=float, default=7.)
    ap.add_argument('--modes', type=int, default=2)
    ap.add_argument('--fit-damping', type=float, default=.001)
    ap.add_argument('--jacobian-backend', choices=['dense', 'chunked'], default='dense',
                    help='chunked retains only the CA Jacobian and adopted full-atom directions')
    ap.add_argument('--jvp-chunk-size', type=int, default=8)
    ap.add_argument('--anm-backend', choices=['dense', 'iterative'], default='dense',
                    help='iterative extracts only required low modes without a dense Hessian; CPU float64')
    ap.add_argument('--anm-seed', type=int, default=0)
    ap.add_argument('--anm-tolerance', type=float, default=1e-9)
    ap.add_argument('--rotations', type=int, default=4)
    ap.add_argument('--top-k', type=int, default=1)
    ap.add_argument('--response-pool-size', type=int, default=50,
                    help='R2v verifies this many response proposals before TopK; R2 uses the same raw pool when both run')
    ap.add_argument('--peaks-per-rotation', type=int, default=4)
    ap.add_argument('--spacing', type=float, default=2.)
    ap.add_argument('--radius', type=float, default=.1)
    ap.add_argument('--max-atom-displacement', type=float, default=.5)
    ap.add_argument('--max-translation', type=float, default=.5)
    ap.add_argument('--iterations', type=int, default=2)
    ap.add_argument('--max-evaluations', type=int, default=20)
    ap.add_argument('--max-grid-points', type=int, default=300000)
    ap.add_argument('--threads', type=int, default=1)
    args = ap.parse_args()
    if min(args.modes, args.rotations, args.top_k, args.peaks_per_rotation, args.iterations,
           args.max_evaluations, args.max_grid_points, args.threads, args.jvp_chunk_size) < 1:
        ap.error('counts must be positive')
    if len(set(args.arms)) != len(args.arms):
        ap.error('arms must be unique')
    if args.response_pool_size < 1 or ('R2v' in args.arms and args.response_pool_size < args.top_k):
        ap.error('response pool must be positive and at least top-k when using R2v')
    if not 0 <= args.seed < 2**32:
        ap.error('seed must be in [0, 2**32)')
    if args.anm_seed < 0 or not math.isfinite(args.anm_tolerance) or not 0 < args.anm_tolerance < 1:
        ap.error('ANM seed must be nonnegative and tolerance finite in (0, 1)')
    for name in ('ph', 'fit_damping', 'spacing', 'radius', 'max_atom_displacement', 'max_translation'):
        if not math.isfinite(getattr(args, name)):
            ap.error(f'{name} must be finite')
    if not 0 <= args.ph <= 14 or min(args.fit_damping, args.radius, args.max_translation) < 0:
        ap.error('pH must be in [0, 14], damping/radius/translation must be nonnegative')
    if min(args.spacing, args.max_atom_displacement) <= 0:
        ap.error('spacing and maximum atom displacement must be positive')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / 'src/zdock/inducedfit').glob('*.py')) + [Path(__file__).resolve(), root / 'src/zdock/geom.py']
    report = {
        'experiment': 'apo_pair_amber_charge_integration_v1', 'status': 'running',
        'arms': {a: {'status': 'not_attempted'} for a in args.arms},
        'scope': 'effective smooth-field energy, Amber charges, CA-spring strain; no full intramolecular force field',
        'accuracy_evaluation': 'not performed; native structures excluded from inference',
        'configuration': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'command': sys.argv, 'platform': platform.platform(), 'python': platform.python_version(),
        'torch': torch.__version__, 'device': 'cpu', 'dtype': 'float64',
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'export': 'heavy atoms only for evaluation; both partners in receptor source frame; combined chains R/L',
    }
    started = time.monotonic()
    def save():
        report['wall_seconds'] = time.monotonic() - started
        usage = resource.getrusage(resource.RUSAGE_SELF)
        report['process_peak_rss_bytes'] = usage.ru_maxrss if sys.platform == 'darwin' else usage.ru_maxrss * 1024
        tmp = args.out / 'report.tmp'
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        tmp.replace(args.out / 'report.json')
    save()
    try:
        run(args, report, save)
    except Exception as error:
        report.update(status='exception', error_type=type(error).__name__, error=str(error))
    save()
    print(json.dumps({k: report[k] for k in ('status', 'stage', 'error', 'wall_seconds') if k in report}, indent=2))
    print(f'Saved {args.out / "report.json"}')
    return 0 if report['status'] in ('completed', 'prepared') else 1


if __name__ == '__main__':
    raise SystemExit(main())
