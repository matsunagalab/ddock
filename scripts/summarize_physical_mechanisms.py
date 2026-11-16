#!/usr/bin/env python3
"""Audit generation, selection and refinement using inference artifacts only.

No native/evaluation manifest, benchmark structure, DockQ or accuracy label is
read. Numerical proximity to a constraint boundary is descriptive and uses
the declared tolerance; solver KKT active sets are reported separately.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from audit_physical_preconditioners import compare_pose_keys, digest, pose_key, read_archive


def summary(values):
    values = np.asarray(list(values), dtype=float)
    if not len(values):
        return dict(count=0)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite mechanism statistic')
    return dict(count=len(values), minimum=float(values.min()), median=float(np.median(values)),
                mean=float(values.mean()), maximum=float(values.max()), total=float(values.sum()))


def active(value, bound):
    return bool(value >= bound - max(1e-9, 1e-6 * bound))


def key_changes(before, after):
    result = compare_pose_keys(before, after)
    positions = {pose_key(row): index for index, row in enumerate(before)}
    shifts = [index - positions[pose_key(row)] for index, row in enumerate(after) if pose_key(row) in positions]
    result.update(common_key_order_changes=sum(shift != 0 for shift in shifts),
                  common_key_absolute_rank_shift=summary(abs(shift) for shift in shifts))
    return result


def verification_details(row, model, radius, max_atom, ball_radius, *, flexible):
    proposal = np.asarray(row['proposed_amplitudes'])
    verified = np.asarray(row['verified_amplitudes'])
    gradient = np.asarray(row['gradient'])
    unconstrained = -np.linalg.solve(model['stiffness'], gradient)
    tangent = np.einsum('m,mnc->nc', proposal, model['basis'])
    accepted = [trial for trial in row['trials'] if trial.get('accepted') is True]
    if len(accepted) > 1:
        raise ValueError('More than one accepted verification trial')
    if flexible and not row['fallback'] and len(accepted) != 1:
        raise ValueError('Non-fallback flexible verification lacks its accepted trial')
    finite = accepted[0] if accepted else dict(ca_rmsd=0., maximum_atom_displacement=0., strain=0.)
    amplitude_norm = float(np.linalg.norm(proposal))
    atom_norm = float(np.linalg.norm(tangent, axis=-1).max())
    relevant_radius = ball_radius if row['constraint'] == 'ball' else radius
    solver = row.get('solver', {})
    return dict(pose_key=pose_key(row), fallback=bool(row['fallback']),
        accepted_scale=row.get('accepted_scale'), trial_count=len(row['trials']),
        geometry_rejected_trials=sum(not trial['geometry_feasible'] for trial in row['trials']),
        energy_rejected_trials=sum(trial['geometry_feasible'] and not trial.get('accepted', False)
                                  for trial in row['trials']),
        proposed_CA_tangent_norm=amplitude_norm, proposed_maximum_atom_tangent=atom_norm,
        mode_gradient_norm=float(np.linalg.norm(gradient)),
        unconstrained_amplitude_norm=float(np.linalg.norm(unconstrained)),
        proposal_minus_unconstrained_norm=float(np.linalg.norm(proposal - unconstrained)),
        proposed_quadratic_strain=float(.5 * proposal @ model['stiffness'] @ proposal),
        verified_CA_tangent_norm=float(np.linalg.norm(verified)),
        verified_finite_CA_rmsd=float(finite['ca_rmsd']),
        verified_finite_maximum_atom_displacement=float(finite['maximum_atom_displacement']),
        verified_actual_strain=float(finite['strain']),
        proposed_usage_CA_radius=amplitude_norm / radius,
        proposed_usage_conservative_ball=amplitude_norm / ball_radius,
        proposed_usage_atom_bound=atom_norm / max_atom,
        amplitude_bound_active=flexible and active(amplitude_norm, relevant_radius),
        linear_atom_bound_active=flexible and active(atom_norm, max_atom),
        finite_CA_bound_active=flexible and active(finite['ca_rmsd'], radius),
        finite_atom_bound_active=flexible and active(finite['maximum_atom_displacement'], max_atom),
        solver_active_constraint_count=len(solver.get('active_constraint_indices', [])),
        solver_active_atom_count=len(solver.get('active_atom_indices', [])),
        quadratic_gain=float(row['quadratic_gain']), verified_gain=float(row['verified_gain']),
        rigid_energy=float(row['rigid_energy']), verified_energy=float(row['verified_energy']))


def summarize_verification(details):
    numeric = ('trial_count', 'geometry_rejected_trials', 'energy_rejected_trials',
        'proposed_CA_tangent_norm', 'proposed_maximum_atom_tangent', 'proposed_quadratic_strain',
        'mode_gradient_norm', 'unconstrained_amplitude_norm', 'proposal_minus_unconstrained_norm',
        'verified_CA_tangent_norm', 'verified_finite_CA_rmsd', 'verified_finite_maximum_atom_displacement',
        'verified_actual_strain', 'proposed_usage_CA_radius', 'proposed_usage_conservative_ball',
        'proposed_usage_atom_bound', 'quadratic_gain', 'verified_gain', 'rigid_energy', 'verified_energy')
    flags = ('amplitude_bound_active', 'linear_atom_bound_active', 'finite_CA_bound_active', 'finite_atom_bound_active')
    return dict(count=len(details), fallback_count=sum(row['fallback'] for row in details),
        accepted_scale_counts=dict(Counter(str(row['accepted_scale']) for row in details if row['accepted_scale'] is not None)),
        boundary_active_counts={key: sum(row[key] for row in details) for key in flags},
        solver_any_active_constraint_count=sum(row['solver_active_constraint_count'] > 0 for row in details),
        solver_any_active_atom_count=sum(row['solver_active_atom_count'] > 0 for row in details),
        statistics={key: summary(row[key] for row in details) for key in numeric}, details=details)


def summarize_refinement(state, config, *, flexible):
    details = []
    for candidate in state.get('candidates', []):
        if candidate.get('export_complete') is not True or candidate.get('status') == 'exception':
            continue
        norm = float(np.linalg.norm(candidate['amplitudes']))
        translation = float(np.linalg.norm(candidate['translation_increment']))
        details.append(dict(pose_key=pose_key(candidate), selection_rank=candidate['selection_rank'],
            final_rank=candidate['final_rank'], status=candidate['status'],
            energy_evaluations=candidate['energy_evaluations'], accepted_steps=candidate['accepted_steps'],
            amplitude_norm=norm, finite_CA_rmsd=candidate['ca_rmsd'],
            finite_maximum_atom_displacement=candidate['maximum_atom_displacement'],
            actual_strain=candidate['strain'], amplitude_radius=candidate['amplitude_radius'],
            amplitude_bound_active=flexible and active(norm, candidate['amplitude_radius']),
            finite_CA_bound_active=flexible and active(candidate['ca_rmsd'], config['radius']),
            finite_atom_bound_active=flexible and active(candidate['maximum_atom_displacement'], config['max_atom']),
            translation_norm=translation, rotation_norm=float(np.linalg.norm(candidate['rotation_increment'])),
            translation_bound_active=active(translation, config['max_translation']),
            energy=candidate['energy'], refinement_gain_from_zero_start=candidate['energy'] - candidate['energy_history'][0],
            final_minus_verified_energy=candidate['energy'] - candidate['verified_energy'],
            energy_history_strictly_decreasing=all(b < a for a, b in zip(candidate['energy_history'], candidate['energy_history'][1:]))))
    numeric = ('energy_evaluations', 'accepted_steps', 'amplitude_norm', 'finite_CA_rmsd',
        'finite_maximum_atom_displacement', 'actual_strain', 'translation_norm', 'rotation_norm',
        'energy', 'refinement_gain_from_zero_start', 'final_minus_verified_energy')
    flags = ('amplitude_bound_active', 'finite_CA_bound_active', 'finite_atom_bound_active', 'translation_bound_active')
    return dict(attempted_candidates=len(state.get('candidates', [])), completed_exports=len(details),
        status_counts=dict(Counter(row.get('status', 'missing') for row in state.get('candidates', []))),
        reached_evaluation_budget_count=sum(row['energy_evaluations'] >= config['max_evaluations'] for row in details),
        max_iterations_status_count=sum(row['status'] == 'max_iterations' for row in details),
        boundary_active_counts={key: sum(row[key] for row in details) for key in flags},
        statistics={key: summary(row[key] for row in details) for key in numeric}, details=details,
        selection_to_final_order=key_changes(state.get('candidates', []),
            sorted((row for row in state.get('candidates', []) if 'final_rank' in row), key=lambda row: row['final_rank'])))


def compare_ball_atom(report, directory):
    ball, atom = (report['arms'].get(arm, {}) for arm in ('ball8', 'atom8'))
    vb, va = ({pose_key(row): row for row in state.get('verification', [])} for state in (ball, atom))
    common = sorted(vb.keys() & va.keys())
    fields = ('gradient', 'proposed_amplitudes', 'verified_amplitudes', 'rigid_energy',
              'verified_energy', 'quadratic_gain', 'verified_gain')
    comparisons = {}
    for field in fields:
        differences, equal = [], []
        for key in common:
            b, a = np.asarray(vb[key][field]), np.asarray(va[key][field])
            equal.append(b.shape == a.shape and b.dtype == a.dtype and b.tobytes() == a.tobytes())
            differences.append(float(np.max(np.abs(b - a))))
        comparisons[field] = dict(common_pose_count=len(common), bitwise_equal_count=sum(equal),
            maximum_absolute_difference=max(differences) if differences else None)
    cb, ca = ({pose_key(row): row for row in state.get('candidates', []) if row.get('export_complete') is True}
              for state in (ball, atom))
    coordinates = []
    for key in sorted(cb.keys() & ca.keys()):
        archives = [read_archive(directory / arm / f"final_{row['selection_rank']:03d}" / 'full_coordinates.npz')
                    for arm, row in (('ball8', cb[key]), ('atom8', ca[key]))]
        comparisons_xyz = {}
        for side in ('receptor', 'ligand'):
            b, a = archives[0][side], archives[1][side]
            delta = b - a
            comparisons_xyz[side] = dict(bitwise_equal=b.shape == a.shape and b.dtype == a.dtype and b.tobytes() == a.tobytes(),
                maximum_absolute_difference=float(np.abs(delta).max()),
                all_atom_rms_difference=float(np.sqrt(np.mean(np.sum(delta**2, axis=-1)))))
        coordinates.append(dict(pose_key=key, ball_selection_rank=cb[key]['selection_rank'],
            atom_selection_rank=ca[key]['selection_rank'], partners=comparisons_xyz))
    return dict(verification_fields=comparisons, common_exported_pose_count=len(coordinates),
        full_coordinates=coordinates,
        all_common_full_coordinates_bitwise_equal=bool(coordinates) and all(
            side['bitwise_equal'] for row in coordinates for side in row['partners'].values()))


def summarize_run(directory):
    report_path = directory / 'report.json'
    report = json.loads(report_path.read_text())
    if report.get('status') not in ('completed', 'partial_failure') or report.get('native_access') is not False:
        raise ValueError('Require a completed/partial-failure native-free inference report')
    config = report['configuration']
    radius, max_atom = config['radius'], config['max_atom']
    first_raw = json.loads((directory / 'first_pass_peaks.json').read_text())
    result = dict(status='completed', inference_status=report['status'], native_access=False,
        source_run=str(directory), source_report_sha256=digest(report_path),
        input_manifest_sha256=report['input_manifest_sha256'], configuration=config,
        boundary_proximity_tolerance='max(1e-9 angstrom, 1e-6 * bound), descriptive; KKT sets reported separately',
        cost=dict(wall_seconds=report['wall_seconds'],
                  shared_preparation_search_seconds=report['shared_preparation_search_seconds'],
                  physical_prior_total_evaluations=report['physical_prior_final_evaluations'],
                  process_peak_rss_bytes=report['process_peak_rss_bytes']), arms={})
    result['cost']['summed_arm_seconds'] = sum(state.get('seconds', 0.) for state in report['arms'].values())
    result['cost']['remaining_shared_and_reporting_seconds'] = (report['wall_seconds']
        - report['shared_preparation_search_seconds'] - result['cost']['summed_arm_seconds'])
    for arm, state in report['arms'].items():
        record = dict(status=state['status'], seconds=state.get('seconds'),
            requested_candidates=state['requested_candidate_count'], pool_count=state.get('pool_count', 0),
            verification_completed_count=len(state.get('verification', [])),
            counts={key: state.get(key) for key in ('exported_count', 'failure_count', 'not_attempted_count', 'missing_requested_count')},
            error=state.get('error'))
        result['arms'][arm] = record
        raw_path = directory / (f'{arm}_second_pass_peaks.json' if arm in ('contact8', 'random8') else 'first_pass_peaks.json')
        if raw_path.exists():
            raw = json.loads(raw_path.read_text())
            record['generation'] = dict(raw_file=raw_path.name, raw_sha256=digest(raw_path),
                used_raw_branches=['rigid'] if arm == 'rigid' else ['rigid', 'response'],
                raw_counts={key: len(value) for key, value in raw.items()},
                rigid_vs_response=key_changes(raw['rigid'], raw['response']),
                first_pass_rigid_vs_arm_rigid=key_changes(first_raw['rigid'], raw['rigid']),
                first_pass_response_vs_arm_response=key_changes(first_raw['response'], raw['response']))
        record['selection'] = {baseline: {field: key_changes(report['arms'][baseline].get(field, []), state.get(field, []))
            for field in ('pool', 'candidates')} for baseline in ('rigid', 'atom8') if baseline in report['arms']}
        for baseline, comparison in record['selection'].items():
            final_orders = [sorted((row for row in item.get('candidates', []) if 'final_rank' in row),
                                   key=lambda row: row['final_rank'])
                            for item in (report['arms'][baseline], state)]
            comparison['final_order'] = key_changes(*final_orders)
        model_path = directory / f'{arm}_model.npz'
        if not model_path.exists():
            record['model_artifact_missing'] = True
            continue
        model = read_archive(model_path)
        ca_basis = model['basis'][:, model['ca_indices']].reshape(len(model['basis']), -1)
        ca_metric_error = float(np.max(np.abs(ca_basis @ ca_basis.T / len(model['ca_indices'])
                                             - np.eye(len(model['basis'])))))
        coefficient = float(np.linalg.svd(model['basis'].transpose(1, 2, 0), compute_uv=False).max())
        ball_radius = min(radius, max_atom / coefficient) if coefficient else radius
        stiffness_eigenvalues = np.linalg.eigvalsh(model['stiffness'])
        record['bounds'] = dict(requested_CA_radius=radius, maximum_atom_displacement=max_atom,
            maximum_atom_coefficient=coefficient, conservative_ball_radius=ball_radius,
            physical_stiffness_eigenvalue_range=[float(stiffness_eigenvalues.min()), float(stiffness_eigenvalues.max())],
            physical_stiffness_unit='kJ/mol/angstrom^2 in CA-RMSD-normalized amplitudes',
            CA_tangent_metric_max_abs_error=ca_metric_error,
            amplitude_norm_interpretation='infinitesimal CA RMSD, subject to the measured CA metric error; finite displacement is separate')
        record['verification'] = summarize_verification([verification_details(row, model, radius,
            max_atom, ball_radius, flexible=arm != 'rigid') for row in state.get('verification', [])])
        record['refinement'] = summarize_refinement(state, config, flexible=arm != 'rigid')
        if 'adaptation' in state:
            adaptation = state['adaptation']
            cg = [row['cg'] for row in adaptation['pose_records'] if row.get('cg')]
            record['adaptation'] = dict(basis_construction_seconds=state.get('basis_construction_seconds'),
                hvp_calls=adaptation['hvp_calls'], response_solver=adaptation.get('response_solver'),
                regularization=adaptation['regularization'], actual_directions=adaptation['actual_directions'],
                cg_iterations=[row['iterations'] for row in cg],
                physical_relative_residuals=[row['relative_residual'] for row in cg])
    result['ball_atom_comparison'] = compare_ball_atom(report, directory)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = summarize_run(args.run)
    result['audit_source_sha256'] = {path.name: digest(path) for path in
        (Path(__file__), Path(__file__).with_name('audit_physical_preconditioners.py'))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(args.out), status=result['inference_status'],
        ball_atom_fullcoords_identical=result['ball_atom_comparison']['all_common_full_coordinates_bitwise_equal'],
        arms={arm: dict(status=value['status'], seconds=value['seconds'],
            fallback_count=value.get('verification', {}).get('fallback_count'),
            refinement_status_counts=value.get('refinement', {}).get('status_counts'))
            for arm, value in result['arms'].items()}), indent=2))


if __name__ == '__main__':
    main()
