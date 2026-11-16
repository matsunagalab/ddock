#!/usr/bin/env python3
"""Compare saved physical-mode subspaces and FFT proposals without native input.

Only inference report.json, common_reference.npz, *_model.npz, and
*_pass_peaks.json are read.
No structures, benchmark labels, native coordinates, or accuracy scores enter
the audit. Coordinate changes in final refinement are intentionally excluded.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        result = {name: archive[name].copy() for name in archive.files}
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError(f'Nonfinite array in {path}')
    return result


def maximum_difference(before, after):
    return float(np.max(np.abs(before - after))) if before.size else 0.


def relative_frobenius(difference, reference):
    return float(np.linalg.norm(difference) / max(np.linalg.norm(reference), np.finfo(float).tiny))


def subspace_alignment(before, after):
    """Columns are CA-normalized modes; return stable angles and Procrustes R.

    atan2 of projected principal-vector residuals avoids loss of resolution
    from arccos(sigma) when the spaces differ by only a few microradians.
    """
    if before.shape != after.shape or before.ndim != 2:
        raise ValueError('This audit requires equal-dimensional mode spaces')
    qb, _ = np.linalg.qr(before, mode='reduced')
    qa, _ = np.linalg.qr(after, mode='reduced')
    left, cosines, right = np.linalg.svd(qb.T @ qa, full_matrices=False)
    principal_after = qa @ right.T
    residual = principal_after - qb @ (qb.T @ principal_after)
    sines = np.linalg.norm(residual, axis=0)
    angles = np.arctan2(sines, np.clip(cosines, 0., 1.))
    left, _, right = np.linalg.svd(before.T @ after, full_matrices=False)
    transform = left @ right
    return transform, dict(principal_angles_degrees=np.sort(np.degrees(angles)).tolist(),
        maximum_principal_angle_degrees=float(np.degrees(angles).max()),
        projector_frobenius_distance=float(np.sqrt(2. * np.sum(np.sin(angles)**2))),
        before_metric_max_abs=float(np.max(np.abs(before.T @ before - np.eye(before.shape[1])))),
        after_metric_max_abs=float(np.max(np.abs(after.T @ after - np.eye(after.shape[1])))))


def compare_models(before_path, after_path):
    before, after = read_archive(before_path), read_archive(after_path)
    if before.keys() != after.keys():
        raise ValueError('Model archive fields differ')
    arrays = {}
    for name in before:
        if before[name].shape != after[name].shape or before[name].dtype != after[name].dtype:
            raise ValueError(f'Model array shape/dtype differs: {name}')
        arrays[name] = dict(bitwise_equal=before[name].tobytes() == after[name].tobytes(),
                           maximum_absolute_difference=maximum_difference(before[name], after[name]))
    for name in ('theta0', 'reference', 'ca_indices'):
        if not arrays[name]['bitwise_equal']:
            raise ValueError(f'Physical reference differs: {name}')
    modes, atoms, _ = before['basis'].shape
    ca = before['ca_indices']
    ub = before['basis'][:, ca].reshape(modes, -1).T / np.sqrt(len(ca))
    ua = after['basis'][:, ca].reshape(modes, -1).T / np.sqrt(len(ca))
    transform, subspace = subspace_alignment(ub, ua)
    full_before = before['basis'].reshape(modes, -1).T
    full_after = after['basis'].reshape(modes, -1).T
    mapped_stiffness = transform.T @ before['stiffness'] @ transform
    stiffness_difference = mapped_stiffness - after['stiffness']
    eigenvalues, eigenvectors = np.linalg.eigh(after['stiffness'])
    if eigenvalues.min() <= 0:
        raise ValueError('Saved physical stiffness is not positive definite')
    inverse_root = (eigenvectors * eigenvalues**-.5) @ eigenvectors.T
    relative_energy = inverse_root @ stiffness_difference @ inverse_root
    relative_energy = .5 * (relative_energy + relative_energy.T)
    return transform, dict(before_sha256=digest(before_path), after_sha256=digest(after_path),
        mode_count=modes, atom_count=atoms, ca_count=len(ca), arrays=arrays, ca_subspace=subspace,
        coordinate_mapping='before_amplitudes = procrustes_transform @ after_amplitudes',
        procrustes_transform=transform.tolist(),
        ca_tangent_relative_frobenius=relative_frobenius(ub @ transform - ua, ua),
        all_atom_tangent_relative_frobenius=relative_frobenius(full_before @ transform - full_after, full_after),
        torsion_tangent_relative_frobenius=relative_frobenius(before['angle_basis'] @ transform - after['angle_basis'], after['angle_basis']),
        stiffness=dict(energy_unit='kJ/mol', amplitude_unit='CA infinitesimal RMSD angstrom',
            before_eigenvalues=np.linalg.eigvalsh(before['stiffness']).tolist(), after_eigenvalues=eigenvalues.tolist(),
            congruence_maximum_absolute_difference=float(np.abs(stiffness_difference).max()),
            congruence_relative_frobenius=relative_frobenius(stiffness_difference, after['stiffness']),
            maximum_relative_quadratic_energy_difference=float(np.abs(np.linalg.eigvalsh(relative_energy)).max()),
            comparison_scope='quadratic physical strain after CA Procrustes mapping; finite nonlinear energies not compared'))


def pose_key(row):
    return int(row['rotation_index']), tuple(float(x) for x in row['translation'])


def compare_pose_keys(before, after):
    bk, ak = [pose_key(row) for row in before], [pose_key(row) for row in after]
    bs, ass = set(bk), set(ak)
    return dict(before_count=len(bk), after_count=len(ak), common_unique_count=len(bs & ass),
        before_unique_count=len(bs), after_unique_count=len(ass), ordered_keys_identical=bk == ak,
        unique_key_sets_identical=bs == ass, removed_keys=sorted(bs - ass), added_keys=sorted(ass - bs))


def compare_peak_file(before_path, after_path, transform):
    before, after = json.loads(before_path.read_text()), json.loads(after_path.read_text())
    result = dict(before_sha256=digest(before_path), after_sha256=digest(after_path), branches={})
    for branch in ('rigid', 'response'):
        rb, ra = before[branch], after[branch]
        record = compare_pose_keys(rb, ra)
        lookup_b, lookup_a = {pose_key(row): row for row in rb}, {pose_key(row): row for row in ra}
        common = sorted(lookup_b.keys() & lookup_a.keys())
        for field in ('coarse_score', 'coarse_rigid_energy', 'coarse_gain'):
            differences = [abs(lookup_b[key][field] - lookup_a[key][field]) for key in common
                           if field in lookup_b[key] and field in lookup_a[key]]
            if differences:
                record[field + '_maximum_absolute_difference'] = max(differences)
        amplitude_differences = [np.linalg.norm(np.asarray(lookup_b[key]['coarse_amplitudes']) -
            transform @ np.asarray(lookup_a[key]['coarse_amplitudes'])) for key in common
            if 'coarse_amplitudes' in lookup_b[key] and 'coarse_amplitudes' in lookup_a[key]]
        if amplitude_differences:
            record['mapped_amplitude_maximum_norm_difference'] = float(max(amplitude_differences))
        result['branches'][branch] = record
    return result


def audit(before_directory, after_directory):
    reports = [json.loads((path / 'report.json').read_text()) for path in (before_directory, after_directory)]
    for report in reports:
        if report.get('status') != 'completed' or report.get('native_access') is not False:
            raise ValueError('Both reports must be completed native-free inference runs')
    if reports[0]['input_manifest_sha256'] != reports[1]['input_manifest_sha256']:
        raise ValueError('Input manifest hashes differ')
    result = dict(status='completed', native_access=False, scope='saved inference models and candidate spaces only',
        before=str(before_directory), after=str(after_directory),
        input_manifest_sha256=reports[0]['input_manifest_sha256'],
        report_sha256=[digest(path / 'report.json') for path in (before_directory, after_directory)],
        configuration_changes={key: [reports[0]['configuration'].get(key), reports[1]['configuration'].get(key)]
            for key in sorted(reports[0]['configuration'].keys() | reports[1]['configuration'].keys())
            if reports[0]['configuration'].get(key) != reports[1]['configuration'].get(key)},
        models={}, peaks={}, arm_pose_keys={})
    common = [read_archive(path / 'common_reference.npz') for path in (before_directory, after_directory)]
    if common[0].keys() != common[1].keys():
        raise ValueError('Common reference archive fields differ')
    result['invariant_checks'] = dict(
        report_fields={field: reports[0][field] == reports[1][field]
            for field in ('potential_configuration', 'typed_parameters', 'grid', 'quaternions')},
        common_reference_arrays={field: common[0][field].dtype == common[1][field].dtype
            and common[0][field].shape == common[1][field].shape
            and common[0][field].tobytes() == common[1][field].tobytes() for field in common[0]},
        physical_source_hashes={name: reports[0]['source_sha256']['src/zdock/inducedfit/' + name]
            == reports[1]['source_sha256']['src/zdock/inducedfit/' + name]
            for name in ('physical_prior.py', 'typed_potential.py', 'typed_fft.py', 'chemistry.py',
                         'atomistic_score.py', 'hydrogen_kinematics.py', 'kinematics.py')})
    transforms = {}
    for arm in ('ball8', 'atom8', 'contact8', 'random8'):
        transforms[arm], result['models'][arm] = compare_models(
            before_directory / f'{arm}_model.npz', after_directory / f'{arm}_model.npz')
    for filename, arm in [('first_pass_peaks.json', 'atom8'),
                          ('contact8_second_pass_peaks.json', 'contact8'),
                          ('random8_second_pass_peaks.json', 'random8')]:
        result['peaks'][filename] = compare_peak_file(before_directory / filename, after_directory / filename, transforms[arm])
    for arm in ('rigid', 'ball8', 'atom8', 'contact8', 'random8'):
        states = [report['arms'][arm] for report in reports]
        result['arm_pose_keys'][arm] = {field: compare_pose_keys(states[0][field], states[1][field])
                                       for field in ('pool', 'candidates')}
    for arm in ('contact8', 'random8'):
        adaptations = [report['arms'][arm]['adaptation'] for report in reports]
        result['models'][arm]['adaptation'] = dict(
            hvp_calls=[value['hvp_calls'] for value in adaptations],
            cg_iterations=[[pose['cg']['iterations'] for pose in value['pose_records'] if pose.get('cg')]
                           for value in adaptations],
            cg_physical_relative_residuals=[[pose['cg']['relative_residual'] for pose in value['pose_records'] if pose.get('cg')]
                                           for value in adaptations])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.before, args.after)
    result['audit_source_sha256'] = digest(__file__)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(args.out), contact_subspace=result['models']['contact8']['ca_subspace'],
        contact_stiffness=result['models']['contact8']['stiffness'],
        random_all_arrays_bitwise_equal=all(value['bitwise_equal'] for value in result['models']['random8']['arrays'].values()),
        all_raw_pose_key_sets_identical=all(branch['unique_key_sets_identical']
            for record in result['peaks'].values() for branch in record['branches'].values())), indent=2))


if __name__ == '__main__':
    main()
