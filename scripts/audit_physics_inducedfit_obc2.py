#!/usr/bin/env python3
"""Secondary OBC2 endpoint audit of every requested frozen docking candidate.

No native structures or accuracy labels are accepted. Source hybrid ranks are
preserved; this diagnostic never reranks or selects candidates. The reference
for both monomer strains is the run's common prepared apo coordinates. Full
OBC2 complex-minus-monomers differs from the hybrid docking Hamiltonian and is
not a binding free energy. No minimization or coordinate changes are performed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

from zdock.inducedfit.atomistic_score import AtomisticScorer, PROTOCOL
from zdock.inducedfit.chemistry import prepare_amber_charges
from zdock.inducedfit.hydrogen_kinematics import HydrogenatedKinematics
from zdock.inducedfit.pipeline import centered_protein, load_pair_manifest


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def save(path, record):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def replace_reference(protein, coordinates, archive_sha256):
    """Replace full and heavy coordinates together, retaining particle order."""
    xyz = torch.from_numpy(np.asarray(coordinates, dtype=np.float64).copy())
    if xyz.shape != protein.monomer.xyz.shape or not bool(torch.isfinite(xyz).all()):
        raise ValueError('Invalid common full-atom reference shape or coordinates')
    indices = protein.heavy_indices
    heavy_keys = tuple(protein.monomer.atom_keys[int(i)] for i in indices)
    if heavy_keys != protein.heavy_monomer.atom_keys:
        raise ValueError('Full/heavy atom order differs in the reconstructed chemistry')
    moved = replace(protein, monomer=replace(protein.monomer, xyz=xyz),
        heavy_monomer=replace(protein.heavy_monomer, xyz=xyz[indices].clone()),
        provenance={**protein.provenance, 'secondary_audit_reference':dict(
            definition='common prepared apo reference from inference artifact',
            archive_sha256=archive_sha256, full_xyz_sha256=array_digest(xyz.numpy()),
            heavy_xyz_sha256=array_digest(xyz[indices].numpy()))})
    if not torch.equal(moved.monomer.xyz[indices], moved.heavy_monomer.xyz):
        raise ValueError('Common reference replacement did not preserve full/heavy consistency')
    return moved


def summarize(rows):
    scored = [r for r in rows if r['audit_status'] == 'scored']
    summary = dict(requested_count=len(rows), scored_count=len(scored),
        missing_count=sum(r['audit_status']=='missing' for r in rows),
        failed_count=sum(r['audit_status']=='failed' for r in rows))
    for key, getter in {
        'interaction_total':lambda r:r['obc2']['interaction']['total'],
        'endpoint_total':lambda r:r['obc2']['endpoint_total'],
        'receptor_strain_total':lambda r:r['obc2']['receptor_strain']['total'],
        'ligand_strain_total':lambda r:r['obc2']['ligand_strain']['total'],
        'severe_lj_pairs':lambda r:r['obc2']['clashes']['severe_lj_pairs'],
        'severe_heavy_lj_pairs':lambda r:r['obc2']['clashes']['severe_heavy_lj_pairs'],
    }.items():
        values = np.asarray([getter(row) for row in scored])
        summary[key] = dict(count=len(values), minimum=float(values.min()),
            median=float(np.median(values)), maximum=float(values.max())) if len(values) else dict(count=0)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-report', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.run_report.read_text())
    if source.get('status') not in {'completed', 'partial_failure', 'exception'}:
        parser.error('Inference must have reached a terminal status before this audit starts')
    if source.get('native_access') is not False:
        parser.error('Expected an inference-only source report with native_access=false')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    repository = Path(__file__).resolve().parents[1]
    run_dir = args.run_report.resolve().parent
    configuration = source['configuration']
    manifest = Path(configuration['manifest'])
    if not manifest.is_absolute():
        manifest = repository/manifest
    common_path = run_dir/'common_reference.npz'
    arm_names = list(configuration['arms'])
    requested = int(configuration['top_k'])
    rows, structure_errors = [], []
    for arm in arm_names:
        state = source['arms'].get(arm, {})
        candidates = state.get('candidates', [])
        by_rank = {}
        for candidate in candidates:
            rank = candidate.get('selection_rank')
            if type(rank) is not int or rank < 1 or rank > requested or rank in by_rank:
                structure_errors.append(f'{arm}: invalid or repeated selection_rank {rank}')
            else:
                by_rank[rank] = candidate
        for rank in range(1, requested+1):
            candidate = by_rank.get(rank)
            relative = f'{arm}/final_{rank:03d}/full_coordinates.npz'
            path = run_dir/relative
            row = dict(arm=arm, selection_rank=rank, source_arm_status=state.get('status'),
                source_final_rank=None if candidate is None else candidate.get('final_rank'),
                source_status=None if candidate is None else candidate.get('status'),
                source_hybrid_energy=None if candidate is None else candidate.get('energy'),
                source_hybrid_prior_strain=None if candidate is None else candidate.get('strain'),
                full_coordinates_file=relative, audit_status='pending')
            if candidate is None:
                row.update(audit_status='missing', reason='requested candidate was not produced')
            elif not candidate.get('export_complete') or candidate.get('status') == 'exception':
                row.update(audit_status='missing', reason='source candidate did not complete export',
                    source_error=candidate.get('error'))
            elif not path.is_file():
                row.update(audit_status='missing', reason='full_coordinates.npz missing')
            else:
                row['full_coordinates_sha256'] = digest(path)
            rows.append(row)
    registration = dict(command=sys.argv, script_sha256=digest(__file__),
        run_report_sha256=digest(args.run_report), source_run_status=source['status'],
        source_inference_hashes=source['source_sha256'], input_manifest=str(manifest),
        input_manifest_sha256=digest(manifest) if manifest.is_file() else None,
        common_reference_sha256=digest(common_path) if common_path.is_file() else None,
        candidate_files={r['full_coordinates_file']:r.get('full_coordinates_sha256') for r in rows},
        requested_count=len(rows), requested_per_arm=requested, arm_order=arm_names,
        seed=configuration['seed'], torch_threads=1, device='CPU Reference', dtype='float64',
        native_access=False, ranking=False, selection=False, minimization=False,
        input_pose_policy='unrounded full_coordinates.npz; original all-H atom order and covalent topology',
        reference_policy='common_reference receptor_xyz and ligand_xyz; both heavy/full monomers replaced together',
        protocol=PROTOCOL,
        protocol_override='PROTOCOL ranking field does not apply: this secondary audit never reranks',
        objective_difference='Full OBC2 complex-minus-monomer interaction and FF-only monomer strain; excludes Cartesian anchor and differs from the screened-Coulomb hybrid inference energy',
        denominator_policy='retain every requested slot, including source failures, missing files and scoring errors')
    save(args.out/'registration.json', registration)
    report = dict(status='running', registration=registration, rows=rows, source_structure_errors=structure_errors)
    save(args.out/'report.json', report)
    started = time.monotonic()
    scorer = None
    try:
        if structure_errors:
            raise ValueError('Source candidate rank structure is invalid')
        if registration['input_manifest_sha256'] != source['input_manifest_sha256']:
            raise ValueError('Input manifest differs from inference')
        for relative, expected in source['source_sha256'].items():
            path = (repository/relative).resolve()
            if not path.is_relative_to(repository) or digest(path) != expected:
                raise ValueError(f'Inference source hash mismatch: {relative}')
        _, inputs = load_pair_manifest(manifest)
        proteins = {side:centered_protein(prepare_amber_charges(inputs[side], pH=7.,
            seed=configuration['seed']))[0] for side in ('receptor','ligand')}
        if digest(common_path) != registration['common_reference_sha256']:
            raise ValueError('Common reference changed after audit registration')
        with np.load(common_path, allow_pickle=False) as archive:
            common = {name:archive[name].copy() for name in archive.files}
        input_receptor = proteins['receptor'].monomer.xyz.clone()
        reconstructed = HydrogenatedKinematics.from_prepared(proteins['receptor']).coordinates(
            torch.from_numpy(common['theta0']))
        if not torch.allclose(reconstructed, torch.from_numpy(common['receptor_xyz']), atol=1e-10, rtol=0.):
            raise ValueError('Common receptor reference does not reconstruct from original atom order/torsions')
        if not torch.equal(proteins['ligand'].monomer.xyz, torch.from_numpy(common['ligand_xyz'])):
            raise ValueError('Common ligand reference differs from reconstructed centered all-H apo ligand')
        proteins = {side:replace_reference(proteins[side], common[f'{side}_xyz'],
            registration['common_reference_sha256']) for side in ('receptor','ligand')}
        scorer = AtomisticScorer.from_proteins(proteins['receptor'], proteins['ligand'])
        for side, parameters in zip(('receptor','ligand'), scorer._particle_parameters):
            for name, column, multiplier in [('charges',0,1.),('sigma',1,10.),('epsilon',2,1.)]:
                if not np.allclose(common[f'{side}_{name}'], parameters[:,column]*multiplier, atol=1e-12, rtol=0.):
                    raise ValueError(f'{side} {name} differs between inference and OBC2 reconstruction')
        expected_receptor = source['physical_prior']['final_components']
        expected_ff_energy = expected_receptor['total']-expected_receptor['anchor']
        reference_error = scorer.reference_energies[0].total-expected_ff_energy
        if abs(reference_error) > 1e-7:
            raise ValueError('Receptor OBC2 baseline differs from common physical preparation')
        report.update(scorer_provenance=scorer.provenance,
            reference_validation=dict(full_heavy_coordinates_consistent=True,
                receptor_torsions_reconstructed=True, ligand_apo_exact=True,
                forcefield_parameters_match=True, receptor_reference_energy_error=reference_error,
                monomer_energies={side:asdict(value) for side,value in
                    zip(('receptor','ligand'),scorer.reference_energies)}))
        anchor_k = configuration['anchor_k']
        reference_anchor = .5*anchor_k*float((proteins['receptor'].monomer.xyz-input_receptor).square().sum())
        for row in rows:
            if row['audit_status'] != 'pending':
                continue
            candidate_started = time.monotonic()
            try:
                path = run_dir/row['full_coordinates_file']
                if digest(path) != row['full_coordinates_sha256']:
                    raise ValueError('Frozen candidate coordinates changed after registration')
                with np.load(path, allow_pickle=False) as archive:
                    rec = torch.from_numpy(archive['receptor'].copy())
                    lig = torch.from_numpy(archive['ligand'].copy())
                for side, xyz in [('receptor',rec),('ligand',lig)]:
                    if xyz.dtype != torch.float64 or xyz.shape != proteins[side].monomer.xyz.shape or not bool(torch.isfinite(xyz).all()):
                        raise ValueError(f'Invalid full-atom {side} coordinates')
                score = scorer.score(rec, lig)
                anchor_change = .5*anchor_k*float((rec-input_receptor).square().sum())-reference_anchor
                reconstructed_strain = score.receptor_strain.total+anchor_change
                strain_error = reconstructed_strain-row['source_hybrid_prior_strain']
                row.update(audit_status='scored', obc2=score.to_dict(),
                    anchor_change_excluded_from_endpoint=anchor_change,
                    source_hybrid_prior_strain_reconstruction_error=strain_error,
                    warnings=[])
                if abs(strain_error) > 1e-5:
                    row['warnings'].append('Hybrid prior strain does not match FF strain plus anchor change within 1e-5 kJ/mol')
                if abs(score.ligand_strain.total) > 1e-6:
                    row['warnings'].append('Rigid ligand OBC2 strain exceeds 1e-6 kJ/mol numerical tolerance')
            except Exception as error:
                row.update(audit_status='failed', error_type=type(error).__name__, error=str(error))
            row['seconds'] = time.monotonic()-candidate_started
            save(args.out/'report.json', report)
        report['status'] = 'completed' if all(r['audit_status']=='scored' for r in rows) else 'partial_failure'
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        for row in rows:
            if row['audit_status']=='pending':
                row.update(audit_status='failed', reason='audit setup failed', error=str(error))
    finally:
        if scorer is not None:
            scorer.close()
        report['summary'] = summarize(rows)
        report['per_arm_summary'] = {arm:summarize([r for r in rows if r['arm']==arm]) for arm in arm_names}
        report['wall_seconds'] = time.monotonic()-started
        report['source_run_report_unchanged'] = digest(args.run_report)==registration['run_report_sha256']
        if not report['source_run_report_unchanged']:
            report.update(status='failed', error='Source inference report changed during the audit')
        save(args.out/'report.json', report)
    print(json.dumps({key:report[key] for key in ['status','summary','wall_seconds','error'] if key in report}, indent=2))
    return 0 if report['status']=='completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
