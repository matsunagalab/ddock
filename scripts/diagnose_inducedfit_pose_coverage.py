#!/usr/bin/env python3
"""Evaluation-only pose quantization probes for a frozen induced-fit run.

Run in the pinned DockQ evaluator environment. Native fits and oracle poses
must never enter inference. Nearest-angle / minimum-CA-RMSD probes are NOT
an upper bound on achievable DockQ or an exhaustive docking search.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np
from score_inducedfit_pilot import chain_alignment, run_dockq, sha256


def fit_row(x, y):
    xc, yc = x - x.mean(0), y - y.mean(0)
    u, values, vt = np.linalg.svd(xc.T @ yc)
    if values[1] < 1e-10 * values[0]:
        raise ValueError('Collinear CA coordinates cannot determine a rigid fit')
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    translation = y.mean(0) - x.mean(0) @ rotation
    return rotation, translation


def quaternion_row(q):
    """Independent NumPy transcription of the recorded zdock.geom convention."""
    q=np.asarray(q,dtype=np.float64)
    if q.shape!=(4,) or not np.isfinite(q).all() or not np.isclose(np.linalg.norm(q),1.,atol=1e-10,rtol=1e-10):
        raise ValueError('Recorded quaternion must be finite and unit length')
    q1, q2, q3, q4 = q
    column = np.array([
        [1-2*q2*q2-2*q3*q3, 2*(q1*q2+q3*q4), 2*(q1*q3-q2*q4)],
        [2*(q1*q2-q3*q4), 1-2*q1*q1-2*q3*q3, 2*(q2*q3+q1*q4)],
        [2*(q1*q3+q2*q4), 2*(q2*q3-q1*q4), 1-2*q1*q1-2*q2*q2],
    ], dtype=np.float64)
    if not np.allclose(column.T @ column, np.eye(3), atol=1e-10):
        raise ValueError('Recorded quaternion is not a rotation')
    return column.T


def rmsd(x, y):
    return float(np.sqrt(np.mean(np.sum((x-y)**2, axis=-1))))


def compose_apo_relative_pose(rr,tr,rl,tl,receptor_center,ligand_center):
    """Compose apo-to-native row transforms into the centered receptor frame."""
    rotation=rl @ rr.T
    translation=(ligand_center @ rl+tl-tr) @ rr.T-receptor_center
    return rotation,translation


def write_oracle(path, apo, centers, rotation, translation):
    from Bio.PDB import Model, PDBIO, Structure
    structure = Structure.Structure('evaluation_only_oracle')
    model = Model.Model(0)
    structure.add(model)
    for side, chain_id in [('receptor', 'R'), ('ligand', 'L')]:
        chain = copy.deepcopy(apo[side])
        chain.id = chain_id
        for atom in chain.get_atoms():
            xyz = np.asarray(atom.coord, dtype=np.float64)
            if side == 'ligand':
                xyz = (xyz-centers[side]) @ rotation + translation + centers['receptor']
            atom.set_coord(xyz)
        model.add(chain)
    writer = PDBIO()
    writer.set_structure(structure)
    writer.save(str(path))


def run(args, result):
    from DockQ.DockQ import load_PDB
    frozen = json.loads(args.run_report.read_text())
    if frozen['status'] != 'completed':
        raise ValueError('Inference must be completed before native evaluation')
    manifest = Path(frozen['configuration']['manifest']).resolve()
    if sha256(manifest) != frozen['input_manifest_sha256']:
        raise ValueError('Manifest hash changed')
    result.update(input_manifest_sha256=sha256(manifest), run_report_sha256=sha256(args.run_report),
                  native_sha256=sha256(args.native), seed=frozen['configuration']['seed'],
                  loss=None, model_or_checkpoint_selection=None)
    native = load_PDB(str(args.native), chains=list(args.native_chains))
    apo, centers, paired, fits, audits = {}, {}, {}, {}, {}
    for side, native_chain in zip(('receptor', 'ligand'), args.native_chains):
        item = frozen['input'][side]
        path = (manifest.parent / item['pdb']).resolve()
        if sha256(path) != item['sha256']:
            raise ValueError(f'{side} input hash changed')
        apo[side] = load_PDB(str(path), chains=[item['chain']])[item['chain']]
        audit, pairs = chain_alignment(apo[side], native[native_chain])
        pairs = [(a,b) for a,b in pairs if 'CA' in a and 'CA' in b]
        x = np.asarray([a['CA'].coord for a,b in pairs], dtype=np.float64)
        y = np.asarray([b['CA'].coord for a,b in pairs], dtype=np.float64)
        rotation, translation = fit_row(x,y)
        centers[side] = np.asarray(frozen['preparation'][side]['center'], dtype=np.float64)
        fits[side] = rotation, translation
        paired[side] = x, y
        audits[side] = dict(path=str(path), sha256=sha256(path), alignment=audit,
                           paired_ca_count=len(x), independent_fit_ca_rmsd=rmsd(x@rotation+translation,y))
    rr, tr = fits['receptor']
    rl, tl = fits['ligand']
    exact_rotation,exact_translation = compose_apo_relative_pose(rr,tr,rl,tl,centers['receptor'],centers['ligand'])
    x = paired['ligand'][0]-centers['ligand']
    target = (paired['ligand'][1]-tr) @ rr.T-centers['receptor']
    if not np.isclose(rmsd(x@exact_rotation+exact_translation,target),
                      audits['ligand']['independent_fit_ca_rmsd'], atol=1e-10):
        raise ValueError('Frame composition did not preserve the independent monomer residual')
    spacing = float(frozen['configuration']['spacing'])
    rows = []
    matrices = []
    for index, quaternion in enumerate(frozen['quaternions']):
        rotation = quaternion_row(quaternion)
        matrices.append(rotation)
        angle = float(np.degrees(np.arccos(np.clip((np.trace(rotation.T@exact_rotation)-1)/2, -1,1))))
        optimal = np.mean(target-x@rotation, axis=0)
        snapped = np.round(optimal/spacing)*spacing
        rows.append(dict(rotation_index=index, angle_to_apo_fit_deg=angle,
                         ca_least_squares_translation=optimal.tolist(), grid_translation=snapped.tolist(),
                         translation_snap_error=float(np.linalg.norm(snapped-optimal)),
                         paired_ligand_ca_rmsd_at_optimal_translation=rmsd(x@rotation+optimal,target),
                         paired_ligand_ca_rmsd_at_grid_translation=rmsd(x@rotation+snapped,target)))
    nearest = min(rows,key=lambda row:row['angle_to_apo_fit_deg'])
    best_grid = min(rows,key=lambda row:row['paired_ligand_ca_rmsd_at_grid_translation'])
    result.update(status='geometry_completed', input_audit=audits, rotations=rows,
                  exact_apo_relative_rotation_row=exact_rotation.tolist(),
                  exact_apo_relative_translation=exact_translation.tolist(),
                  nearest_angle_probe=nearest, minimum_ca_rmsd_grid_probe=best_grid,
                  configured_local_rotation_bound_deg=float(np.degrees(.15)),
                  exact_apo_grid_snap_error=float(np.linalg.norm(np.round(exact_translation/spacing)*spacing-exact_translation)),
                  final_candidate_probes={arm:[dict(selection_rank=row['selection_rank'],
                      rotation_index=row['rotation_index'],
                      angle_to_apo_fit_deg=rows[row['rotation_index']]['angle_to_apo_fit_deg'],
                      selected_translation_distance_to_ca_optimum=float(np.linalg.norm(
                        np.asarray(row['translation'])-rows[row['rotation_index']]['ca_least_squares_translation'])))
                      for row in value['candidates']] for arm,value in frozen['arms'].items()})
    probes = {'apo_fit_continuous':(exact_rotation, exact_translation),
              'apo_fit_grid_translation':(exact_rotation,np.round(exact_translation/spacing)*spacing),
              'nearest_angle_grid_translation':(matrices[nearest['rotation_index']],np.asarray(nearest['grid_translation'])),
              'minimum_ca_rmsd_grid_translation':(matrices[best_grid['rotation_index']],np.asarray(best_grid['grid_translation']))}
    result['oracle_probes'] = {}
    if args.dockq:
        sanity = run_dockq(args.dockq, args.native, args.native, args.native_chains+':'+args.native_chains,
                           args.out,'native_sanity',0,60.)
        result['native_sanity'] = sanity
        if sanity['status'] != 'scored' or sanity['metrics']['DockQ'] < .999:
            raise ValueError('Native-vs-native sanity failed')
    for name,(rotation,translation) in probes.items():
        path = args.out / (name+'.pdb')
        write_oracle(path,apo,centers,rotation,translation)
        probe = dict(pdb=str(path),sha256=sha256(path),evaluation_only=True)
        if args.dockq:
            probe.update(run_dockq(args.dockq,path,args.native,'RL:'+args.native_chains,
                                   args.out,name,args.allowed_mismatches,60.))
        result['oracle_probes'][name] = probe
    result['status'] = ('completed' if all(p.get('status','scored')=='scored'
                              for p in result['oracle_probes'].values()) else 'evaluation_failure')


def score_frozen_probe(args, result):
    """Score one already recorded native-informed probe; no native read here."""
    import torch
    from zdock.geom import rotate
    from zdock.inducedfit.replay import load_replay
    from zdock.inducedfit.fields import (build_fields, receptor_channels, ligand_channels,
        correlation_spectra, correlate_from_spectra, valid_translation_mask, physical_energy)
    from zdock.inducedfit.response import QuadraticResponse
    from zdock.inducedfit.search import PreparedResponseReceptor, search_response

    torch.set_num_threads(1)
    coverage=json.loads(args.score_only_coverage.read_text())
    if coverage['status']!='completed' or coverage['run_report_sha256']!=sha256(args.run_report):
        raise ValueError('Coverage diagnostic is incomplete or belongs to another frozen run')
    context=load_replay(args.run_report)
    probe=coverage['nearest_angle_probe']
    index=probe['rotation_index']
    quaternion=context.quaternions[index]
    translation=context.receptor.xyz.new_tensor(probe['grid_translation'])
    receptor,ligand,grid,parameters=context.receptor,context.ligand,context.grid,context.parameters
    with torch.no_grad():
        rf=build_fields(receptor.xyz,receptor.radii,receptor.charges,grid,parameters)
        rfft=correlation_spectra(receptor_channels(rf,grid,parameters))
        xyz=torch.stack(rotate(*ligand.xyz.unbind(-1),quaternion),dim=-1)
        lf=build_fields(xyz,ligand.radii,ligand.charges,grid,parameters)
        correlation=correlate_from_spectra(rfft,correlation_spectra(ligand_channels(lf)),grid.shape)
        mask=valid_translation_mask(lf,grid)
        lag=translation/grid.spacing
        if not torch.equal(lag,lag.round()):
            raise ValueError('Probe is not on the original translation lattice')
        tensor_index=tuple(int(lag[d])+grid.shape[d]-1 for d in range(3))
        valid=bool(mask[tensor_index])
        energy=float(correlation[tensor_index])
        lower_count=int(((correlation < energy) & mask).sum())
        shifted=build_fields(xyz+translation,ligand.radii,ligand.charges,grid,parameters)
        direct=float(physical_energy(rf,shifted,grid,parameters))
        if valid and not np.isclose(energy,direct,atol=1e-7,rtol=1e-9):
            raise ValueError('FFT and directly shifted probe energies disagree')
        # Use the actual production peak picker, without computing unused
        # response derivative FFTs. The stiffness object is unused in rigid selection.
        prepared=PreparedResponseReceptor(grid,parameters,rfft,(),QuadraticResponse(context.model.stiffness),0.)
        peaks=search_response(prepared,ligand,quaternion.unsqueeze(0),radius=0.,top_k=4,
                              peaks_per_rotation=4,min_peak_distance=3.,selection='rigid')
    old=context.report['arms']['R0']['candidates']
    cutoff=max(row['selected_rigid_energy'] for row in old)
    result.update(status='completed',native_access='no native file read; consumes a native-informed evaluation probe',
        coverage_report_sha256=sha256(args.score_only_coverage),run_report_sha256=sha256(args.run_report),
        input_manifest_sha256=context.source_audit['manifest_sha256'],source_audit=context.source_audit,
        rotation_index=index,translation=translation.tolist(),full_support_valid=valid,
        rigid_fft_energy=energy,rigid_direct_energy=direct,fft_direct_difference=energy-direct,
        valid_translation_count=int(mask.sum()),strictly_better_translations_in_this_rotation=lower_count,
        original_R0_top5_cutoff_energy=cutoff,probe_would_beat_original_global_top5_cutoff=energy<cutoff,
        production_rotation_top4=[dict(translation=p.translation.tolist(),energy=p.rigid_energy) for p in peaks],
        probe_is_in_rotation_top4=any(torch.equal(p.translation,translation) for p in peaks),
        torch=torch.__version__,threads_policy='torch.set_num_threads(1); BLAS thread counts not separately verified')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-report',type=Path,required=True)
    parser.add_argument('--native',type=Path)
    parser.add_argument('--native-chains',default='AD')
    parser.add_argument('--allowed-mismatches',type=int,default=1)
    parser.add_argument('--dockq',type=Path)
    parser.add_argument('--score-only-coverage',type=Path,
                        help='in ddock environment, inspect one saved oracle probe on the original score/grid')
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    if len(args.native_chains)!=2 or len(set(args.native_chains))!=2:
        parser.error('native-chains must name two distinct chains')
    if args.score_only_coverage is None and args.native is None:
        parser.error('--native is required for geometry/DockQ evaluation')
    if args.score_only_coverage and (args.native or args.dockq):
        parser.error('score-only mode consumes the saved diagnostic, not native files or DockQ')
    if args.dockq:
        args.dockq = args.dockq.resolve()
    args.out.mkdir(parents=True,exist_ok=False)
    result = dict(status='running',command=sys.argv, native_access='evaluation only',
                  scope='posthoc geometric probes; not an upper bound on DockQ; no inference tuning',
                  source_sha256={str(Path(__file__).name):sha256(Path(__file__)),
                    'score_inducedfit_pilot.py':sha256(Path(__file__).with_name('score_inducedfit_pilot.py')),
                    'geom.py':sha256(Path(__file__).resolve().parents[1]/'src/zdock/geom.py')},
                  platform=platform.platform(),python=platform.python_version(),numpy=np.__version__)
    start=time.monotonic()
    try:
        if args.score_only_coverage:
            score_frozen_probe(args,result)
        else:
            run(args,result)
    except Exception as error:
        result.update(status='failed',error_type=type(error).__name__,error=str(error))
    result['wall_seconds']=time.monotonic()-start
    (args.out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','wall_seconds','nearest_angle_probe','minimum_ca_rmsd_grid_probe','error') if k in result},indent=2))
    return 0 if result['status']=='completed' else 1


if __name__=='__main__':
    raise SystemExit(main())
