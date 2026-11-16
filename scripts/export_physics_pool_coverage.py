#!/usr/bin/env python3
"""Export a completed physics pilot's pool union for evaluation-only coverage.

No native structure or quality metric is accepted or read. Pose selection is
unchanged: every stored pool row is exported once at the common zero-deformation
reference. DockQ argv is saved with a native-path placeholder for a separate
evaluation process. Its oracle results must never become prediction ranks.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
import torch

from zdock.inducedfit.pipeline import load_pair_manifest, export_heavy_pair
from zdock.inducedfit.physics_pipeline import rotate_xyz


TERMINAL = {'completed', 'partial_failure', 'failed', 'exception'}
NATIVE_PLACEHOLDER = '__EVALUATION_ONLY_NATIVE_PDB__'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_signature(array):
    if array.dtype.hasobject or not np.isfinite(array).all():
        raise ValueError('arrays must be finite and non-object')
    return dict(shape=list(array.shape), dtype=str(array.dtype),
                sha256=hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest())


def save_json(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def checked_pose_key(row, quaternion_count):
    index=row['rotation_index']; translation=row['translation']
    if type(index) is not int or not 0<=index<quaternion_count:
        raise ValueError('invalid rotation index')
    if (not isinstance(translation,list) or len(translation)!=3
            or any(type(x) not in (int,float) or not math.isfinite(x) for x in translation)):
        raise ValueError('translation must be three finite numbers')
    return index,tuple(float(x) for x in translation)


def build_pool_mapping(report):
    if report.get('status') not in TERMINAL:
        raise ValueError('predictor report must have terminal status before coverage export')
    arm_names=report['configuration']['arms']
    if len(set(arm_names))!=len(arm_names) or set(arm_names)!=set(report['arms']):
        raise ValueError('configured arms and recorded arms differ')
    quaternion_count=len(report['quaternions'])
    entries=[]; lookup={}; arms={}
    for arm in arm_names:
        state=report['arms'][arm]
        if state.get('status') not in TERMINAL:
            raise ValueError(f'{arm} is not terminal')
        pool=state.get('pool',[])
        if state.get('pool_count',len(pool))!=len(pool):
            raise ValueError(f'{arm} recorded pool count differs')
        if state['status']=='completed' and len(pool)!=report['configuration']['pool_size']:
            raise ValueError(f'{arm} completed without its registered pool size')
        selected={}; ranks=set(); final_ranks=set()
        for candidate in state.get('candidates',[]):
            rank=candidate['selection_rank']; final_rank=candidate.get('final_rank')
            if type(rank) is not int or rank<1 or rank in ranks:
                raise ValueError(f'{arm} has invalid/duplicate selection rank')
            if final_rank is not None and (type(final_rank) is not int or final_rank<1 or final_rank in final_ranks):
                raise ValueError(f'{arm} has invalid/duplicate final rank')
            key=checked_pose_key(candidate,quaternion_count)
            if key in selected:
                raise ValueError(f'{arm} selected the same pose more than once')
            ranks.add(rank);final_ranks.add(final_rank);selected[key]=candidate
        pool_keys=set();ordered=[];top5=[];final5=[]
        for pool_rank,row in enumerate(pool,1):
            key=checked_pose_key(row,quaternion_count)
            if key in pool_keys:
                raise ValueError(f'{arm} contains duplicate pool poses')
            pool_keys.add(key)
            if key not in lookup:
                label=f'pool_{len(entries)+1:04d}'
                lookup[key]=len(entries)
                entries.append(dict(label=label,rotation_index=key[0],translation=list(key[1]),memberships=[]))
            entry=entries[lookup[key]];label=entry['label'];ordered.append(label)
            candidate=selected.get(key)
            membership=dict(arm=arm,pool_rank=pool_rank,pool_row=row,selected_top5=False,
                            selection_rank=None,final_rank=None)
            if candidate is not None:
                rank=candidate['selection_rank'];final_rank=candidate.get('final_rank')
                membership.update(selection_rank=rank,final_rank=final_rank,selected_top5=rank<=5,
                                  initial_complex=candidate.get('initial_files',{}).get('complex'),
                                  prediction_status=candidate.get('status'),
                                  prediction_export_complete=candidate.get('export_complete'))
                if rank<=5:top5.append((rank,label))
                if final_rank is not None and final_rank<=5:final5.append((final_rank,label))
            entry['memberships'].append(membership)
        if not set(selected).issubset(pool_keys):
            raise ValueError(f'{arm} selected a pose absent from its pool')
        arms[arm]=dict(predictor_status=state['status'],requested_pool_size=report['configuration']['pool_size'],
                       available_pool_count=len(pool),pool_order=ordered,
                       selected_top5_order=[label for _,label in sorted(top5)],
                       final_top5_pose_keys_in_final_rank_order=[label for _,label in sorted(final5)],
                       final_rank_note='These labels identify starting poses; pool exports contain no final refinement.')
    return entries,arms


def restore_heavy_indices(prepared, provenance, full_xyz):
    rows=provenance['heavy_atom_mapping'];keys=prepared.monomer.atom_keys
    if full_xyz.dtype!=np.float64 or full_xyz.ndim!=2 or full_xyz.shape[1]!=3 or not np.isfinite(full_xyz).all():
        raise ValueError('full coordinates must be finite float64 (N,3)')
    if provenance['n_atoms']!=len(full_xyz) or provenance['n_heavy_atoms']!=len(keys) or len(rows)!=len(keys):
        raise ValueError('heavy/full atom counts differ from preparation provenance')
    if provenance['input_sha256']!=prepared.source_sha256:
        raise ValueError('preparation provenance differs from input SHA')
    indices=[]
    for original_index,(row,key) in enumerate(zip(rows,keys,strict=True)):
        index=row['full_atom_index']
        if (row['original_heavy_index']!=original_index or row['atom_key']!=asdict(key)
                or type(index) is not int or not 0<=index<len(full_xyz)):
            raise ValueError('full-to-heavy mapping identity/order/index mismatch')
        indices.append(index)
    if len(set(indices))!=len(indices):
        raise ValueError('full-to-heavy mapping is not injective')
    return torch.tensor(indices,dtype=torch.long)


def run(args, output):
    report_path=args.run_report.resolve();report_hash=digest(report_path)
    output.update(predictor_report=str(report_path),predictor_report_sha256=report_hash)
    report=json.loads(report_path.read_text())
    entries,arms=build_pool_mapping(report)
    output.update(predictor_status=report['status'],arms=arms,models=entries,
                  predictor_configuration=report['configuration'])
    manifest=args.manifest.resolve()
    if digest(manifest)!=report['input_manifest_sha256']:
        raise ValueError('manifest SHA differs from predictor report')
    metadata,inputs=load_pair_manifest(manifest)
    if metadata!=report['input']:
        raise ValueError('input metadata differs from predictor report')
    output.update(input_manifest=str(manifest),input_manifest_sha256=digest(manifest),input=metadata)
    run_dir=report_path.parent;common_path=run_dir/'common_reference.npz'
    with np.load(common_path,allow_pickle=False) as archive:
        common={name:archive[name] for name in archive.files}
    output['common_reference']=dict(path=str(common_path),sha256=digest(common_path),
                                   arrays={name:array_signature(value) for name,value in common.items()})
    quaternions=np.asarray(report['quaternions'],dtype=np.float64)
    if quaternions.shape!=(len(quaternions),4) or not np.isfinite(quaternions).all():
        raise ValueError('invalid quaternion array')
    if not np.allclose(np.linalg.norm(quaternions,axis=1),1.,rtol=0,atol=1e-12):
        raise ValueError('quaternions are not unit length')
    output['quaternions']=array_signature(quaternions)
    output['reference_model_checks']={}
    for arm in arms:
        model_path=run_dir/f'{arm}_model.npz'
        with np.load(model_path,allow_pickle=False) as archive:
            if not np.array_equal(archive['reference'],common['receptor_xyz']):
                raise ValueError(f'{arm} receptor reference differs from common reference')
        output['reference_model_checks'][arm]=dict(path=str(model_path),sha256=digest(model_path),
                                                  receptor_reference_identical=True)
    coordinates={};indices={};centers={};output['heavy_mapping_checks']={}
    for side in ('receptor','ligand'):
        coordinates[side]=torch.from_numpy(common[side+'_xyz'].copy())
        indices[side]=restore_heavy_indices(inputs[side],report['preparation'][side],common[side+'_xyz'])
        centers[side]=torch.tensor(report['preparation'][side]['center'],dtype=torch.float64)
        if centers[side].shape!=(3,) or not torch.isfinite(centers[side]).all():
            raise ValueError('invalid recorded centering translation')
        torch.testing.assert_close(centers[side],inputs[side].monomer.xyz.mean(0),atol=1e-12,rtol=0)
        changes=coordinates[side][indices[side]]+centers[side]-inputs[side].monomer.xyz
        if side=='ligand':torch.testing.assert_close(changes,torch.zeros_like(changes),atol=1e-10,rtol=0)
        output['heavy_mapping_checks'][side]=dict(n_full=len(coordinates[side]),n_heavy=len(indices[side]),
            indices=indices[side].tolist(),index_array=array_signature(indices[side].numpy()),
            input_sha256=inputs[side].source_sha256,atom_keys_identical=True,
            maximum_heavy_displacement_from_input_angstrom=float(changes.norm(dim=-1).max()),
            role='common prepared receptor reference' if side=='receptor' else 'unchanged centered ligand')
    root=Path(__file__).resolve().parents[1]
    source_paths=[Path(__file__),root/'scripts/score_inducedfit_pilot.py',root/'src/zdock/geom.py',
                  *sorted((root/'src/zdock/inducedfit').glob('*.py'))]
    output['source_sha256']={str(path.relative_to(root)):digest(path) for path in source_paths}
    for relative,sha in output['source_sha256'].items():
        if relative in report['source_sha256'] and sha!=report['source_sha256'][relative]:
            raise ValueError(f'source differs from frozen predictor: {relative}')
        target=args.out/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/relative,target)
    save_json(args.out/'mapping.json',output)
    for entry in entries:
        quaternion=torch.from_numpy(quaternions[entry['rotation_index']].copy())
        ligand=rotate_xyz(coordinates['ligand'],quaternion)+torch.tensor(entry['translation'],dtype=torch.float64)
        paths=export_heavy_pair(args.out/'models'/entry['label'],inputs['receptor'],inputs['ligand'],
            coordinates['receptor'][indices['receptor']],ligand[indices['ligand']],centers['receptor'])
        entry['files']={key:str(Path(path).relative_to(args.out)) for key,path in paths.items()}
        entry['file_sha256']={key:digest(path) for key,path in paths.items()}
        entry['selected_initial_export_checks']=[]
        for member in entry['memberships']:
            if member.get('initial_complex'):
                previous=run_dir/member['initial_complex']
                if not previous.is_file() or digest(previous)!=entry['file_sha256']['complex']:
                    raise ValueError('reconstructed selected pose differs from primary initial export')
                entry['selected_initial_export_checks'].append(dict(arm=member['arm'],
                    selection_rank=member['selection_rank'],identical_complex_sha256=True))
        entry['status']='exported'
        output['exported_count']=sum(row.get('status')=='exported' for row in entries)
        save_json(args.out/'mapping.json',output)
    if digest(report_path)!=report_hash or digest(common_path)!=output['common_reference']['sha256']:
        raise ValueError('predictor report or common array file changed during export')
    # Do not resolve the interpreter symlink: venv identity depends on argv[0].
    argv=[str(args.evaluator_python.absolute()),str((root/'scripts/score_inducedfit_pilot.py').resolve()),
          '--native',NATIVE_PLACEHOLDER,'--mapping','RL:AD','--allowed-mismatches','1',
          '--out',str((args.out/'coverage_scores.json').resolve())]
    for row in entries:
        argv.extend(['--model',row['label']+'='+str((args.out/row['files']['complex']).resolve())])
    save_json(args.out/'dockq_command.json',dict(argv=argv,native_placeholder=NATIVE_PLACEHOLDER,
        instruction='Replace the placeholder only in the separate evaluation process. Each union model occurs once.',
        mapping_and_mismatch_scope='Historical 1BRS pilot only; fixed RL:AD and mismatch allowance 1.',
        scorer_sha256=digest(root/'scripts/score_inducedfit_pilot.py'),
        predictor_report_sha256=report_hash,union_model_count=len(entries),
        evaluation_scope='Zero-deformation pool coverage diagnostic; no reranking or prediction selection.'))
    output.update(status='completed',unique_pose_count=len(entries),
                  total_pool_memberships=sum(len(row['memberships']) for row in entries),
                  selected_top5_membership_count=sum(m['selected_top5'] for row in entries for m in row['memberships']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-report',required=True,type=Path)
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--evaluator-python',type=Path,default=Path('data/inducedfit/dockq_env/bin/python'))
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    output=dict(status='running',command=sys.argv,platform=platform.platform(),python=platform.python_version(),
                torch=torch.__version__,native_access=False,quality_metrics_read=False,exported_count=0,
                script_sha256=digest(Path(__file__)),
                purpose='Evaluation-only pool union at common zero-deformation reference; no reranking.',
                coordinate_frame='Receptor input frame; add receptor center to both partners.')
    save_json(args.out/'mapping.json',output)
    try:run(args,output)
    except Exception as error:
        output.update(status='failed',error_type=type(error).__name__,error=str(error))
    output['seconds']=time.monotonic()-started
    save_json(args.out/'mapping.json',output)
    print(json.dumps({key:output[key] for key in ('status','unique_pose_count','exported_count','error') if key in output},indent=2))
    return 0 if output['status']=='completed' else 1


if __name__=='__main__':
    raise SystemExit(main())
