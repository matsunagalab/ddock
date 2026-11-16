"""Pool coverage export is exhaustive, native-free, and preserves rank provenance."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from zdock.inducedfit.pipeline import load_pair_manifest, export_heavy_pair
from test_inducedfit_pipeline import pair


spec=importlib.util.spec_from_file_location('pool_coverage_export',Path(__file__).parents[1]/'scripts/export_physics_pool_coverage.py')
exporter=importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)
summary_spec=importlib.util.spec_from_file_location('pool_coverage_summary',Path(__file__).parents[1]/'scripts/summarize_physics_pool_coverage.py')
summary_module=importlib.util.module_from_spec(summary_spec)
summary_spec.loader.exec_module(summary_module)


def pose(x):
    return dict(rotation_index=0,translation=[float(x),0.,0.],coarse_score=-float(x))


def report_fixture():
    return dict(status='completed',configuration=dict(arms=['a','b'],pool_size=2,top_k=2),
                quaternions=[[0.,0.,0.,1.]],arms={
        'a':dict(status='completed',pool_count=2,pool=[pose(1),pose(2)],candidates=[
            dict(pose(2),selection_rank=1,final_rank=2),dict(pose(1),selection_rank=2,final_rank=1)]),
        'b':dict(status='completed',pool_count=2,pool=[pose(2),pose(3)],candidates=[
            dict(pose(3),selection_rank=1,final_rank=1)])})


def test_union_deduplicates_without_sorting_by_score_and_preserves_separate_ranks():
    entries,arms=exporter.build_pool_mapping(report_fixture())
    assert [row['translation'][0] for row in entries]==[1.,2.,3.]
    assert len(entries[1]['memberships'])==2
    assert arms['a']['pool_order']==['pool_0001','pool_0002']
    assert arms['b']['pool_order']==['pool_0002','pool_0003']
    assert arms['a']['selected_top5_order']==['pool_0002','pool_0001']
    assert arms['a']['final_top5_pose_keys_in_final_rank_order']==['pool_0001','pool_0002']
    assert entries[1]['memberships'][0]['selected_top5']
    assert not entries[1]['memberships'][1]['selected_top5']


@pytest.mark.parametrize('change', ['running','arm_running','duplicate_pool','absent_selected','bad_rotation','bad_translation'])
def test_invalid_or_nonterminal_pool_rejected(change):
    report=report_fixture()
    if change=='running':report['status']='running'
    elif change=='arm_running':report['arms']['b']['status']='running'
    elif change=='duplicate_pool':report['arms']['a']['pool'][1]=pose(1)
    elif change=='absent_selected':report['arms']['a']['candidates'][0].update(pose(7))
    elif change=='bad_rotation':report['arms']['a']['pool'][0]['rotation_index']=True
    elif change=='bad_translation':report['arms']['a']['pool'][0]['translation'][0]=float('nan')
    with pytest.raises(ValueError):exporter.build_pool_mapping(report)


def provenance(prepared):
    return dict(n_atoms=3,n_heavy_atoms=1,input_sha256=prepared.source_sha256,
        center=prepared.monomer.xyz.mean(0).tolist(),heavy_atom_mapping=[
            dict(original_heavy_index=0,full_atom_index=1,atom_key=asdict(prepared.monomer.atom_keys[0]))])


def test_mapping_rebuilt_from_checked_atom_identity_not_full_atom_position(tmp_path):
    manifest,_=pair(tmp_path);_,inputs=load_pair_manifest(manifest)
    prepared=inputs['receptor'];metadata=provenance(prepared)
    coordinates=np.zeros((3,3),dtype=np.float64)
    assert exporter.restore_heavy_indices(prepared,metadata,coordinates).tolist()==[1]
    bad=deepcopy(metadata);bad['heavy_atom_mapping'][0]['atom_key']['atom_name']='BAD'
    with pytest.raises(ValueError,match='identity/order/index'):
        exporter.restore_heavy_indices(prepared,bad,coordinates)


def test_complete_synthetic_export_and_scorer_argv_need_no_native_file(tmp_path):
    manifest,metadata=pair(tmp_path);_,inputs=load_pair_manifest(manifest)
    run_dir=tmp_path/'run';run_dir.mkdir();out=tmp_path/'export';out.mkdir()
    report=report_fixture()
    coordinates=np.array([[.1,0.,0.],[0.,0.,0.],[-.1,0.,0.]],dtype=np.float64)
    np.savez(run_dir/'common_reference.npz',receptor_xyz=coordinates,ligand_xyz=coordinates)
    report.update(input=metadata,input_manifest_sha256=exporter.digest(manifest),
                  preparation={side:provenance(inputs[side]) for side in inputs},source_sha256={})
    for arm in report['arms']:
        np.savez(run_dir/f'{arm}_model.npz',reference=coordinates)
        for row in report['arms'][arm]['candidates']:
            previous=export_heavy_pair(run_dir/f'{arm}_{row["selection_rank"]}',inputs['receptor'],inputs['ligand'],
                torch.zeros((1,3),dtype=torch.float64),torch.tensor([row['translation']],dtype=torch.float64),
                inputs['receptor'].monomer.xyz[0])
            row['initial_files']={'complex':str(Path(previous['complex']).relative_to(run_dir))}
    report_path=run_dir/'report.json';report_path.write_text(json.dumps(report))
    interpreter_target=tmp_path/'base_python';interpreter_target.write_text('not executed')
    interpreter=tmp_path/'venv_python';interpreter.symlink_to(interpreter_target)
    args=argparse.Namespace(run_report=report_path,manifest=manifest,out=out,evaluator_python=interpreter)
    result={'native_access':False,'quality_metrics_read':False}
    exporter.run(args,result)
    assert result['status']=='completed' and result['exported_count']==3
    assert result['total_pool_memberships']==4 and result['selected_top5_membership_count']==3
    assert len(result['models'][1]['selected_initial_export_checks'])==1
    assert (out/'source_snapshot/scripts/export_physics_pool_coverage.py').is_file()
    command=json.loads((out/'dockq_command.json').read_text())
    assert command['argv'].count('--model')==3
    assert command['argv'][0]==str(interpreter.absolute())
    assert command['argv'][0]!=str(interpreter.resolve())
    assert exporter.NATIVE_PLACEHOLDER in command['argv']
    assert all(row['status']=='exported' for row in result['models'])
    assert report_path.read_text()==json.dumps(report)


def evaluation_fixture():
    report=report_fixture()
    report['arms']['a']['candidates']=[dict(pose(2),selection_rank=1,final_rank=1)]
    entries,arms=exporter.build_pool_mapping(report)
    for row in entries:row['file_sha256']={'complex':row['label']+'_sha'}
    mapping=dict(status='completed',models=entries,arms=arms,source_sha256={'scripts/score_inducedfit_pilot.py':'scorer_sha'})
    evaluation=dict(status='completed',script_sha256='scorer_sha',models=[
        dict(arm=row['label'],stage='explicit',status='scored',model_sha256=row['file_sha256']['complex'],
             metrics={'DockQ':score}) for row,score in zip(entries,[.8,.1,.3],strict=True)])
    return mapping,evaluation


def test_oracle_summary_preserves_prediction_ranks_and_describes_initial_selection_gap():
    mapping,evaluation=evaluation_fixture();before=deepcopy(mapping)
    result=summary_module.summarize(mapping,evaluation)
    arm=result['arms']['a']
    assert arm['oracle_pool_best_zero_deformation']['selection_rank'] is None
    assert arm['oracle_selected_top5_best_initial']['selection_rank']==1
    assert arm['oracle_pool_minus_oracle_selected_initial_DockQ']==pytest.approx(.7)
    assert arm['selection_rank1_initial']['metrics']['DockQ']==.1
    assert mapping==before and not result['identical_pool_sets']


@pytest.mark.parametrize('change',['nonterminal','missing','different_structure','different_scorer'])
def test_oracle_summary_rejects_unverified_evaluations(change):
    mapping,evaluation=evaluation_fixture()
    if change=='nonterminal':evaluation['status']='running'
    elif change=='missing':evaluation['models'].pop()
    elif change=='different_structure':evaluation['models'][0]['model_sha256']='wrong'
    elif change=='different_scorer':evaluation['script_sha256']='wrong'
    with pytest.raises(ValueError):summary_module.summarize(mapping,evaluation)
