#!/usr/bin/env python3
"""Evaluation-only pool coverage and initial Top5 selection diagnostic.

Reads completed export and official scorer JSON, never protein coordinates.
Oracle quality is descriptive only: no prediction reranking, selected-pose
replacement, or bound on accuracy after deformation/refinement is produced.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(mapping, evaluation):
    if mapping.get('status')!='completed' or evaluation.get('status')!='completed':
        raise ValueError('export and evaluator must both be completed')
    if evaluation.get('script_sha256')!=mapping['source_sha256']['scripts/score_inducedfit_pilot.py']:
        raise ValueError('scorer script differs from export registration')
    entries={row['label']:row for row in mapping['models']}
    if len(entries)!=len(mapping['models']):
        raise ValueError('duplicate union label')
    scores={}
    for row in evaluation['models']:
        label=row['arm']
        if label not in entries or label in scores or row.get('stage')!='explicit':
            raise ValueError('evaluator labels do not match the unique exported union')
        if row.get('status')!='scored':
            raise ValueError('every union model must have a valid score')
        if row.get('model_sha256')!=entries[label]['file_sha256']['complex']:
            raise ValueError('evaluated structure SHA differs from exported pool')
        dockq=row['metrics']['DockQ']
        if type(dockq) not in (int,float) or not math.isfinite(dockq) or not 0<=dockq<=1.0000000001:
            raise ValueError('invalid DockQ value')
        scores[label]=dict(row['metrics'])
    if set(scores)!=set(entries):
        raise ValueError('missing evaluated union candidates')
    arms={}
    for arm,state in mapping['arms'].items():
        pool=state['pool_order'];selected=state['selected_top5_order']
        if not pool or not selected or not set(selected).issubset(pool):
            raise ValueError('arm has empty or inconsistent pool/selection')
        pool_best=max(pool,key=lambda label:scores[label]['DockQ'])
        selected_best=max(selected,key=lambda label:scores[label]['DockQ'])
        def details(label):
            membership=next(row for row in entries[label]['memberships'] if row['arm']==arm)
            return dict(label=label,rotation_index=entries[label]['rotation_index'],
                        translation=entries[label]['translation'],metrics=scores[label],
                        pool_rank=pool.index(label)+1,
                        selection_rank=membership['selection_rank'])
        arms[arm]=dict(pool_count=len(pool),selected_initial_count=len(selected),
            oracle_pool_best_zero_deformation=details(pool_best),
            oracle_selected_top5_best_initial=details(selected_best),
            selection_rank1_initial=details(selected[0]),
            oracle_pool_minus_oracle_selected_initial_DockQ=scores[pool_best]['DockQ']-scores[selected_best]['DockQ'],
            pool_candidates_DockQ_ge_0_23=sum(scores[label]['DockQ']>=.23 for label in pool),
            selected_initial_candidates_DockQ_ge_0_23=sum(scores[label]['DockQ']>=.23 for label in selected))
    first=next(iter(mapping['arms'].values()))
    return dict(arms=arms,unique_union_count=len(entries),scored_union_count=len(scores),
        identical_pool_sets=all(set(state['pool_order'])==set(first['pool_order']) for state in mapping['arms'].values()),
        identical_pool_order=all(state['pool_order']==first['pool_order'] for state in mapping['arms'].values()),
        identical_selected_top5_initial_order=all(state['selected_top5_order']==first['selected_top5_order'] for state in mapping['arms'].values()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapping',type=Path,required=True)
    parser.add_argument('--scores',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--csv',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists() or args.csv.exists():
        raise FileExistsError('refusing to overwrite earlier diagnostic')
    result=dict(status='running',command=sys.argv,script_sha256=digest(__file__),
        export_mapping=str(args.mapping.absolute()),export_mapping_sha256=digest(args.mapping),
        scorer_report=str(args.scores.absolute()),scorer_report_sha256=digest(args.scores),
        purpose='Evaluation-only zero-deformation pool coverage versus unchanged initial Top5 selection.',
        limitations='Not prediction success, not a bound on final refined accuracy, not an HDOCK comparison. All oracle labels are diagnostic only.',
        input_coordinates_read=False,inference_or_selection_modified=False)
    try:
        mapping=json.loads(args.mapping.read_text());evaluation=json.loads(args.scores.read_text())
        result.update(summarize(mapping,evaluation),status='completed',
            predictor_report_sha256=mapping['predictor_report_sha256'],
            native_sha256=evaluation['native_sha256'],dockq_version=evaluation['versions']['DockQ'],
            scorer_script_sha256=evaluation['script_sha256'])
        rows=[]
        for arm,state in result['arms'].items():
            rows.append(dict(arm=arm,pool_count=state['pool_count'],selected_initial_count=state['selected_initial_count'],
                oracle_pool_best_zero_deformation_DockQ=state['oracle_pool_best_zero_deformation']['metrics']['DockQ'],
                oracle_selected_top5_best_initial_DockQ=state['oracle_selected_top5_best_initial']['metrics']['DockQ'],
                selection_rank1_initial_DockQ=state['selection_rank1_initial']['metrics']['DockQ'],
                oracle_pool_minus_oracle_selected_initial_DockQ=state['oracle_pool_minus_oracle_selected_initial_DockQ'],
                pool_candidates_DockQ_ge_0_23=state['pool_candidates_DockQ_ge_0_23'],
                selected_initial_candidates_DockQ_ge_0_23=state['selected_initial_candidates_DockQ_ge_0_23']))
        args.csv.parent.mkdir(parents=True,exist_ok=True)
        with args.csv.open('x',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator='\n')
            writer.writeheader();writer.writerows(rows)
        result['summary_csv_sha256']=digest(args.csv)
    except Exception as error:
        result.update(status='failed',error_type=type(error).__name__,error=str(error))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as handle:json.dump(result,handle,indent=2,allow_nan=False)
    print(json.dumps({key:result[key] for key in ['status','unique_union_count','scored_union_count','identical_pool_sets','identical_pool_order','identical_selected_top5_initial_order','error'] if key in result},indent=2))
    return 0 if result['status']=='completed' else 1


if __name__=='__main__':
    raise SystemExit(main())
