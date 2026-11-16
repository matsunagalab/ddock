#!/usr/bin/env python3
"""Experimental items1–4: typed FFT / physical strain / constraints / adaptation.

Inference-only input manifest; native coordinates and DockQ are never read.
All generated arrays, full per-rotation peaks and failures are saved.
"""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import shutil
import sys
import time

import numpy as np
import torch

from zdock.inducedfit.pipeline import load_pair_manifest,centered_protein,rotation_sample,export_heavy_pair
from zdock.inducedfit.chemistry import prepare_amber_charges
from zdock.inducedfit.modes import build_torsion_modes
from zdock.inducedfit.physical_prior import (prepare_physical_prior,build_physical_modes,
                                           load_prepared_physical_prior)
from zdock.inducedfit.typed_potential import TypedPotentialConfig,extract_typed_parameters,direct_pair_energy
from zdock.inducedfit.typed_fft import build_probe_types
from zdock.inducedfit.physics_pipeline import (typed_pair_grid,prepare_typed_cache,typed_search,mixed_pool,
    representative_poses,rotate_xyz,verify_physical_pool,refine_physical_pose)
from zdock.inducedfit.adaptive_contact import contact_torsion_mask,build_contact_directions

ARMS=('rigid','ball8','atom8','contact8','random8')


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    root=Path(__file__).resolve().parents[1]
    paths=[Path(__file__),*sorted((root/'src/zdock/inducedfit').glob('*.py')),root/'src/zdock/geom.py']
    return {str(p.relative_to(root)):digest(p) for p in paths}


def save_json(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def finalize_arm_counts(state, requested):
    """Failed exports never enter ranks; missing requests stay in the denominator."""
    candidates=state.get('candidates',[])
    ranked=sorted((r for r in candidates if r.get('export_complete') is True
                   and r.get('status')!='exception' and math.isfinite(r.get('energy',math.nan))),
                  key=lambda r:(r['energy'],r['selection_rank']))
    for row in candidates:row.pop('final_rank',None)
    for rank,row in enumerate(ranked,1):row['final_rank']=rank
    state.update(exported_count=len(ranked),failure_count=len(candidates)-len(ranked),
                 not_attempted_count=max(0,requested-len(candidates)),
                 missing_requested_count=max(0,requested-len(ranked)))
    return len(ranked)==requested


def run(args,report,save):
    start=time.monotonic()
    metadata,inputs=load_pair_manifest(args.manifest)
    report.update(input=metadata,input_manifest_sha256=digest(args.manifest),stage='chemical_preparation')
    save()
    proteins={}; centers={}
    for side in ('receptor','ligand'):
        protein=prepare_amber_charges(inputs[side],seed=args.seed)
        proteins[side],centers[side]=centered_protein(protein)
        report.setdefault('preparation',{})[side]={**protein.provenance,'center':centers[side].tolist()}
        save()
    receptor,ligand=proteins['receptor'].monomer,proteins['ligand'].monomer
    rp,lp=(extract_typed_parameters(proteins[side]) for side in ('receptor','ligand'))
    probes=build_probe_types(lp); config=TypedPotentialConfig()
    report.update(typed_parameters={'receptor':rp.diagnostics,'ligand':lp.diagnostics},
                  potential_configuration=config.to_dict(),stage='physical_apo_preparation')
    save()
    prior=None
    try:
        if args.prepared_prior is None:
            prior=prepare_physical_prior(proteins['receptor'],anchor_k=args.anchor_k,
                max_iterations=args.preparation_iterations,max_evaluations=args.preparation_evaluations,
                stationarity_tolerance=args.stationarity_tolerance,max_heavy_displacement=1.)
        else:
            prior=load_prepared_physical_prior(proteins['receptor'],args.prepared_prior,
                anchor_k=args.anchor_k,stationarity_tolerance=args.stationarity_tolerance,
                max_heavy_displacement=1.)
        report['physical_prior']=prior.diagnostics
        report['stage']='physical_mode_construction'; save()
        anm=build_torsion_modes(prior.prepared_kinematics,n_modes=8,jacobian_backend='chunked',
                                jvp_chunk_size=8,fit_relative_damping=.001)
        full=build_physical_modes(prior,anm.angle_basis)
        base=build_physical_modes(prior,anm.angle_basis[:,:4])
        report['anm_diagnostics']=anm.diagnostics
        models={'ball8':full,'atom8':full,'rigid':full}
        grid=typed_pair_grid(prior.reference,ligand.xyz,spacing=args.spacing,atom_margin=args.max_atom,
                            translation_margin=args.max_translation)
        if grid.n_points>args.max_grid_points: raise MemoryError(f'grid {grid.shape} exceeds budget')
        quaternions=rotation_sample(args.rotations,args.seed)
        report.update(grid=asdict(grid),quaternions=quaternions.tolist(),stage='first_fft_pass')
        save()
        cache=prepare_typed_cache(prior.reference,rp,probes,grid,config,full,max_bytes=args.max_cache_bytes)
        report['first_fft_cache']=cache.diagnostics
        def progress(done,total):
            report['rotation_progress']=[done,total];save()
        raw=typed_search(cache,ligand.xyz,lp,probes,quaternions,radius=args.radius,
                         peaks_per_rotation=4,progress=progress)
        del cache;gc.collect()
        save_json(args.out/'first_pass_peaks.json',raw)
        report['first_pass_raw_counts']={k:len(v) for k,v in raw.items()}
        common_pool=mixed_pool(raw,args.pool_size)
        report['first_pass_mixed_pool']=common_pool
        # Select contact representatives by continuous rigid score, not native
        # quality or a coarse interpolated surrogate.
        seeds=[]
        with torch.no_grad():
            for row in raw['rigid'][:args.pool_size]:
                ly=rotate_xyz(ligand.xyz,quaternions[row['rotation_index']])+ligand.xyz.new_tensor(row['translation'])
                value=float(direct_pair_energy(prior.reference,ly,rp,lp,config))
                seeds.append(dict(row,direct_rigid_energy=value))
        seeds.sort(key=lambda row:row['direct_rigid_energy'])
        seeds=representative_poses(seeds,4)
        report['adaptation_representatives']=seeds
        np.savez(args.out/'common_reference.npz',theta0=prior.theta0.numpy(),
            receptor_xyz=prior.reference.numpy(),ligand_xyz=ligand.xyz.numpy(),
            receptor_sigma=rp.sigma.numpy(),receptor_epsilon=rp.epsilon.numpy(),receptor_charges=rp.charges.numpy(),
            ligand_sigma=lp.sigma.numpy(),ligand_epsilon=lp.epsilon.numpy(),ligand_charges=lp.charges.numpy())
        report['shared_preparation_search_seconds']=time.monotonic()-start
        gradients=[];masks=[]
        preconditioner_metric=None
        if any(a in args.arms for a in ('contact8','random8')):
            report['stage']='contact_force_preparation';save()
            for row in seeds:
                ly=rotate_xyz(ligand.xyz,quaternions[row['rotation_index']])+ligand.xyz.new_tensor(row['translation'])
                theta=prior.theta0.detach().clone().requires_grad_(True)
                energy=direct_pair_energy(prior.kinematics.coordinates(theta),ly,rp,lp,config)
                gradients.append(torch.autograd.grad(energy,theta)[0].detach())
                masks.append(contact_torsion_mask(prior.kinematics,prior.reference,ly,contact_radius=8.))
            if args.adaptive_preconditioner=='anchor':
                transform,_,diagnostics=prior._variable_transform('anchor_metric')
                inverse=torch.linalg.solve(transform,torch.eye(len(transform),dtype=transform.dtype))
                preconditioner_metric=inverse.T@inverse
                report['adaptive_preconditioner']={**diagnostics,
                    'metric_sha256':hashlib.sha256(preconditioner_metric.numpy().tobytes()).hexdigest(),
                    'scope':'numerical local response preconditioner only; original apo input Jacobian'}
                save()
        for arm in args.arms:
            state=report['arms'][arm]
            arm_start=time.monotonic(); report['stage']=arm;state['status']='running';save()
            try:
                arm_raw=raw
                if arm in ('contact8','random8'):
                    if not gradients: raise ValueError('no contact representatives')
                    state['adaptive_hvp_count']=0
                    def hvp(vector):
                        result=prior.hessian_vector(prior.theta0,vector)
                        state['adaptive_hvp_count']+=1
                        if state['adaptive_hvp_count']%25==0:save()
                        return result
                    adaptive=build_contact_directions(prior.kinematics,prior.theta0,base.angle_basis,
                        torch.stack(gradients),torch.stack(masks),
                        hvp,max_directions=4,preconditioner_metric=preconditioner_metric,
                        regularization=0.,seed=args.seed,method='response' if arm=='contact8' else 'random')
                    state['adaptation']=adaptive.diagnostics;save()
                    if adaptive.angle_basis.shape[1]!=4:
                        raise ValueError('adaptive/random rank shortfall; equal8-mode comparison not met')
                    models[arm]=build_physical_modes(prior,torch.cat((base.angle_basis,adaptive.angle_basis),dim=1))
                    state['basis_construction_seconds']=time.monotonic()-arm_start
                    cache=prepare_typed_cache(prior.reference,rp,probes,grid,config,models[arm],max_bytes=args.max_cache_bytes)
                    state['cache']=cache.diagnostics
                    arm_raw=typed_search(cache,ligand.xyz,lp,probes,quaternions,radius=args.radius,
                                         peaks_per_rotation=4,progress=progress)
                    del cache;gc.collect()
                    save_json(args.out/f'{arm}_second_pass_peaks.json',arm_raw)
                model=models[arm];state['mode_diagnostics']=model.diagnostics
                np.savez(args.out/f'{arm}_model.npz',theta0=prior.theta0.numpy(),
                    reference=model.reference.numpy(),angle_basis=model.angle_basis.numpy(),
                    basis=model.basis.numpy(),stiffness=model.stiffness.numpy(),ca_indices=model.ca_indices.numpy())
                pool=arm_raw['rigid'][:args.pool_size] if arm=='rigid' else mixed_pool(arm_raw,args.pool_size)
                state.update(pool=pool,verification=[],pool_count=len(pool))
                def record(index,row):
                    state['verification'].append(row);save()
                def started(index,row):
                    state['current_verification']={'pool_index':index,'pose':row};save()
                selected=verify_physical_pool(pool,quaternions,ligand.xyz,model,rp,lp,config,
                    constraint='ball' if arm=='ball8' else 'atom',radius=args.radius,max_atom=args.max_atom,
                    rigid=arm=='rigid',record=record,started=started)[:args.top_k]
                state.pop('current_verification',None)
                state.update(candidate_shortfall=max(0,args.top_k-len(selected)),candidates=[])
                for index,candidate in enumerate(selected,1):
                    row={'selection_rank':index,**candidate};state['candidates'].append(row);save()
                    try:
                        ly=rotate_xyz(ligand.xyz,quaternions[row['rotation_index']])+ligand.xyz.new_tensor(row['translation'])
                        paths=export_heavy_pair(args.out/arm/f'initial_{index:03d}',inputs['receptor'],inputs['ligand'],
                            prior.reference[proteins['receptor'].heavy_indices],ly[proteins['ligand'].heavy_indices],centers['receptor'])
                        row['initial_files']={k:str(Path(v).relative_to(args.out)) for k,v in paths.items()}
                        result,rx,ly=refine_physical_pose(candidate,quaternions[row['rotation_index']],ligand.xyz,
                            model,rp,lp,config,flexible=arm!='rigid',radius=args.radius,max_atom=args.max_atom,
                            constraint='ball' if arm=='ball8' else 'atom',
                            max_translation=args.max_translation,max_iterations=args.iterations,
                            max_evaluations=args.max_evaluations,preconditioner=args.refinement_preconditioner)
                        paths=export_heavy_pair(args.out/arm/f'final_{index:03d}',inputs['receptor'],inputs['ligand'],
                            rx[proteins['receptor'].heavy_indices],ly[proteins['ligand'].heavy_indices],centers['receptor'])
                        row.update(result,files={k:str(Path(v).relative_to(args.out)) for k,v in paths.items()})
                        np.savez(args.out/arm/f'final_{index:03d}'/'full_coordinates.npz',receptor=rx.numpy(),ligand=ly.numpy())
                        row['export_complete']=True
                    except Exception as error:
                        row.update(status='exception',export_complete=False,error_type=type(error).__name__,error=str(error))
                        if 'files' in row:row['partial_files']=row.pop('files')
                    save()
                complete=finalize_arm_counts(state,args.top_k)
                state['status']='completed' if complete else 'partial_failure'
            except Exception as error:
                state.update(status='failed',error_type=type(error).__name__,error=str(error))
                if hasattr(error,'diagnostics'):state['error_diagnostics']=error.diagnostics
                if 'current_verification' in state:
                    state['failed_verification']=state.pop('current_verification')
                finalize_arm_counts(state,args.top_k)
            state['seconds']=time.monotonic()-arm_start;save()
        report['status']='completed' if all(report['arms'][a]['status']=='completed' for a in args.arms) else 'partial_failure'
        report['stage']='complete'
    finally:
        if prior is not None:
            report['physical_prior_final_evaluations']=prior._context.evaluations
            prior.close()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--prepared-prior',type=Path,
                    help='Reuse a passed apo-only prior after hashes, forces and geometry are rechecked')
    ap.add_argument('--adaptive-preconditioner',choices=['none','anchor'],default='anchor')
    ap.add_argument('--refinement-preconditioner',choices=['fixed','physical'],default='physical')
    ap.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS))
    for name,default in [('rotations',128),('top-k',5),('pool-size',50),('iterations',8),('max-evaluations',64),
                         ('seed',0),('threads',1),('max-grid-points',750000),('max-cache-bytes',4000000000),
                         ('preparation-iterations',1000),('preparation-evaluations',5000)]:
        ap.add_argument('--'+name,type=int,default=default)
    for name,default in [('spacing',2.),('radius',.5),('max-atom',1.5),('max-translation',4.),('anchor-k',10.),
                         ('stationarity-tolerance',.001)]:ap.add_argument('--'+name,type=float,default=default)
    args=ap.parse_args()
    if len(set(args.arms))!=len(args.arms):ap.error('duplicate arms')
    if args.pool_size<args.top_k:ap.error('pool must contain at least top-k')
    for key,value in vars(args).items():
        if isinstance(value,(int,float)) and (not math.isfinite(value) or value<0 or (key!='seed' and value==0)):
            ap.error(f'invalid {key}')
    if args.seed>=2**32:ap.error('seed out of range')
    args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(args.threads)
    report=dict(status='running',experiment='physical_inducedfit_v1',configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        command=sys.argv,platform=platform.platform(),python=platform.python_version(),torch=torch.__version__,
        dtype='float64',device='CPU',source_sha256=source_hashes(),native_access=False,energy_unit='kJ/mol',
        objective='C2 LJ + dielectric78.5 screened Coulomb cross-partner + stationary ff14SB/OBC2/ACE monomer strain and anchor; hybrid, not binding free energy',
        arms={a:dict(status='pending',requested_candidate_count=args.top_k) for a in args.arms})
    started=time.monotonic()
    repository=Path(__file__).resolve().parents[1]
    for relative in report['source_sha256']:
        target=args.out/'source_snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(repository/relative,target)
    def save():
        report['wall_seconds']=time.monotonic()-started
        usage=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['process_peak_rss_bytes']=usage if sys.platform=='darwin' else usage*1024
        save_json(args.out/'report.json',report)
    save()
    try:run(args,report,save)
    except Exception as error:
        report.update(status='exception',error_type=type(error).__name__,error=str(error))
        if hasattr(error,'diagnostics'):report['failure_diagnostics']=error.diagnostics
        for state in report['arms'].values():
            if state.get('status')=='pending':state['status']='not_attempted'
            finalize_arm_counts(state,args.top_k)
    if source_hashes()!=report['source_sha256']:
        report.update(status='exception',error='Implementation source changed during inference')
    save();print(json.dumps({k:report[k] for k in ('status','stage','wall_seconds','error') if k in report},indent=2))
    return 0 if report['status']=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
