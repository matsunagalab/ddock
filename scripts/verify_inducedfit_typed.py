#!/usr/bin/env python3
"""Native-free fixed ff14SB LJ type-pair and grid-assignment verification.

The 91-pair interpolation experiment evaluates the actual cubic assignment.
Cardinal interpolation builds the full zero-padded potential/coefficient grid;
the mass-assignment ablation samples just the 64 required nodes. FFT-versus-
grid-sum/JVP correctness is covered by the independent small test suite.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
from pathlib import Path
import shutil
import sys
import time
import xml.etree.ElementTree as ET

import torch

from zdock.inducedfit.types import GridSpec
from zdock.inducedfit.typed_potential import TypedPotentialConfig, lj_potential
from zdock.inducedfit.typed_fft import assignment_stencil, cardinal_prefilter


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_pair_samples(sigma, epsilon, spacing, phase, config):
    # Full cutoff support and zero exterior: exactly the same BC as the
    # production cardinal coefficient grid, without unrelated charge channel.
    half=math.ceil((config.cutoff+spacing)/spacing)
    grid=GridSpec(tuple([2*half+1]*3),spacing,tuple([-half*spacing]*3))
    rx=torch.tensor(phase,dtype=torch.float64)*spacing
    coefficients=None
    if config.grid_interpolation=='cardinal':
        with torch.no_grad():
            values=lj_potential((grid.points(dtype=torch.float64)-rx).square().sum(-1),sigma,epsilon,config)
            coefficients=cardinal_prefilter(values.reshape(1,*grid.shape)).reshape(-1)
    return grid,rx,coefficients


def sampled_pair_energy(distances, sigma, epsilon, spacing, phase, config, samples=None):
    grid,rx,coefficients=prepare_pair_samples(sigma,epsilon,spacing,phase,config) if samples is None else samples
    lx=rx[None,:]+torch.stack((distances,torch.zeros_like(distances),torch.zeros_like(distances)),dim=-1)
    index,weights=assignment_stencil(lx,grid)
    if coefficients is not None:
        return (weights*coefficients[index]).sum(-1)
    iz=index%grid.shape[2]
    iy=(index//grid.shape[2])%grid.shape[1]
    ix=index//(grid.shape[1]*grid.shape[2])
    nodes=torch.stack((ix,iy,iz),dim=-1).to(torch.float64)*spacing+torch.tensor(grid.origin)
    d2=(nodes-rx).square().sum(-1)
    return (weights*lj_potential(d2,sigma,epsilon,config)).sum(-1)


def run(args, result):
    from openmm import app
    torch.set_num_threads(1)
    xml=Path(app.__file__).parent/'data/amber14/protein.ff14SB.xml'
    kinds=sorted({(float(a.attrib['sigma'])*10,float(a.attrib['epsilon']))
                  for a in ET.parse(xml).getroot().findall('NonbondedForce/Atom')})
    active=[x for x in kinds if x[1]>0]
    config=TypedPotentialConfig(grid_interpolation=args.interpolation)
    phases=list(itertools.product([0.,.5],repeat=3))
    result.update(forcefield_xml=str(xml),forcefield_xml_sha256=sha(xml),parameters=config.to_dict(),
        n_parameter_types=len(kinds),n_nonzero_lj_types=len(active),n_zero_lj_types=len(kinds)-len(active),
        registration=dict(spacings=args.spacings,phases_fractional=phases,
            radial_interval=f'max(0,r_min-{args.minimum_scan_half_width}) through r_min+{args.minimum_scan_half_width} Angstrom',
            radial_step_angstrom=args.radial_step,minimum_distance_gate_angstrom=.1,
            full_radial_scan='0 through min(11 Angstrom,r_min+3 Angstrom), step<=.01 Angstrom',
            maximum_extra_well_depth_fraction_gate=.05,
            gate_applies_to_spacings_at_or_below=.5,seed=None,native_access=False,
            parameter_selection='fixed before measurement; no fitted parameters',
            scope='single LJ pair interpolation; no PPI accuracy or solvation validation'),
        analytic_rows=[],grid_rows=[])
    (args.out/'registration.json').write_text(json.dumps(result,indent=2)+'\n')
    start=time.monotonic()
    for first,(s1,e1) in enumerate(active):
        for second in range(first,len(active)):
            s2,e2=active[second]
            sigma=torch.tensor((s1+s2)/2,dtype=torch.float64)
            epsilon=torch.tensor((e1*e2)**.5,dtype=torch.float64)
            minimum=float(2**(1/6)*sigma)
            r=torch.tensor(minimum,dtype=torch.float64,requires_grad=True)
            value=lj_potential(r.square(),sigma,epsilon,config)
            gradient=torch.autograd.grad(value,r)[0]
            zero=float(lj_potential(sigma.square(),sigma,epsilon,config))
            analytic=dict(type_i=first,type_j=second,sigma_angstrom=float(sigma),epsilon_kj_per_mol=float(epsilon),
                minimum_angstrom=minimum,zero_energy=zero,minimum_energy_error=float(value.detach()+epsilon),
                force_at_minimum=-float(gradient))
            analytic['passed']=abs(zero)<1e-10 and abs(analytic['minimum_energy_error'])<1e-10 and abs(float(gradient))<1e-10
            result['analytic_rows'].append(analytic)
            for spacing in args.spacings:
                for phase in phases:
                    samples=prepare_pair_samples(sigma,epsilon,spacing,phase,config)
                    left=max(0.,minimum-args.minimum_scan_half_width)
                    right=minimum+args.minimum_scan_half_width
                    steps=int(round((right-left)/args.radial_step))+1
                    distances=torch.linspace(left,right,steps,dtype=torch.float64)
                    with torch.no_grad():
                        energies=sampled_pair_energy(distances,sigma,epsilon,spacing,phase,config,samples)
                        wide=torch.linspace(0.,min(11.,minimum+3),math.ceil(min(11.,minimum+3)/.01)+1,dtype=torch.float64)
                        wide_energy=sampled_pair_energy(wide,sigma,epsilon,spacing,phase,config,samples)
                        wide_index=int(wide_energy.argmin())
                    selected=int(energies.argmin())
                    actual=float(distances[selected])
                    location=torch.tensor([minimum],dtype=torch.float64,requires_grad=True)
                    sampled=sampled_pair_energy(location,sigma,epsilon,spacing,phase,config,samples)
                    force=-float(torch.autograd.grad(sampled.sum(),location)[0])
                    row=dict(type_i=first,type_j=second,spacing=spacing,phase_x=phase[0],phase_y=phase[1],phase_z=phase[2],
                        analytic_minimum_angstrom=minimum,sampled_minimum_angstrom=actual,
                        minimum_distance_error_angstrom=actual-minimum,sampled_minimum_energy=float(energies[selected]),
                        force_at_analytic_minimum_kj_per_mol_angstrom=force,
                        minimum_at_scan_boundary=selected in (0,len(distances)-1),
                        global_sampled_minimum_angstrom=float(wide[wide_index]),
                        global_sampled_minimum_energy=float(wide_energy[wide_index]),
                        maximum_extra_well_depth_fraction=max(0.,-float(wide_energy[wide_index]/epsilon)-1))
                    row['distance_gate_passed']=abs(actual-minimum)<=.1+1e-12 and not row['minimum_at_scan_boundary']
                    row['depth_gate_passed']=row['maximum_extra_well_depth_fraction']<=.05
                    result['grid_rows'].append(row)
    zero=torch.tensor([0.],dtype=torch.float64)
    result['zero_lj_max_abs_energy']=float(lj_potential(torch.tensor([0.,1.,10.,100.],dtype=torch.float64),zero,zero).abs().max())
    result['summary']={}
    for spacing in args.spacings:
        rows=[r for r in result['grid_rows'] if r['spacing']==spacing]
        result['summary'][str(spacing)]=dict(n=len(rows),distance_gate_pass_count=sum(r['distance_gate_passed'] for r in rows),
            boundary_minimum_count=sum(r['minimum_at_scan_boundary'] for r in rows),
            maximum_absolute_minimum_distance_error_angstrom=max(abs(r['minimum_distance_error_angstrom']) for r in rows),
            depth_gate_pass_count=sum(r['depth_gate_passed'] for r in rows),
            maximum_extra_well_depth_fraction=max(r['maximum_extra_well_depth_fraction'] for r in rows),
            maximum_absolute_force_at_analytic_minimum=max(abs(r['force_at_analytic_minimum_kj_per_mol_angstrom']) for r in rows))
    result['analytic_passed']=all(r['passed'] for r in result['analytic_rows']) and result['zero_lj_max_abs_energy']==0
    fine_rows=[r for r in result['grid_rows'] if r['spacing']<=.5]
    result['fine_grid_gate_applicable']=bool(fine_rows)
    result['fine_grid_distance_gate_passed']=all(r['distance_gate_passed'] for r in fine_rows) if fine_rows else None
    result['fine_grid_depth_gate_passed']=all(r['depth_gate_passed'] for r in fine_rows) if fine_rows else None
    result.update(status='completed',measurement_seconds=time.monotonic()-start,
        failure_count=0,skip_count=0,
        interpretation='Completed measurements do not mean all interpolation gates passed; retain every failed spacing/phase.')
    for name,rows in [('analytic_pairs',result['analytic_rows']),('grid_pairs',result['grid_rows'])]:
        with (args.out/(name+'.csv')).open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator='\n')
            writer.writeheader();writer.writerows(rows)
    if args.summary:
        args.summary.parent.mkdir(parents=True,exist_ok=True)
        with args.summary.open('w',newline='') as handle:
            rows=[dict(spacing_angstrom=spacing,**values) for spacing,values in result['summary'].items()]
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator='\n')
            writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--summary',type=Path)
    parser.add_argument('--interpolation',choices=['mass','cardinal'],default='mass')
    parser.add_argument('--spacings',type=float,nargs='+',default=[1.,.5,.25])
    parser.add_argument('--minimum-scan-half-width',type=float,default=.8)
    parser.add_argument('--radial-step',type=float,default=.005)
    args=parser.parse_args()
    if (not all(math.isfinite(x) and x>0 for x in args.spacings)
            or len(set(args.spacings))!=len(args.spacings)
            or not 0<args.minimum_scan_half_width<=4 or not 0<args.radial_step<=.05):
        parser.error('require distinct positive spacings, half width in (0,4], radial step in (0,.05]')
    args.out.mkdir(parents=True,exist_ok=False)
    if args.summary and args.summary.exists():
        parser.error('summary exists; do not overwrite an earlier measurement')
    sources=[Path(__file__),Path(__file__).resolve().parents[1]/'src/zdock/inducedfit/typed_potential.py',
             Path(__file__).resolve().parents[1]/'src/zdock/inducedfit/typed_fft.py']
    result=dict(status='running',command=sys.argv,platform=platform.platform(),python=platform.python_version(),
                torch=torch.__version__,threads_policy='torch.set_num_threads(1); BLAS pools not independently measured',
                source_sha256={p.name:sha(p) for p in sources},seed=None,loss=None,checkpoint_selection=None)
    for p in sources: shutil.copy2(p,args.out/('executed_'+p.name))
    try:
        run(args,result)
    except Exception as error:
        result.update(status='failed',error_type=type(error).__name__,error=str(error))
    (args.out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['status','analytic_passed','fine_grid_distance_gate_passed','measurement_seconds','summary','error'] if k in result},indent=2))
    return 0 if result['status']=='completed' and result['analytic_passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
