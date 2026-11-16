"""Experimental typed FFT proposals and continuous physical verification.

Coarse-grid energies propose poses, never stand in for continuous pair
energies. All final ranking/refinement uses the declared kJ/mol hybrid
potential. No native coordinates or accuracy labels are accepted.
"""
from dataclasses import dataclass
import math

import torch

from zdock.geom import rotate
from .fields import correlation_spectra, correlate_from_spectra
from .response import QuadraticResponse
from .search import _lag_at
from .types import GridSpec
from .typed_potential import direct_pair_energy
from .typed_fft import (receptor_channels, ligand_channels, jvp_receptor_channels,
                        valid_translation_mask)
from .atom_constraints import AtomwiseResponse
from .relax import _rotate_vector


def rotate_xyz(xyz, quaternion):
    return torch.stack(rotate(*xyz.unbind(-1), quaternion), dim=-1)


def typed_pair_grid(rx, lx, *, spacing=2., atom_margin=1.5, translation_margin=.5, potential_cutoff=12.):
    extent = float(rx.norm(dim=-1).max() + 2 * lx.norm(dim=-1).max())
    extent = max(extent + 2 * spacing + atom_margin + translation_margin + 2.,
                 float(rx.norm(dim=-1).max()) + potential_cutoff + atom_margin + spacing)
    half = math.ceil(extent / spacing)
    return GridSpec((2 * half + 1,) * 3, spacing, (-half * spacing,) * 3)


def prior_gradient(model):
    return model.reference.new_tensor(model.diagnostics['baseline_projected_gradient'])


@dataclass
class TypedCache:
    grid: GridSpec
    rigid: torch.Tensor
    derivatives: tuple
    response: object
    baseline_gradient: torch.Tensor
    diagnostics: dict


def prepare_typed_cache(rx, params, probes, grid, config, model=None, *, max_bytes=4_000_000_000):
    modes = 0 if model is None else model.n_modes
    shape = tuple(2*n-1 for n in grid.shape)
    spectrum_bytes = (probes.n_types+1)*shape[0]*shape[1]*(shape[2]//2+1)*16
    if spectrum_bytes*(modes+1) > max_bytes:
        raise MemoryError(f'typed FFT cache needs {spectrum_bytes*(modes+1)} bytes, budget {max_bytes}')
    with torch.no_grad():
        channels = receptor_channels(rx, params, probes, grid, config)
        rigid = correlation_spectra(channels).detach()
        del channels
    derivatives = []
    if model is not None:
        if not torch.allclose(rx, model.reference, atol=1e-12, rtol=0):
            raise ValueError('typed FFT receptor differs from mode reference')
        for direction in model.basis:
            _, tangent = jvp_receptor_channels(rx, params, probes, grid, config, direction)
            derivatives.append(correlation_spectra(tangent.detach()).detach())
        response = QuadraticResponse(model.stiffness)
        gradient = prior_gradient(model)
    else:
        response, gradient = None, rx.new_zeros(0)
    return TypedCache(grid, rigid, tuple(derivatives), response, gradient,
        dict(retained_fft_bytes=sum(x.numel()*x.element_size() for x in [rigid,*derivatives]),
             channels=probes.n_types+1, modes=modes, score_role='coarse proposal only'))


def _peaks(score, rigid, rotation_index, grid, count, ref, gradients=None, response=None, radius=0.):
    values = score.clone().reshape(-1)
    output = []
    for _ in range(count):
        value, index = values.min(0)
        if not torch.isfinite(value):
            break
        index = int(index)
        translation = _lag_at(index, grid.shape, ref) * grid.spacing
        row = dict(rotation_index=rotation_index, translation=translation.tolist(),
                   coarse_score=float(value), coarse_rigid_energy=float(rigid.reshape(-1)[index]))
        if gradients is not None:
            solved = response.solve(gradients[index], radius)
            row.update(coarse_gain=float(solved.gain), coarse_amplitudes=solved.amplitudes.tolist())
        output.append(row)
        sizes = score.shape
        x, rest = divmod(index, sizes[1]*sizes[2]); y, z = divmod(rest, sizes[2])
        center = [x,y,z]; cells = math.ceil(3./grid.spacing)
        lo = [max(0, c-cells) for c in center]
        hi = [min(sizes[d], center[d]+cells+1) for d in range(3)]
        axes = [torch.arange(lo[d], hi[d])-center[d] for d in range(3)]
        distance2 = sum(v.square() for v in torch.meshgrid(*axes,indexing='ij'))*grid.spacing**2
        values.reshape(sizes)[tuple(slice(lo[d],hi[d]) for d in range(3))].masked_fill_(distance2<=9., torch.inf)
    return output


@torch.no_grad()
def typed_search(cache, lx, lparams, probes, quaternions, *, radius=.5, peaks_per_rotation=4, progress=None):
    """Preserve ALL per-rotation peaks, including those later discarded."""
    rigid_rows, response_rows = [], []
    for index, quaternion in enumerate(quaternions):
        xyz = rotate_xyz(lx, quaternion)
        spectrum = correlation_spectra(ligand_channels(xyz,lparams,probes,cache.grid))
        rigid = correlate_from_spectra(cache.rigid,spectrum,cache.grid.shape)
        valid = valid_translation_mask(xyz,cache.grid)
        if not bool(valid.any()) or not bool(torch.isfinite(rigid[valid]).all()):
            raise FloatingPointError('typed search has no finite valid translations')
        rigid_score = rigid.clone().masked_fill_(~valid,torch.inf)
        rigid_rows += _peaks(rigid_score,rigid,index,cache.grid,peaks_per_rotation,lx)
        if cache.derivatives:
            gradients = torch.stack([correlate_from_spectra(d,spectrum,cache.grid.shape)
                                     for d in cache.derivatives],-1).reshape(-1,len(cache.derivatives))
            gradients += cache.baseline_gradient
            score = rigid_score.clone().reshape(-1)
            for indices in valid.reshape(-1).nonzero().flatten().split(4096):
                score[indices] += cache.response.solve(gradients[indices],radius).gain
            if not bool(torch.isfinite(score[valid.reshape(-1)]).all()):
                raise FloatingPointError('nonfinite typed response score')
            response_rows += _peaks(score.reshape(rigid.shape),rigid,index,cache.grid,
                peaks_per_rotation,lx,gradients,cache.response,radius)
        if progress is not None:
            progress(index+1,len(quaternions))
    key=lambda p:(p['coarse_score'],p['rotation_index'],tuple(p['translation']))
    return dict(rigid=sorted(rigid_rows,key=key),response=sorted(response_rows,key=key))


def pose_key(row):
    return row['rotation_index'],tuple(row['translation'])


def mixed_pool(raw, size):
    """Fixed quotas, unique keys, then deterministic alternating backfill."""
    if size<1:
        raise ValueError('pool size must be positive')
    left,right=raw['rigid'],raw['response']
    selected=[]; seen=set()
    def append(row):
        if pose_key(row) not in seen and len(selected)<size:
            seen.add(pose_key(row)); selected.append(dict(row))
    for rows,quota in ((left,(size+1)//2),(right,size//2)):
        for row in rows[:quota]: append(row)
    for i in range(max(len(left),len(right))):
        for rows in (left,right):
            if i<len(rows): append(rows[i])
    return selected


def representative_poses(rows, count=4, minimum_distance=6.):
    selected=[]
    for row in rows:
        t=torch.tensor(row['translation'])
        if all(row['rotation_index']!=old['rotation_index'] or
               float((t-torch.tensor(old['translation'])).norm())>=minimum_distance for old in selected):
            selected.append(row)
        if len(selected)==count: break
    return selected


def geometry(model,a,radius,max_atom):
    displacement=model.coordinates(a)-model.reference
    ca=float(displacement[model.ca_indices].square().sum(-1).mean().sqrt())
    atom=float(displacement.norm(dim=-1).max())
    return ca<=radius+1e-9 and atom<=max_atom+1e-9,ca,atom


def pair_gradient(model,lx,rparams,lparams,config):
    zero=model.reference.new_zeros(model.n_modes,requires_grad=True)
    energy=direct_pair_energy(model.coordinates(zero),lx,rparams,lparams,config)
    gradient=torch.autograd.grad(energy,zero)[0].detach()+prior_gradient(model)
    return float(energy.detach()),gradient


def verify_physical_pool(pool,quaternions,lx,model,rparams,lparams,config,*,constraint='atom',
                         radius=.5,max_atom=1.5,rigid=False,record=None,started=None):
    if constraint not in {'atom','ball'}: raise ValueError('unknown constraint')
    solver=AtomwiseResponse(model.stiffness,model.basis) if constraint=='atom' else QuadraticResponse(model.stiffness)
    norm=float(torch.linalg.matrix_norm(model.basis.permute(1,2,0),ord=2).max())
    ball=min(radius,max_atom/norm) if norm else radius
    output=[]
    for index,pose in enumerate(pool):
        if started: started(index+1,pose)
        xyz=rotate_xyz(lx,quaternions[pose['rotation_index']])+lx.new_tensor(pose['translation'])
        if rigid:
            with torch.no_grad(): e0=float(direct_pair_energy(model.reference,xyz,rparams,lparams,config))
            gradient=model.reference.new_zeros(model.n_modes)
            amplitude=gradient.clone(); gain=0.; diagnostic={}
        else:
            e0,gradient=pair_gradient(model,xyz,rparams,lparams,config)
            solved=solver.solve(gradient,radius=radius,max_atom_displacement=max_atom) if constraint=='atom' else solver.solve(gradient,ball)
            amplitude=solved.amplitudes; gain=float(solved.gain)
            diagnostic=getattr(solved,'diagnostics',{})
        row=dict(pose,rigid_energy=e0,gradient=gradient.tolist(),quadratic_gain=gain,
                 proposed_amplitudes=amplitude.tolist(),constraint=constraint,solver=diagnostic,
                 verified_energy=e0,verified_gain=0.,verified_amplitudes=[0.]*model.n_modes,
                 fallback=not rigid,trials=[])
        for trial in range(0 if rigid else 6):
            scale=.5**trial; a=scale*amplitude
            feasible,ca,atom=geometry(model,a,radius,max_atom)
            result=dict(scale=scale,ca_rmsd=ca,maximum_atom_displacement=atom,geometry_feasible=feasible)
            row['trials'].append(result)
            if not feasible: continue
            with torch.no_grad():
                strain=float(model.deformation_energy(a))
                # A finite stationary tolerance allows a tiny *unclipped*
                # linear baseline residual. Record its explicit allowance.
                allowance=1e-6+2*abs(float(prior_gradient(model)@a))
                if strain < -allowance:
                    raise ValueError(f'physical prior decreased beyond stationarity allowance: {strain}')
                energy=float(direct_pair_energy(model.coordinates(a),xyz,rparams,lparams,config))+strain
            predicted=float(gradient@a+.5*a@model.stiffness@a)
            actual=energy-e0
            if not all(math.isfinite(x) for x in (strain,energy,predicted,actual)):
                raise FloatingPointError('nonfinite physical verification')
            result.update(strain=strain,strain_negative_allowance=allowance,energy=energy,
                          predicted_gain=predicted,actual_gain=actual)
            if predicted<0 and actual<0 and actual/predicted>=.75:
                row.update(verified_energy=energy,verified_gain=actual,verified_amplitudes=a.tolist(),
                           accepted_scale=scale,fallback=False)
                result['accepted']=True; break
        output.append(row)
        if record: record(index+1,row)
    return sorted(output,key=lambda r:(r['verified_energy'],*pose_key(r)))


def refine_physical_pose(pose,quaternion,lx,model,rparams,lparams,config,*,flexible=True,
                         constraint='atom',radius=.5,max_atom=1.5,max_translation=4.,max_rotation=.15,
                         max_iterations=8,max_evaluations=64,preconditioner='physical'):
    """Projected Armijo descent on the same continuous physical objective."""
    if constraint not in {'atom','ball'}: raise ValueError('unknown constraint')
    if preconditioner not in {'fixed','physical'}: raise ValueError('unknown preconditioner')
    rotated=rotate_xyz(lx,quaternion); center=rotated.mean(0)
    initial_t=lx.new_tensor(pose['translation']); m=model.n_modes if flexible else 0
    amplitude_radius=radius
    if flexible and constraint=='ball':
        norm=float(torch.linalg.matrix_norm(model.basis.permute(1,2,0),ord=2).max())
        amplitude_radius=min(radius,max_atom/norm) if norm else radius
    values=lx.new_zeros(6+m); evaluations=0
    factor=torch.linalg.cholesky(model.stiffness) if m and preconditioner=='physical' else None
    def coordinates(z):
        a=z[6:] if flexible else lx.new_zeros(model.n_modes)
        return model.coordinates(a),_rotate_vector(rotated-center,z[:3])+center+initial_t+z[3:6],a
    def energy(z,gradient):
        nonlocal evaluations
        z=z.detach().requires_grad_(gradient)
        rx,ly,a=coordinates(z)
        result=direct_pair_energy(rx,ly,rparams,lparams,config)+model.deformation_energy(a)
        evaluations+=1
        if not bool(torch.isfinite(result)): raise FloatingPointError('nonfinite refinement energy')
        grad=torch.autograd.grad(result,z)[0].detach() if gradient else None
        if grad is not None and not bool(torch.isfinite(grad).all()): raise FloatingPointError('nonfinite gradient')
        return float(result.detach()),grad
    current,grad=energy(values,True); history=[current]; accepted=0; status='max_iterations'
    def project(z):
        result=z.clone()
        for begin,end,bound in ((0,3,max_rotation),(3,6,max_translation),(6,6+m,amplitude_radius)):
            if end>begin:
                norm=result[begin:end].norm()
                if norm>bound: result[begin:end]*=bound/norm
        return result
    for iteration in range(max_iterations):
        if evaluations>=max_evaluations: status='max_evaluations'; break
        # Physical K is a numerical metric, never an added energy term.
        weights=values.new_tensor([.01]*3+[1.]*3+[.1]*m)
        direction=-weights*grad
        if factor is not None:
            direction[6:]=-torch.cholesky_solve(grad[6:,None],factor).flatten()
        step=1.; improved=False
        while evaluations<max_evaluations and step>=2**-24:
            trial=project(values+step*direction); delta=trial-values
            if float(delta.norm())<1e-12: status='stationary'; break
            if float(grad@delta)>=0: step*=.5; continue
            a=trial[6:] if flexible else lx.new_zeros(model.n_modes)
            feasible,_,_=geometry(model,a,radius,max_atom)
            if not feasible: step*=.5; continue
            with torch.no_grad(): proposed,_=energy(trial,False)
            if proposed<=current+1e-4*float(grad@delta) and proposed<current:
                values=trial; current=proposed; accepted+=1; history.append(current); improved=True; break
            step*=.5
        if not improved:
            if status!='stationary': status='max_evaluations' if evaluations>=max_evaluations else 'line_search_failed'
            break
        if iteration+1<max_iterations and evaluations<max_evaluations:
            current,grad=energy(values,True)
    with torch.no_grad():
        rx,ly,a=coordinates(values); strain=float(model.deformation_energy(a))
        feasible,ca,atom=geometry(model,a,radius,max_atom)
    if not feasible: raise ValueError('final refinement violates nonlinear geometry')
    allowance=1e-6+2*abs(float(prior_gradient(model)@a))
    if strain < -allowance:
        raise ValueError('final refinement prior decreased beyond stationarity allowance')
    return dict(status=status,energy=current,strain=strain,amplitudes=a.tolist(),ca_rmsd=ca,
                strain_negative_allowance=allowance,constraint=constraint,amplitude_radius=amplitude_radius,
                preconditioner=preconditioner,
                maximum_atom_displacement=atom,rotation_increment=values[:3].tolist(),
                translation_increment=values[3:6].tolist(),energy_evaluations=evaluations,
                accepted_steps=accepted,energy_history=history),rx.detach(),ly.detach()
