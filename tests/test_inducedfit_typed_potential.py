from dataclasses import replace
import itertools
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
import torch

from zdock.inducedfit.typed_potential import (TypedParameters, TypedPotentialConfig,
    lj_potential, charge_kernel, direct_pair_energy, extract_typed_parameters)


def params(sigma=(3.4,), epsilon=(.5,), charges=(0.,)):
    return TypedParameters(*(torch.tensor(x, dtype=torch.float64) for x in (sigma, epsilon, charges)))


def test_all_ff14sb_lj_pairs_preserve_exact_zero_minimum_and_force():
    app = pytest.importorskip('openmm.app')
    p = Path(app.__file__).parent/'data/amber14/protein.ff14SB.xml'
    kinds = sorted({(float(a.attrib['sigma'])*10, float(a.attrib['epsilon']))
                    for a in ET.parse(p).getroot().findall('NonbondedForce/Atom')
                    if float(a.attrib['epsilon']) > 0})
    assert len(kinds) == 13
    pairs = list(itertools.combinations_with_replacement(kinds, 2))
    sigma = torch.tensor([(a[0]+b[0])/2 for a,b in pairs], dtype=torch.float64)
    eps = torch.tensor([math.sqrt(a[1]*b[1]) for a,b in pairs], dtype=torch.float64)
    r = (2**(1/6)*sigma).requires_grad_()
    e = lj_potential(r.square(), sigma, eps)
    force = -torch.autograd.grad(e.sum(),r)[0]
    torch.testing.assert_close(e, -eps, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(force, torch.zeros_like(force), atol=1e-11, rtol=0)
    torch.testing.assert_close(lj_potential(sigma.square(), sigma, eps), torch.zeros_like(e), atol=1e-12, rtol=0)


@pytest.mark.parametrize('ratio', [.5, .899, .901, 1., 1.3, 2.])
def test_direct_lj_gradient_matches_finite_difference(ratio):
    p = params()
    r = torch.tensor([[0., 0., 0.]], dtype=torch.float64, requires_grad=True)
    l = torch.tensor([[ratio*3.4, .17, -.09]], dtype=torch.float64)
    f = direct_pair_energy(r,l,p,p)
    grad = torch.autograd.grad(f,r)[0]
    h=1e-5
    shift=torch.tensor([[h,0.,0.]],dtype=torch.float64)
    fd=(direct_pair_energy(r.detach()+shift,l,p,p)-direct_pair_energy(r.detach()-shift,l,p,p))/(2*h)
    torch.testing.assert_close(grad[0,0],fd,atol=2e-6,rtol=2e-7)


def derivatives(kernel, location):
    r = torch.tensor(location, dtype=torch.float64, requires_grad=True)
    v=kernel(r.square())
    g=torch.autograd.grad(v,r,create_graph=True)[0]
    h=torch.autograd.grad(g,r)[0]
    return torch.stack((v.detach(),g.detach(),h.detach()))


@pytest.mark.parametrize('kind,join', [('lj',3.06),('charge',.5),('lj',10.),('lj',12.),('charge',10.),('charge',12.)])
def test_joins_are_c2(kind,join):
    p=params()
    kernel=(lambda d:lj_potential(d,p.sigma[0],p.epsilon[0])) if kind=='lj' else charge_kernel
    left,right=derivatives(kernel,join-1e-8),derivatives(kernel,join+1e-8)
    torch.testing.assert_close(left,right,atol=2e-4,rtol=2e-6)


def test_exact_overlap_is_finite_and_zero_lj_is_exactly_zero():
    sigma=torch.tensor([0.,3.4],dtype=torch.float64)
    eps=torch.tensor([0.,.5],dtype=torch.float64)
    d=torch.zeros(2,dtype=torch.float64,requires_grad=True)
    v=lj_potential(d,sigma,eps)+charge_kernel(d)
    assert bool(torch.isfinite(v).all())
    assert bool(torch.isfinite(torch.autograd.grad(v.sum(),d)[0]).all())
    assert lj_potential(d,sigma,torch.zeros_like(eps)).abs().max()==0
    x=torch.zeros((1,3),dtype=torch.float64,requires_grad=True)
    e=direct_pair_energy(x,x.detach(),params(charges=(1.,)),params(charges=(-1.,)))
    assert torch.isfinite(e)
    torch.testing.assert_close(torch.autograd.grad(e,x)[0],torch.zeros_like(x),atol=0,rtol=0)


def test_translation_rotation_invariance_and_chunk_parity():
    g=torch.Generator().manual_seed(17)
    r=torch.randn((4,3),generator=g,dtype=torch.float64)
    l=torch.randn((3,3),generator=g,dtype=torch.float64)+4.
    rp=params((2.,3.,3.4,1.),(.2,.3,.4,0.),(.2,-.4,.3,.1))
    lp=params((3.1,2.7,1.),(.3,.2,0.),(.2,-.3,.1))
    rotation=torch.linalg.qr(torch.randn((3,3),generator=g,dtype=torch.float64))[0]
    e=direct_pair_energy(r,l,rp,lp)
    torch.testing.assert_close(e,direct_pair_energy(r@rotation+8,l@rotation+8,rp,lp),atol=1e-11,rtol=1e-12)
    torch.testing.assert_close(e,direct_pair_energy(r,l,rp,lp,replace(TypedPotentialConfig(),atom_chunk_size=1)))


def checkpoint_fixture():
    generator=torch.Generator().manual_seed(41)
    r=torch.randn((24,3),generator=generator,dtype=torch.float64)
    l=torch.randn((19,3),generator=generator,dtype=torch.float64)+3
    rp=params((3.4,)*24,(.5,)*24,(.2,)*24)
    lp=params((2.8,)*19,(.3,)*19,(-.1,)*19)
    return r,l,rp,lp


def test_checkpoint_multichunk_energy_gradients_and_hvp_match():
    r,l,rp,lp=checkpoint_fixture()
    results=[]
    for enabled in (False,True):
        rx=r.clone().requires_grad_();lx=l.clone().requires_grad_()
        config=replace(TypedPotentialConfig(),atom_chunk_size=8,checkpoint_direct=enabled)
        energy=direct_pair_energy(rx,lx,rp,lp,config)
        gradients=torch.autograd.grad(energy,(rx,lx),create_graph=True)
        hvp=torch.autograd.grad(gradients[0][:,0].sum(),(rx,lx))
        results.append((energy.detach(),*[x.detach() for x in gradients],*hvp))
    for ordinary,checkpointed in zip(*results):
        torch.testing.assert_close(ordinary,checkpointed,atol=1e-10,rtol=1e-12)


@pytest.mark.parametrize('requires_grad',[False,True])
@pytest.mark.filterwarnings('ignore:`torch.jit.script` is deprecated:DeprecationWarning')
def test_direct_jvp_bypasses_checkpoint_and_matches_finite_difference(requires_grad):
    r,l,rp,lp=checkpoint_fixture()
    r.requires_grad_(requires_grad);l.requires_grad_(requires_grad)
    dr=torch.ones_like(r)*.11;dl=torch.ones_like(l)*-.13
    config=replace(TypedPotentialConfig(),atom_chunk_size=8)
    function=lambda rx,lx:direct_pair_energy(rx,lx,rp,lp,config)
    value,tangent=torch.func.jvp(function,(r,l),(dr,dl))
    h=1e-5
    with torch.no_grad():
        difference=(function(r+h*dr,l+h*dl)-function(r-h*dr,l-h*dl))/(2*h)
    torch.testing.assert_close(value,function(r,l),atol=1e-10,rtol=1e-12)
    torch.testing.assert_close(tangent,difference,atol=1e-6,rtol=1e-7)


def test_checkpoint_saves_vectors_and_coordinates_not_all_pair_intermediates():
    r,l,rp,lp=checkpoint_fixture()
    r.requires_grad_();l.requires_grad_()
    saved=[]
    def pack(value):
        saved.append(value.shape)
        return value
    config=replace(TypedPotentialConfig(),atom_chunk_size=8)
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda x:x):
        energy=direct_pair_energy(r,l,rp,lp,config)
    assert saved and all(shape in (torch.Size([0]),torch.Size([]),torch.Size([24,3]),torch.Size([8,3]),
                                  torch.Size([19,3]),torch.Size([8]),torch.Size([19])) for shape in saved)
    gradients=torch.autograd.grad(energy,(r,l))
    assert all(torch.isfinite(x).all() for x in gradients)


def test_screened_charge_has_explicit_physical_units():
    d=torch.tensor(16.,dtype=torch.float64)
    expected=1389.35456/78.5*math.exp(-.4)/4
    assert float(charge_kernel(d))==pytest.approx(expected,rel=1e-13)


def test_extract_ff14sb_keeps_identity_parameters_and_original_coordinates(tmp_path):
    pytest.importorskip('openmm')
    from test_inducedfit_chemistry import make_dipeptide_pdb
    from zdock.inducedfit.pdb_input import read_pdb_monomer
    from zdock.inducedfit.chemistry import prepare_amber_charges
    protein=prepare_amber_charges(read_pdb_monomer(make_dipeptide_pdb(tmp_path/'input.pdb'),chain='A',charge_policy='zero_diagnostic'))
    xyz=protein.monomer.xyz.clone()
    p=extract_typed_parameters(protein)
    assert len(p.sigma)==20 and p.diagnostics['n_exceptions']>0
    torch.testing.assert_close(p.charges,protein.monomer.charges,atol=1e-12,rtol=0)
    assert torch.equal(xyz,protein.monomer.xyz)
    assert bool((p.epsilon>=0).all()) and bool((p.sigma>0).all())


@pytest.mark.parametrize('change',[{'dielectric':0.},{'cutoff':9.},{'lj_core_ratio':1.},{'kappa':-1.}])
def test_invalid_configuration_rejected(change):
    with pytest.raises(ValueError):
        direct_pair_energy(torch.zeros((1,3),dtype=torch.float64),torch.ones((1,3),dtype=torch.float64),params(),params(),replace(TypedPotentialConfig(),**change))
