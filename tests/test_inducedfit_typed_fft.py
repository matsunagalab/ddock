import pytest
import torch
from dataclasses import replace

from zdock.inducedfit.typed_potential import TypedParameters,TypedPotentialConfig,direct_pair_energy
from zdock.inducedfit.typed_fft import (build_probe_types,receptor_channels,ligand_channels,
    jvp_receptor_channels,valid_translation_mask,grid_pair_energy,cardinal_prefilter)
from zdock.inducedfit.fields import correlation_spectra,correlate_from_spectra,direct_correlation_at_shift
from zdock.inducedfit.types import GridSpec

pytestmark=pytest.mark.filterwarnings('ignore:`torch.jit.script` is deprecated.*:DeprecationWarning')


def fixture():
    rp=TypedParameters(torch.tensor([2.4,2.8],dtype=torch.float64),torch.tensor([.2,.3],dtype=torch.float64),torch.tensor([.2,-.3],dtype=torch.float64))
    lp=TypedParameters(torch.tensor([2.2,1.],dtype=torch.float64),torch.tensor([.25,0.],dtype=torch.float64),torch.tensor([-.2,.2],dtype=torch.float64))
    r=torch.tensor([[-1.2,.1,.3],[.4,-.3,-.1]],dtype=torch.float64)
    l=torch.tensor([[2.6,.5,.2],[3.2,1.2,-.4]],dtype=torch.float64)
    grid=GridSpec((19,19,19),1.,(-9.,-9.,-9.))
    return r,l,rp,lp,grid,replace(TypedPotentialConfig(),switch_start=4.,cutoff=5.)


def test_mass_and_charge_conservation_including_zero_lj():
    r,l,rp,lp,grid,c=fixture()
    probes=build_probe_types(lp)
    assert probes.n_types==1 and probes.atom_type.tolist()==[0,-1]
    fields=ligand_channels(l,lp,probes,grid)
    torch.testing.assert_close(fields.sum((1,2,3)),torch.tensor([1.,0.],dtype=torch.float64),atol=2e-15,rtol=0)


@pytest.mark.parametrize('interpolation',['mass','cardinal'])
def test_arbitrary_channel_fft_matches_independent_overlap_and_shifted_grid(interpolation):
    r,l,rp,lp,grid,c=fixture()
    c=replace(c,grid_interpolation=interpolation)
    probes=build_probe_types(lp)
    rc=receptor_channels(r,rp,probes,grid,c)
    lc=ligand_channels(l,lp,probes,grid)
    corr=correlate_from_spectra(correlation_spectra(rc),correlation_spectra(lc),grid.shape)
    for shift in [(0,0,0),(1,-1,2),(-2,1,0)]:
        index=tuple(n-1+t for n,t in zip(grid.shape,shift))
        same=direct_correlation_at_shift(rc,lc,shift)
        moved=(rc*ligand_channels(l+l.new_tensor(shift)*grid.spacing,lp,probes,grid)).sum()
        torch.testing.assert_close(corr[index],same,atol=1e-11,rtol=1e-11)
        torch.testing.assert_close(corr[index],moved,atol=1e-11,rtol=1e-11)
    valid=valid_translation_mask(l,grid)
    assert valid[tuple(n-1 for n in grid.shape)]
    assert not valid[0,0,0]


@pytest.mark.parametrize('offset',[0.,1e-7,-1e-7])
@pytest.mark.parametrize('interpolation',['mass','cardinal'])
def test_jvp_matches_finite_difference_through_grid_cell(offset,interpolation):
    r,l,rp,lp,grid,c=fixture()
    c=replace(c,grid_interpolation=interpolation)
    r=r.clone();r[0,0]=offset
    probes=build_probe_types(lp)
    d=torch.tensor([[.2,-.1,.3],[-.2,.1,.1]],dtype=torch.float64)
    value,tangent=jvp_receptor_channels(r,rp,probes,grid,c,d)
    h=1e-5
    fd=(receptor_channels(r+h*d,rp,probes,grid,c)-receptor_channels(r-h*d,rp,probes,grid,c))/(2*h)
    torch.testing.assert_close(value,receptor_channels(r,rp,probes,grid,c),atol=1e-11,rtol=1e-11)
    torch.testing.assert_close(tangent,fd,atol=1e-6,rtol=3e-6)


@pytest.mark.parametrize('interpolation',['mass','cardinal'])
def test_ligand_assignment_reverse_gradient_matches_finite_difference(interpolation):
    r,l,rp,lp,grid,c=fixture()
    c=replace(c,grid_interpolation=interpolation)
    l.requires_grad_()
    value=grid_pair_energy(r,l,rp,lp,grid,c)
    grad=torch.autograd.grad(value,l)[0]
    d=torch.tensor([[.2,-.1,.3],[-.2,.1,.1]],dtype=torch.float64)
    h=1e-5
    fd=(grid_pair_energy(r,l.detach()+h*d,rp,lp,grid,c)-grid_pair_energy(r,l.detach()-h*d,rp,lp,grid,c))/(2*h)
    torch.testing.assert_close((grad*d).sum(),fd,atol=1e-7,rtol=1e-6)


def test_support_violation_fails_instead_of_clipping_or_renormalizing():
    r,l,rp,lp,grid,c=fixture()
    with pytest.raises(ValueError,match='support'):
        ligand_channels(l+100,lp,build_probe_types(lp),grid)


def test_charge_only_ligand_has_one_channel_and_zero_lj():
    r,l,rp,lp,grid,c=fixture()
    lp=TypedParameters(lp.sigma,torch.zeros_like(lp.epsilon),lp.charges)
    probes=build_probe_types(lp)
    assert probes.n_types==0
    rc=receptor_channels(r,rp,probes,grid,c)
    assert rc.shape[0]==1
    assert torch.isfinite((rc*ligand_channels(l,lp,probes,grid)).sum())


def test_wrong_probe_parameters_are_rejected():
    r,l,rp,lp,grid,c=fixture()
    probes=build_probe_types(lp)
    changed=TypedParameters(lp.sigma+.1,lp.epsilon,lp.charges)
    with pytest.raises(ValueError,match='differ'):
        ligand_channels(l,changed,probes,grid)


def test_cardinal_coefficients_reproduce_nodes_without_periodic_wrap():
    g=torch.Generator().manual_seed(19)
    values=torch.randn((2,21,5,7),generator=g,dtype=torch.float64)
    coefficients=cardinal_prefilter(values)
    reconstructed=coefficients
    for axis in (1,2,3):
        moved=reconstructed.movedim(axis,0)
        zero=torch.zeros_like(moved[:1])
        reconstructed=((2/3)*moved+(1/6)*torch.cat((zero,moved[:-1]))
                       +(1/6)*torch.cat((moved[1:],zero))).movedim(0,axis)
    torch.testing.assert_close(reconstructed,values,atol=4e-15,rtol=4e-15)
    impulse=torch.zeros((1,21,5,7),dtype=torch.float64);impulse[0,0,2,3]=1.
    coeff=cardinal_prefilter(impulse)
    assert float(coeff[0,-1].abs().max())<1e-10
    assert float(coeff[0,1].abs().max())>.1


def test_cardinal_requires_complete_radial_support_and_mass_ablation_is_explicit():
    r,l,rp,lp,grid,c=fixture()
    c=replace(c,grid_interpolation='cardinal')
    r=r+6.
    probes=build_probe_types(lp)
    with pytest.raises(ValueError,match='receptor potential support'):
        receptor_channels(r,rp,probes,grid,c)
    assert torch.isfinite(receptor_channels(r,rp,probes,grid,replace(c,grid_interpolation='mass'))).all()
