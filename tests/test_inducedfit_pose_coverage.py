"""Guard the evaluation-only frame algebra without native structures."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from zdock.geom import rotate


@pytest.fixture
def coverage(monkeypatch):
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location('coverage_diagnostic',scripts/'diagnose_inducedfit_pose_coverage.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quaternion_row_matches_production_convention(coverage):
    rng=np.random.default_rng(23)
    points=rng.normal(size=(20,3))
    quaternions=[np.array([0.,0.,0.,1.])]
    for _ in range(16):
        q=rng.normal(size=4)
        quaternions.append(q/np.linalg.norm(q))
    for q in quaternions:
        actual=torch.stack(rotate(*torch.tensor(points).unbind(-1),torch.tensor(q)),dim=-1).numpy()
        np.testing.assert_allclose(points@coverage.quaternion_row(q),actual,atol=1e-12,rtol=1e-12)


@pytest.mark.parametrize('invalid',[[0.,0.,0.,0.],[0.,0.,0.,2.],[0.,0.,np.nan,1.]])
def test_invalid_quaternion_cannot_masquerade_as_identity(coverage,invalid):
    with pytest.raises(ValueError,match='finite and unit length'):
        coverage.quaternion_row(invalid)


def test_noncommuting_apo_fits_compose_to_original_receptor_frame(coverage):
    # Known x and z rotations, deliberately noncommuting and with nonzero centers.
    ax,az=.73,-1.11
    rr=np.array([[1.,0.,0.],[0.,np.cos(ax),-np.sin(ax)],[0.,np.sin(ax),np.cos(ax)]])
    rl=np.array([[np.cos(az),-np.sin(az),0.],[np.sin(az),np.cos(az),0.],[0.,0.,1.]])
    tr=np.array([3.,-2.,7.]); tl=np.array([-11.,4.,2.])
    cr=np.array([8.,3.,-5.]); cl=np.array([-6.,9.,4.])
    x=np.random.default_rng(17).normal(size=(13,3))
    r,t=coverage.compose_apo_relative_pose(rr,tr,rl,tl,cr,cl)
    native_ligand=x@rl+tl
    expected=(native_ligand-tr)@np.linalg.inv(rr)-cr
    np.testing.assert_allclose((x-cl)@r+t,expected,atol=1e-12,rtol=1e-12)
    assert np.linalg.norm(rr.T@rl-r)>.1


def test_kabsch_recovers_known_proper_transform(coverage):
    rng=np.random.default_rng(29)
    x=rng.normal(size=(19,3))
    q=rng.normal(size=4); q/=np.linalg.norm(q)
    expected=coverage.quaternion_row(q)
    shift=np.array([3.,-8.,1.])
    r,t=coverage.fit_row(x,x@expected+shift)
    np.testing.assert_allclose(r,expected,atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(t,shift,atol=1e-12,rtol=1e-12)


def test_kabsch_never_uses_reflection_to_hide_shape_error(coverage):
    x=np.random.default_rng(31).normal(size=(23,3))
    reflection=np.diag([-1.,1.,1.])
    y=x@reflection+np.array([4.,1.,-7.])
    r,t=coverage.fit_row(x,y)
    np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-12)
    assert np.linalg.det(r)==pytest.approx(1.,abs=1e-12)
    assert coverage.rmsd(x@r+t,y)>.1


def test_collinear_fit_is_rejected(coverage):
    x=np.arange(15,dtype=float).reshape(5,3)
    with pytest.raises(ValueError,match='Collinear'):
        coverage.fit_row(x,x+7.)
