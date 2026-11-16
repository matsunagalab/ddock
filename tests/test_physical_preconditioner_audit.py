import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def audit_module():
    path = Path(__file__).resolve().parents[1] / 'scripts/audit_physical_preconditioners.py'
    spec = importlib.util.spec_from_file_location('physical_audit_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_principal_angle_resolves_microradian_spaces(audit_module):
    angle = 1e-8
    before = np.eye(4)[:, :2]
    after = before.copy()
    after[1, 1], after[2, 1] = np.cos(angle), np.sin(angle)
    _, result = audit_module.subspace_alignment(before, after)
    assert result['maximum_principal_angle_degrees'] == pytest.approx(np.degrees(angle), rel=1e-8)
    assert result['projector_frobenius_distance'] == pytest.approx(np.sqrt(2.) * np.sin(angle), rel=1e-8)


@pytest.mark.parametrize('physical_scale,expected_energy_difference', [(1., 0.), (2., .5)])
def test_basis_rotation_preserves_physics_but_changed_K_is_detected(audit_module, tmp_path,
                                                                  physical_scale, expected_energy_difference):
    angle = .37
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    basis = np.zeros((2, 2, 3))
    basis[0, 0, 0] = basis[1, 1, 1] = np.sqrt(2.)
    angles = np.array([[1., 0.], [0., 2.], [.5, .7]])
    stiffness = np.diag([2., 5.])
    common = dict(theta0=np.zeros(3), reference=np.zeros((2, 3)), ca_indices=np.arange(2))
    before, after = tmp_path / 'before.npz', tmp_path / 'after.npz'
    np.savez(before, **common, basis=basis, angle_basis=angles, stiffness=stiffness)
    np.savez(after, **common, basis=np.einsum('ij,inc->jnc', rotation, basis),
             angle_basis=angles @ rotation, stiffness=physical_scale * rotation.T @ stiffness @ rotation)
    transform, result = audit_module.compare_models(before, after)
    np.testing.assert_allclose(transform, rotation, atol=1e-14)
    assert result['ca_subspace']['maximum_principal_angle_degrees'] < 1e-12
    assert result['all_atom_tangent_relative_frobenius'] < 1e-14
    assert result['torsion_tangent_relative_frobenius'] < 1e-14
    assert result['stiffness']['maximum_relative_quadratic_energy_difference'] == pytest.approx(
        expected_energy_difference, abs=1e-13)
