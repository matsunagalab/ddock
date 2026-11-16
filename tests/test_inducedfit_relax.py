"""Nonlinear geometry, feasibility, and energy checks for local relaxation."""

import math

import pytest
import torch

from zdock.inducedfit.fields import FieldParameters, build_fields, physical_energy
from zdock.inducedfit.relax import _rotate_vector, relax_pose
from zdock.inducedfit.types import AtomKey, GridSpec, MonomerInput


class HingeModel:
    """Independent analytic one-angle geometry for solver verification."""

    n_modes = 1

    def __init__(self, reference):
        self.reference = reference
        self.ca_indices = torch.arange(len(reference))

    def coordinates(self, a):
        angle = math.sqrt(2) * a[0]
        x, y, z = self.reference.unbind(-1)
        moved = torch.stack((x, angle.cos() * y - angle.sin() * z,
                             angle.sin() * y + angle.cos() * z), -1)
        return torch.where((torch.arange(len(x)) >= 2)[:, None], moved, self.reference)

    def deformation_energy(self, a):
        return .5 * (a * a).sum()


@pytest.fixture
def setup():
    dtype = torch.float64

    def monomer(chain, coordinates, charges):
        xyz = torch.tensor(coordinates, dtype=dtype)
        return MonomerInput(xyz, torch.full((len(xyz),), .6, dtype=dtype),
                            torch.tensor(charges, dtype=dtype),
                            tuple(AtomKey(chain, i + 1, "", "SYN", "X") for i in range(len(xyz))))

    rec = monomer("R", [[-1, 0, 0], [0, 0, 0], [1, 1, 0], [2, 1, 0]], [.2, 0, -.2, .1])
    lig = monomer("L", [[-.4, 2, 1], [.4, 2, 1]], [-.3, .2])
    grid = GridSpec((15, 15, 15), 1., (-7., -7., -7.))
    params = FieldParameters(surface_width=.3, charge_sigma=.45,
                             screening_switch_start=2., screening_cutoff=3.,
                             repulsion_weight=3., charge_weight=.2)
    q = torch.tensor([0., 0., 0., 1.], dtype=dtype)
    t = torch.zeros(3, dtype=dtype)
    return rec, lig, HingeModel(rec.xyz), grid, params, q, t


def test_rotation_increment_has_finite_correct_derivative_at_identity():
    points = torch.tensor([[1., 2., -1.], [0., -1., 2.]], dtype=torch.float64)
    omega = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    jac = torch.autograd.functional.jacobian(lambda w: _rotate_vector(points, w), omega)
    assert torch.isfinite(jac).all()
    for i in range(3):
        unit = torch.eye(3, dtype=torch.float64)[i]
        expected = torch.linalg.cross(unit.expand_as(points), points, dim=-1)
        torch.testing.assert_close(jac[..., i], expected)


def test_nonlinear_relaxation_lowers_actual_energy_and_preserves_hinge_geometry(setup):
    rec, lig, model, grid, params, q, t = setup
    result = relax_pose(rec, lig, model, grid, params, q, t,
                        radius=.1, max_iterations=12, max_translation=0., max_rotation=0.)
    assert result.accepted_steps > 0
    assert result.energy_history[-1] < result.energy_history[0]
    assert all(b <= a + 1e-12 for a, b in zip(result.energy_history, result.energy_history[1:]))
    assert result.ca_rmsd <= .1 + 1e-10
    assert result.amplitudes.norm() <= .1 + 1e-12
    initial_lengths = torch.linalg.vector_norm(rec.xyz[1:] - rec.xyz[:-1], dim=-1)
    final_lengths = torch.linalg.vector_norm(result.receptor_xyz[1:] - result.receptor_xyz[:-1], dim=-1)
    torch.testing.assert_close(initial_lengths, final_lengths, atol=1e-12, rtol=0)
    rf = build_fields(result.receptor_xyz, rec.radii, rec.charges, grid, params)
    lf = build_fields(result.ligand_xyz, lig.radii, lig.charges, grid, params)
    independent = physical_energy(rf, lf, grid, params, direct=True) + model.deformation_energy(result.amplitudes)
    assert result.energy == pytest.approx(float(independent), abs=1e-12)


def test_rigid_and_zero_radius_leave_receptor_fixed(setup):
    rec, lig, model, grid, params, q, t = setup
    for kwargs in (dict(radius=.1, flexible=False), dict(radius=0.)):
        result = relax_pose(rec, lig, model, grid, params, q, t, max_iterations=3, **kwargs)
        torch.testing.assert_close(result.receptor_xyz, rec.xyz, atol=0, rtol=0)
        assert result.energy <= result.energy_history[0]


def test_evaluation_budget_and_bad_initial_state_are_explicit(setup):
    rec, lig, model, grid, params, q, t = setup
    result = relax_pose(rec, lig, model, grid, params, q, t, radius=.1, max_evaluations=1)
    assert result.energy_evaluations == 1
    assert result.status in ("evaluation_budget", "converged_gradient")
    assert result.accepted_steps == 0
    with pytest.raises(ValueError, match="bounds"):
        relax_pose(rec, lig, model, grid, params, q, t, radius=-.1)
    with pytest.raises(ValueError, match="within the radius"):
        relax_pose(rec, lig, model, grid, params, q, t, radius=.1, initial_amplitudes=q.new_tensor([.2]))
    with pytest.raises(ValueError, match="support"):
        relax_pose(rec, lig, model, grid, params, q, t + 100, radius=.1)
