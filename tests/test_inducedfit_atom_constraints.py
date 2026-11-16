import types

import numpy as np
import pytest
import torch

from zdock.inducedfit.atom_constraints import AtomwiseConvergenceError, AtomwiseResponse


DTYPE = torch.float64


def test_one_dimensional_active_atom_constraint_matches_analytic_solution():
    solver = AtomwiseResponse(torch.tensor([[2.]], dtype=DTYPE), torch.tensor([[[1., 0., 0.]]], dtype=DTYPE))
    result = solver.solve(torch.tensor([4.], dtype=DTYPE), 1., .2)
    torch.testing.assert_close(result.amplitudes, torch.tensor([-.2], dtype=DTYPE), atol=1e-9, rtol=0)
    assert float(result.gain) == pytest.approx(-.76, abs=1e-10)
    assert float(result.strain) == pytest.approx(.04, abs=1e-10)
    assert result.diagnostics['status'] == 'certified'
    assert result.diagnostics['stationarity_relative'] < 1e-8
    assert result.diagnostics['finite_geometry_verified'] is False


def test_anisotropy_recovers_motion_lost_by_the_global_conservative_ball():
    basis = torch.zeros((2, 2, 3), dtype=DTYPE)
    basis[0, 0, 0], basis[1, 1, 1] = 10., 1.
    result = AtomwiseResponse(torch.eye(2, dtype=DTYPE), basis).solve(torch.tensor([0., -2.], dtype=DTYPE), 1., .5)
    torch.testing.assert_close(result.amplitudes, torch.tensor([0., .5], dtype=DTYPE), atol=1e-8, rtol=0)
    assert result.diagnostics['conservative_ball_radius'] == pytest.approx(.05)
    assert float(result.gain) == pytest.approx(-.875, abs=1e-8)
    assert result.diagnostics['gain_minus_conservative_reference'] < -.7
    assert result.diagnostics['maximum_linear_atom_displacement'] <= .5 + 1e-9


def test_inactive_atom_bounds_reduce_to_the_unconstrained_spd_solution():
    matrix = torch.tensor([[4., .4], [.4, 2.]], dtype=DTYPE)
    gradient = torch.tensor([.2, -.3], dtype=DTYPE)
    basis = torch.ones((2, 3, 3), dtype=DTYPE) * .1
    result = AtomwiseResponse(matrix, basis).solve(gradient, 1., 2.)
    torch.testing.assert_close(result.amplitudes, -torch.linalg.solve(matrix, gradient), atol=1e-8, rtol=1e-8)
    assert result.diagnostics['active_constraint_indices'] == []


def test_mode_rotation_does_not_change_the_physical_solution():
    generator = torch.Generator().manual_seed(22)
    basis = torch.randn(3, 6, 3, generator=generator, dtype=DTYPE)
    factor = torch.randn(3, 3, generator=generator, dtype=DTYPE)
    matrix = factor.T @ factor + torch.eye(3, dtype=DTYPE)
    gradient = torch.tensor([1.2, -.6, 1.7], dtype=DTYPE)
    orthogonal, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=DTYPE))
    original = AtomwiseResponse(matrix, basis).solve(gradient, .6, .3)
    rotated = AtomwiseResponse(orthogonal.T @ matrix @ orthogonal,
        torch.einsum('mk,mnc->knc', orthogonal, basis)).solve(orthogonal.T @ gradient, .6, .3)
    torch.testing.assert_close(orthogonal @ rotated.amplitudes, original.amplitudes, atol=2e-7, rtol=2e-7)
    assert float(rotated.gain) == pytest.approx(float(original.gain), abs=1e-10)


def test_zero_atom_bound_uses_exact_nullspace():
    basis = torch.zeros((2, 1, 3), dtype=DTYPE)
    basis[0, 0, 0] = 1.
    result = AtomwiseResponse(torch.eye(2, dtype=DTYPE), basis).solve(torch.tensor([1., 2.], dtype=DTYPE), .4, 0.)
    torch.testing.assert_close(result.amplitudes, torch.tensor([0., -.4], dtype=DTYPE))
    assert result.diagnostics['atom_nullity'] == 1


def test_zero_radius_and_no_modes_are_exact():
    solver = AtomwiseResponse(torch.eye(1, dtype=DTYPE), torch.ones(1, 2, 3, dtype=DTYPE))
    result = solver.solve(torch.ones(1, dtype=DTYPE), 0., 1.)
    assert not bool(result.amplitudes.any()) and float(result.gain) == 0.
    result = AtomwiseResponse(torch.empty(0, 0, dtype=DTYPE), torch.empty(0, 2, 3, dtype=DTYPE)).solve(torch.empty(0, dtype=DTYPE), 1., 1.)
    assert result.amplitudes.shape == (0,) and float(result.gain) == 0.


def test_nonconvergence_is_not_returned_as_a_success():
    basis = torch.zeros((2, 2, 3), dtype=DTYPE)
    basis[0, 0, 0], basis[1, 1, 1] = 10., 1.
    with pytest.raises(AtomwiseConvergenceError, match='did not converge') as caught:
        AtomwiseResponse(torch.eye(2, dtype=DTYPE), basis).solve(torch.tensor([0., -2.], dtype=DTYPE), 1., .5, max_iterations=1)
    assert caught.value.diagnostics['optimizer_success'] is False


def test_optimizer_success_without_kkt_certificate_is_rejected(monkeypatch):
    import zdock.inducedfit.atom_constraints as module
    monkeypatch.setattr(module, 'minimize', lambda *args, **kwargs: types.SimpleNamespace(
        x=np.zeros(1), success=True, status=0, message='fake success', nit=1))
    solver = AtomwiseResponse(torch.eye(1, dtype=DTYPE), torch.ones(1, 1, 3, dtype=DTYPE))
    with pytest.raises(AtomwiseConvergenceError, match='certificate'):
        solver.solve(torch.ones(1, dtype=DTYPE), 1., 1.)


def test_line_search_warning_requires_and_records_independent_certificate(monkeypatch):
    import zdock.inducedfit.atom_constraints as module
    monkeypatch.setattr(module, 'minimize', lambda *args, **kwargs: types.SimpleNamespace(
        x=np.array([-.2]), success=False, status=8, message='line-search warning', nit=5))
    solver = AtomwiseResponse(torch.tensor([[2.]], dtype=DTYPE), torch.tensor([[[1., 0., 0.]]], dtype=DTYPE))
    result = solver.solve(torch.tensor([4.], dtype=DTYPE), 1., .2)
    assert result.diagnostics['status'] == 'certified_optimizer_warning'
    assert result.diagnostics['success_basis'] == 'independent_certificate'
    assert result.diagnostics['optimizer_success'] is False


@pytest.mark.parametrize('invalid', [float('nan'), -1.])
def test_invalid_bounds_are_rejected(invalid):
    solver = AtomwiseResponse(torch.eye(1, dtype=DTYPE), torch.ones(1, 1, 3, dtype=DTYPE))
    with pytest.raises(ValueError):
        solver.solve(torch.ones(1, dtype=DTYPE), invalid, 1.)
    with pytest.raises(ValueError):
        solver.solve(torch.ones(1, dtype=DTYPE), 1., invalid)


def test_solver_rejects_indefinite_stiffness_and_does_not_mutate_inputs():
    with pytest.raises(ValueError, match='positive definite'):
        AtomwiseResponse(-torch.eye(1, dtype=DTYPE), torch.ones(1, 1, 3, dtype=DTYPE))
    matrix, basis = torch.eye(1, dtype=DTYPE), torch.ones(1, 1, 3, dtype=DTYPE)
    solver = AtomwiseResponse(matrix, basis)
    matrix.mul_(10)
    basis.mul_(10)
    result = solver.solve(torch.tensor([.1], dtype=DTYPE), 1., 1.)
    torch.testing.assert_close(result.amplitudes, torch.tensor([-.1], dtype=DTYPE))
