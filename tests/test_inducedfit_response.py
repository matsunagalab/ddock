"""Numerical contracts for the inference-only induced-fit response solver."""

import pytest
import torch

from zdock.inducedfit.response import QuadraticResponse


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_one_dimensional_analytic_response(dtype):
    stiffness = torch.tensor([[2.0]], dtype=dtype)
    gradient = torch.tensor([[4.0], [-4.0], [0.0], [0.2]], dtype=dtype)
    result = QuadraticResponse(stiffness).solve(gradient, 0.5)
    expected = torch.tensor([[-0.5], [0.5], [0.0], [-0.1]], dtype=dtype)
    torch.testing.assert_close(result.amplitudes, expected)
    torch.testing.assert_close(result.gain, torch.tensor([-1.75, -1.75, 0.0, -0.01], dtype=dtype))
    torch.testing.assert_close(result.strain, torch.tensor([0.25, 0.25, 0.0, 0.01], dtype=dtype))
    torch.testing.assert_close(result.lagrange, torch.tensor([6.0, 6.0, 0.0, 0.0], dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dense_stiffness_kkt_and_objective(dtype):
    generator = torch.Generator().manual_seed(17)
    factor = torch.randn((7, 7), generator=generator, dtype=dtype)
    stiffness = factor.T @ factor + 0.25 * torch.eye(7, dtype=dtype)
    gradient = torch.randn((3, 5, 7), generator=generator, dtype=dtype)
    radius = torch.tensor([[0.01], [0.4], [100.0]], dtype=dtype)
    result = QuadraticResponse(stiffness).solve(gradient, radius)
    norm = torch.linalg.vector_norm(result.amplitudes, dim=-1)
    assert bool((norm <= radius * (1 + 4 * torch.finfo(dtype).eps)).all())
    residual = result.amplitudes @ stiffness + result.lagrange[..., None] * result.amplitudes + gradient
    tolerance = 2e-5 if dtype == torch.float32 else 1e-11
    assert float(torch.linalg.vector_norm(residual) / torch.linalg.vector_norm(gradient)) < tolerance
    torch.testing.assert_close(result.lagrange * (norm - radius), torch.zeros_like(norm), atol=tolerance, rtol=0)
    expected_strain = 0.5 * torch.einsum("...i,ij,...j->...", result.amplitudes, stiffness, result.amplitudes)
    torch.testing.assert_close(result.strain, expected_strain)
    torch.testing.assert_close(result.gain, (gradient * result.amplitudes).sum(-1) + expected_strain)
    assert bool((result.strain >= 0).all())
    assert bool((result.gain <= 0).all())
    torch.testing.assert_close(result.amplitudes[2], -torch.linalg.solve(stiffness, gradient[2].T).T)


def test_constrained_objective_is_not_unconstrained_bonus():
    result = QuadraticResponse(torch.eye(2, dtype=torch.float64)).solve(
        torch.tensor([3.0, 4.0], dtype=torch.float64), 0.2,
    )
    torch.testing.assert_close(result.gain, torch.tensor(-0.98, dtype=torch.float64))
    assert result.gain.item() != pytest.approx(-12.5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_radius_and_zero_gradient(dtype):
    gradient = torch.tensor([[2.0, -3.0], [0.0, 0.0]], dtype=dtype)
    result = QuadraticResponse(torch.eye(2, dtype=dtype)).solve(gradient, 0.0)
    assert not bool(result.amplitudes.any())
    assert not bool(result.gain.any())
    assert not bool(result.strain.any())
    assert torch.isposinf(result.lagrange[0])
    assert result.lagrange[1].item() == 0


@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3), (0,)])
def test_zero_modes(batch_shape):
    result = QuadraticResponse(torch.empty((0, 0), dtype=torch.float64)).solve(
        torch.empty((*batch_shape, 0), dtype=torch.float64), 0.5,
    )
    assert result.amplitudes.shape == (*batch_shape, 0)
    for value in (result.gain, result.strain, result.lagrange):
        assert value.shape == batch_shape
        assert not bool(value.any())


def test_empty_batch():
    result = QuadraticResponse(torch.eye(2, dtype=torch.float64)).solve(torch.empty((0, 2), dtype=torch.float64), 1.0)
    assert result.amplitudes.shape == (0, 2)
    assert result.gain.shape == (0,)


def test_extremely_stiff_rigid_limit():
    gradient = torch.tensor([1.0, -2.0], dtype=torch.float64)
    result = QuadraticResponse(torch.eye(2, dtype=torch.float64) * 1e12).solve(gradient, 0.5)
    torch.testing.assert_close(result.amplitudes, -gradient * 1e-12, atol=1e-25, rtol=1e-12)
    assert result.gain.abs().item() < 3e-12
    assert result.lagrange.item() == 0


def test_representable_constrained_solution_when_unconstrained_division_overflows():
    solver = QuadraticResponse(torch.tensor([[1e-20]], dtype=torch.float32))
    result = solver.solve(torch.tensor([1e20], dtype=torch.float32), 0.5)
    torch.testing.assert_close(result.amplitudes, torch.tensor([-0.5], dtype=torch.float32))
    torch.testing.assert_close(result.gain, torch.tensor(-5e19, dtype=torch.float32))
    assert bool(torch.isfinite(result.lagrange))


def test_nonrepresentable_multiplier_reports_failure():
    solver = QuadraticResponse(torch.eye(1, dtype=torch.float32))
    with pytest.raises(FloatingPointError, match="multiplier bound"):
        solver.solve(torch.tensor([1e30], dtype=torch.float32), 1e-20)


def test_matches_independent_scipy_constrained_optimizer():
    scipy_optimize = pytest.importorskip("scipy.optimize")
    stiffness = torch.tensor([[5.0, 1.2, -0.4], [1.2, 3.0, 0.3], [-0.4, 0.3, 1.5]], dtype=torch.float64)
    gradient = torch.tensor([2.5, -1.1, 0.4], dtype=torch.float64)
    radius = 0.31
    result = QuadraticResponse(stiffness).solve(gradient, radius)
    matrix, vector = stiffness.numpy(), gradient.numpy()
    oracle = scipy_optimize.minimize(
        lambda a: vector @ a + 0.5 * a @ matrix @ a,
        x0=vector * 0,
        jac=lambda a: vector + matrix @ a,
        constraints={"type": "ineq", "fun": lambda a: radius**2 - a @ a, "jac": lambda a: -2 * a},
        method="SLSQP", options={"ftol": 1e-13, "maxiter": 200},
    )
    assert oracle.success, oracle.message
    torch.testing.assert_close(result.amplitudes, torch.as_tensor(oracle.x), atol=2e-8, rtol=2e-8)
    assert result.gain.item() == pytest.approx(oracle.fun, abs=1e-11)


def test_cached_stiffness_is_detached_and_copied():
    stiffness = torch.eye(2, dtype=torch.float64, requires_grad=True)
    solver = QuadraticResponse(stiffness)
    with torch.no_grad():
        stiffness.mul_(20)
    result = solver.solve(torch.tensor([0.2, -0.3], dtype=torch.float64, requires_grad=True), 1.0)
    torch.testing.assert_close(result.amplitudes, torch.tensor([-0.2, 0.3], dtype=torch.float64))
    assert not any(value.requires_grad for value in (result.amplitudes, result.gain, result.strain, result.lagrange))


@pytest.mark.parametrize("matrix, message", [
    ([[1.0, 0.5], [0.0, 1.0]], "symmetric"),
    ([[1.0, 0.0], [0.0, 0.0]], "positive definite"),
    ([[1.0, 2.0], [2.0, 1.0]], "positive definite"),
    ([[float("nan")]], "finite"),
    ([[float("inf")]], "finite"),
    ([[1.0, 2.0]], "square"),
])
def test_invalid_stiffness(matrix, message):
    with pytest.raises(ValueError, match=message):
        QuadraticResponse(torch.tensor(matrix, dtype=torch.float64))


@pytest.mark.parametrize("radius", [-0.5, float("nan"), float("inf"), torch.tensor([-1.0, 1.0])])
def test_invalid_radius(radius):
    solver = QuadraticResponse(torch.eye(2, dtype=torch.float64))
    with pytest.raises(ValueError, match="radius"):
        solver.solve(torch.ones((2, 2), dtype=torch.float64), radius)


def test_invalid_gradient_and_solver_arguments():
    with pytest.raises(TypeError, match="float32 or float64"):
        QuadraticResponse(torch.eye(2, dtype=torch.float16))
    solver = QuadraticResponse(torch.eye(2, dtype=torch.float64))
    with pytest.raises(TypeError, match="same dtype"):
        solver.solve(torch.ones(2, dtype=torch.float32), 1.0)
    with pytest.raises(ValueError, match="shape"):
        solver.solve(torch.ones(3, dtype=torch.float64), 1.0)
    with pytest.raises(ValueError, match="finite"):
        solver.solve(torch.tensor([float("nan"), 0.0], dtype=torch.float64), 1.0)
    with pytest.raises(ValueError, match="broadcast"):
        solver.solve(torch.ones((2, 2), dtype=torch.float64), torch.ones(3))
    for iterations in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            solver.solve(torch.ones(2, dtype=torch.float64), 1.0, max_iter=iterations)
