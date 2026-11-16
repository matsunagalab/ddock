"""Sparse-edge ANM against the existing dense physical-model oracle."""
import numpy as np
import pytest
from scipy.sparse.linalg import ArpackNoConvergence
import torch

from zdock.inducedfit import anm_iterative
from zdock.inducedfit.anm_iterative import build_anm_edge_operator, build_anm_iterative
from zdock.inducedfit.modes import build_anm
from test_inducedfit_modes import tetrahedron


def test_edge_definition_and_hv_match_dense_including_cutoff_boundary():
    xyz = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                        [0., 0., 1.], [1., 1., 1.]], dtype=torch.float64)
    dense = build_anm(xyz, cutoff=2**.5, gamma=2.7)
    edge = build_anm_edge_operator(xyz, cutoff=2**.5, gamma=2.7)
    assert torch.equal(edge.pairs, dense.pairs)
    torch.testing.assert_close(edge.reference_distances, dense.reference_distances, atol=0, rtol=0)
    vectors = torch.randn((15, 4), dtype=torch.float64, generator=torch.Generator().manual_seed(19))
    torch.testing.assert_close(torch.from_numpy(edge.matmat(vectors.numpy())), dense.hessian @ vectors,
                               atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(torch.from_numpy(edge.matvec(vectors[:, 0].numpy())), dense.hessian @ vectors[:, 0],
                               atol=2e-14, rtol=2e-14)


@pytest.mark.parametrize("n_modes", [1, 3, 6])
def test_low_eigenvalues_and_subspace_match_dense_without_full_hessian(n_modes):
    xyz = tetrahedron()
    dense = build_anm(xyz)
    actual = build_anm_iterative(xyz, n_modes=n_modes, seed=43)
    assert actual.hessian is None
    assert actual.n_modes == n_modes
    assert actual.diagnostics["n_zero_modes"] == 6
    assert actual.diagnostics["n_positive_modes"] == 6
    torch.testing.assert_close(actual.eigenvalues, dense.eigenvalues[:n_modes], atol=1e-12, rtol=1e-10)
    u, v = actual.modes.reshape(n_modes, -1).T, dense.modes[:n_modes].reshape(n_modes, -1).T
    torch.testing.assert_close(u @ u.T, v @ v.T, atol=2e-10, rtol=2e-10)
    torch.testing.assert_close(dense.hessian @ u, u * actual.eigenvalues, atol=1e-11, rtol=1e-10)
    assert max(actual.diagnostics["selected_relative_residuals"]) < 1e-10
    assert actual.diagnostics["orthogonality_max_abs_error"] < 1e-12


@pytest.mark.parametrize("xyz,cutoff,n_modes,zero_count", [
    (torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]], dtype=torch.float64), 1.1, 2, 7),
    (torch.tensor([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]], dtype=torch.float64), 2., 5, 7),
])
def test_connected_floppy_networks_remove_additional_zeros(xyz, cutoff, n_modes, zero_count):
    dense = build_anm(xyz, cutoff=cutoff)
    actual = build_anm_iterative(xyz, n_modes=n_modes, cutoff=cutoff, seed=3)
    assert actual.diagnostics["n_components"] == 1
    assert actual.diagnostics["n_zero_modes"] == zero_count
    assert actual.diagnostics["n_zero_modes"] > actual.diagnostics["n_rigid_modes"]
    torch.testing.assert_close(actual.eigenvalues, dense.eigenvalues, atol=1e-12, rtol=1e-11)
    u, v = actual.modes.flatten(1), dense.modes.flatten(1)
    torch.testing.assert_close(u.T @ u, v.T @ v, atol=1e-11, rtol=1e-11)


def test_disconnected_graph_is_rejected_or_reported_with_its_extra_zeros():
    xyz = torch.cat((tetrahedron(), tetrahedron() + 20.))
    with pytest.raises(ValueError, match="disconnected"):
        build_anm_iterative(xyz, n_modes=2, cutoff=3.)
    actual = build_anm_iterative(xyz, n_modes=2, cutoff=3., reject_disconnected=False)
    dense = build_anm(xyz, cutoff=3., reject_disconnected=False)
    assert actual.diagnostics["n_components"] == 2
    assert actual.diagnostics["n_zero_modes"] == 12
    torch.testing.assert_close(actual.eigenvalues, dense.eigenvalues[:2], atol=1e-12, rtol=1e-10)


def test_two_atom_smallest_supported_system_has_one_stretch_mode():
    xyz = torch.tensor([[0., 0., 0.], [1., .3, .2]], dtype=torch.float64)
    actual = build_anm_iterative(xyz, n_modes=1, gamma=2.)
    assert actual.diagnostics["n_rigid_modes"] == 5
    assert actual.diagnostics["n_zero_modes"] == 5
    torch.testing.assert_close(actual.eigenvalues, torch.tensor([4.], dtype=torch.float64))


def test_requested_too_many_and_budget_exhaustion_are_explicit():
    with pytest.raises(ValueError, match="at most 6"):
        build_anm_iterative(tetrahedron(), n_modes=7)
    reduced = build_anm_iterative(tetrahedron(), n_modes=7, allow_rank_reduction=True)
    assert reduced.n_modes == 6 and reduced.diagnostics["rank_reduced"]
    square = torch.tensor([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]], dtype=torch.float64)
    with pytest.raises(ValueError, match="only 5 positive"):
        build_anm_iterative(square, n_modes=6)
    assert build_anm_iterative(square, n_modes=6, allow_rank_reduction=True).n_modes == 5
    chain = torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]], dtype=torch.float64)
    with pytest.raises(RuntimeError, match="exhaust max_eigenpairs"):
        build_anm_iterative(chain, n_modes=1, cutoff=1.1, max_eigenpairs=1)


def test_zero_request_skips_spectrum_but_still_validates_graph(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("zero-request must not solve an eigensystem")
    monkeypatch.setattr(anm_iterative, "eigsh", forbidden)
    actual = build_anm_iterative(tetrahedron(), n_modes=0)
    assert actual.n_modes == 0 and actual.hessian is None
    assert actual.diagnostics["n_zero_modes"] is None
    assert actual.diagnostics["n_positive_modes"] is None
    assert not actual.diagnostics["spectral_analysis_performed"]
    with pytest.raises(ValueError, match="disconnected"):
        build_anm_iterative(tetrahedron(), n_modes=0, cutoff=.01)


def test_no_edges_has_explicit_empty_rank_reduction():
    xyz = tetrahedron()
    with pytest.raises(ValueError, match="no springs"):
        build_anm_iterative(xyz, n_modes=1, cutoff=.01, reject_disconnected=False)
    result = build_anm_iterative(xyz, n_modes=1, cutoff=.01, reject_disconnected=False, allow_rank_reduction=True)
    assert result.n_modes == 0 and result.diagnostics["n_zero_modes"] == 12


def test_no_dense_hessian_or_all_ca_pairs_fallback(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("dense Hessian/all-pair fallback forbidden")
    monkeypatch.setattr(torch, "triu_indices", forbidden)
    monkeypatch.setattr(torch.linalg, "eigh", forbidden)
    monkeypatch.setattr(np.linalg, "eigh", forbidden)
    assert build_anm_iterative(tetrahedron(), n_modes=2).n_modes == 2


def test_seed_is_local_and_fixed_seed_is_reproducible():
    np.random.seed(51)
    before = np.random.get_state()
    first = build_anm_iterative(tetrahedron(), n_modes=3, seed=7)
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    second = build_anm_iterative(tetrahedron(), n_modes=3, seed=7)
    torch.testing.assert_close(first.modes, second.modes, atol=0, rtol=0)
    torch.testing.assert_close(first.eigenvalues, second.eigenvalues, atol=0, rtol=0)


def test_nonconvergence_is_a_hard_error_not_partial_output(monkeypatch):
    def fail(*args, **kwargs):
        raise ArpackNoConvergence("deliberate test failure", np.empty(0), np.empty((12, 0)))
    monkeypatch.setattr(anm_iterative, "eigsh", fail)
    with pytest.raises(RuntimeError, match="did not converge.*no partial result or dense fallback"):
        build_anm_iterative(tetrahedron(), n_modes=2)


@pytest.mark.parametrize("keyword,value", [("seed", True), ("tolerance", 0.), ("tolerance", 1.),
    ("n_modes", -1), ("max_iterations", 0), ("max_eigenpairs", 0), ("zero_eigenvalue_rtol", -1.)])
def test_bad_parameters_fail(keyword, value):
    arguments = {"n_modes": 1, keyword: value}
    with pytest.raises((ValueError, TypeError)):
        build_anm_iterative(tetrahedron(), **arguments)


def test_invalid_dtype_or_coincident_coordinates_fail():
    with pytest.raises(TypeError, match="float64 CPU"):
        build_anm_iterative(tetrahedron(torch.float32), n_modes=1)
    with pytest.raises(ValueError, match="coincident"):
        build_anm_iterative(torch.zeros((3, 3), dtype=torch.float64), n_modes=1)
