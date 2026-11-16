"""Bounded forward AD preserves the anchored mode map and nonlinear prior."""
from dataclasses import replace
import sys
from types import SimpleNamespace

import pytest
import torch

from zdock.inducedfit.hydrogen_kinematics import HydrogenatedKinematics
from zdock.inducedfit.modes import build_anm, build_torsion_modes
from test_inducedfit_hydrogen_kinematics import make_prepared
from test_inducedfit_modes import ToyTorsions


pytestmark = pytest.mark.filterwarnings("ignore:`torch.jit.script` is deprecated.*:DeprecationWarning")


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("chunk_size", [1, 2, 8])
def test_chunked_preserves_all_atom_basis_coordinates_and_prior(dtype, chunk_size):
    # Four interleaved CAs among heavy atoms and H, not a CA-only output.
    kin = HydrogenatedKinematics.from_prepared(make_prepared(dtype=dtype))
    dense = build_torsion_modes(kin, n_modes=2)
    chunked = build_torsion_modes(kin, n_modes=2, jacobian_backend="chunked", jvp_chunk_size=chunk_size)
    tolerance = 4e-5 if dtype == torch.float32 else 2e-11
    for key in ("angle_basis", "basis", "stiffness"):
        torch.testing.assert_close(getattr(chunked, key), getattr(dense, key), atol=tolerance, rtol=tolerance)
    assert chunked.regularization_strength == pytest.approx(dense.regularization_strength, rel=tolerance)
    assert chunked.diagnostics["fit_absolute_damping"] == pytest.approx(dense.diagnostics["fit_absolute_damping"], rel=tolerance)
    assert not chunked.diagnostics["materialized_full_coordinate_jacobian"]
    amplitudes = kin.reference.new_tensor([.031, -.017])
    torch.testing.assert_close(chunked.coordinates(amplitudes), dense.coordinates(amplitudes), atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(chunked.deformation_energy(amplitudes), dense.deformation_energy(amplitudes), atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(torch.func.grad(chunked.deformation_energy)(amplitudes),
                               torch.func.grad(dense.deformation_energy)(amplitudes), atol=tolerance, rtol=tolerance)
    origin = torch.zeros_like(amplitudes)
    torch.testing.assert_close(chunked.stiffness, torch.func.hessian(chunked.deformation_energy)(origin),
                               atol=tolerance, rtol=tolerance)
    ca = chunked.basis[:, kin.ca_indices].reshape(2, -1)
    torch.testing.assert_close(ca @ ca.T / len(kin.ca_indices), torch.eye(2, dtype=dtype), atol=tolerance, rtol=tolerance)
    # A non-CA hydrogen derivative must survive the reduced construction.
    hydrogen_indices = sorted(set(range(len(kin.reference))) - set(kin.heavy_indices.tolist()))
    assert chunked.basis[:, hydrogen_indices].abs().max() > .1


def test_chunked_bounds_ad_seed_blocks_and_never_calls_full_jacfwd(monkeypatch):
    kin = HydrogenatedKinematics.from_prepared(make_prepared())
    original_vmap = torch.vmap
    sizes = []

    def watched_vmap(function, *args, **kwargs):
        mapped = original_vmap(function, *args, **kwargs)

        def call(directions, *call_args, **call_kwargs):
            sizes.append(tuple(directions.shape))
            return mapped(directions, *call_args, **call_kwargs)
        return call

    monkeypatch.setattr(torch, "vmap", watched_vmap)
    monkeypatch.setattr(torch.func, "jacfwd", lambda *a, **k: pytest.fail("full jacfwd must not be called"))
    model = build_torsion_modes(kin, n_modes=2, jacobian_backend="chunked", jvp_chunk_size=2)
    assert sizes == [(2, kin.n_torsions)] * (kin.n_torsions // 2) + [(2, kin.n_torsions)]
    assert all(batch <= 2 and width == kin.n_torsions for batch, width in sizes)
    assert model.basis.shape == (2, len(kin.reference), 3)


@pytest.mark.parametrize("backend", ["dense", "chunked"])
def test_zero_mode_and_zero_torsion_rules_are_preserved(backend, monkeypatch):
    kin = ToyTorsions(n_torsions=0)
    monkeypatch.setattr(torch.func, "jvp", lambda *a, **k: pytest.fail("zero-mode path must not take JVP"))
    model = build_torsion_modes(kin, n_modes=0, jacobian_backend=backend)
    empty = kin.reference.new_empty(0)
    torch.testing.assert_close(model.coordinates(empty), kin.reference)
    assert model.deformation_energy(empty).item() == pytest.approx(0., abs=1e-25)
    assert model.basis.shape == (0, len(kin.reference), 3)
    assert not model.diagnostics["materialized_full_coordinate_jacobian"]
    with pytest.raises(ValueError, match="torsion degrees"):
        build_torsion_modes(kin, n_modes=1, jacobian_backend=backend)


def test_chunked_rank_reduction_remains_explicit_and_matches_dense():
    kin = ToyTorsions(n_torsions=2)
    with pytest.raises(ValueError, match="numerical rank 2"):
        build_torsion_modes(kin, n_modes=4, jacobian_backend="chunked")
    dense = build_torsion_modes(kin, n_modes=4, allow_rank_reduction=True)
    reduced = build_torsion_modes(kin, n_modes=4, allow_rank_reduction=True, jacobian_backend="chunked")
    assert reduced.n_modes == 2 and reduced.diagnostics["rank_reduced"]
    torch.testing.assert_close(reduced.basis, dense.basis)
    torch.testing.assert_close(reduced.stiffness, dense.stiffness)


def test_chunked_rejects_zero_ca_rank():
    kin = ToyTorsions()
    kin.coordinates = lambda angles: kin.reference + angles.square().sum() * torch.ones_like(kin.reference)
    with pytest.raises(ValueError, match="zero numerical rank"):
        build_torsion_modes(kin, n_modes=1, jacobian_backend="chunked")


@pytest.mark.parametrize("keyword,value", [
    ("jacobian_backend", "batched"), ("jvp_chunk_size", 0), ("jvp_chunk_size", True),
    ("jvp_chunk_size", 1.5), ("anm_backend", "approximate"), ("anm_seed", -1),
    ("anm_seed", True), ("anm_tolerance", 0.), ("anm_tolerance", float("nan")),
])
def test_backend_parameters_are_validated_even_for_zero_modes(keyword, value):
    with pytest.raises(ValueError, match=keyword):
        build_torsion_modes(ToyTorsions(), n_modes=0, **{keyword: value})


def test_iterative_dispatch_passes_explicit_solver_controls(monkeypatch):
    captured = {}

    def iterative(ca_xyz, **kwargs):
        captured.update(kwargs)
        dense = build_anm(ca_xyz, cutoff=kwargs["cutoff"], gamma=kwargs["gamma"])
        count = kwargs["n_modes"]
        return replace(dense, hessian=None, modes=dense.modes[:count], eigenvalues=dense.eigenvalues[:count])

    monkeypatch.setitem(sys.modules, "zdock.inducedfit.anm_iterative", SimpleNamespace(build_anm_iterative=iterative))
    model = build_torsion_modes(ToyTorsions(), n_modes=2, anm_backend="iterative", anm_seed=17,
                                anm_tolerance=3e-8, allow_rank_reduction=True, jacobian_backend="chunked")
    assert captured == {"n_modes": 2, "cutoff": 12., "gamma": 1., "seed": 17,
                        "tolerance": 3e-8, "allow_rank_reduction": True}
    assert model.anm.hessian is None
    assert model.n_modes == 2


def test_iterative_and_dense_nonlinear_models_agree_after_basis_gauge():
    kin = ToyTorsions(torch.float64)
    dense = build_torsion_modes(kin, n_modes=2)
    iterative = build_torsion_modes(kin, n_modes=2, jacobian_backend="chunked", anm_backend="iterative")
    # Eigenvector signs and rotations within the fitted mode subspace are a
    # basis gauge. Compare both coordinates and prior in the same amplitudes.
    d = dense.basis[:, kin.ca_indices].reshape(2, -1)
    i = iterative.basis[:, kin.ca_indices].reshape(2, -1)
    gauge = d @ i.T / len(kin.ca_indices)
    tolerance = 2e-8
    torch.testing.assert_close(gauge.T @ gauge, torch.eye(2, dtype=kin.reference.dtype), atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(dense.angle_basis @ gauge, iterative.angle_basis, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(gauge.T @ dense.stiffness @ gauge, iterative.stiffness, atol=tolerance, rtol=tolerance)
    amplitude = kin.reference.new_tensor([.04, -.02])
    torch.testing.assert_close(dense.coordinates(gauge @ amplitude), iterative.coordinates(amplitude), atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(dense.deformation_energy(gauge @ amplitude), iterative.deformation_energy(amplitude), atol=tolerance, rtol=tolerance)
    assert iterative.anm.hessian is None


def test_iterative_float32_is_rejected_without_silent_promotion():
    with pytest.raises(TypeError, match="float64 CPU"):
        build_torsion_modes(ToyTorsions(torch.float32), n_modes=2,
                            jacobian_backend="chunked", anm_backend="iterative")


def test_iterative_zero_modes_skip_spectral_counts_and_preserve_coordinates():
    kin = ToyTorsions(n_torsions=0)
    model = build_torsion_modes(kin, n_modes=0, jacobian_backend="chunked", anm_backend="iterative")
    torch.testing.assert_close(model.coordinates(kin.reference.new_empty(0)), kin.reference)
    assert model.anm.hessian is None
    assert model.anm.diagnostics["n_zero_modes"] is None
    assert model.anm.diagnostics["n_positive_modes"] is None
