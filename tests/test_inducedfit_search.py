"""Independent end-to-end checks of the experimental response FFT search."""

import math

import pytest
import torch

from zdock.geom import rotate
from zdock.inducedfit.fields import FieldParameters, build_fields, ligand_channels, receptor_channels
from zdock.inducedfit.search import prepare_response_receptor, search_response
from zdock.inducedfit.types import AtomKey, GridSpec, MonomerInput


# PyTorch 2.13's JVP lazy import scripts an internal helper and emits this
# upstream deprecation. Keep all other warnings under the repository's error policy.
pytestmark = pytest.mark.filterwarnings("ignore:`torch.jit.script` is deprecated.*:DeprecationWarning")


@pytest.fixture(scope="module")
def system():
    dtype = torch.float64

    def monomer(chain, xyz, charges):
        xyz = torch.tensor(xyz, dtype=dtype)
        return MonomerInput(
            xyz, torch.full((len(xyz),), 0.6, dtype=dtype),
            torch.tensor(charges, dtype=dtype),
            tuple(AtomKey(chain, i + 1, "", "SYN", "X") for i in range(len(xyz))),
        )

    receptor = monomer("R", [[-1, 0, 0], [1, 0, 0], [0, 1, 0]], [.5, -.3, .2])
    ligand = monomer("L", [[-.5, 0, 0], [.5, 0, 0]], [-.3, .2])
    grid = GridSpec((17, 15, 19), 1.0, (-8., -7., -9.))
    params = FieldParameters(
        surface_width=.3, charge_sigma=.45,
        screening_switch_start=2., screening_cutoff=3.,
        repulsion_weight=3., charge_weight=.2,
    )
    basis = torch.tensor([
        [[-1., 0, 0], [1., 0, 0], [0., 0, 0]],
        [[0., -1., 0], [0., -1., 0], [0., 2., 0]],
    ], dtype=dtype)
    basis = basis / torch.sqrt(basis.square().sum((1, 2)) / len(receptor.xyz))[:, None, None]
    stiffness = torch.tensor([[4., .3], [.3, 7.]], dtype=dtype)
    quats = torch.tensor([[0., 0., 0., 1.], [0., 0., math.sqrt(.5), math.sqrt(.5)]], dtype=dtype)
    prepared = prepare_response_receptor(receptor, grid, params, basis, stiffness)
    return receptor, ligand, grid, params, basis, stiffness, quats, prepared


def _energy(rec_xyz, lig_xyz, receptor, ligand, grid, params):
    r = build_fields(rec_xyz, receptor.radii, receptor.charges, grid, params)
    l = build_fields(lig_xyz, ligand.radii, ligand.charges, grid, params)
    return (receptor_channels(r, grid, params) * ligand_channels(l)).sum()


@pytest.mark.parametrize("radius", [.01, .1])
def test_selected_scores_match_rebuilt_energy_and_response_finite_difference(system, radius):
    receptor, ligand, grid, params, basis, stiffness, quats, prepared = system
    result = search_response(prepared, ligand, quats, radius=radius, top_k=3, peaks_per_rotation=2)
    assert len(result) == 3
    for candidate in result:
        lig_xyz = torch.stack(rotate(*ligand.xyz.unbind(-1), quats[candidate.rotation_index]), -1)
        lig_xyz = lig_xyz + candidate.translation
        direct = _energy(receptor.xyz, lig_xyz, receptor, ligand, grid, params)
        assert candidate.rigid_energy == pytest.approx(float(direct), abs=1e-10)
        h = 1e-4
        gradient = torch.stack([
            (_energy(receptor.xyz + h * mode, lig_xyz, receptor, ligand, grid, params)
             - _energy(receptor.xyz - h * mode, lig_xyz, receptor, ligand, grid, params)) / (2 * h)
            for mode in basis
        ])
        a = candidate.amplitudes
        expected = gradient @ a + .5 * (a @ stiffness @ a)
        assert candidate.predicted_gain == pytest.approx(float(expected), rel=1e-5, abs=1e-8)
        assert torch.linalg.vector_norm(a) <= radius + 1e-12
        assert candidate.strain >= 0
        assert candidate.predicted_gain <= 1e-12
        assert candidate.score == pytest.approx(candidate.rigid_energy + candidate.predicted_gain, abs=1e-12)


def test_zero_radius_and_no_modes_recover_rigid_selection(system):
    receptor, ligand, grid, params, basis, stiffness, quats, prepared = system
    options = dict(radius=0., top_k=3, peaks_per_rotation=2)
    rigid = search_response(prepared, ligand, quats, selection="rigid", **options)
    zero = search_response(prepared, ligand, quats, **options)
    empty_prepared = prepare_response_receptor(receptor, grid, params, basis[:0], stiffness[:0, :0])
    empty = search_response(empty_prepared, ligand, quats, **options)
    for ref, constrained, no_modes in zip(rigid, zero, empty, strict=True):
        assert ref.rotation_index == constrained.rotation_index == no_modes.rotation_index
        torch.testing.assert_close(ref.translation, constrained.translation, rtol=0, atol=0)
        torch.testing.assert_close(ref.translation, no_modes.translation, rtol=0, atol=0)
        assert ref.score == constrained.score == no_modes.score
        assert torch.equal(constrained.amplitudes, torch.zeros_like(constrained.amplitudes))
        assert no_modes.amplitudes.numel() == 0


def test_only_ligand_spectra_are_built_inside_rotation_loop(system, monkeypatch):
    import zdock.inducedfit.search as search
    _, ligand, _, _, _, _, quats, prepared = system
    calls = []
    original = search.correlation_spectra

    def count(channels):
        calls.append(channels.shape)
        return original(channels)

    monkeypatch.setattr(search, "correlation_spectra", count)
    search_response(prepared, ligand, quats, radius=.1, top_k=2)
    assert len(calls) == len(quats)
    assert prepared.rigid_spectrum.grad_fn is None
    assert all(t.grad_fn is None for t in prepared.derivative_spectra)


def test_translation_tiling_does_not_change_selection(system):
    _, ligand, _, _, _, _, quats, prepared = system
    small = search_response(prepared, ligand, quats[:1], radius=.1, top_k=2, response_tile_size=101)
    large = search_response(prepared, ligand, quats[:1], radius=.1, top_k=2, response_tile_size=100_000)
    for left, right in zip(small, large, strict=True):
        assert left.score == pytest.approx(right.score, abs=1e-12)
        torch.testing.assert_close(left.translation, right.translation)
        torch.testing.assert_close(left.amplitudes, right.amplitudes)


@pytest.mark.parametrize("kwargs", [dict(radius=-1), dict(radius=float("nan")), dict(top_k=0),
                                   dict(peaks_per_rotation=1.5), dict(selection="native")])
def test_invalid_search_configuration_fails(system, kwargs):
    *_, quats, prepared = system
    with pytest.raises(ValueError):
        search_response(prepared, system[1], quats, **(dict(radius=.1) | kwargs))


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_valid_energy_is_not_silent_candidate_exhaustion(system, monkeypatch, nonfinite):
    import zdock.inducedfit.search as search
    *_, quats, prepared = system
    shape = tuple(2 * n - 1 for n in prepared.grid.shape)
    monkeypatch.setattr(search, "correlate_from_spectra", lambda *args: torch.full(shape, nonfinite, dtype=torch.float64))
    with pytest.raises(FloatingPointError, match="nonfinite rigid"):
        search_response(prepared, system[1], quats, radius=.1)
