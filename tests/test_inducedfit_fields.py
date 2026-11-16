"""Independent numerical checks for the experimental smooth physical core."""

from dataclasses import replace
import itertools

import pytest
import torch

from zdock.inducedfit.fields import (
    FieldParameters,
    build_fields,
    correlate_from_spectra,
    correlation_spectra,
    direct_correlation_at_shift,
    direct_linear_convolve_same,
    fft_correlate_channels,
    jvp_receptor_fields,
    kernel_grid,
    ligand_channels,
    linear_convolve_same,
    physical_energy,
    receptor_channels,
    signed_lag_axes,
    valid_translation_mask,
)
from zdock.inducedfit.types import GridSpec


# PyTorch 2.13 imports its own scripted helpers lazily on first torch.func JVP.
# The upstream deprecation is unrelated to these numerical assertions; retain
# the repository's warnings-as-errors policy for every other warning.
pytestmark = pytest.mark.filterwarnings(
    "ignore:`torch.jit.script` is deprecated.*:DeprecationWarning"
)


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def system():
    grid = GridSpec((13, 15, 13), 0.8, (-4.8, -5.6, -4.8))
    params = FieldParameters(
        surface_width=0.3, occupancy_switch_start=2.0, occupancy_cutoff_sigma=3.0,
        charge_sigma=0.4, charge_switch_start=2.0, charge_cutoff_sigma=3.0,
        screening_switch_start=2.4, screening_cutoff=3.2,
        atom_chunk_size=1, point_chunk_size=257,
        repulsion_weight=2.3, charge_weight=0.7,
    )
    rec = torch.tensor([[-0.7, 0.2, 0.1], [0.6, -0.3, -0.2]], dtype=torch.float64)
    lig = torch.tensor([[-0.3, 0.1, 0.4], [0.5, 0.4, -0.1]], dtype=torch.float64)
    radii = torch.tensor([0.5, 0.6], dtype=torch.float64)
    charges = torch.tensor([1.0, -0.4], dtype=torch.float64)
    return grid, params, rec, lig, radii, charges


def test_unit_peak_occupancy_positive_surface_and_charge_conservation(system):
    grid, params, rec, _, radii, charges = system
    fields = build_fields(rec, radii, charges, grid, params)
    assert bool((fields.occupancy >= 0).all())
    assert bool((fields.occupancy < 1).all())
    assert float(fields.surface.min()) >= -1e-15
    torch.testing.assert_close(fields.charge_density.sum() * grid.voxel_volume,
                               charges.sum(), atol=1e-13, rtol=1e-13)
    center = torch.zeros((1, 3), dtype=torch.float64)
    single = build_fields(center, radii[:1], charges[:1], grid, params)
    # The center is a grid node and its occupancy Gaussian has height ONE.
    torch.testing.assert_close(single.occupancy[6, 7, 6], -torch.expm1(torch.tensor(-1., dtype=torch.float64)))
    torch.testing.assert_close(single.surface[6, 7, 6], torch.tensor(0., dtype=torch.float64), atol=1e-15, rtol=0)


def test_chunked_fields_are_identical(system):
    grid, params, rec, _, radii, charges = system
    small = build_fields(rec, radii, charges, grid, params)
    large = build_fields(rec, radii, charges, grid,
                         replace(params, atom_chunk_size=8, point_chunk_size=4096))
    for name in ("occupancy", "surface", "charge_density"):
        torch.testing.assert_close(getattr(small, name), getattr(large, name), atol=1e-14, rtol=1e-13)


def test_linear_convolution_matches_spatial_oracle_and_has_no_wrap():
    density = torch.zeros((7, 5, 3), dtype=torch.float64)
    density[0, 0, 0] = 1.0
    kernel = torch.zeros((3, 3, 3), dtype=torch.float64)
    kernel[1, 1, 1] = 2.0
    kernel[0, 1, 1] = 0.3
    kernel[2, 1, 1] = 0.7
    fft = linear_convolve_same(density, kernel)
    direct = direct_linear_convolve_same(density, kernel)
    torch.testing.assert_close(fft, direct, atol=1e-14, rtol=1e-13)
    assert float(fft[-1, 0, 0].abs()) < 1e-14
    assert float(fft[1, 0, 0]) == pytest.approx(0.7)


def test_fft_correlation_all_signed_lags_against_direct_overlap():
    generator = torch.Generator().manual_seed(37)
    rec = torch.randn((3, 4, 3, 5), generator=generator, dtype=torch.float64)
    lig = torch.randn((3, 4, 3, 5), generator=generator, dtype=torch.float64)
    fft = fft_correlate_channels(rec, lig)
    axes = signed_lag_axes(tuple(rec.shape[-3:]))
    for index in itertools.product(*(range(len(axis)) for axis in axes)):
        shift = tuple(int(axes[d][index[d]]) for d in range(3))
        expected = direct_correlation_at_shift(rec, lig, shift)
        torch.testing.assert_close(fft[index], expected, atol=2e-13, rtol=1e-11)
    cached = correlate_from_spectra(correlation_spectra(rec), correlation_spectra(lig), (4, 3, 5))
    torch.testing.assert_close(cached, fft, atol=0, rtol=0)


def test_positive_lag_means_positive_ligand_translation():
    rec = torch.zeros((1, 5, 3, 3), dtype=torch.float64)
    lig = torch.zeros_like(rec)
    rec[0, 3, 1, 1] = 1
    lig[0, 1, 1, 1] = 1
    result = fft_correlate_channels(rec, lig)
    peak = torch.unravel_index(result.argmax(), result.shape)
    assert tuple(int(v) for v in peak) == (6, 2, 2)  # zero index (4,2,2), shift (+2,0,0)


def test_physical_fft_matches_actual_coordinate_direct_energy(system):
    grid, params, rec, lig, radii, charges = system
    rec_fields = build_fields(rec, radii, charges, grid, params)
    lig_fields = build_fields(lig, radii, -charges, grid, params)
    rec_channels = receptor_channels(rec_fields, grid, params)
    scores = fft_correlate_channels(rec_channels, ligand_channels(lig_fields))
    mask = valid_translation_mask(lig_fields, grid)
    for shift in ((0, 0, 0), (1, -1, 0), (-1, 0, 1), (0, 2, -1)):
        idx = tuple(n - 1 + t for n, t in zip(grid.shape, shift))
        assert bool(mask[idx])
        moved = lig + torch.tensor(shift, dtype=lig.dtype) * grid.spacing
        moved_fields = build_fields(moved, radii, -charges, grid, params)
        expected = physical_energy(rec_fields, moved_fields, grid, params, direct=True)
        torch.testing.assert_close(scores[idx], expected, atol=2e-12, rtol=1e-10)
    # Negative and positive support violations are explicitly invalid.
    assert not bool(mask[0].any())
    assert not bool(mask[-1].any())


def test_support_truncation_is_rejected_and_invalid_parameters_fail(system):
    grid, params, rec, _, radii, charges = system
    with pytest.raises(ValueError, match="compact support"):
        build_fields(rec + torch.tensor([4., 0., 0.]), radii, charges, grid, params)
    with pytest.raises(ValueError, match="switch_start"):
        build_fields(rec, radii, charges, grid, replace(params, screening_cutoff=1.0))


def test_jvp_matches_rebuilt_fields_and_energy_finite_difference(system):
    grid, params, rec, lig, radii, charges = system
    direction = torch.tensor([[0.12, -0.18, 0.07], [-0.06, 0.09, 0.15]], dtype=torch.float64)
    primal, tangent = jvp_receptor_fields(rec, radii, charges, direction, grid, params)
    ligand = build_fields(lig, radii, -charges, grid, params)
    ligand_values = ligand_channels(ligand)
    jvp_energy = (tangent * ligand_values).sum()
    errors = []
    for h in (1e-2, 1e-3, 1e-4):
        plus = build_fields(rec + h * direction, radii, charges, grid, params)
        minus = build_fields(rec - h * direction, radii, charges, grid, params)
        plus_channels = receptor_channels(plus, grid, params)
        minus_channels = receptor_channels(minus, grid, params)
        difference = (plus_channels - minus_channels) / (2 * h)
        errors.append(float(torch.linalg.vector_norm(difference - tangent) / torch.linalg.vector_norm(tangent)))
        if h == 1e-4:
            torch.testing.assert_close(difference, tangent, atol=1e-8, rtol=1e-4)
            direct_difference = (physical_energy(plus, ligand, grid, params, direct=True)
                                 - physical_energy(minus, ligand, grid, params, direct=True)) / (2 * h)
            torch.testing.assert_close(jvp_energy, direct_difference, atol=1e-8, rtol=1e-4)
    assert errors[-1] < 1e-6
    assert errors[1] < errors[0] / 20
    assert errors[2] < errors[1] / 20
    torch.testing.assert_close(primal, receptor_channels(build_fields(rec, radii, charges, grid, params),
                                                        grid, params), atol=1e-13, rtol=1e-13)


def test_jvp_fft_gradient_maps_match_rebuilt_energy_at_signed_shifts(system):
    grid, params, rec, lig, radii, charges = system
    direction = torch.tensor([[0.1, 0.2, -0.1], [-0.2, 0.05, 0.1]], dtype=torch.float64)
    _, tangent = jvp_receptor_fields(rec, radii, charges, direction, grid, params)
    ligand = build_fields(lig, radii, -charges, grid, params)
    gradient_map = fft_correlate_channels(tangent, ligand_channels(ligand))
    h = 1e-4
    plus = build_fields(rec + h * direction, radii, charges, grid, params)
    minus = build_fields(rec - h * direction, radii, charges, grid, params)
    for shift in ((1, -1, 0), (-1, 0, 1)):
        translated = build_fields(lig + grid.spacing * torch.tensor(shift, dtype=lig.dtype),
                                  radii, -charges, grid, params)
        fd = (physical_energy(plus, translated, grid, params, direct=True)
              - physical_energy(minus, translated, grid, params, direct=True)) / (2 * h)
        idx = tuple(n - 1 + t for n, t in zip(grid.shape, shift))
        torch.testing.assert_close(gradient_map[idx], fd, atol=1e-8, rtol=1e-4)


def test_continuous_coordinate_autograd_matches_direct_finite_difference(system):
    grid, params, rec, lig, radii, charges = system
    rec_fields = build_fields(rec, radii, charges, grid, params)
    shift = torch.tensor([0.13, -0.07, 0.09], dtype=torch.float64, requires_grad=True)
    moved = build_fields(lig + shift, radii, -charges, grid, params)
    energy = physical_energy(rec_fields, moved, grid, params)
    gradient, = torch.autograd.grad(energy, shift)
    direction = torch.tensor([0.3, -0.2, 0.4], dtype=torch.float64)
    h = 1e-4
    plus = build_fields(lig + shift.detach() + h * direction, radii, -charges, grid, params)
    minus = build_fields(lig + shift.detach() - h * direction, radii, -charges, grid, params)
    fd = (physical_energy(rec_fields, plus, grid, params, direct=True)
          - physical_energy(rec_fields, minus, grid, params, direct=True)) / (2 * h)
    torch.testing.assert_close(gradient @ direction, fd, atol=1e-8, rtol=1e-4)


def test_screened_kernel_cutoff_and_regularization(system):
    grid, params, *_ = system
    kernel = kernel_grid(grid, params, dtype=torch.float64)
    assert bool(torch.isfinite(kernel).all())
    assert bool((kernel >= 0).all())
    middle = tuple(n // 2 for n in kernel.shape)
    expected = torch.exp(torch.tensor(-params.screening_kappa * params.screening_softening)) / params.screening_softening
    assert float(kernel[middle]) == pytest.approx(float(expected), rel=1e-7)
    assert float(kernel[0, 0, 0]) == 0
