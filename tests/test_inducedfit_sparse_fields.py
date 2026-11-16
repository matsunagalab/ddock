"""Dense-oracle parity for exact compact-support atom deposition."""

from dataclasses import replace

import pytest
import torch

from zdock.inducedfit.fields import FieldParameters, build_fields, physical_energy, receptor_channels, support_radii
from zdock.inducedfit.sparse_fields import build_sparse_fields
from zdock.inducedfit.types import GridSpec


pytestmark = pytest.mark.filterwarnings("ignore:`torch.jit.script` is deprecated.*:DeprecationWarning")


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def system():
    grid = GridSpec((15, 17, 15), 0.8, (-5.6, -6.4, -5.6))
    params = FieldParameters(
        surface_width=0.25, occupancy_switch_start=2.0, occupancy_cutoff_sigma=3.0,
        charge_sigma=0.35, charge_switch_start=2.0, charge_cutoff_sigma=3.0,
        screening_switch_start=1.6, screening_cutoff=2.4,
        atom_chunk_size=2, point_chunk_size=131, repulsion_weight=2.1, charge_weight=0.8,
    )
    xyz = torch.tensor([[-0.73, 0.21, 0.13], [0.6, -0.31, -0.23], [0.18, 0.72, -0.4],
                        [-0.3, -0.6, 0.7], [0.62, 0.41, 0.39]], dtype=torch.float64)
    radii = torch.tensor([0.47, 0.61, 0.53, 0.49, 0.57], dtype=torch.float64)
    charges = torch.tensor([0.7, -0.9, 0.3, -0.2, 0.5], dtype=torch.float64)
    return grid, params, xyz, radii, charges


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fields_and_charge_integral_equal_dense_oracle(system, dtype):
    grid, params, xyz, radii, charges = system
    xyz, radii, charges = (value.to(dtype) for value in (xyz, radii, charges))
    expected = build_fields(xyz, radii, charges, grid, params)
    actual = build_sparse_fields(xyz, radii, charges, grid, params)
    tolerance = 2e-6 if dtype == torch.float32 else 2e-13
    for name in ("occupancy", "surface", "charge_density", "support_lower", "support_upper"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(actual.charge_density.sum() * grid.voxel_volume, charges.sum(), atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("checkpointed", [False, True])
def test_energy_and_xyz_radii_charge_gradients_equal_dense_spatial_oracle(system, checkpointed):
    grid, params, xyz, radii, charges = system
    ligand = build_fields(xyz + xyz.new_tensor([0.8, 0.2, -0.1]), radii, -charges, grid, params)

    def evaluate(sparse):
        inputs = tuple(value.clone().requires_grad_() for value in (xyz, radii, charges))
        if sparse:
            fields = build_sparse_fields(*inputs, grid, params, checkpoint_chunks=checkpointed)
        else:
            fields = build_fields(*inputs, grid, params)
        energy = physical_energy(fields, ligand, grid, params, direct=not sparse)
        return energy.detach(), torch.autograd.grad(energy, inputs)

    expected, expected_gradients = evaluate(False)
    actual, actual_gradients = evaluate(True)
    torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=2e-10, rtol=2e-10)


def test_normalizer_is_differentiated_and_charge_conserved(system):
    grid, params, xyz, radii, charges = system
    xyz, charges = xyz.requires_grad_(), charges.requires_grad_()
    fields = build_sparse_fields(xyz, radii, charges, grid, params)
    integral = fields.charge_density.sum() * grid.voxel_volume
    gradient_xyz, gradient_charge = torch.autograd.grad(integral, (xyz, charges))
    torch.testing.assert_close(gradient_xyz, torch.zeros_like(xyz), atol=2e-13, rtol=0)
    torch.testing.assert_close(gradient_charge, torch.ones_like(charges), atol=2e-13, rtol=0)


def test_jvp_of_all_receptor_channels_matches_dense_and_finite_difference(system):
    grid, params, xyz, radii, charges = system
    direction = torch.tensor([[.1, -.2, .3], [-.3, .2, .1], [.2, -.1, -.2], [.1, .3, -.1], [-.1, .2, .3]], dtype=xyz.dtype)

    def channels(coordinates, builder):
        return receptor_channels(builder(coordinates, radii, charges, grid, params), grid, params)

    value, derivative = torch.func.jvp(lambda value: channels(value, build_sparse_fields), (xyz,), (direction,))
    expected_value, expected_derivative = torch.func.jvp(lambda value: channels(value, build_fields), (xyz,), (direction,))
    torch.testing.assert_close(value, expected_value, atol=2e-12, rtol=2e-11)
    torch.testing.assert_close(derivative, expected_derivative, atol=2e-11, rtol=2e-10)
    finite_difference = (channels(xyz + 1e-5 * direction, build_fields) - channels(xyz - 1e-5 * direction, build_fields)) / 2e-5
    torch.testing.assert_close(derivative, finite_difference, atol=2e-8, rtol=2e-7)


@pytest.mark.parametrize("position", [-0.8, 0.0, 0.8])
def test_crossing_integer_cell_boundary_preserves_value_and_derivative(system, position):
    grid, params, _, radii, charges = system
    xyz = torch.tensor([[position, 0.11, -0.23]], dtype=torch.float64)
    direction = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    for delta in (-1e-8, 0.0, 1e-8):
        coordinate = xyz + delta * direction
        _, actual = torch.func.jvp(lambda point: build_sparse_fields(point, radii[:1], charges[:1], grid, params).charge_density,
                                   (coordinate,), (direction,))
        _, expected = torch.func.jvp(lambda point: build_fields(point, radii[:1], charges[:1], grid, params).charge_density,
                                     (coordinate,), (direction,))
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-11)
    plus = build_sparse_fields(xyz + 1e-6 * direction, radii[:1], charges[:1], grid, params).charge_density
    minus = build_sparse_fields(xyz - 1e-6 * direction, radii[:1], charges[:1], grid, params).charge_density
    torch.testing.assert_close((plus - minus) / 2e-6, expected, atol=2e-6, rtol=2e-6)


def test_stencil_can_extend_outside_grid_only_when_true_support_is_complete(system):
    grid, params, _, radii, charges = system
    support = support_radii(radii[:1], params)[0]
    lower, upper = grid.bounds(dtype=radii.dtype)
    xyz = torch.stack((lower + support, upper - support))
    atom_radii, atom_charges = radii[:1].expand(2), charges[:1].expand(2)
    sparse = build_sparse_fields(xyz, atom_radii, atom_charges, grid, params)
    dense = build_fields(xyz, atom_radii, atom_charges, grid, params)
    for name in ("occupancy", "surface", "charge_density"):
        torch.testing.assert_close(getattr(sparse, name), getattr(dense, name), atol=2e-13, rtol=2e-12)
    with pytest.raises(ValueError, match="compact support"):
        build_sparse_fields(xyz + xyz.new_tensor([0.01, 0.0, 0.0]), atom_radii, atom_charges, grid, params)


def test_chunk_size_does_not_change_field_or_gradient(system):
    grid, params, xyz, radii, charges = system
    gradients = []
    for size in (1, 2, 7):
        coordinate = xyz.clone().requires_grad_()
        fields = build_sparse_fields(coordinate, radii, charges, grid, replace(params, atom_chunk_size=size), checkpoint_chunks=True)
        objective = fields.occupancy.square().sum() + fields.surface.square().sum() + fields.charge_density.square().sum()
        gradients.append(torch.autograd.grad(objective, coordinate)[0])
    for actual in gradients[1:]:
        torch.testing.assert_close(actual, gradients[0], atol=2e-12, rtol=2e-12)


def test_sparse_checkpoint_is_reverse_only_and_parameter_validation_remains(system):
    grid, params, xyz, radii, charges = system
    with pytest.raises(ValueError, match="reverse-mode only"):
        torch.func.jvp(lambda point: build_sparse_fields(point, radii, charges, grid, params, checkpoint_chunks=True).occupancy,
                       (xyz,), (torch.ones_like(xyz),))
    with pytest.raises(TypeError, match="bool"):
        build_sparse_fields(xyz, radii, charges, grid, params, checkpoint_chunks="yes")
