"""Checkpoint parity, independent gradients, and retained-activation checks."""

from dataclasses import replace

import pytest
import torch

from zdock.inducedfit.fields import FieldParameters, build_fields, physical_energy
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
    grid = GridSpec((11, 11, 11), 0.9, (-4.5, -4.5, -4.5))
    params = FieldParameters(
        surface_width=0.2, occupancy_switch_start=2.0, occupancy_cutoff_sigma=3.0,
        charge_sigma=0.35, charge_switch_start=2.0, charge_cutoff_sigma=3.0,
        screening_switch_start=1.8, screening_cutoff=2.7,
        atom_chunk_size=2, point_chunk_size=127,
        repulsion_weight=2.3, charge_weight=0.7,
    )
    xyz = torch.tensor([
        [-0.8, 0.23, 0.1], [0.6, -0.37, -0.2], [0.11, 0.7, -0.5],
        [-0.5, -0.6, 0.35], [0.7, 0.2, 0.6],
    ], dtype=torch.float64)
    radii = torch.tensor([0.47, 0.62, 0.55, 0.49, 0.61], dtype=torch.float64)
    charges = torch.tensor([1.0, -0.4, 0.7, -0.9, 0.2], dtype=torch.float64)
    return grid, params, xyz, radii, charges


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_forward_fields_and_no_grad_are_identical(system, dtype):
    grid, params, xyz, radii, charges = system
    xyz, radii, charges = (tensor.to(dtype).requires_grad_() for tensor in (xyz, radii, charges))
    plain = build_fields(xyz, radii, charges, grid, params)
    checkpointed = build_fields(xyz, radii, charges, grid, params, checkpoint_chunks=True)
    with torch.no_grad():
        no_grad = build_fields(xyz, radii, charges, grid, params, checkpoint_chunks=True)
    for name in ("occupancy", "surface", "charge_density", "support_lower", "support_upper"):
        torch.testing.assert_close(getattr(plain, name), getattr(checkpointed, name), atol=0, rtol=0)
        torch.testing.assert_close(getattr(plain, name), getattr(no_grad, name), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_energy_and_all_input_gradients_match_independent_spatial_energy(system, dtype):
    grid, params, xyz, radii, charges = system
    xyz, radii, charges = (tensor.to(dtype) for tensor in (xyz, radii, charges))
    ligand = build_fields(xyz + xyz.new_tensor([0.6, -0.1, 0.3]), radii, -charges, grid, params)

    def compute(checkpointed, direct):
        inputs = tuple(value.clone().requires_grad_() for value in (xyz, radii, charges))
        fields = build_fields(*inputs, grid, params, checkpoint_chunks=checkpointed)
        energy = physical_energy(fields, ligand, grid, params, direct=direct)
        return energy.detach(), torch.autograd.grad(energy, inputs)

    expected, expected_gradients = compute(False, True)
    actual, actual_gradients = compute(True, False)
    tolerance = 2e-5 if dtype == torch.float32 else 2e-11
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=tolerance, rtol=tolerance)


def test_checkpointed_charge_conservation_includes_normalizer_gradient(system):
    grid, params, xyz, radii, charges = system
    xyz, charges = xyz.requires_grad_(), charges.requires_grad_()
    fields = build_fields(xyz, radii, charges, grid, params, checkpoint_chunks=True)
    total_charge = fields.charge_density.sum() * grid.voxel_volume
    coordinate_gradient, charge_gradient = torch.autograd.grad(total_charge, (xyz, charges))
    torch.testing.assert_close(total_charge, charges.sum(), atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(coordinate_gradient, torch.zeros_like(xyz), atol=2e-13, rtol=0)
    torch.testing.assert_close(charge_gradient, torch.ones_like(charges), atol=2e-13, rtol=0)


def test_weighted_charge_gradient_matches_rebuilt_finite_difference(system):
    grid, params, xyz, radii, charges = system
    # A coarse grid relative to charge_sigma gives a strongly position-
    # dependent discrete normalizer; detaching it would fail this check.
    coordinate = xyz.clone().requires_grad_()
    points = grid.points(dtype=xyz.dtype)
    weight = (points[:, 0].sin() + 0.3 * points[:, 1].square() - points[:, 2].cos()).reshape(grid.shape)
    fields = build_fields(coordinate, radii, charges, grid, params, checkpoint_chunks=True)
    objective = (fields.charge_density * weight).sum() * grid.voxel_volume
    gradient, = torch.autograd.grad(objective, coordinate)
    direction = torch.tensor([
        [0.1, -0.2, 0.3], [-0.3, 0.1, 0.2], [0.4, -0.3, 0.1],
        [0.2, 0.1, -0.4], [-0.1, -0.2, 0.2],
    ], dtype=xyz.dtype)

    def value(shift):
        rebuilt = build_fields(xyz + shift * direction, radii, charges, grid, params)
        return (rebuilt.charge_density * weight).sum() * grid.voxel_volume

    finite_difference = (value(1e-5) - value(-1e-5)) / 2e-5
    torch.testing.assert_close((gradient * direction).sum(), finite_difference, atol=2e-9, rtol=2e-9)


def test_checkpointing_does_not_capture_last_loop_chunk(system):
    grid, params, xyz, radii, charges = system
    results = []
    for atom_chunk, point_chunk in ((1, 67), (2, 127), (3, 211), (20, 2000)):
        config = replace(params, atom_chunk_size=atom_chunk, point_chunk_size=point_chunk)
        inputs = tuple(value.clone().requires_grad_() for value in (xyz, radii, charges))
        fields = build_fields(*inputs, grid, config, checkpoint_chunks=True)
        objective = fields.occupancy.square().sum() + fields.surface.square().sum() + fields.charge_density.square().sum()
        results.append(torch.autograd.grad(objective, inputs))
    for gradients in results[1:]:
        for actual, expected in zip(gradients, results[0]):
            torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)


def test_checkpointing_reduces_unique_storage_retained_for_backward(system):
    grid, params, xyz, radii, charges = system

    def saved_bytes(checkpointed):
        coordinate = xyz.clone().requires_grad_()
        storages = {}

        def pack(tensor):
            storage = tensor.untyped_storage()
            storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            fields = build_fields(coordinate, radii, charges, grid, params, checkpoint_chunks=checkpointed)
            objective = fields.occupancy.square().sum() + fields.surface.square().sum() + fields.charge_density.square().sum()
        # This measures unique forward-saved tensor storage, not process RSS
        # or backward recomputation peak; the paired process benchmark does that.
        return sum(storages.values()), objective

    plain_bytes, plain_objective = saved_bytes(False)
    checkpoint_bytes, checkpoint_objective = saved_bytes(True)
    torch.testing.assert_close(checkpoint_objective, plain_objective, atol=0, rtol=0)
    assert checkpoint_bytes < plain_bytes / 4


def test_default_jvp_remains_supported_and_checkpoint_jvp_is_explicitly_rejected(system):
    grid, params, xyz, radii, charges = system

    def plain(coordinate):
        return build_fields(coordinate, radii, charges, grid, params).charge_density

    direction = torch.ones_like(xyz) * 0.1
    _, derivative = torch.func.jvp(plain, (xyz,), (direction,))
    assert bool(torch.isfinite(derivative).all())
    with pytest.raises(ValueError, match="reverse-mode only"):
        torch.func.jvp(lambda coordinate: build_fields(coordinate, radii, charges, grid, params, checkpoint_chunks=True).charge_density,
                       (xyz,), (direction,))
    with pytest.raises(TypeError, match="bool"):
        build_fields(xyz, radii, charges, grid, params, checkpoint_chunks="yes")
