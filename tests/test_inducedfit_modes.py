"""Apo-only ANM, anchored projection, and consistent nonlinear prior checks."""

import pytest
import torch

from zdock.inducedfit.modes import build_anm, build_torsion_modes

pytestmark = pytest.mark.filterwarnings("ignore:`torch.jit.script` is deprecated.*:DeprecationWarning")


class ToyTorsions:
    """Numerical articulated chain, not a chemical protein model.

    Three initial points are anchored.  Every subsequent point is connected
    through a rotatable joint; distances and angles along the chain persist.
    This independent small map tests mode projection without PDB fixtures.
    """

    def __init__(self, dtype=torch.float64, n_torsions=4):
        self.reference = torch.tensor([
            [0.0, 0.0, 0.0], [1.3, 0.2, 0.1], [1.8, 1.3, -0.1],
            [2.7, 1.9, 1.1], [3.5, 1.0, 1.8], [4.6, 1.5, 2.0],
            [5.1, 2.8, 1.2],
        ], dtype=dtype)
        self.ca_indices = torch.arange(len(self.reference))
        self.n_torsions = n_torsions

    def coordinates(self, angles):
        xyz = self.reference
        for index in range(self.n_torsions):
            first, second = index + 1, index + 2
            axis = xyz[second] - xyz[first]
            axis = axis / torch.linalg.vector_norm(axis)
            relative = xyz - xyz[second]
            cosine, sine = torch.cos(angles[index]), torch.sin(angles[index])
            rotated = relative * cosine + torch.linalg.cross(axis.expand_as(relative), relative) * sine
            rotated = rotated + (relative * axis).sum(-1, keepdim=True) * axis * (1 - cosine) + xyz[second]
            moving = torch.arange(len(xyz)) > second
            xyz = torch.where(moving[:, None], rotated, xyz)
        return xyz


def tetrahedron(dtype=torch.float64):
    return torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 1.1, 0.0], [0.3, 0.4, 1.2]], dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_anm_removes_all_rigid_zero_modes(dtype):
    xyz = tetrahedron(dtype)
    model = build_anm(xyz)
    assert model.n_modes == 6
    assert model.diagnostics["n_zero_modes"] == 6
    assert model.diagnostics["n_components"] == 1
    assert model.diagnostics["n_edges"] == 6
    eigenvectors = model.modes.reshape(6, 12)
    torch.testing.assert_close(eigenvectors @ eigenvectors.T, torch.eye(6, dtype=dtype))
    torch.testing.assert_close(model.hessian @ eigenvectors.T, eigenvectors.T * model.eigenvalues)
    for translation in torch.eye(3, dtype=dtype):
        vector = translation.expand_as(xyz).reshape(-1)
        torch.testing.assert_close(model.hessian @ vector, torch.zeros_like(vector), atol=1e-6 if dtype == torch.float32 else 1e-13, rtol=0)
    for axis in torch.eye(3, dtype=dtype):
        vector = torch.linalg.cross(axis.expand_as(xyz), xyz).reshape(-1)
        torch.testing.assert_close(model.hessian @ vector, torch.zeros_like(vector), atol=1e-6 if dtype == torch.float32 else 1e-13, rtol=0)


def test_anm_hessian_matches_nonlinear_central_spring_energy():
    xyz = tetrahedron()
    model = build_anm(xyz, gamma=2.7)

    def energy(flat):
        coordinates = flat.reshape(-1, 3)
        differences = coordinates[model.pairs[:, 1]] - coordinates[model.pairs[:, 0]]
        residual = torch.linalg.vector_norm(differences, dim=-1) - model.reference_distances
        return 0.5 * 2.7 * residual.square().sum()

    hessian = torch.func.hessian(energy)(xyz.reshape(-1))
    torch.testing.assert_close(model.hessian, hessian, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(torch.func.grad(energy)(xyz.reshape(-1)), torch.zeros(12, dtype=xyz.dtype), atol=1e-13, rtol=0)


def test_connected_underconstrained_network_removes_extra_zero_modes():
    model = build_anm(torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64), cutoff=1.1)
    assert model.diagnostics["n_components"] == 1
    assert model.diagnostics["n_zero_modes"] == 7
    assert model.n_modes == 2


def test_disconnected_network_is_rejected_or_explicitly_reported():
    xyz = torch.cat([tetrahedron(), tetrahedron() + 20])
    with pytest.raises(ValueError, match="disconnected"):
        build_anm(xyz, cutoff=3)
    model = build_anm(xyz, cutoff=3, reject_disconnected=False)
    assert model.diagnostics["n_components"] == 2
    assert model.diagnostics["n_zero_modes"] == 12
    assert model.n_modes == 12


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_actual_torsion_jacobian_is_rms_normalized(dtype):
    kinematics = ToyTorsions(dtype)
    model = build_torsion_modes(kinematics, n_modes=3)
    assert model.n_modes == 3
    assert model.basis.shape == (3, 7, 3)
    assert model.angle_basis.shape == (4, 3)
    assert model.reference is kinematics.reference
    assert model.ca_indices is kinematics.ca_indices
    jacobian = torch.func.jacrev(model.coordinates)(torch.zeros(3, dtype=dtype))
    torch.testing.assert_close(model.basis, jacobian.permute(2, 0, 1))
    vectors = model.basis[:, model.ca_indices].reshape(3, -1)
    tolerance = 2e-4 if dtype == torch.float32 else 1e-11
    torch.testing.assert_close(vectors @ vectors.T / 7, torch.eye(3, dtype=dtype), atol=tolerance, rtol=tolerance)
    assert model.diagnostics["gauge"] == "anchored_actual_CA_Jacobian_no_alignment"
    assert len(model.diagnostics["fit_relative_residuals"]) == 3
    assert not model.diagnostics["rank_reduced"]


def test_coordinate_basis_matches_finite_differences_and_preserves_anchors():
    model = build_torsion_modes(ToyTorsions(), n_modes=3)
    step = 1e-6
    for index in range(3):
        delta = torch.zeros(3, dtype=torch.float64)
        delta[index] = step
        difference = (model.coordinates(delta) - model.coordinates(-delta)) / (2 * step)
        torch.testing.assert_close(difference, model.basis[index], atol=2e-8, rtol=2e-8)
    amplitudes = torch.tensor([0.05, -0.1, 0.03], dtype=torch.float64)
    moved = model.coordinates(amplitudes)
    torch.testing.assert_close(moved[:3], model.reference[:3], atol=0, rtol=0)
    original_bonds = torch.linalg.vector_norm(model.reference[1:] - model.reference[:-1], dim=-1)
    moved_bonds = torch.linalg.vector_norm(moved[1:] - moved[:-1], dim=-1)
    torch.testing.assert_close(moved_bonds, original_bonds, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("regularization", [0.0, 1e-4, 0.2])
def test_stiffness_is_hessian_of_same_nonlinear_prior(regularization):
    model = build_torsion_modes(ToyTorsions(), n_modes=3, relative_regularization=regularization)
    origin = torch.zeros(3, dtype=torch.float64)
    assert model.deformation_energy(origin).item() == pytest.approx(0, abs=1e-25)
    torch.testing.assert_close(torch.func.grad(model.deformation_energy)(origin), origin, atol=1e-13, rtol=0)
    torch.testing.assert_close(torch.func.hessian(model.deformation_energy)(origin), model.stiffness, atol=1e-10, rtol=1e-10)
    assert bool((torch.linalg.eigvalsh(model.stiffness) > 0).all())
    assert model.regularization_strength == model.diagnostics["regularization_strength"]
    perturbation = torch.tensor([1.0, -0.7, 0.3], dtype=torch.float64) * 1e-4
    quadratic = 0.5 * perturbation @ model.stiffness @ perturbation
    assert model.deformation_energy(perturbation).item() == pytest.approx(quadratic.item(), rel=1e-3)
    gradient_fd = torch.stack([
        (torch.func.grad(model.deformation_energy)(torch.eye(3, dtype=torch.float64)[index] * 1e-5)
         - torch.func.grad(model.deformation_energy)(-torch.eye(3, dtype=torch.float64)[index] * 1e-5)) / 2e-5
        for index in range(3)
    ], dim=1)
    torch.testing.assert_close(gradient_fd, model.stiffness, atol=2e-7, rtol=2e-7)


def test_rank_reduction_requires_explicit_opt_in_and_is_reported():
    kinematics = ToyTorsions(n_torsions=2)
    with pytest.raises(ValueError, match="numerical rank 2"):
        build_torsion_modes(kinematics, n_modes=4)
    model = build_torsion_modes(kinematics, n_modes=4, allow_rank_reduction=True)
    assert model.n_modes == 2
    assert model.diagnostics["requested_modes"] == 4
    assert model.diagnostics["actual_modes"] == 2
    assert model.diagnostics["rank_reduced"]


def test_angular_damping_avoids_inverting_weak_leverage_directions():
    class WeakLeverageMap:
        def __init__(self):
            self.reference = tetrahedron()
            self.ca_indices = torch.arange(4)
            self.n_torsions = 2
            anm = build_anm(self.reference)
            # The desired lowest ANM direction requires both coordinate
            # directions, but the second has extremely weak atomic leverage.
            self.strong = (anm.modes[0] + anm.modes[1]) / 2**0.5
            self.weak = 1e-6 * (anm.modes[0] - anm.modes[1]) / 2**0.5

        def coordinates(self, angles):
            return self.reference + angles[0].sin() * self.strong + angles[1].sin() * self.weak

    undamped = build_torsion_modes(WeakLeverageMap(), n_modes=1, fit_relative_damping=0)
    damped = build_torsion_modes(WeakLeverageMap(), n_modes=1, fit_relative_damping=1e-3)
    assert undamped.angle_basis.abs().amax() > 1e6
    assert damped.angle_basis.abs().amax() < 3
    assert damped.diagnostics["fit_relative_residuals"][0] > undamped.diagnostics["fit_relative_residuals"][0]
    origin = torch.zeros(1, dtype=torch.float64)
    torch.testing.assert_close(torch.func.hessian(damped.deformation_energy)(origin), damped.stiffness)
    a = torch.tensor([0.1], dtype=torch.float64)
    actual_rmsd = ((damped.coordinates(a) - damped.reference).square().sum() / 4).sqrt()
    assert actual_rmsd.item() == pytest.approx(0.1, rel=0.01)
    assert damped.diagnostics["max_angle_per_unit_amplitude_rad_per_A"] < 3


def test_zero_modes_have_unchanged_coordinates_and_zero_prior():
    model = build_torsion_modes(ToyTorsions(), n_modes=0)
    amplitudes = torch.empty(0, dtype=torch.float64)
    torch.testing.assert_close(model.coordinates(amplitudes), model.reference)
    assert model.deformation_energy(amplitudes).item() == pytest.approx(0, abs=1e-25)
    assert model.basis.shape == (0, 7, 3)
    assert model.stiffness.shape == (0, 0)


@pytest.mark.parametrize("keyword, value", [("cutoff", 0), ("gamma", -1), ("cutoff", float("nan"))])
def test_anm_invalid_parameters(keyword, value):
    with pytest.raises(ValueError, match=keyword):
        build_anm(tetrahedron(), **{keyword: value})


def test_bad_coordinates_and_map_metadata_fail_clearly():
    with pytest.raises(ValueError, match="coincident"):
        build_anm(torch.zeros((3, 3), dtype=torch.float64))
    with pytest.raises(TypeError, match="float32 or float64"):
        build_anm(tetrahedron(torch.float16))
    broken = ToyTorsions()
    broken.ca_indices = torch.tensor([0, 1, 1])
    with pytest.raises(ValueError, match="unique"):
        build_torsion_modes(broken)
    with pytest.raises(ValueError, match="nonnegative"):
        build_torsion_modes(ToyTorsions(), n_modes=-1)
    with pytest.raises(ValueError, match="relative_regularization"):
        build_torsion_modes(ToyTorsions(), relative_regularization=-0.1)
    with pytest.raises(ValueError, match="fit_relative_damping"):
        build_torsion_modes(ToyTorsions(), fit_relative_damping=-0.1)
    with pytest.raises(ValueError, match="torsion degrees"):
        build_torsion_modes(ToyTorsions(n_torsions=0))


def test_parameters_and_coordinates_do_not_depend_on_target_complex():
    # Rebuilding from identical apo coordinates produces the same model;
    # the API accepts no native/reference-complex labels or interface mask.
    first = build_torsion_modes(ToyTorsions(), n_modes=2)
    second = build_torsion_modes(ToyTorsions(), n_modes=2)
    torch.testing.assert_close(first.stiffness, second.stiffness)
    torch.testing.assert_close(first.angle_basis, second.angle_basis)
