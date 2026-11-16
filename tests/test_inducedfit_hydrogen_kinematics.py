"""Graph-cut hydrogen motion, covalent geometry, and derivative contracts."""

from dataclasses import replace
import itertools
import math
from types import SimpleNamespace

import pytest
import torch

from zdock.inducedfit.hydrogen_kinematics import HydrogenatedKinematics
from zdock.inducedfit.kinematics import TorsionKinematics
from zdock.inducedfit.modes import build_torsion_modes
from zdock.inducedfit.types import AtomKey, MonomerInput


pytestmark = pytest.mark.filterwarnings("ignore:`torch.jit.script` is deprecated.*:DeprecationWarning")


def make_prepared(sequence=("ALA", "GLY", "ALA", "ALA"), *, dtype=torch.float64):
    """Independent synthetic topology, not an Amber charge/valence fixture.

    Includes H on N, CA and sidechain C, with deliberately interleaved atom
    ordering.  Backbone and optional proline ring have valid heavy geometry.
    """
    coordinates, keys = [], []

    def backbone(index):
        block = index // 2
        if index % 2 == 0:
            n = torch.tensor([5.83 * block, 4.015 * block, 0.], dtype=dtype)
            ca = n + n.new_tensor([1.45, 0., 0.])
            c = ca + ca.new_tensor([.55, 1.42, 0.])
        else:
            n = torch.tensor([5.83 * block + 3.32, 4.015 * block + 1.42, 0.], dtype=dtype)
            ca = n + n.new_tensor([.53, 1.35, 0.])
            c = ca + ca.new_tensor([1.52, 0., 0.])
        return n, ca, c

    for index, name in enumerate(sequence):
        n, ca, c = backbone(index)
        next_n, _, _ = backbone(index + 1)
        direction = (ca - c) / (ca - c).norm() + (next_n - c) / (next_n - c).norm()
        atoms = {"N": n, "CA": ca, "C": c, "O": c - 1.23 * direction / direction.norm()}
        if name == "PRO":
            axis, vertical, last = (ca - n) / (ca - n).norm(), n.new_tensor([0., 0., 1.]), ca
            for atom, step in zip(("CB", "CG", "CD"), (1, 2, 3)):
                delta = math.cos(step * 2 * math.pi / 5) * axis + math.sin(step * 2 * math.pi / 5) * vertical
                last = last + (ca - n).norm() * delta
                atoms[atom] = last
        elif name != "GLY":
            atoms["CB"] = ca + ca.new_tensor([0., 0., 1.53])
        for atom_name, coordinate in atoms.items():
            coordinates.append(coordinate)
            keys.append(AtomKey("A", index + 1, "", name, atom_name))
    heavy_xyz = torch.stack(coordinates)
    heavy = MonomerInput(heavy_xyz, torch.full((len(keys),), 1.5, dtype=dtype), torch.zeros(len(keys), dtype=dtype), tuple(keys))
    initial = TorsionKinematics.from_monomer(heavy)
    # Make a nonplanar reference while preserving the independent bond graph.
    initial_angles = torch.linspace(-0.2, 0.3, initial.n_torsions, dtype=dtype)
    heavy = replace(heavy, xyz=initial.coordinates(initial_angles))
    index_by_name = {(key.residue_number, key.atom_name): index for index, key in enumerate(keys)}
    heavy_bonds = []
    for residue in range(1, len(sequence) + 1):
        for first, second in (("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"), ("CG", "CD")):
            if (residue, first) in index_by_name and (residue, second) in index_by_name:
                heavy_bonds.append((index_by_name[residue, first], index_by_name[residue, second]))
        if sequence[residue - 1] == "PRO":
            heavy_bonds.append((index_by_name[residue, "CD"], index_by_name[residue, "N"]))
        if residue < len(sequence):
            heavy_bonds.append((index_by_name[residue, "C"], index_by_name[residue + 1, "N"]))

    full_xyz, full_keys, heavy_indices, hydrogen_bonds = [], [], [], []
    for index, (coordinate, key) in enumerate(zip(heavy.xyz, keys)):
        parent = len(full_xyz)
        heavy_indices.append(parent)
        full_xyz.append(coordinate)
        full_keys.append(key)
        h_name = {"N": "H", "CA": "HA", "CB": "HB1", "CG": "HG1", "CD": "HD1"}.get(key.atom_name)
        if h_name and not (key.residue_name == "PRO" and key.atom_name == "N"):
            direction = coordinate.new_tensor([0.31 + 0.02 * index, -0.7, 0.6])
            full_xyz.append(coordinate + 1.03 * direction / direction.norm())
            full_keys.append(replace(key, atom_name=h_name))
            hydrogen_bonds.append((parent, len(full_xyz) - 1))
    xyz = torch.stack(full_xyz)
    full = MonomerInput(xyz, torch.full((len(xyz),), 1.2, dtype=dtype), torch.linspace(-0.3, 0.3, len(xyz), dtype=dtype), tuple(full_keys))
    bonds = tuple((heavy_indices[a], heavy_indices[b]) for a, b in heavy_bonds) + tuple(hydrogen_bonds)
    return SimpleNamespace(monomer=full, heavy_monomer=heavy, heavy_indices=torch.tensor(heavy_indices), bonds=bonds, provenance={"fixture": "synthetic_geometry_only"})


def geometry(xyz, bonds):
    neighbors = [set() for _ in range(len(xyz))]
    for first, second in bonds:
        neighbors[first].add(second)
        neighbors[second].add(first)
    edge_indices = torch.tensor(bonds)
    triples = torch.tensor([(first, middle, second) for middle, linked in enumerate(neighbors) for first, second in itertools.combinations(sorted(linked), 2)])
    lengths = (xyz[edge_indices[:, 0]] - xyz[edge_indices[:, 1]]).norm(dim=-1)
    first = xyz[triples[:, 0]] - xyz[triples[:, 1]]
    second = xyz[triples[:, 2]] - xyz[triples[:, 1]]
    cosines = (first * second).sum(-1) / (first.norm(dim=-1) * second.norm(dim=-1))
    return lengths, cosines


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_full_map_preserves_bond_lengths_angles_and_heavy_motion(dtype):
    prepared = make_prepared(dtype=dtype)
    charges_before = prepared.monomer.charges.clone()
    model = HydrogenatedKinematics.from_prepared(prepared)
    angles = torch.linspace(-0.45, 0.35, model.n_torsions, dtype=dtype)
    moved = model.coordinates(angles)
    heavy_moved = model.heavy_kinematics.coordinates(angles)
    tolerance = 2e-5 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(moved[model.heavy_indices], heavy_moved, atol=tolerance, rtol=tolerance)
    for before, after in zip(geometry(model.reference, model.bonds), geometry(moved, model.bonds)):
        torch.testing.assert_close(after, before, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(prepared.monomer.charges, charges_before, atol=0, rtol=0)
    torch.testing.assert_close(model.coordinates(torch.zeros_like(angles)), prepared.monomer.xyz, atol=tolerance, rtol=tolerance)
    assert model.provenance["n_hydrogens"] == prepared.monomer.n_atoms - prepared.heavy_monomer.n_atoms


def test_phi_moves_alpha_hydrogen_but_keeps_amide_hydrogen():
    model = HydrogenatedKinematics.from_prepared(make_prepared())
    index = {(key.residue_number, key.atom_name): i for i, key in enumerate(model.atom_keys)}
    joint_index = next(i for i, joint in enumerate(model.joints) if joint.kind == "phi" and joint.residue_index == 1)
    angles = torch.zeros(model.n_torsions, dtype=torch.float64)
    angles[joint_index] = 0.5
    moved = model.coordinates(angles)
    torch.testing.assert_close(moved[index[2, "H"]], model.reference[index[2, "H"]], atol=0, rtol=0)
    torch.testing.assert_close(moved[index[2, "CA"]], model.reference[index[2, "CA"]], atol=1e-13, rtol=0)
    assert (moved[index[2, "HA"]] - model.reference[index[2, "HA"]]).norm() > 0.1
    assert index[2, "CA"] in model.joints[joint_index].moving_indices


def test_psi_keeps_alpha_hydrogen_and_moves_carbonyl_and_downstream_hydrogen():
    model = HydrogenatedKinematics.from_prepared(make_prepared())
    index = {(key.residue_number, key.atom_name): i for i, key in enumerate(model.atom_keys)}
    joint_index = next(i for i, joint in enumerate(model.joints) if joint.kind == "psi" and joint.residue_index == 0)
    angles = torch.zeros(model.n_torsions, dtype=torch.float64)
    angles[joint_index] = 0.5
    moved = model.coordinates(angles)
    torch.testing.assert_close(moved[index[1, "HA"]], model.reference[index[1, "HA"]], atol=0, rtol=0)
    assert (moved[index[1, "O"]] - model.reference[index[1, "O"]]).norm() > 0.1
    assert (moved[index[2, "H"]] - model.reference[index[2, "H"]]).norm() > 0.1


def test_proline_phi_omitted_and_ring_preserved():
    prepared = make_prepared(("ALA", "PRO", "GLY", "ALA"))
    model = HydrogenatedKinematics.from_prepared(prepared)
    assert not any(joint.kind == "phi" and joint.residue_index == 1 for joint in model.joints)
    moved = model.coordinates(torch.linspace(-0.3, 0.4, model.n_torsions, dtype=torch.float64))
    for before, after in zip(geometry(model.reference, model.bonds), geometry(moved, model.bonds)):
        torch.testing.assert_close(after, before, atol=2e-12, rtol=2e-12)


def test_full_map_derivatives_and_mode_model_compatibility():
    model = HydrogenatedKinematics.from_prepared(make_prepared())
    origin = torch.zeros(model.n_torsions, dtype=torch.float64)
    direction = torch.linspace(-0.2, 0.3, model.n_torsions, dtype=torch.float64)
    _, derivative = torch.func.jvp(model.coordinates, (origin,), (direction,))
    finite_difference = (model.coordinates(direction * 1e-6) - model.coordinates(-direction * 1e-6)) / 2e-6
    torch.testing.assert_close(derivative, finite_difference, atol=2e-8, rtol=2e-8)
    forward = torch.func.jacfwd(model.coordinates)(origin)
    reverse = torch.func.jacrev(model.coordinates)(origin)
    torch.testing.assert_close(forward, reverse, atol=1e-12, rtol=1e-12)
    modes = build_torsion_modes(model, n_modes=2)
    assert modes.basis.shape == (2, len(model.reference), 3)
    torch.testing.assert_close(torch.func.hessian(modes.deformation_energy)(torch.zeros(2, dtype=torch.float64)), modes.stiffness)


def test_full_graph_crosslink_that_closes_joint_is_rejected():
    prepared = make_prepared()
    # Add a topology-only link without moving coordinates.  Distance-only
    # heavy validation cannot see it, whereas the full graph cut must fail.
    prepared.bonds += ((int(prepared.heavy_indices[0]), int(prepared.heavy_indices[-1])),)
    with pytest.raises(ValueError, match="ring or crosslink"):
        HydrogenatedKinematics.from_prepared(prepared)


@pytest.mark.parametrize("mutation", ["unbound", "double_bound", "H_H"])
def test_hydrogens_require_exactly_one_heavy_parent(mutation):
    prepared = make_prepared()
    hydrogens = [i for i, key in enumerate(prepared.monomer.atom_keys) if key.atom_name.startswith("H")]
    first, second = hydrogens[:2]
    bonds = [pair for pair in prepared.bonds if first not in pair]
    if mutation == "double_bound":
        bonds += [(first, int(prepared.heavy_indices[0])), (first, int(prepared.heavy_indices[1]))]
    elif mutation == "H_H":
        bonds += [(first, second)]
    prepared.bonds = tuple(bonds)
    with pytest.raises(ValueError, match="exactly one bond to a heavy atom"):
        HydrogenatedKinematics.from_prepared(prepared)


def test_mapping_and_graph_errors_are_not_silently_repaired():
    prepared = make_prepared()
    prepared.heavy_indices = prepared.heavy_indices.flip(0)
    with pytest.raises(ValueError, match="identities/order"):
        HydrogenatedKinematics.from_prepared(prepared)
    prepared = make_prepared()
    changed = prepared.monomer.xyz.clone()
    changed[prepared.heavy_indices[0], 0] += 0.01
    prepared.monomer = replace(prepared.monomer, xyz=changed)
    with pytest.raises(ValueError, match="changed the heavy"):
        HydrogenatedKinematics.from_prepared(prepared)
    prepared = make_prepared()
    prepared.bonds += (prepared.bonds[0],)
    with pytest.raises(ValueError, match="duplicate"):
        HydrogenatedKinematics.from_prepared(prepared)
    prepared = make_prepared()
    prepared.bonds = prepared.bonds[:-1]
    with pytest.raises(ValueError, match="exactly one bond"):
        HydrogenatedKinematics.from_prepared(prepared)


def test_nonheavy_nonhydrogen_atom_and_residue_mismatch_rejected():
    prepared = make_prepared()
    index = next(i for i, key in enumerate(prepared.monomer.atom_keys) if key.atom_name == "H")
    keys = list(prepared.monomer.atom_keys)
    keys[index] = replace(keys[index], atom_name="OX")
    prepared.monomer = replace(prepared.monomer, atom_keys=tuple(keys))
    with pytest.raises(ValueError, match="must be a hydrogen"):
        HydrogenatedKinematics.from_prepared(prepared)
    prepared = make_prepared()
    keys = list(prepared.monomer.atom_keys)
    keys[index] = replace(keys[index], residue_number=999)
    prepared.monomer = replace(prepared.monomer, atom_keys=tuple(keys))
    with pytest.raises(ValueError, match="same residue"):
        HydrogenatedKinematics.from_prepared(prepared)
