"""Geometry invariants and independent derivatives of backbone torsions."""

from dataclasses import replace
import itertools
import math

import pytest
import torch

from zdock.inducedfit.kinematics import TorsionKinematics
from zdock.inducedfit.types import AtomKey, MonomerInput


pytestmark = pytest.mark.filterwarnings(
    "ignore:`torch.jit.script` is deprecated.*:DeprecationWarning"
)


def make_peptide(sequence=("ALA", "GLY", "SER", "ALA"), *, dtype=torch.float64):
    """Independent extended backbone with valid C--N distances and sidechains."""
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
        carbonyl_direction = (ca - c) / torch.linalg.vector_norm(ca - c)
        carbonyl_direction += (next_n - c) / torch.linalg.vector_norm(next_n - c)
        oxygen = c - 1.23 * carbonyl_direction / torch.linalg.vector_norm(carbonyl_direction)
        atoms = {"N": n, "CA": ca, "C": c, "O": oxygen}
        if name == "PRO":
            # A closed regular pentagon containing the existing N--CA edge.
            axis = (ca - n) / torch.linalg.vector_norm(ca - n)
            vertical = ca.new_tensor([0., 0., 1.])
            edge_length = torch.linalg.vector_norm(ca - n)
            last = ca
            for atom, step in zip(("CB", "CG", "CD"), (1, 2, 3)):
                direction = math.cos(step * 2 * math.pi / 5) * axis + math.sin(step * 2 * math.pi / 5) * vertical
                last = last + edge_length * direction
                atoms[atom] = last
        elif name != "GLY":
            atoms["CB"] = ca + ca.new_tensor([0., 0., 1.53])
            if name == "SER":
                atoms["OG"] = atoms["CB"] + ca.new_tensor([0., 0., 1.42])
            if name == "CYS":
                atoms["SG"] = atoms["CB"] + ca.new_tensor([0., 0., 1.81])
        for atom, xyz in atoms.items():
            coordinates.append(xyz)
            keys.append(AtomKey("A", index + 1, "", name, atom))
    xyz = torch.stack(coordinates)
    return MonomerInput(xyz, torch.full((len(xyz),), 1.5, dtype=dtype),
                        torch.zeros(len(xyz), dtype=dtype), tuple(keys))


def _index(monomer):
    return {(key.residue_number, key.atom_name): i for i, key in enumerate(monomer.atom_keys)}


def _bond_graph(monomer):
    index = _index(monomer)
    residues = sorted({key.residue_number for key in monomer.atom_keys})
    edges = []
    for rid in residues:
        for a, b in (("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"),
                     ("CB", "OG"), ("CB", "SG"), ("CB", "CG"), ("CG", "CD")):
            if (rid, a) in index and (rid, b) in index:
                edges.append((index[rid, a], index[rid, b]))
        key = monomer.atom_keys[index[rid, "N"]]
        if key.residue_name == "PRO":
            edges.append((index[rid, "CD"], index[rid, "N"]))
    for left, right in zip(residues[:-1], residues[1:]):
        edges.append((index[left, "C"], index[right, "N"]))
    neighbors = {i: [] for i in range(len(monomer.xyz))}
    for a, b in edges:
        neighbors[a].append(b)
        neighbors[b].append(a)
    angles = [(a, middle, b) for middle, ends in neighbors.items() for a, b in itertools.combinations(ends, 2)]
    return torch.tensor(edges), torch.tensor(angles)


def _geometry(coordinates, edges, angles):
    lengths = torch.linalg.vector_norm(coordinates[edges[:, 0]] - coordinates[edges[:, 1]], dim=-1)
    first = coordinates[angles[:, 0]] - coordinates[angles[:, 1]]
    second = coordinates[angles[:, 2]] - coordinates[angles[:, 1]]
    cosines = ((first * second).sum(-1)
               / (torch.linalg.vector_norm(first, dim=-1) * torch.linalg.vector_norm(second, dim=-1)))
    return lengths, cosines


def _dihedral(points):
    first = points[0] - points[1]
    axis = points[2] - points[1]
    axis = axis / torch.linalg.vector_norm(axis)
    last = points[3] - points[2]
    first = first - (first * axis).sum() * axis
    last = last - (last * axis).sum() * axis
    return torch.atan2((torch.linalg.cross(axis, first) * last).sum(), (first * last).sum())


def _wrapped(value):
    return torch.atan2(torch.sin(value), torch.cos(value))


def test_zero_angles_and_root_gauge_and_torsion_order():
    monomer = make_peptide()
    model = TorsionKinematics.from_monomer(monomer)
    assert model.n_torsions == 6
    assert [(joint.kind, joint.residue_index) for joint in model.joints] == [
        ("psi", 0), ("phi", 1), ("psi", 1), ("phi", 2), ("psi", 2), ("phi", 3),
    ]
    zero = torch.zeros(model.n_torsions, dtype=torch.float64)
    torch.testing.assert_close(model.coordinates(zero), monomer.xyz, atol=1e-14, rtol=1e-14)
    moved = model.coordinates(torch.linspace(-.5, .6, model.n_torsions, dtype=torch.float64))
    root = [_index(monomer)[1, atom] for atom in ("N", "CA", "C")]
    torch.testing.assert_close(moved[root], monomer.xyz[root], atol=0, rtol=0)
    assert bool(torch.isfinite(moved).all())
    assert float(torch.linalg.vector_norm(moved - monomer.xyz)) > 1


@pytest.mark.parametrize("sequence", [("ALA", "GLY", "SER", "ALA"), ("ALA", "PRO", "SER", "ALA")])
def test_all_bond_lengths_angles_and_peptide_omega_are_preserved(sequence):
    monomer = make_peptide(sequence)
    model = TorsionKinematics.from_monomer(monomer)
    angles = torch.linspace(-.7, .8, model.n_torsions, dtype=torch.float64)
    moved = model.coordinates(angles)
    edges, triples = _bond_graph(monomer)
    for before, after in zip(_geometry(monomer.xyz, edges, triples), _geometry(moved, edges, triples)):
        torch.testing.assert_close(before, after, atol=5e-13, rtol=5e-13)
    index = _index(monomer)
    for rid in range(1, len(sequence)):
        indices = [index[rid, "CA"], index[rid, "C"], index[rid + 1, "N"], index[rid + 1, "CA"]]
        assert abs(float(_wrapped(_dihedral(moved[indices]) - _dihedral(monomer.xyz[indices])))) < 1e-12
    for rid in range(1, len(sequence) + 1):
        if (rid, "CB") in index:
            ends = [index[rid, atom] for atom in ("N", "C", "CB")]
            ca = index[rid, "CA"]
            before = torch.linalg.det(monomer.xyz[ends] - monomer.xyz[ca])
            after = torch.linalg.det(moved[ends] - moved[ca])
            torch.testing.assert_close(before, after, atol=5e-12, rtol=5e-12)
    if "PRO" in sequence:
        assert not any(j.kind == "phi" and j.residue_index == sequence.index("PRO") for j in model.joints)


def test_independent_measured_phi_psi_changes_equal_requested_increments():
    monomer = make_peptide()
    model = TorsionKinematics.from_monomer(monomer)
    increments = torch.linspace(-.4, .5, model.n_torsions, dtype=torch.float64)
    moved, index = model.coordinates(increments), _index(monomer)
    for angle, joint in zip(increments, model.joints):
        rid = joint.residue_index + 1
        atoms = ([(rid - 1, "C"), (rid, "N"), (rid, "CA"), (rid, "C")] if joint.kind == "phi"
                 else [(rid, "N"), (rid, "CA"), (rid, "C"), (rid + 1, "N")])
        indices = [index[key] for key in atoms]
        measured = _wrapped(_dihedral(moved[indices]) - _dihedral(monomer.xyz[indices]))
        torch.testing.assert_close(measured, angle, atol=1e-12, rtol=1e-12)


def test_jvp_jacrev_jacfwd_and_nested_hessian_match_independent_differences():
    model = TorsionKinematics.from_monomer(make_peptide())
    angles = torch.linspace(-.15, .2, model.n_torsions, dtype=torch.float64)
    direction = torch.linspace(.1, -.13, model.n_torsions, dtype=torch.float64)
    _, tangent = torch.func.jvp(model.coordinates, (angles,), (direction,))
    h = 1e-5
    difference = (model.coordinates(angles + h * direction) - model.coordinates(angles - h * direction)) / (2 * h)
    torch.testing.assert_close(tangent, difference, atol=3e-10, rtol=1e-7)
    jacobian = torch.func.jacrev(model.coordinates)(angles)
    forward_jacobian = torch.func.jacfwd(model.coordinates)(angles)
    torch.testing.assert_close(jacobian, forward_jacobian, atol=5e-13, rtol=5e-13)
    torch.testing.assert_close(jacobian @ direction, tangent, atol=5e-13, rtol=5e-13)

    def objective(a):
        coordinates = model.coordinates(a)
        # Coordinate-dependent objective with a known independent finite difference.
        return .5 * (coordinates[-1] - coordinates[0]).square().sum()

    hessian = torch.func.hessian(objective)(angles)
    finite_difference = (torch.func.grad(objective)(angles + h * direction)
                         - torch.func.grad(objective)(angles - h * direction)) / (2 * h)
    torch.testing.assert_close(hessian @ direction, finite_difference, atol=5e-9, rtol=1e-7)
    torch.testing.assert_close(hessian, hessian.T, atol=1e-11, rtol=1e-11)


def test_shuffled_atom_order_retains_correct_chain_connectivity():
    monomer = make_peptide()
    permutation = torch.randperm(len(monomer.xyz), generator=torch.Generator().manual_seed(19))
    shuffled = MonomerInput(monomer.xyz[permutation], monomer.radii[permutation], monomer.charges[permutation],
                            tuple(monomer.atom_keys[i] for i in permutation.tolist()))
    original_model = TorsionKinematics.from_monomer(monomer)
    shuffled_model = TorsionKinematics.from_monomer(shuffled)
    angles = torch.linspace(-.3, .4, original_model.n_torsions, dtype=torch.float64)
    torch.testing.assert_close(shuffled_model.coordinates(angles), original_model.coordinates(angles)[permutation],
                               atol=1e-12, rtol=1e-12)


def test_single_residue_is_a_valid_zero_degree_map():
    monomer = make_peptide(("ALA",))
    model = TorsionKinematics.from_monomer(monomer)
    assert model.n_torsions == 0
    torch.testing.assert_close(model.coordinates(torch.empty(0, dtype=torch.float64)), monomer.xyz)
    assert torch.func.jacrev(model.coordinates)(torch.empty(0, dtype=torch.float64)).shape == (5, 3, 0)


def test_missing_backbone_hydrogen_nonstandard_and_multiple_chains_fail():
    monomer = make_peptide()
    keep = [i for i, key in enumerate(monomer.atom_keys) if (key.residue_number, key.atom_name) != (2, "O")]
    incomplete = MonomerInput(monomer.xyz[keep], monomer.radii[keep], monomer.charges[keep],
                              tuple(monomer.atom_keys[i] for i in keep))
    with pytest.raises(ValueError, match="Incomplete backbone"):
        TorsionKinematics.from_monomer(incomplete)
    for replacement, expected in ((replace(monomer.atom_keys[-1], atom_name="H"), "heavy atoms"),
                                   (replace(monomer.atom_keys[-1], residue_name="MSE"), "Unsupported residue"),
                                   (replace(monomer.atom_keys[-1], chain="B"), "single chain")):
        changed = replace(monomer, atom_keys=monomer.atom_keys[:-1] + (replacement,))
        with pytest.raises(ValueError, match=expected):
            TorsionKinematics.from_monomer(changed)


def test_peptide_break_disulfide_and_other_interresidue_link_fail():
    monomer = make_peptide(("CYS", "GLY", "CYS", "ALA"))
    changed = monomer.xyz.clone()
    changed[[i for i, key in enumerate(monomer.atom_keys) if key.residue_number >= 3]] += 4.0
    with pytest.raises(ValueError, match="Peptide break"):
        TorsionKinematics.from_monomer(replace(monomer, xyz=changed))
    index = _index(monomer)
    changed = monomer.xyz.clone()
    changed[index[3, "SG"]] = changed[index[1, "SG"]] + changed.new_tensor([2.05, 0, 0])
    with pytest.raises(ValueError, match="disulfide|crosslink"):
        TorsionKinematics.from_monomer(replace(monomer, xyz=changed))
    changed = monomer.xyz.clone()
    changed[index[3, "CB"]] = changed[index[1, "CB"]] + changed.new_tensor([1.52, 0, 0])
    with pytest.raises(ValueError, match="crosslink"):
        TorsionKinematics.from_monomer(replace(monomer, xyz=changed))


def test_external_angle_validation_rejects_nonfinite_and_metadata_errors():
    model = TorsionKinematics.from_monomer(make_peptide())
    finite = torch.zeros(model.n_torsions, dtype=torch.float64)
    model.validate_angles(finite)
    invalid = finite.clone()
    invalid[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model.validate_angles(invalid)
    with pytest.raises(ValueError, match="shape"):
        model.coordinates(finite[:-1])
    with pytest.raises(ValueError, match="dtype"):
        model.coordinates(finite.float())
