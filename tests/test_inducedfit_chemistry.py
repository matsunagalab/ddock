"""Optional real OpenMM charge preparation, with no network or hidden repair."""

from dataclasses import replace
import itertools
import random

import pytest
import torch

from zdock.inducedfit import chemistry
from zdock.inducedfit.chemistry import prepare_amber_charges
from zdock.inducedfit.pdb_input import read_pdb_monomer


def make_dipeptide_pdb(path):
    """Complete GLY--ALA with both charged termini and no input hydrogen."""
    residues = [
        ("GLY", {"N": (0., 0., 0.), "CA": (1.45, 0., 0.), "C": (2., 1.42, 0.), "O": (1.30, 2.43, 0.)}),
        ("ALA", {"N": (3.32, 1.42, 0.), "CA": (3.85, 2.77, 0.), "C": (5.37, 2.77, 0.),
                 "O": (6.07, 1.76, 0.), "CB": (3.85, 2.77, 1.53), "OXT": (5.90, 3.88, 0.)}),
    ]
    lines = []
    for rid, (residue, atoms) in enumerate(residues, 1):
        for name, (x, y, z) in atoms.items():
            lines.append(f"ATOM  {len(lines)+1:5d} {name:^4s} {residue:3s} A{rid:4d}    "
                         f"{x:8.3f}{y:8.3f}{z:8.3f}{1.:6.2f}{20.:6.2f}          {name[0]:>2s}  ")
    path.write_text("\n".join(lines + ["TER", "END"]) + "\n")
    return path


@pytest.fixture
def prepared(tmp_path):
    return read_pdb_monomer(make_dipeptide_pdb(tmp_path / "gly_ala.pdb"), chain="A", charge_policy="zero_diagnostic")


@pytest.fixture
def openmm():
    return pytest.importorskip("openmm", reason="optional OpenMM chemistry dependency is not installed")


def test_optional_dependency_is_loaded_lazily_with_clear_error(prepared, monkeypatch):
    original = chemistry.importlib.import_module

    def unavailable(name, *args, **kwargs):
        if name.startswith("openmm"):
            raise ModuleNotFoundError("synthetic missing optional dependency")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(chemistry.importlib, "import_module", unavailable)
    with pytest.raises(ImportError, match="optional OpenMM"):
        prepare_amber_charges(prepared)


def test_real_ff14sb_charges_retain_all_h_and_exact_heavy_coordinates(prepared, openmm):
    charged = prepare_amber_charges(prepared, pH=7., seed=23)
    assert len(charged.heavy_monomer.xyz) == 10
    assert len(charged.monomer.xyz) == 20
    assert charged.provenance["n_hydrogens"] == 10
    assert charged.heavy_indices.dtype == torch.long
    assert len(set(charged.heavy_indices.tolist())) == 10
    assert tuple(charged.monomer.atom_keys[i] for i in charged.heavy_indices) == prepared.monomer.atom_keys
    assert torch.equal(charged.monomer.xyz[charged.heavy_indices], prepared.monomer.xyz)
    assert torch.equal(charged.heavy_monomer.xyz, prepared.monomer.xyz)
    assert torch.equal(charged.heavy_monomer.charges, charged.monomer.charges[charged.heavy_indices])
    assert charged.provenance["heavy_coordinate_output_max_abs_change_angstrom"] == 0
    assert abs(charged.provenance["net_charge_e"]) < 1e-12
    assert float(charged.monomer.charges.sum()) == pytest.approx(0., abs=1e-12)
    assert charged.provenance["heavy_charge_sum_e"] + charged.provenance["hydrogen_charge_sum_e"] == pytest.approx(0., abs=1e-12)
    assert abs(charged.provenance["hydrogen_charge_sum_e"]) > .1
    assert abs(float(charged.heavy_monomer.charges.sum())) > .1  # not a substitute for all-atom electrostatics
    assert charged.provenance["forcefield_file"] == "amber14/protein.ff14SB.xml"
    assert charged.provenance["openmm_version"] == openmm.__version__
    assert all(len(value) == 64 for value in charged.provenance["forcefield_xml_sha256"].values())
    assert len(charged.provenance["variants"]) == 2
    assert bool(torch.isfinite(charged.monomer.xyz).all())
    assert bool(torch.isfinite(charged.monomer.charges).all())


def test_fixed_seed_preserves_caller_rng_and_reproduces_preparation(prepared, openmm):
    random.seed(197)
    before = random.getstate()
    first = prepare_amber_charges(prepared, seed=9)
    assert random.getstate() == before
    second = prepare_amber_charges(prepared, seed=9)
    assert random.getstate() == before
    assert first.monomer.atom_keys == second.monomer.atom_keys
    assert first.provenance["variants"] == second.provenance["variants"]
    torch.testing.assert_close(first.monomer.xyz, second.monomer.xyz, atol=1e-12, rtol=0)
    torch.testing.assert_close(first.monomer.charges, second.monomer.charges, atol=0, rtol=0)


def test_every_hydrogen_has_one_heavy_parent_and_explicit_geometry_radius(prepared, openmm):
    charged = prepare_amber_charges(prepared)
    heavy = set(charged.heavy_indices.tolist())
    neighbors = {i: [] for i in range(len(charged.monomer.xyz))}
    for a, b in charged.bonds:
        neighbors[a].append(b)
        neighbors[b].append(a)
    for index, key in enumerate(charged.monomer.atom_keys):
        if index not in heavy:
            assert key.atom_name.startswith("H")
            assert len(neighbors[index]) == 1 and neighbors[index][0] in heavy
            assert float(charged.monomer.radii[index]) == 1.2
            length = float((charged.monomer.xyz[index] - charged.monomer.xyz[neighbors[index][0]]).norm())
            assert .7 < length < 1.5


@pytest.mark.parametrize("missing", ["CB", "OXT"])
def test_missing_heavy_atoms_are_rejected_not_repaired(prepared, openmm, missing):
    keep = [i for i, key in enumerate(prepared.monomer.atom_keys) if key.atom_name != missing]
    monomer = prepared.monomer
    changed = replace(monomer, xyz=monomer.xyz[keep], radii=monomer.radii[keep], charges=monomer.charges[keep],
                       atom_keys=tuple(monomer.atom_keys[i] for i in keep))
    incomplete = replace(prepared, monomer=changed, records=tuple(prepared.records[i] for i in keep))
    with pytest.raises(ValueError, match="missing=.*" + missing):
        prepare_amber_charges(incomplete)


def test_explicit_crosslink_and_invalid_parameters_are_rejected(prepared):
    crosslinked = replace(prepared, provenance={**prepared.provenance, "link_records": ["LINK    unsupported"]})
    with pytest.raises(ValueError, match="LINK"):
        prepare_amber_charges(crosslinked)
    for bad_pH in (float("nan"), -1., 15., True):
        with pytest.raises(ValueError, match="pH"):
            prepare_amber_charges(prepared, pH=bad_pH)
    for bad_seed in (-1, 2**32, True, .5):
        with pytest.raises(ValueError, match="seed"):
            prepare_amber_charges(prepared, seed=bad_seed)


def test_heavy_input_order_is_preserved_even_when_openmm_groups_residues(prepared, openmm):
    permutation = torch.tensor([4, 0, 5, 1, 6, 2, 7, 3, 9, 8])
    monomer = prepared.monomer
    reordered = replace(monomer, xyz=monomer.xyz[permutation], radii=monomer.radii[permutation], charges=monomer.charges[permutation],
                         atom_keys=tuple(monomer.atom_keys[i] for i in permutation))
    changed = replace(prepared, monomer=reordered, records=tuple(prepared.records[i] for i in permutation))
    charged = prepare_amber_charges(changed, seed=1)
    assert charged.heavy_monomer.atom_keys == reordered.atom_keys
    assert torch.equal(charged.monomer.xyz[charged.heavy_indices], reordered.xyz)


def test_heavy_movement_is_an_error_before_roundoff_restoration(prepared, openmm, monkeypatch):
    from openmm import app, unit
    original = app.Modeller.addHydrogens

    def deliberately_move_heavy(self, *args, **kwargs):
        variants = original(self, *args, **kwargs)
        positions = self.positions.value_in_unit(unit.nanometer)
        positions[0] = positions[0] + openmm.Vec3(1e-6, 0, 0)
        self.positions = positions * unit.nanometer
        return variants

    monkeypatch.setattr(app.Modeller, "addHydrogens", deliberately_move_heavy)
    with pytest.raises(ValueError, match="moved existing heavy atoms"):
        prepare_amber_charges(prepared)


def test_real_charge_preparation_connects_to_full_hydrogen_torsion_map(prepared, openmm):
    from zdock.inducedfit.hydrogen_kinematics import HydrogenatedKinematics

    charged = prepare_amber_charges(prepared, seed=4)
    kinematics = HydrogenatedKinematics.from_prepared(charged)
    angles = torch.linspace(-.2, .3, kinematics.n_torsions, dtype=torch.float64)
    moved = kinematics.coordinates(angles)
    expected_heavy = kinematics.heavy_kinematics.coordinates(angles)
    torch.testing.assert_close(moved[charged.heavy_indices], expected_heavy, atol=1e-12, rtol=1e-12)
    edges = torch.tensor(charged.bonds)
    before = torch.linalg.vector_norm(charged.monomer.xyz[edges[:, 0]] - charged.monomer.xyz[edges[:, 1]], dim=-1)
    after = torch.linalg.vector_norm(moved[edges[:, 0]] - moved[edges[:, 1]], dim=-1)
    torch.testing.assert_close(before, after, atol=1e-12, rtol=1e-12)
    neighbors = {i: [] for i in range(len(moved))}
    for a, b in charged.bonds:
        neighbors[a].append(b)
        neighbors[b].append(a)
    triples = torch.tensor([(a, middle, b) for middle, ends in neighbors.items()
                            for a, b in itertools.combinations(ends, 2)])

    def cosines(xyz):
        first = xyz[triples[:, 0]] - xyz[triples[:, 1]]
        second = xyz[triples[:, 2]] - xyz[triples[:, 1]]
        return (first * second).sum(-1) / (first.norm(dim=-1) * second.norm(dim=-1))

    torch.testing.assert_close(cosines(charged.monomer.xyz), cosines(moved), atol=1e-12, rtol=1e-12)
    hydrogen_indices = sorted(set(range(len(moved))) - set(charged.heavy_indices.tolist()))
    assert float((moved[hydrogen_indices] - charged.monomer.xyz[hydrogen_indices]).norm()) > .1
