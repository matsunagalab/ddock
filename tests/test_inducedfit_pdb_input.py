"""Handcrafted PDB tests: identity, frame, chemistry policy and malformed data."""

import hashlib

import pytest
import torch

from zdock.inducedfit.pdb_input import read_pdb_monomer, validate_torsion_input, write_pdb_monomer
from zdock.inducedfit.types import AtomKey


def _atom(serial=1, name="CA", *, residue="ALA", chain="A", number=1,
          insertion="", altloc="", occupancy=1.0, xyz=(1.0, 2.0, 3.0),
          element="C", tag="ATOM"):
    x, y, z = xyz
    return (f"{tag:<6}{serial:5d} {name:>4}{altloc:1}{residue:>3} {chain:1}"
            f"{number:4d}{insertion:1}   {x:8.3f}{y:8.3f}{z:8.3f}"
            f"{occupancy:6.2f}{10.0:6.2f}{'':10}{element:>2}  ")


def _write(tmp_path, lines, name="input.pdb"):
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def _read(path, **kwargs):
    return read_pdb_monomer(path, **({"chain": "A", "charge_policy": "zero_diagnostic"} | kwargs))


def test_explicit_charge_policy_and_chain_are_required(tmp_path):
    path = _write(tmp_path, [_atom()])
    with pytest.raises(TypeError):
        read_pdb_monomer(path, charge_policy="zero_diagnostic")
    with pytest.raises(ValueError, match="Explicit charge"):
        read_pdb_monomer(path, chain="A")
    with pytest.raises(ValueError, match="Explicit charge"):
        read_pdb_monomer(path, chain="A", charge_policy="zdock")
    prepared = _read(path)
    assert prepared.provenance["electrostatics_status"] == "disabled_diagnostic"
    assert torch.equal(prepared.monomer.charges, torch.zeros(1, dtype=torch.float64))
    assert prepared.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert prepared.monomer.atom_keys == (AtomKey("A", 1, "", "ALA", "CA"),)
    torch.testing.assert_close(prepared.monomer.xyz, torch.tensor([[1., 2., 3.]], dtype=torch.float64))


def test_altloc_occupancy_then_blank_then_a_and_insertion_identity(tmp_path):
    path = _write(tmp_path, [
        _atom(1, altloc="B", occupancy=.8, xyz=(9., 2., 3.)),
        _atom(2, altloc="A", occupancy=.2),
        _atom(3, "N", element="N", occupancy=.5),
        _atom(4, "N", element="N", altloc="A", occupancy=.5),
        _atom(5, "O", element="O", altloc="C", occupancy=.5),
        _atom(6, "O", element="O", altloc="A", occupancy=.5),
        _atom(7, insertion="A"),
    ])
    prepared = _read(path)
    assert [r.serial for r in prepared.records] == [1, 3, 6, 7]
    assert [r.altloc for r in prepared.records] == ["B", "", "A", ""]
    assert prepared.monomer.xyz[0, 0] == 9
    assert prepared.monomer.atom_keys[-1].insertion_code == "A"
    torch.testing.assert_close(prepared.monomer.radii,
                               torch.tensor([1.7, 1.55, 1.52, 1.7], dtype=torch.float64))
    assert len(prepared.provenance["altloc_choices"]) == 3


def test_first_model_and_explicit_chain_selection(tmp_path):
    path = _write(tmp_path, [
        "MODEL        7", _atom(1, chain="A"), _atom(2, chain="B", xyz=(5., 6., 7.)), "ENDMDL",
        "MODEL        8", _atom(1, chain="A", xyz=(99., 0., 0.)), "ENDMDL",
    ])
    prepared = _read(path, chain="B")
    assert prepared.provenance["selected_model_serial"] == 7
    assert prepared.provenance["explicit_model_count"] == 2
    assert prepared.provenance["ignored_record_counts"]["other_chain_atoms"] == 1
    torch.testing.assert_close(prepared.monomer.xyz, torch.tensor([[5., 6., 7.]], dtype=torch.float64))
    with pytest.raises(ValueError, match="No standard heavy atoms"):
        _read(path, chain="Z")


def test_water_hydrogen_anisou_and_missing_element_are_declared(tmp_path):
    path = _write(tmp_path, [
        _atom(element=""), _atom(2, "1HB", element=""),
        _atom(3, "D", element="D"),
        _atom(4, "O", residue="HOH", tag="HETATM", element="O"), "ANISOU dummy ignored",
    ])
    prepared = _read(path)
    assert prepared.monomer.n_atoms == 1
    assert prepared.records[0].element == "C"  # CA means carbon, not calcium.
    assert prepared.provenance["ignored_record_counts"] == {
        "other_chain_atoms": 0, "water_hetatm": 1, "hydrogen_atoms": 2, "anisou": 1,
    }
    assert prepared.provenance["element_inferred_from_atom_name_lines"] == [1, 2]


def test_ter_numbering_gap_and_roundtrip_preserve_bijection(tmp_path):
    source = _write(tmp_path, [
        _atom(3, "N", number=10, insertion="A", element="N"),
        _atom(8, "CA", number=10, insertion="A"), "TER",
        _atom(12, "C", number=12), "END",
    ])
    prepared = _read(source)
    assert [r.segment_index for r in prepared.records] == [0, 0, 1]
    assert prepared.provenance["ter_breaks"][0]["after"]["atom_name"] == "CA"
    assert len(prepared.provenance["residue_number_gaps"]) == 1
    moved = prepared.monomer.xyz + torch.tensor([2.125, -1.375, .625])
    output = tmp_path / "moved.pdb"
    write_pdb_monomer(output, prepared, moved)
    reread = _read(output)
    assert reread.monomer.atom_keys == prepared.monomer.atom_keys
    assert [r.serial for r in reread.records] == [3, 8, 12]
    assert [r.segment_index for r in reread.records] == [0, 0, 1]
    torch.testing.assert_close(reread.monomer.xyz, moved, rtol=0, atol=.00051)
    for before, after in zip(prepared.records, reread.records, strict=True):
        assert before.raw_line[:30] == after.raw_line[:30]
        assert before.raw_line[54:] == after.raw_line[54:]
    assert reread.source_sha256 != prepared.source_sha256


def test_explicit_mapping_must_cover_exact_selected_atom_keys(tmp_path):
    path = _write(tmp_path, [_atom(), _atom(2, "N", element="N")])
    keys = _read(path).monomer.atom_keys
    mapping = {keys[0]: .2, keys[1]: -.1}
    prepared = read_pdb_monomer(path, chain="A", charge_mapping=mapping)
    torch.testing.assert_close(prepared.monomer.charges, torch.tensor([.2, -.1], dtype=torch.float64))
    assert prepared.provenance["charge_policy"] == "explicit_mapping"
    with pytest.raises(ValueError, match="alone"):
        _read(path, charge_mapping=mapping)
    with pytest.raises(ValueError, match="bijectively"):
        read_pdb_monomer(path, chain="A", charge_mapping={keys[0]: .2})
    with pytest.raises(ValueError, match="bijectively"):
        read_pdb_monomer(path, chain="A", charge_mapping=mapping | {AtomKey("A", 2, "", "ALA", "CA"): .3})
    with pytest.raises(ValueError, match="nonfinite"):
        read_pdb_monomer(path, chain="A", charge_mapping={keys[0]: float("nan"), keys[1]: .2})


@pytest.mark.parametrize("line,match", [
    (_atom(residue="MSE"), "nonstandard"),
    (_atom(residue="MSE", tag="HETATM"), "non-water HETATM"),
    (_atom(element="FE"), "unsupported heavy-atom element"),
    (_atom(xyz=(float("nan"), 0, 0)), "nonfinite coordinate"),
    (_atom(occupancy=float("nan")), "nonfinite occupancy"),
    (_atom(occupancy=1.5), "occupancy must"),
    ("ATOM      1  CA", "truncated"),
])
def test_malformed_or_unsupported_chemistry_rejected(tmp_path, line, match):
    with pytest.raises(ValueError, match=match):
        _read(_write(tmp_path, [line]))


def test_duplicates_and_residue_microheterogeneity_are_not_hidden(tmp_path):
    with pytest.raises(ValueError, match="duplicate atom key and altloc"):
        _read(_write(tmp_path, [_atom(), _atom(2)]))
    with pytest.raises(ValueError, match="duplicate atom serials"):
        _read(_write(tmp_path, [_atom(), _atom(1, "N", element="N")]))
    with pytest.raises(ValueError, match="multiple residue names"):
        _read(_write(tmp_path, [_atom(), _atom(2, residue="CYS", altloc="B")]))


@pytest.mark.parametrize("lines", [["MODEL        1", _atom()], ["ENDMDL", _atom()],
                                   ["MODEL        1", _atom(), "MODEL        2", "ENDMDL"]])
def test_malformed_model_boundaries_rejected(tmp_path, lines):
    with pytest.raises(ValueError, match="MODEL"):
        _read(_write(tmp_path, lines))


def test_blank_chain_and_blank_occupancy_are_explicit(tmp_path):
    line = _atom(chain="")
    line = line[:54] + " " * 6 + line[60:]
    prepared = _read(_write(tmp_path, [line]), chain="")
    assert prepared.monomer.atom_keys[0].chain == ""
    assert prepared.records[0].occupancy == 0
    assert prepared.provenance["blank_occupancy_lines"] == [1]


def test_blank_ter_after_another_chain_does_not_split_selected_chain(tmp_path):
    path = _write(tmp_path, [_atom(), _atom(2, chain="B"), "TER", _atom(3, "N", element="N")])
    prepared = _read(path)
    assert [record.segment_index for record in prepared.records] == [0, 0]
    assert prepared.provenance["ter_breaks"] == []


def test_export_rejects_nonfinite_or_unrepresentable_coordinates(tmp_path):
    prepared = _read(_write(tmp_path, [_atom()]))
    for xyz in (torch.full((1, 3), float("nan")), torch.full((1, 3), 100000.0)):
        with pytest.raises(ValueError):
            write_pdb_monomer(tmp_path / "bad.pdb", prepared, xyz)
    assert not (tmp_path / "bad.pdb").exists()


def test_torsion_preflight_allows_terminal_ter_but_refuses_internal_break(tmp_path):
    terminal = _read(_write(tmp_path, [_atom(), _atom(2, "N", element="N"), "TER", "END"]))
    assert validate_torsion_input(terminal) is None
    internal = _read(_write(tmp_path, [_atom(), "TER", _atom(2, "N", element="N"), "TER", "END"]))
    with pytest.raises(ValueError, match="internal TER"):
        validate_torsion_input(internal)


@pytest.mark.parametrize("record,key", [
    ("SSBOND   1 CYS A    1    CYS A    2", "ssbond_records"),
    ("LINK         SG  CYS A   1                  C1  LIG A 100", "link_records"),
    ("CONECT    1    2", "conect_records"),
])
def test_torsion_preflight_refuses_declared_connectivity(tmp_path, record, key):
    prepared = _read(_write(tmp_path, [record, _atom()]))
    assert prepared.provenance[key] == [record]
    with pytest.raises(ValueError, match="explicit.*connectivity"):
        validate_torsion_input(prepared)
