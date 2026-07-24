"""Unit tests for `zdock.io.parse_pdb_ms`.

Validate:
 - Receptor PDB has expected atom count and known first-atom fields.
 - Ligand PDB uses the 4-token tail (no ACE type column) and still parses.
 - Parse error paths (short lines, wrong tail length).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zdock.io import parse_pdb_ms


_BENCH_1KXQ = Path(__file__).resolve().parent / "data" / "1KXQ"


@pytest.fixture(scope="module")
def receptor_path() -> Path:
    p = _BENCH_1KXQ / "1KXQ_r_u.pdb.ms"
    if not p.exists():
        pytest.skip(f"missing fixture: {p}")
    return p


@pytest.fixture(scope="module")
def ligand_path() -> Path:
    p = _BENCH_1KXQ / "1KXQ_l_u.pdb.ms"
    if not p.exists():
        pytest.skip(f"missing fixture: {p}")
    return p


def test_receptor_atom_count_and_first_atom(receptor_path: Path):
    atoms = parse_pdb_ms(receptor_path)

    # Known from a direct `wc -l` on 1KXQ_r_u.pdb.ms.
    assert len(atoms) == 3908

    assert atoms.atomname[0] == "N"
    assert atoms.resname[0] == "GLN"
    assert atoms.resseq[0] == 1

    # Coordinates from the file's first ATOM record.
    np.testing.assert_allclose(
        atoms.xyz[0], np.asarray([14.376, -5.731, -7.538]), atol=1e-3,
    )

    # Extended tail is the 5-token receptor form: occ / temp / radius / ACE / charge.
    assert atoms.ace_type_col[0] == "1PPI"
    assert atoms.radius_col[0] == pytest.approx(1.63, abs=1e-3)
    assert atoms.charge_col[0] == pytest.approx(-0.15, abs=1e-3)


def test_ligand_parses_with_four_token_tail(ligand_path: Path):
    atoms = parse_pdb_ms(ligand_path)
    assert len(atoms) == 916

    # Ligand records use chain 'H' in 1KXQ.
    assert atoms.chain[0] == "H"
    assert atoms.atomname[0] == "N"
    assert atoms.resname[0] == "GLN"

    # Ligand has no ACE atom-type tag.
    assert atoms.ace_type_col[0] is None
    assert atoms.radius_col[0] == pytest.approx(1.63, abs=1e-3)
    assert atoms.charge_col[0] == pytest.approx(-0.15, abs=1e-3)


def test_parse_rejects_bad_tail_length(tmp_path: Path):
    bad = tmp_path / "bad.pdb"
    # Construct a line with only 2 tail tokens — should raise.
    bad.write_text(
        "ATOM      1  N   GLN     1      14.376  -5.731  -7.538  only two\n"
    )
    with pytest.raises(ValueError, match="tail tokens"):
        parse_pdb_ms(bad)


def test_parse_rejects_empty_file(tmp_path: Path):
    empty = tmp_path / "empty.pdb"
    empty.write_text("HEADER  nothing here\n")
    with pytest.raises(ValueError, match="no ATOM"):
        parse_pdb_ms(empty)
