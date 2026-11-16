"""Explicit, reference-free PDB geometry input for diagnostic field calculations.

This reader preserves the input Cartesian frame. It does not infer protonation,
force-field charges, connectivity, or missing atoms. The fixed C/N/O/S radii
below are geometric van der Waals radii for the effective field, not an Amber
parameterization. Charges must be supplied, or explicitly disabled with
``charge_policy='zero_diagnostic'``; the latter is not a charged physical model.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import math
from numbers import Real
from pathlib import Path

import torch

from .types import AtomKey, MonomerInput


STANDARD_AMINO_ACIDS = frozenset(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()
)
GEOMETRIC_RADII_ANGSTROM = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80}
IGNORED_WATER_RESIDUES = frozenset({"HOH", "WAT", "DOD"})


@dataclass(frozen=True, slots=True)
class PDBAtomRecord:
    raw_line: str
    line_number: int
    serial: int
    atom_key: AtomKey
    altloc: str
    occupancy: float
    element: str
    segment_index: int


@dataclass(frozen=True, slots=True)
class PreparedPDB:
    monomer: MonomerInput
    source_sha256: str
    provenance: dict
    records: tuple[PDBAtomRecord, ...]


def _number(text: str, field: str, line_number: int, *, default=None) -> float:
    if not text.strip() and default is not None:
        return default
    try:
        value = float(text)
    except ValueError as error:
        raise ValueError(f"PDB line {line_number}: invalid {field}") from error
    if not math.isfinite(value):
        raise ValueError(f"PDB line {line_number}: nonfinite {field}")
    return value


def _integer(text: str, field: str, line_number: int) -> int:
    try:
        return int(text)
    except ValueError as error:
        raise ValueError(f"PDB line {line_number}: invalid {field}; hybrid-36 IDs are unsupported") from error


def _first_model(lines: list[str]) -> tuple[list[tuple[int, str]], int | None, int]:
    starts = [i for i, line in enumerate(lines) if line[:6].strip() == "MODEL"]
    if not starts:
        if any(line[:6].strip() == "ENDMDL" for line in lines):
            raise ValueError("ENDMDL without MODEL")
        return list(enumerate(lines, 1)), None, 0
    start = starts[0]
    serial = _integer(lines[start][10:14], "MODEL serial", start + 1)
    end = next((i for i in range(start + 1, len(lines))
                if lines[i][:6].strip() in {"MODEL", "ENDMDL"}), None)
    if end is None or lines[end][:6].strip() != "ENDMDL":
        raise ValueError("First MODEL must end with ENDMDL before another MODEL")
    return [(i + 1, lines[i]) for i in range(start + 1, end)], serial, len(starts)


def _altloc_order(record: PDBAtomRecord):
    preference = 0 if record.altloc == "" else 1 if record.altloc == "A" else 2
    return (-record.occupancy, preference, record.altloc, record.line_number)


def read_pdb_monomer(
    path: str | Path, *, chain: str,
    charge_policy: str | None = None,
    charge_mapping: Mapping[AtomKey, float] | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> PreparedPDB:
    """Read the explicitly selected chain from the first MODEL only.

    Alternate locations are selected per atom by highest occupancy, then blank,
    then A, then lexical altloc. Blank occupancy is explicitly interpreted as
    zero. Duplicate rows with the SAME atom key and altloc are errors. Per-atom
    selection does not guarantee a coherent residue conformer; downstream
    geometry checks must establish that. Non-water HETATM on the selected chain
    and nonstandard ATOM residues are rejected, not silently removed.

    Exactly one charge source is required: ``zero_diagnostic`` or a complete
    AtomKey-to-charge mapping for the selected heavy atoms (no extra keys).
    Neither hydrogen addition nor force-field preparation is performed.
    """
    if not isinstance(chain, str) or len(chain) > 1 or (chain and chain.isspace()):
        raise ValueError("chain must be an explicit one-character PDB chain ID, or '' for blank")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("PDB diagnostic input supports float32 or float64")
    if charge_mapping is None:
        if charge_policy != "zero_diagnostic":
            raise ValueError("Explicit charge_mapping or charge_policy='zero_diagnostic' is required")
    elif charge_policy is not None or not isinstance(charge_mapping, Mapping):
        raise ValueError("Supply a charge_mapping alone, or zero_diagnostic alone")

    path = Path(path)
    source = path.read_bytes()
    try:
        lines = source.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("PDB input must be ASCII text") from error
    model_lines, model_serial, model_count = _first_model(lines)
    atom_rows: dict[AtomKey, list[tuple[PDBAtomRecord, tuple[float, float, float]]]] = {}
    residues: dict[tuple, str] = {}
    ignored = {"other_chain_atoms": 0, "water_hetatm": 0, "hydrogen_atoms": 0, "anisou": 0}
    inferred_elements, blank_occupancy = [], []
    ter_breaks = []
    segment_index = 0
    last_selected = None
    last_atom_chain = None
    available_chains = set()
    for line_number, line in model_lines:
        tag = line[:6].strip()
        if tag == "ANISOU":
            ignored["anisou"] += 1
            continue
        if tag == "TER":
            ter_chain = line[21:22].strip()
            matches_chain = ter_chain == chain if ter_chain else last_atom_chain == chain
            if last_selected is not None and matches_chain:
                ter_breaks.append({"line_number": line_number, "after": asdict(last_selected)})
                segment_index += 1
                last_selected = None
            last_atom_chain = None
            continue
        if tag not in {"ATOM", "HETATM"}:
            continue
        if len(line) < 54:
            raise ValueError(f"PDB line {line_number}: truncated atom record")
        row_chain, residue_name = line[21:22].strip(), line[17:20].strip()
        last_atom_chain = row_chain
        available_chains.add(row_chain)
        if row_chain != chain:
            ignored["other_chain_atoms"] += 1
            continue
        if tag == "HETATM":
            if residue_name in IGNORED_WATER_RESIDUES:
                ignored["water_hetatm"] += 1
                continue
            raise ValueError(f"PDB line {line_number}: unsupported non-water HETATM {residue_name}")
        if residue_name not in STANDARD_AMINO_ACIDS:
            raise ValueError(f"PDB line {line_number}: nonstandard amino acid {residue_name!r}")
        serial = _integer(line[6:11], "atom serial", line_number)
        if serial <= 0:
            raise ValueError(f"PDB line {line_number}: atom serial must be positive")
        atom_name = line[12:16].strip()
        key = AtomKey(row_chain, _integer(line[22:26], "residue number", line_number),
                      line[26:27].strip(), residue_name, atom_name)
        position = tuple(_number(line[a:b], "coordinate", line_number)
                         for a, b in ((30, 38), (38, 46), (46, 54)))
        occupancy = _number(line[54:60], "occupancy", line_number, default=0.0)
        if not 0.0 <= occupancy <= 1.0:
            raise ValueError(f"PDB line {line_number}: occupancy must be in [0, 1]")
        _number(line[60:66], "temperature factor", line_number, default=0.0)
        if not line[54:60].strip():
            blank_occupancy.append(line_number)
        element = line[76:78].strip().upper()
        if not element:
            element = atom_name.lstrip("0123456789")[:1].upper()
            inferred_elements.append(line_number)
        if element in {"H", "D"}:
            ignored["hydrogen_atoms"] += 1
            continue
        if element not in GEOMETRIC_RADII_ANGSTROM:
            raise ValueError(f"PDB line {line_number}: unsupported heavy-atom element {element!r}")
        residue_id = (key.chain, key.residue_number, key.insertion_code)
        if residue_id in residues and residues[residue_id] != key.residue_name:
            raise ValueError(f"PDB line {line_number}: multiple residue names at the same residue ID")
        residues[residue_id] = key.residue_name
        record = PDBAtomRecord(line, line_number, serial, key, line[16:17].strip(),
                               occupancy, element, segment_index)
        alternatives = atom_rows.setdefault(key, [])
        if any(old.altloc == record.altloc for old, _ in alternatives):
            raise ValueError(f"PDB line {line_number}: duplicate atom key and altloc {key}")
        if any(old.element != record.element or old.segment_index != record.segment_index for old, _ in alternatives):
            raise ValueError(f"PDB line {line_number}: alternatives disagree on element or TER segment")
        alternatives.append((record, position))
        last_selected = key

    if not atom_rows:
        raise ValueError(f"No standard heavy atoms for chain {chain!r}; available chains: {sorted(available_chains)!r}")
    selected = [min(rows, key=lambda row: _altloc_order(row[0])) for rows in atom_rows.values()]
    selected.sort(key=lambda row: row[0].line_number)
    records = tuple(record for record, _ in selected)
    keys = tuple(record.atom_key for record in records)
    if len({record.serial for record in records}) != len(records):
        raise ValueError("Selected heavy atoms have duplicate atom serials")
    if charge_mapping is None:
        charges = [0.0] * len(keys)
        resolved_charge_policy = "zero_diagnostic"
    else:
        if set(charge_mapping) != set(keys):
            raise ValueError("charge_mapping must bijectively match the selected heavy-atom AtomKeys")
        charges = []
        for key in keys:
            value = charge_mapping[key]
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"charge_mapping has a nonfinite or nonnumeric charge for {key}")
            charges.append(float(value))
        resolved_charge_policy = "explicit_mapping"
    monomer = MonomerInput(
        xyz=torch.tensor([position for _, position in selected], dtype=dtype, device=device),
        radii=torch.tensor([GEOMETRIC_RADII_ANGSTROM[record.element] for record in records], dtype=dtype, device=device),
        charges=torch.tensor(charges, dtype=dtype, device=device), atom_keys=keys,
    )
    ordered_residues = []
    for record in records:
        residue = (record.atom_key.residue_number, record.atom_key.insertion_code, record.atom_key.residue_name)
        if not ordered_residues or residue != ordered_residues[-1]:
            ordered_residues.append(residue)
    number_gaps = [{"previous": previous, "next": current}
                   for previous, current in zip(ordered_residues, ordered_residues[1:])
                   if current[0] - previous[0] not in (0, 1)]
    source_sha256 = hashlib.sha256(source).hexdigest()
    provenance = {
        "reader_version": 1, "source_name": path.name, "source_sha256": source_sha256,
        "selected_chain": chain, "available_chains_first_model": sorted(available_chains),
        "model_policy": "first_model_only",
        "selected_model_serial": model_serial, "explicit_model_count": model_count,
        "coordinates": "original_input_frame_no_centering_or_geometry_repair",
        "charge_policy": resolved_charge_policy,
        "electrostatics_status": "disabled_diagnostic" if charge_mapping is None else "caller_supplied_not_validated_as_forcefield",
        "radii_angstrom": dict(GEOMETRIC_RADII_ANGSTROM), "radii_role": "geometric_effective_field_not_forcefield",
        "standard_amino_acids_only": True, "hydrogen_policy": "omit_H_and_D_no_addition",
        "ignored_water_residues": sorted(IGNORED_WATER_RESIDUES), "ignored_record_counts": ignored,
        "altloc_policy": "per_atom_highest_occupancy_then_blank_then_A_then_lexical",
        "altloc_choices": [{"atom_key": asdict(record.atom_key), "selected_altloc": record.altloc,
                            "selected_line": record.line_number,
                            "discarded_lines": [old.line_number for old, _ in atom_rows[record.atom_key]
                                                if old.line_number != record.line_number]}
                           for record in records if len(atom_rows[record.atom_key]) > 1],
        "blank_occupancy_policy": "zero", "blank_occupancy_lines": blank_occupancy,
        "element_inferred_from_atom_name_lines": inferred_elements,
        "ter_breaks": ter_breaks, "residue_number_gaps": number_gaps,
        "ssbond_records": [line for line in lines if line[:6].strip() == "SSBOND"],
        "link_records": [line for line in lines if line[:6].strip() == "LINK"],
        "conect_records": [line for line in lines if line[:6].strip() == "CONECT"],
        "geometry_validation": "not_performed_backbone_bonds_disulfides_missing_atoms_and_altloc_coherence_require_validation",
    }
    return PreparedPDB(monomer, source_sha256, provenance, records)


def validate_torsion_input(prepared: PreparedPDB) -> None:
    """Reject unsupported explicit connectivity before discarding provenance.

    Call this on the reader result BEFORE passing ``.monomer`` to a torsion
    model. This is a conservative preflight, not a chemistry parser: ANY SSBOND,
    LINK, or CONECT record is refused, even if it describes another chain or
    ordinary rather than crosslinked bonds. Internal TER breaks are refused;
    a terminal TER after the last selected atom is allowed. Coordinate-based
    disulfide detection, missing backbone atoms and peptide geometry still
    require validation by the kinematics implementation.
    """
    if not isinstance(prepared, PreparedPDB):
        raise TypeError("Expected PreparedPDB with connectivity provenance")
    for key in ("ssbond_records", "link_records", "conect_records"):
        if key not in prepared.provenance:
            raise ValueError(f"Missing connectivity provenance: {key}")
        if prepared.provenance[key]:
            raise ValueError(f"Torsion prototype does not support explicit {key.removesuffix('_records').upper()} connectivity")
    if tuple(record.atom_key for record in prepared.records) != prepared.monomer.atom_keys:
        raise ValueError("Record/monomer atom identity mapping is not bijective")
    if len({record.segment_index for record in prepared.records}) > 1:
        raise ValueError("Torsion prototype does not support internal TER chain breaks")


def write_pdb_monomer(path: str | Path, prepared: PreparedPDB, xyz: torch.Tensor | None = None) -> None:
    """Write selected ATOM records with coordinates replaced in the same frame.

    Atom names, serials, chain/residue IDs, chosen altloc, occupancies and other
    source ATOM columns are preserved. MODEL wrappers and unrelated headers are
    not copied. TER boundaries are retained between selected segments. PDB
    output rounds coordinates to 0.001 angstrom; values that do not fit fail.
    """
    xyz = prepared.monomer.xyz if xyz is None else xyz
    if not isinstance(xyz, torch.Tensor) or not xyz.is_floating_point() or xyz.shape != prepared.monomer.xyz.shape:
        raise ValueError("xyz must be a floating tensor with the prepared atom shape")
    if not bool(torch.isfinite(xyz).all()):
        raise ValueError("xyz must be finite")
    if tuple(record.atom_key for record in prepared.records) != prepared.monomer.atom_keys:
        raise ValueError("Record/monomer atom identity mapping is not bijective")
    lines = []
    previous_segment = None
    for record, position in zip(prepared.records, xyz.detach().cpu().tolist(), strict=True):
        if previous_segment is not None and previous_segment != record.segment_index:
            lines.append("TER")
        coordinates = [f"{value:8.3f}" for value in position]
        if any(len(value) != 8 for value in coordinates):
            raise ValueError("Coordinate does not fit the PDB 8.3 field")
        raw = record.raw_line.ljust(80)
        lines.append(raw[:30] + "".join(coordinates) + raw[54:])
        previous_segment = record.segment_index
    lines.extend(["TER", "END"])
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")
