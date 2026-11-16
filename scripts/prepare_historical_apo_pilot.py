#!/usr/bin/env python3
"""Declare and prepare the historical barnase/barstar apo development pair.

Fixed inputs: 1A2P chain B and 1A19 chain A.  This script does not download or
inspect a bound complex, select by accuracy, fabricate missing coordinates,
or modify retained protein ATOM coordinates/occupancies.  It explicitly
excludes crystallographic Zn and solvent, documenting each removed selected-
chain heteroatom, its distances, links, missing termini and zero occupancy.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch

from zdock.inducedfit.chemistry import prepare_amber_charges
from zdock.inducedfit.pdb_input import read_pdb_monomer, validate_torsion_input


INPUTS = {
    "receptor": {"pdb_id": "1A2P", "chain": "B", "sha256": "f666c1dc491d6994a859f647953c7c8bc9546fdaa4a9be6dfb520955ffbbf903"},
    "ligand": {"pdb_id": "1A19", "chain": "A", "sha256": "0cd033f1646ae3da136276ebaa0eee6de825f647910308d5a39c22c683162d4e"},
}


def atom_record(line):
    return {"serial": int(line[6:11]), "atom_name": line[12:16].strip(), "altloc": line[16].strip(),
            "residue_name": line[17:20].strip(), "chain": line[21].strip(),
            "residue_number": int(line[22:26]), "insertion_code": line[26].strip(),
            "xyz": [float(line[30:38]), float(line[38:46]), float(line[46:54])],
            "occupancy": float(line[54:60]), "source_line": line}


def derive_protein_only(source: Path, destination: Path, chain: str) -> dict:
    lines = source.read_text().splitlines()
    protein_lines = [line for line in lines if line.startswith("ATOM  ") and line[21] == chain]
    protein = [atom_record(line) for line in protein_lines]
    hetero = [atom_record(line) for line in lines if line.startswith("HETATM") and line[21] == chain]
    if not protein:
        raise ValueError("Selected chain has no protein ATOM coordinates")
    if any(row["residue_name"] not in {"HOH", "WAT", "DOD", "ZN"} for row in hetero):
        raise ValueError("Unexpected non-water/non-Zn heteroatom: explicit policy revision required")
    if any(line.startswith("SSBOND") for line in lines):
        raise ValueError("This declared policy does not permit disulfide removal")
    # The only supported LINK removal is protein/solvent--Zn coordination.
    links = [line for line in lines if line.startswith("LINK")]
    if any("ZN" not in (line[17:20].strip(), line[47:50].strip()) for line in links):
        raise ValueError("Non-Zn LINK found: cannot silently discard connectivity")
    selected_serials = {row["serial"] for row in protein}
    conect = [line for line in lines if line.startswith("CONECT")]
    for line in conect:
        serials = [int(line[i:i+5]) for i in range(6, len(line), 5) if line[i:i+5].strip()]
        if serials and serials[0] in selected_serials and any(v in selected_serials for v in serials[1:]):
            raise ValueError("Selected protein has an explicit internal CONECT bond; inspect before derivation")
    if hetero:
        distances = torch.cdist(torch.tensor([row["xyz"] for row in hetero], dtype=torch.float64),
                                torch.tensor([row["xyz"] for row in protein], dtype=torch.float64))
        for row, values in zip(hetero, distances):
            nearest = int(values.argmin())
            row["nearest_selected_protein_atom"] = {k: v for k, v in protein[nearest].items() if k != "source_line"}
            row["nearest_selected_protein_distance_angstrom"] = float(values[nearest])
            row["selected_protein_contacts_within_3_angstrom"] = [
                {"serial": protein[i]["serial"], "atom_name": protein[i]["atom_name"],
                 "residue_name": protein[i]["residue_name"], "residue_number": protein[i]["residue_number"],
                 "distance_angstrom": float(values[i])} for i in (values <= 3.).nonzero().flatten().tolist()
            ]
    metadata = [
        "REMARK 900 DECLARED PROTEIN-ONLY APO DEVELOPMENT INPUT; NOT A NATIVE COMPLEX",
        f"REMARK 900 SOURCE {source.stem} CHAIN {chain}; RETAINED ATOM RECORDS UNCHANGED",
        "REMARK 900 EXCLUDED SOLVENT/CRYSTAL CONTACT ZN; SEE INPUT_POLICY.JSON",
        "REMARK 900 UNRESOLVED N-TERMINAL RESIDUES NOT REBUILT; OBSERVED END CHARGED",
        "REMARK 900 ZERO-OCCUPANCY DEPOSITED ATOMS RETAINED WITH EXPLICIT CAVEATS",
    ]
    sequence_metadata = [line for line in lines if (line.startswith("SEQRES") and line[11] == chain)
                         or (line.startswith("SEQADV") and line[16] == chain)]
    anisou = [line for line in lines if line.startswith("ANISOU") and line[21] == chain]
    destination.write_text("\n".join(metadata + sequence_metadata + protein_lines + anisou + ["TER", "END"]) + "\n")
    reread = [line for line in destination.read_text().splitlines() if line.startswith("ATOM  ")]
    if reread != protein_lines:
        raise RuntimeError("Retained protein ATOM records changed during derivation")
    return {
        "retained_protein_ATOM_records": len(protein_lines), "retained_ATOM_records_byte_identical": True,
        "removed_selected_chain_heteroatom_counts": dict(Counter(row["residue_name"] for row in hetero)),
        "removed_selected_chain_heteroatoms": hetero,
        "removed_LINK_records": links, "removed_CONECT_records": conect,
        "excluded_other_chain_ATOM_count": sum(line.startswith("ATOM  ") and line[21] != chain for line in lines),
        "source_missing_residue_records": [line for line in lines if line.startswith("REMARK 465")],
        "source_zero_occupancy_records": [line for line in lines if line.startswith("REMARK 480")],
        "selected_zero_occupancy_ATOMs": [row for row in protein if row["occupancy"] == 0],
        "source_sequence_difference_records": [line for line in lines if line.startswith("SEQADV")],
        "source_title_records": [line for line in lines if line.startswith("TITLE")],
        "crystal_symmetry_note": "Distances above use deposited coordinates only; symmetry-related contacts remain in original LINK records",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/inducedfit/inputs"))
    parser.add_argument("--out", type=Path, default=Path("data/inducedfit/apo_pilot_v1"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    report = {
        "policy_version": 1, "status": "running", "command": sys.argv,
        "system_choice": "historical barnase/barstar pair fixed before docking, chosen for registered development pilot",
        "selection_uses_native_or_accuracy": False, "native_complex_opened": False,
        "platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
        "device": "cpu", "dtype": "float64", "threads": 1,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "policy": {
            "scope": "protein-protein-unbound experimental monomers; protein-only pilot approximation",
            "heteroatoms": "remove selected-chain crystallographic Zn and all solvent, with atom records and distances retained here",
            "zinc_rationale": "1A2P primary structural paper identifies Zn as bridging crystallographic symmetry contacts; this is not a claim that metal removal leaves the electrostatic state unchanged",
            "missing_heavy_atoms": "do not fabricate missing atom coordinates; fail if resolved-residue heavy atom template incomplete",
            "N_terminus": "1A2P Ala1/Gln2 unresolved; model deposited Val3 as charged N-terminus. 1A19 expression Met0 unresolved; model Lys1 as charged N-terminus. These are declared truncation/protonation approximations, not assertions of experimental cleavage",
            "zero_occupancy": "retain deposited 1A2P_B zero-occupancy modeled coordinates unchanged; locations experimentally uncertain, no claim of reliable observed coordinates",
            "mutation": "1A19 is uncomplexed C82A barstar; no mutation correction",
            "partner_oligomers": "use one selected monomer per side; exclude crystallographic/biological other chains for binary docking approximation",
        },
        "sources": [
            {"url": "https://www.rcsb.org/structure/1A2P", "title": "BARNASE WILDTYPE STRUCTURE AT 1.5 ANGSTROMS RESOLUTION",
             "primary_doi": "10.1107/S0907444998010865", "primary_title": "Refinement and structural analysis of barnase at 1.5 A resolution."},
            {"url": "https://www.rcsb.org/structure/1A19", "title": "BARSTAR (FREE), C82A MUTANT",
             "primary_doi": "10.1021/bi972857n", "primary_title": "Discrepancies between the NMR and X-ray structures of uncomplexed barstar: analysis suggests that packing densities of protein structures determined by NMR are unreliable."},
        ],
        "rejected_alternative": {"pdb_id": "1BNF", "url": "https://www.rcsb.org/structure/1BNF",
                                 "reason": "T70C/S92C engineered disulfide mutant; title/primary paper override misleading generic mutation badge; not downloaded or used"},
        "inputs": {},
    }
    target = args.out / "input_policy.json"
    def save():
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    started = time.monotonic()
    save()
    manifest = {"format_version": 1, "system_id": "historical_1BRS_apo_protein_only_v1",
                "input_kind": "apo_protein_only", "split": "historical_development"}
    try:
        for side, specification in INPUTS.items():
            pdb_id, chain = specification["pdb_id"], specification["chain"]
            source = args.source_dir / f"{pdb_id}.pdb"
            raw = source.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if digest != specification["sha256"]:
                raise ValueError(f"Original {pdb_id} source hash differs from registered input")
            source_copy = args.out / f"{pdb_id}_original.pdb"
            source_copy.write_bytes(raw)
            derived = args.out / f"{pdb_id}_{chain}_protein_only.pdb"
            row = {**specification, "source_url": f"https://files.rcsb.org/download/{pdb_id}.pdb",
                   "derivative": derived.name, "chemistry_status": "not_attempted"}
            report["inputs"][side] = row
            row.update(derive_protein_only(source_copy, derived, chain))
            row["derived_sha256"] = hashlib.sha256(derived.read_bytes()).hexdigest()
            save()
            try:
                prepared = read_pdb_monomer(derived, chain=chain, charge_policy="zero_diagnostic")
                validate_torsion_input(prepared)
                began = time.monotonic()
                charged = prepare_amber_charges(prepared, pH=7.0, seed=0)
                row.update(chemistry_status="passed", chemistry_seconds=time.monotonic() - began,
                           chemistry_provenance=charged.provenance,
                           n_heavy_atoms=charged.heavy_monomer.n_atoms, n_all_atoms=charged.monomer.n_atoms,
                           retained_residue_count=len({(key.residue_number, key.insertion_code) for key in charged.heavy_monomer.atom_keys}),
                           selected_zero_occupancy_after_altloc=[asdict(record.atom_key) for record in prepared.records if record.occupancy == 0])
                manifest[side] = {"pdb": derived.name, "chain": chain, "sha256": row["derived_sha256"]}
            except Exception as error:
                row.update(chemistry_status="failed", error_type=type(error).__name__, error=str(error))
            save()
        report["status"] = "ready" if all(row["chemistry_status"] == "passed" for row in report["inputs"].values()) else "not_ready"
        if report["status"] == "ready":
            manifest_path = args.out / "inference_manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            report["manifest"] = manifest_path.name
            report["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    except Exception as error:
        report.update(status="exception", error_type=type(error).__name__, error=str(error))
    report["wall_seconds"] = time.monotonic() - started
    save()
    print(json.dumps({"status": report["status"], "wall_seconds": report["wall_seconds"],
                      "inputs": {side: {k: row[k] for k in ("chemistry_status", "n_heavy_atoms", "n_all_atoms", "retained_residue_count", "error") if k in row}
                                 for side, row in report["inputs"].items()}}, indent=2))
    print(f"Saved {target}")
    return 0 if report["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
