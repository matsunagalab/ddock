#!/usr/bin/env python3
"""Register and audit a small, deterministic PINDER train apo input cohort.

Selection reads a CACHED official index (optional pyarrow dependency). Audit
reads only the registered apo files, never native structures, and records all
failures. Neither mode downloads files or runs docking. Output paths must be
new: reruns cannot silently replace a registration or an audit.

See docs/inducedfit_input_audit.md for pinned sources and download commands.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
from urllib.parse import quote


INDEX_URL = "https://pinderdata.org/2024-02/index.parquet"
STRUCTURE_BASE_URL = "https://pinderdata.org/2024-02/pdbs/"
INDEX_COLUMNS = ["id", "apo_R_pdb", "apo_L_pdb", "apo_R_quality", "apo_L_quality",
                 "cluster_id", "uniprot_R", "uniprot_L"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _filename(name: str) -> str:
    if (not isinstance(name, str) or not name.endswith(".pdb")
            or name != Path(name).name or any(c in name for c in ("/", "\\", "\n", "\r"))):
        raise ValueError(f"Invalid registered PDB basename: {name!r}")
    return name


def _new_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite registered artifact: {path}")


def _save_new(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")


def select(args) -> int:
    _new_output(args.out)
    if args.count < 1 or args.max_residues < 1:
        raise ValueError("count and max-residues must be positive")
    # Only this command requires pyarrow; pandas/PINDER are not required.
    import pyarrow.parquet as pq

    table = pq.read_table(args.index, columns=INDEX_COLUMNS,
                          filters=[("split", "=", "train"), ("apo_R", "=", True),
                                   ("apo_L", "=", True)])
    rows = table.sort_by([("id", "ascending")]).slice(0, args.count).to_pylist()
    if len(rows) != args.count or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Requested number of unique train apo systems is unavailable")
    for row in rows:
        row["receptor_url"] = STRUCTURE_BASE_URL + quote(_filename(row["apo_R_pdb"]))
        row["ligand_url"] = STRUCTURE_BASE_URL + quote(_filename(row["apo_L_pdb"]))
    manifest = {
        "schema_version": 1, "purpose": "development_input_eligibility_only_no_docking_or_native_access",
        "dataset_release": "2024-02", "index_url": INDEX_URL,
        "index_sha256": _sha256(args.index), "index_columns": INDEX_COLUMNS,
        "train_both_apo_pool_size": table.num_rows,
        "selection_rule": f"split=train AND apo_R=True AND apo_L=True; first {args.count} system IDs in lexicographic order, frozen before PDB retrieval",
        "eligibility_rule": f"each supplied apo chain <={args.max_residues} resolved standard amino-acid residues; strict reader and ff14SB completeness; no crosslinks or internal breaks; no outcomes used",
        "max_residues_per_partner": args.max_residues, "chain": "A",
        "attempt_count": args.count, "native_files_downloaded": False,
        "chain_selection": "A (PINDER standardized single-chain apo)", "systems": rows,
        "selection_command": sys.argv,
    }
    _save_new(args.out, manifest)
    print(json.dumps({"saved": str(args.out), "pool_size": table.num_rows,
                      "registered_ids": [row["id"] for row in rows]}, indent=2))
    return 0


def audit(args) -> int:
    _new_output(args.out)  # Fail before any preparation if the result exists.
    import torch
    from zdock.inducedfit.chemistry import prepare_amber_charges
    from zdock.inducedfit.pdb_input import read_pdb_monomer

    if args.threads < 1 or not 0 <= args.pH <= 14 or not 0 <= args.seed < 2**32:
        raise ValueError("Invalid threads, pH, or preparation seed")
    torch.set_num_threads(args.threads)
    manifest = json.loads(args.manifest.read_text())
    # The initial, manually registered v1 manifest had these same fixed values
    # in its eligibility text, before explicit scalar schema fields were added.
    max_residues = manifest.get("max_residues_per_partner", 150)
    chain = manifest.get("chain", "A")
    rows = manifest["systems"]
    if (manifest.get("native_files_downloaded") is not False or chain != "A"
            or type(max_residues) is not int or max_residues < 1
            or len(rows) != manifest["attempt_count"]):
        raise ValueError("Unsupported or inconsistent registration manifest")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Registration must contain unique system IDs")
    repo = Path(__file__).resolve().parents[1]
    source_paths = [repo / "src/zdock/inducedfit" / name
                    for name in ("pdb_input.py", "chemistry.py", "kinematics.py", "types.py")]
    source_paths.append(Path(__file__).resolve())
    source_hashes = {str(path.relative_to(repo)): _sha256(path) for path in source_paths}
    results = []
    for row in rows:
        result = {"id": row["id"], "partners": {}, "docking_attempted": False}
        prepared = {}
        for side, key in (("receptor", "apo_R_pdb"), ("ligand", "apo_L_pdb")):
            path = args.pdb_dir / _filename(row[key])
            entry = {"path": str(path)}
            try:
                entry.update(sha256=_sha256(path), bytes=path.stat().st_size)
                pdb = read_pdb_monomer(path, chain=chain, charge_policy="zero_diagnostic")
                n = len({(k.chain, k.residue_number, k.insertion_code, k.residue_name)
                         for k in pdb.monomer.atom_keys})
                entry.update(n_residues=n, n_heavy_atoms=pdb.monomer.n_atoms,
                             reader_status="passed", size_eligible=n <= max_residues,
                             oxt_count=sum(k.atom_name == "OXT" for k in pdb.monomer.atom_keys))
                prepared[side] = pdb
            except Exception as error:
                entry.update(reader_status="failed", error_type=type(error).__name__,
                             error=str(error), size_eligible=False)
            result["partners"][side] = entry
        if all(entry["size_eligible"] for entry in result["partners"].values()):
            for side, pdb in prepared.items():
                started = time.perf_counter()
                try:
                    charged = prepare_amber_charges(pdb, pH=args.pH, seed=args.seed)
                    result["partners"][side].update(chemistry_status="passed",
                        n_full_atoms=charged.monomer.n_atoms, chemistry_provenance=charged.provenance)
                except Exception as error:
                    result["partners"][side].update(chemistry_status="failed",
                        error_type=type(error).__name__, error=str(error))
                result["partners"][side]["preparation_seconds"] = time.perf_counter() - started
            result["pair_status"] = ("eligible" if all(entry.get("chemistry_status") == "passed"
                for entry in result["partners"].values()) else "chemistry_ineligible")
        else:
            result["pair_status"] = "input_size_or_reader_ineligible"
            for entry in result["partners"].values():
                entry["chemistry_status"] = "not_attempted_pair_fails_size_or_reader"
        # Full atom maps are retained in JSON, never dumped to the terminal.
        print(json.dumps({"id": result["id"], "pair_status": result["pair_status"],
                          "partners": {side: {k: v for k, v in entry.items() if k != "chemistry_provenance"}
                                       for side, entry in result["partners"].items()}}), flush=True)
        results.append(result)
    try:
        openmm_version = importlib.metadata.version("openmm")
    except importlib.metadata.PackageNotFoundError:
        openmm_version = None
    report = {
        "scope": "registered_input_only_eligibility_not_docking", "command": sys.argv,
        "manifest": str(args.manifest), "manifest_sha256": _sha256(args.manifest),
        "torch": torch.__version__, "openmm": openmm_version, "platform": platform.platform(),
        "device": "cpu", "dtype": "float64", "threads": torch.get_num_threads(),
        "pH": args.pH, "seed": args.seed, "max_residues_per_partner": max_residues,
        "source_sha256": source_hashes,
        "source_unchanged_during_audit": all(_sha256(path) == source_hashes[str(path.relative_to(repo))]
                                             for path in source_paths),
        "systems": results,
        "summary": {"registered_pairs": len(rows),
                    "eligible_pairs": sum(result["pair_status"] == "eligible" for result in results),
                    "docking_attempts": 0},
    }
    _save_new(args.out, report)
    print(json.dumps({"saved": str(args.out), **report["summary"]}, indent=2))
    return 0  # Completion of the audit, NOT evidence that the inputs are eligible.


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    selection = commands.add_parser("select", help="Register IDs from a cached official index before reading PDBs")
    selection.add_argument("--index", type=Path, required=True)
    selection.add_argument("--out", type=Path, required=True)
    selection.add_argument("--count", type=int, default=5)
    selection.add_argument("--max-residues", type=int, default=150)
    selection.set_defaults(run=select)
    validation = commands.add_parser("audit", help="Read registered apo files and attempt strict ff14SB preparation")
    validation.add_argument("--manifest", type=Path, required=True)
    validation.add_argument("--pdb-dir", type=Path, required=True)
    validation.add_argument("--out", type=Path, required=True)
    validation.add_argument("--pH", type=float, default=7.0)
    validation.add_argument("--seed", type=int, default=0)
    validation.add_argument("--threads", type=int, default=1)
    validation.set_defaults(run=audit)
    args = parser.parse_args()
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
