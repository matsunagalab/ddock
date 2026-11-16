#!/usr/bin/env python3
"""Evaluate a completed apo pilot using the independently installed official DockQ.

Run with data/inducedfit/dockq_env/bin/python. No ddock scoring/geometry code is
imported. Native files are evaluation-only, never supplied to the predictor.
Every requested model, including missing/failed exports, remains in the report.
DockQ quality thresholds are reported as DockQ categories, not PINDER results.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time


DOCKQ_VERSION = "2.1.3"
PRIMARY_SOURCES = {
    "dockq": "https://github.com/wallnerlab/DockQ",
    "dockq_paper": "https://doi.org/10.1093/bioinformatics/btae586",
    "benchmark_chain_pair": "https://zlab.wenglab.org/zdock/benchmark.pdf",
    "native_1BRS": "https://files.rcsb.org/download/1BRS.pdb",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_mapping(mapping: str) -> tuple[str, str]:
    parts = mapping.split(":")
    if (len(parts) != 2 or any(len(part) != 2 or len(set(part)) != 2 for part in parts)
            or any(not c.isalnum() for part in parts for c in part)):
        raise ValueError("Use an explicit two-chain mapping MODEL:NATIVE, for example RL:AD; no wildcards")
    return parts[0], parts[1]


def quality(dockq: float) -> str:
    if not math.isfinite(dockq) or not 0 <= dockq <= 1.0000000001:
        raise ValueError("DockQ must be finite and in [0, 1]")
    return "high" if dockq >= .80 else "medium" if dockq >= .49 else "acceptable" if dockq >= .23 else "incorrect"


def checked_metrics(data: dict, mapping: str) -> dict:
    model_chains, native_chains = parse_mapping(mapping)
    expected = dict(zip(native_chains, model_chains))  # Official JSON is native:model.
    if data.get("best_mapping") != expected:
        raise ValueError("Official scorer did not retain the registered chain mapping")
    interfaces = data.get("best_result", {})
    if len(interfaces) != 1:
        raise ValueError("Expected exactly one native interface")
    interface, metrics = next(iter(interfaces.items()))
    if set(interface) != set(native_chains):
        raise ValueError("Scored a different native interface")
    for name in ("DockQ", "iRMSD", "LRMSD", "fnat"):
        if not isinstance(metrics.get(name), (int, float)) or not math.isfinite(metrics[name]):
            raise ValueError(f"Nonfinite or missing {name}")
    if metrics["iRMSD"] < 0 or metrics["LRMSD"] < 0 or not 0 <= metrics["fnat"] <= 1:
        raise ValueError("Invalid RMSD or contact fraction")
    result = dict(metrics)
    result["dockq_quality"] = quality(metrics["DockQ"])
    result["acceptable_or_better_by_dockq"] = metrics["DockQ"] >= .23
    return result


def _residue_id(residue) -> dict:
    return {"number": int(residue.id[1]), "insertion_code": residue.id[2].strip(),
            "residue_name": residue.resname}


def chain_alignment(model_chain, native_chain) -> tuple[dict, list[tuple]]:
    # The SAME pinned official sequence alignment as the evaluator, including
    # its exclusion of mismatched residues from matched-residue measurements.
    from DockQ.DockQ import align_chains, format_alignment
    alignment = format_alignment(align_chains(model_chain, native_chain))
    model_residues, native_residues = list(model_chain.get_residues()), list(native_chain.get_residues())
    mi = ni = 0
    pairs, mismatches, absent_native, absent_model = [], [], [], []
    for model_letter, native_letter in zip(alignment["seqA"], alignment["seqB"], strict=True):
        if model_letter != "-" and native_letter != "-":
            if model_letter == native_letter:
                pairs.append((model_residues[mi], native_residues[ni]))
            else:
                mismatches.append({"model": _residue_id(model_residues[mi]),
                                   "native": _residue_id(native_residues[ni])})
        elif model_letter != "-":
            absent_native.append(_residue_id(model_residues[mi]))
        elif native_letter != "-":
            absent_model.append(_residue_id(native_residues[ni]))
        mi += model_letter != "-"
        ni += native_letter != "-"
    if mi != len(model_residues) or ni != len(native_residues):
        raise ValueError("Sequence alignment does not account for every parsed residue")
    return {"model_chain": model_chain.id, "native_chain": native_chain.id,
            "model_residues": mi, "native_residues": ni, "identical_aligned_residues": len(pairs),
            "alignment": alignment, "mismatches": mismatches,
            "model_residues_missing_from_native": absent_native,
            "native_residues_missing_from_model": absent_model,
            "mismatch_metric_policy": "official DockQ matching-residue calculation excludes mismatches"}, pairs


def sequence_audit(model_path: Path, native_path: Path, mapping: str) -> list[dict]:
    from DockQ.DockQ import load_PDB
    model_chains, native_chains = parse_mapping(mapping)
    model = load_PDB(str(model_path), chains=list(model_chains))
    native = load_PDB(str(native_path), chains=list(native_chains))
    return [chain_alignment(model[m], native[n])[0] for m, n in zip(model_chains, native_chains)]


def apo_deformation(native_path: Path, native_chains: str, apo_paths: tuple[Path, Path],
                    apo_chains: tuple[str, str]) -> dict:
    """Evaluation-only CA deformation after independent monomer alignment.

    Interface residues have any native heavy-atom distance <=5 A. Each monomer
    is fitted using all sequence-identical paired CAs. Interface residual
    squares are then pooled. This is study-specific D, not CAPRI iRMSD.
    """
    import numpy as np
    from DockQ.DockQ import load_PDB
    native = load_PDB(str(native_path), chains=list(native_chains))
    residue_sets = [list(native[c].get_residues()) for c in native_chains]
    atom_coords, atom_residue_indices = [], []
    for residues in residue_sets:
        atoms = [(i, atom) for i, residue in enumerate(residues) for atom in residue.get_atoms()
                 if str(getattr(atom, "element", "")).upper() not in {"H", "D"}]
        atom_coords.append(np.asarray([atom.coord for _, atom in atoms], dtype=np.float64))
        atom_residue_indices.append(np.asarray([i for i, _ in atoms]))
    close = ((atom_coords[0][:, None, :] - atom_coords[1][None, :, :]) ** 2).sum(-1) <= 25.0
    native_interface_ids = [set(atom_residue_indices[side][close.any(axis=1 - side)].tolist()) for side in (0, 1)]
    pooled = []
    chain_reports = []
    for side, (path, apo_chain, native_chain) in enumerate(zip(apo_paths, apo_chains, native_chains)):
        model = load_PDB(str(path), chains=[apo_chain])
        audit, pairs = chain_alignment(model[apo_chain], native[native_chain])
        pairs = [(a, b) for a, b in pairs if "CA" in a and "CA" in b]
        if len(pairs) < 3:
            raise ValueError("At least three matching CAs are required for monomer alignment")
        x = np.asarray([a["CA"].coord for a, _ in pairs], dtype=np.float64)
        y = np.asarray([b["CA"].coord for _, b in pairs], dtype=np.float64)
        xc, yc = x - x.mean(0), y - y.mean(0)
        u, _, vt = np.linalg.svd(xc.T @ yc)
        correction = np.eye(3)
        correction[-1, -1] = np.linalg.det(u @ vt)
        residual2 = ((xc @ (u @ correction @ vt) - yc) ** 2).sum(-1)
        interface_residues = {residue_sets[side][i] for i in native_interface_ids[side]}
        mask = np.asarray([b in interface_residues for _, b in pairs])
        interface_values = residual2[mask]
        if not len(interface_values):
            raise ValueError("No matching native interface CAs")
        pooled.extend(interface_values.tolist())
        chain_reports.append({"apo": str(path), "apo_sha256": sha256(path), "alignment": audit,
                              "aligned_ca_count": len(pairs), "all_matched_ca_rmsd": float(np.sqrt(residual2.mean())),
                              "native_interface_residue_count": len(interface_residues),
                              "matched_interface_ca_count": len(interface_values),
                              "interface_ca_rmsd": float(np.sqrt(interface_values.mean()))})
    return {"definition": "native heavy-atom interface <=5 A; independent full matched-CA monomer fit; pooled interface CA RMSD",
            "D_angstrom": math.sqrt(sum(pooled) / len(pooled)), "chains": chain_reports,
            "caveat": "sequence mismatches and missing residues excluded and counted; evaluation-only diagnostic, not an input-selection or tuning criterion"}


def models_from_report(path: Path) -> tuple[list[dict], dict]:
    report = json.loads(path.read_text())
    if report.get("status") not in {"completed", "partial_failure", "exception"}:
        raise ValueError("Predictor report must be finished before native scoring")
    entries, arms = [], {}
    for arm, state in report["arms"].items():
        rows = state.get("candidates", [])
        expected = state.get("requested_candidate_count", report.get("configuration", {}).get("top_k", len(rows)))
        if type(expected) is not int or expected < 0:
            raise ValueError("Invalid registered candidate count")
        arms[arm] = {"predictor_status": state.get("status"), "requested_candidates": expected,
                     "candidate_shortfall": state.get("candidate_shortfall"),
                     "failure_count": state.get("failure_count")}
        seen_selection, seen_final = set(), set()
        for row in rows:
            selection_rank, final_rank = row["selection_rank"], row.get("final_rank")
            if (type(selection_rank) is not int or selection_rank < 1
                    or (final_rank is not None and (type(final_rank) is not int or final_rank < 1))):
                raise ValueError("Predictor ranks must be positive integers")
            if selection_rank in seen_selection or (final_rank is not None and final_rank in seen_final):
                raise ValueError("Duplicate predictor rank")
            seen_selection.add(selection_rank)
            if final_rank is not None:
                seen_final.add(final_rank)
            for stage, key, rank in (("initial", "initial_files", selection_rank), ("final", "files", final_rank)):
                relative = row.get(key, {}).get("complex")
                entries.append({"arm": arm, "stage": stage, "rank": rank,
                                "selection_rank": selection_rank, "final_rank": final_rank,
                                "path": str((path.parent / relative).resolve()) if relative else None,
                                "predictor_status": row.get("status"),
                                "predictor_error": row.get("error")})
        for missing in sorted(set(range(1, expected + 1)) - seen_selection):
            for stage in ("initial", "final"):
                entries.append({"arm": arm, "stage": stage, "rank": missing if stage == "initial" else None,
                                "selection_rank": missing, "final_rank": None, "path": None,
                                "predictor_status": "candidate_shortfall"})
    return entries, arms


def summarize(entries: list[dict]) -> dict:
    output = {}
    for arm, stage in sorted({(row["arm"], row["stage"]) for row in entries}):
        rows = [row for row in entries if (row["arm"], row["stage"]) == (arm, stage)]
        scored = [row for row in rows if row["status"] == "scored"]
        first = next((row for row in rows if row.get("rank") == 1), None)
        top1 = first.get("metrics", {}).get("DockQ", 0.0) if first else 0.0
        within_five = [row for row in scored if row.get("rank") is not None and row["rank"] <= 5]
        best = max(within_five, key=lambda row: row["metrics"]["DockQ"], default=None)
        first_metrics = first.get("metrics", {}) if first else {}
        output[f"{arm}_{stage}"] = {
            "arm": arm, "stage": stage, "registered_candidate_rows": len(rows),
            "scored_candidates": len(scored), "failed_or_missing_candidates": len(rows) - len(scored),
            "top1_DockQ": top1, "top1_scored": bool(first and first["status"] == "scored"),
            "top1_iRMSD": first_metrics.get("iRMSD"), "top1_LRMSD": first_metrics.get("LRMSD"),
            "top1_fnat": first_metrics.get("fnat"),
            "top1_selection_rank": first.get("selection_rank") if first else None,
            "top1_acceptable_or_better_by_dockq": top1 >= .23,
            "oracle_at_5_DockQ": best["metrics"]["DockQ"] if best else 0.0,
            "oracle_at_5_rank": best["rank"] if best else None,
            "oracle_at_5_selection_rank": best.get("selection_rank") if best else None,
            "ranking_basis": ("selection_rank" if stage == "initial" else "final_energy_rank"
                              if stage == "final" else "explicit_model_label"),
        }
    return output


def run_dockq(executable: Path, model: Path, native: Path, mapping: str, artifacts: Path,
              label: str, allowed_mismatches: int, timeout: float) -> dict:
    raw_json = artifacts / f"{label}.json"
    command = [str(executable), str(model), str(native), "--mapping", mapping,
               "--allowed_mismatches", str(allowed_mismatches), "--n_cpu", "1", "--short",
               "--json", str(raw_json)]
    result = {"command": command, "raw_json": str(raw_json), "model_sha256": sha256(model)}
    started = time.monotonic()
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        (artifacts / f"{label}.stdout.txt").write_text(process.stdout)
        (artifacts / f"{label}.stderr.txt").write_text(process.stderr)
        result["returncode"] = process.returncode
        if process.returncode != 0:
            raise RuntimeError(f"DockQ exited {process.returncode}: {process.stderr[-2000:]}")
        result["metrics"] = checked_metrics(json.loads(raw_json.read_text()), mapping)
        result["status"] = "scored"
    except Exception as error:
        result.update(status="evaluation_failure", error_type=type(error).__name__, error=str(error))
    result["wall_seconds"] = time.monotonic() - started
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--run-report", type=Path)
    parser.add_argument("--model", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--mapping", default="RL:AD")
    parser.add_argument("--allowed-mismatches", type=int, default=1,
                        help="Fixed allowance for 1A19 Cys40 versus 1BRS Ala40; default 1")
    parser.add_argument("--dockq", type=Path, default=Path(sys.executable).parent / "DockQ")
    parser.add_argument("--timeout", type=float, default=120.)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--apo-receptor", type=Path)
    parser.add_argument("--apo-receptor-chain", default="B")
    parser.add_argument("--apo-ligand", type=Path)
    parser.add_argument("--apo-ligand-chain", default="A")
    args = parser.parse_args()
    _, native_chains = parse_mapping(args.mapping)
    if args.allowed_mismatches < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("Invalid mismatch allowance or timeout")
    if bool(args.apo_receptor) != bool(args.apo_ligand):
        parser.error("Supply both apo partners or neither")
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    artifacts = args.out.parent / f"{args.out.stem}_dockq_artifacts"
    artifacts.mkdir(parents=True, exist_ok=False)
    report = {"scope": "single historical apo development pilot; no benchmark success-rate claim",
              "status": "running", "command": sys.argv, "primary_sources": PRIMARY_SOURCES,
              "mapping_model_to_native": args.mapping, "allowed_mismatches": args.allowed_mismatches,
              "alignment_policy": "official DockQ default sequence alignment; fixed chain assignment, no mapping search",
              "native": str(args.native.resolve()), "native_sha256": sha256(args.native),
              "native_access": "evaluation process only; no inference parameter or coordinate update",
              "script_sha256": sha256(Path(__file__)), "python": platform.python_version(),
              "platform": platform.platform(), "models": []}

    def save():
        temporary = args.out.with_suffix(args.out.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.out)

    save()
    try:
        from DockQ import DockQ as official
        versions = {package: importlib.metadata.version(package) for package in ("DockQ", "numpy", "biopython")}
        report["versions"] = versions
        if versions["DockQ"] != DOCKQ_VERSION:
            raise ValueError(f"Registered evaluator requires DockQ {DOCKQ_VERSION}")
        report["dockq_source_sha256"] = sha256(Path(official.__file__))
        report["dockq_executable_sha256"] = sha256(args.dockq)
        if args.run_report:
            entries, arms = models_from_report(args.run_report)
            report.update(predictor_report=str(args.run_report), predictor_report_sha256=sha256(args.run_report),
                          registered_arms=arms)
        else:
            entries = []
        labels = set()
        for item in args.model:
            label, path = item.split("=", 1)
            if not label or label in labels:
                raise ValueError("Explicit model labels must be nonempty and unique")
            labels.add(label)
            entries.append({"arm": label, "stage": "explicit", "rank": 1, "path": str(Path(path).resolve())})
        sanity = run_dockq(args.dockq.resolve(), args.native.resolve(), args.native.resolve(),
                           f"{native_chains}:{native_chains}", artifacts.resolve(), "native_sanity", 0, args.timeout)
        report["native_sanity"] = sanity
        if (sanity["status"] != "scored" or sanity["metrics"]["DockQ"] < .99
                or sanity["metrics"]["fnat"] < .999999):
            raise RuntimeError("Native-versus-native DockQ sanity failed")
        if args.apo_receptor:
            report["apo_deformation"] = apo_deformation(args.native, native_chains,
                (args.apo_receptor, args.apo_ligand), (args.apo_receptor_chain, args.apo_ligand_chain))
        for index, entry in enumerate(entries):
            row = dict(entry)
            report["models"].append(row)
            if not entry.get("path") or not Path(entry["path"]).is_file():
                row.update(status="missing_prediction", error="No exported model file", failure_DockQ=0.0)
                save()
                continue
            try:
                row["sequence_alignment"] = sequence_audit(Path(entry["path"]), args.native, args.mapping)
                if any(len(a["mismatches"]) > args.allowed_mismatches for a in row["sequence_alignment"]):
                    raise ValueError("Sequence mismatch count exceeds the predeclared evaluator allowance")
                row.update(run_dockq(args.dockq.resolve(), Path(entry["path"]).resolve(), args.native.resolve(),
                                     args.mapping, artifacts.resolve(), f"model_{index:04d}",
                                     args.allowed_mismatches, args.timeout))
            except Exception as error:
                row.update(status="evaluation_failure", error_type=type(error).__name__, error=str(error))
            save()
        report["summary"] = summarize(report["models"])
        report["status"] = ("evaluation_incomplete" if any(row["status"] == "evaluation_failure" for row in report["models"])
                            else "completed")
    except Exception as error:
        report.update(status="evaluation_failure", error_type=type(error).__name__, error=str(error))
    save()
    print(json.dumps({key: report[key] for key in ("status", "error", "summary") if key in report}, indent=2))
    print(f"Saved {args.out}")
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
