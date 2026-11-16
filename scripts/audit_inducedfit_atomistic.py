#!/usr/bin/env python3
"""Re-score every frozen initial/final candidate without native/accuracy input.

Example:
  .venv/bin/python scripts/audit_inducedfit_atomistic.py \
    --report data/inducedfit/apo_pilot_v1/pilot_run/report.json \
    --out data/inducedfit/apo_pilot_v1/atomistic_obc2_v1.json

The fixed protocol is written before replay/scoring. No parameter selection,
optimization, new hydrogen placement on a pose, or accuracy evaluation occurs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time
import traceback

import torch

from zdock.inducedfit.atomistic_score import AtomisticScorer, PROTOCOL
from zdock.inducedfit.replay import load_replay


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def coordinate_sha256(xyz):
    return hashlib.sha256(xyz.detach().cpu().to(torch.float64).contiguous().numpy().tobytes()).hexdigest()


def save(path, record):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def rankings(rows):
    result = {}
    for stage in ("initial", "final"):
        stage_rows = [row for row in rows if row["status"] == "completed" and row["stage"] == stage]
        result[stage] = {}
        for group in ("all", *sorted({row["arm"] for row in stage_rows})):
            selected = [row for row in stage_rows if group == "all" or row["arm"] == group]
            result[stage][group] = {}
            for name in ("interaction", "endpoint"):
                key = (lambda row: row["score"]["interaction"]["total"]) if name == "interaction" else (lambda row: row["score"]["endpoint_total"])
                ordered = sorted(selected, key=lambda row: (key(row), row["arm"], row["selection_rank"]))
                result[stage][group][name] = [
                    {"arm": row["arm"], "selection_rank": row["selection_rank"],
                     "original_final_rank": row["original_final_rank"], "score_kj_mol": key(row)}
                    for row in ordered
                ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    if args.out.exists():
        parser.error("output already exists; choose a new path to retain prior results")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    repo = Path(__file__).resolve().parents[1]
    sources = [Path(__file__).resolve(), *sorted((repo / "src/zdock/inducedfit").glob("*.py"))]
    record = {
        "status": "registered", "command": [sys.executable, *sys.argv],
        "report_path": str(args.report.resolve()), "report_sha256": sha256(args.report),
        "protocol": dict(PROTOCOL), "native_or_accuracy_input_accessed": False,
        "selection": "all exported initial and final candidates; no oracle filtering",
        "source_sha256": {str(path.relative_to(repo)): sha256(path) for path in sources},
        "platform": platform.platform(), "processor": platform.machine(), "python": sys.version,
        "torch_version": torch.__version__, "threads": args.threads,
        "rows": [], "failures": [],
    }
    save(args.out, record)
    try:
        context = load_replay(args.report)
        record["replay_source_audit"] = context.source_audit
        with AtomisticScorer.from_proteins(context.proteins["receptor"], context.proteins["ligand"]) as scorer:
            record["scorer_provenance"] = scorer.provenance
            record["status"] = "scoring"
            save(args.out, record)
            for arm, arm_report in context.report["arms"].items():
                for candidate in arm_report.get("candidates", []):
                    for stage in ("initial", "final"):
                        row = {"arm": arm, "stage": stage, "selection_rank": candidate["selection_rank"],
                               "original_final_rank": candidate.get("final_rank")}
                        pose_started = time.perf_counter()
                        try:
                            pose = getattr(context, f"{stage}_pose")(arm, candidate)
                            result = scorer.score(pose.receptor_xyz, pose.ligand_xyz)
                            row.update({"status": "completed", "score": asdict(result),
                                "full_atom_coordinate_sha256": {
                                    "receptor": coordinate_sha256(pose.receptor_xyz),
                                    "ligand": coordinate_sha256(pose.ligand_xyz)},
                                "heavy_export_errors": pose.heavy_export_errors,
                                "amplitudes": pose.amplitudes.tolist(),
                                "original_effective_energy": pose.stored_energy,
                                "original_effective_strain": pose.strain})
                        except Exception as error:
                            row.update({"status": "failed", "error_type": type(error).__name__,
                                        "error": str(error), "traceback": traceback.format_exc()})
                            record["failures"].append({k: row[k] for k in ("arm", "stage", "selection_rank", "error_type", "error")})
                        row["wall_seconds"] = time.perf_counter() - pose_started
                        record["rows"].append(row)
                        record["rankings"] = rankings(record["rows"])
                        save(args.out, record)
                        print(json.dumps({k: row[k] for k in ("arm", "stage", "selection_rank", "status", "wall_seconds")}), flush=True)
        record["status"] = "partial_failure" if record["failures"] else "completed"
    except Exception as error:
        record.update({"status": "failed", "error_type": type(error).__name__,
                       "error": str(error), "traceback": traceback.format_exc()})
        record["failures"].append({"stage": "initialization", "error_type": type(error).__name__, "error": str(error)})
    record["total_wall_seconds"] = time.perf_counter() - started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record["peak_rss_bytes"] = int(rss if sys.platform == "darwin" else rss * 1024)
    save(args.out, record)
    print(json.dumps({"status": record["status"], "out": str(args.out),
                      "wall_seconds": record["total_wall_seconds"], "failures": len(record["failures"])}), flush=True)
    return 0 if record["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
