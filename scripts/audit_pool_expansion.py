"""Fail-loud completeness audit for an incrementally expanded pool cache.

The miner can reuse compatible read-only caches from a smaller, prefix-nested
cohort and write only new complexes to another directory.  This audit checks
the union against the exact fit/validation IDs in the new split, verifies that
duplicate cache entries carry identical pose identities, counts complexes with
at least one acceptable pose, and records OOM/rescue outcomes.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from types import SimpleNamespace

import torch


def expand_globs(patterns: list[str]) -> tuple[list[Path], list[str]]:
    paths: list[Path] = []
    unmatched: list[str] = []
    for pattern in patterns:
        matches = [Path(p) for p in sorted(glob.glob(pattern))]
        if not matches:
            unmatched.append(pattern)
        paths.extend(matches)
    return (list(dict.fromkeys(path.resolve() for path in paths)), unmatched)


def load_split(path: Path) -> tuple[dict, set[str]]:
    split = json.loads(path.read_text())
    fit = split["fit_ids"]
    val = split["val_ids"]
    if len(fit) != len(set(fit)) or len(val) != len(set(val)):
        raise SystemExit(f"duplicate ID in {path}")
    if set(fit) & set(val):
        raise SystemExit(f"fit/validation overlap in {path}")
    expected = set(fit) | set(val)
    if len(expected) != split["n_total"]:
        raise SystemExit(
            f"{path}: n_total={split['n_total']} but split union={len(expected)}"
        )
    return split, expected


def reconstruct_split(args: argparse.Namespace) -> dict:
    """Rebuild the deterministic split even if every worker failed early."""
    from run_pinder_scaling import select_split

    ns = SimpleNamespace(
        n_fit=args.n_fit,
        n_total=args.n_total,
        val_frac=args.val_frac,
        master_ids=args.master_ids,
        prep_manifest=args.prep_manifest,
        max_grid_voxels=args.max_grid_voxels,
        exclude_bad_geometry=args.exclude_bad_geometry,
        exclude_homodimer=args.exclude_homodimer,
        grid_voxels=args.grid_voxels,
        spacing=args.spacing,
    )
    selected, fit_ids, val_ids, _, info = select_split(ns)
    return {
        "n_fit": len(fit_ids),
        "n_val": len(val_ids),
        "n_total": len(selected),
        "selection": info,
        "fit_ids": fit_ids,
        "val_ids": val_ids,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--split-check", type=Path, action="append", default=[])
    ap.add_argument("--pool-glob", action="append", required=True)
    ap.add_argument("--skip-log-glob", action="append", default=[])
    ap.add_argument("--dockq-threshold", type=float, default=0.23)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-fit", type=int, default=0,
                    help="if nonzero, reconstruct the canonical split")
    ap.add_argument("--n-total", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--master-ids", default="data/scaling/master_ids.txt")
    ap.add_argument("--prep-manifest", default="data/scaling/prep_manifest.jsonl")
    ap.add_argument("--grid-voxels", default="data/scaling/grid_voxels_1.2.json")
    ap.add_argument("--max-grid-voxels", type=int, default=31_250_000)
    ap.add_argument("--exclude-bad-geometry",
                    default="data/scaling/excluded_bad_geometry.txt")
    ap.add_argument("--exclude-homodimer", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--spacing", type=float, default=1.2)
    args = ap.parse_args()

    missing_split_files = []
    if args.n_fit:
        split = reconstruct_split(args)
        expected = set(split["fit_ids"]) | set(split["val_ids"])
        split_paths = [args.split, *args.split_check]
    else:
        split, expected = load_split(args.split)
        split_paths = args.split_check
    split_mismatches: list[str] = []
    for other_path in split_paths:
        if not other_path.exists():
            missing_split_files.append(str(other_path))
            continue
        other, other_expected = load_split(other_path)
        if (other["fit_ids"] != split["fit_ids"]
                or other["val_ids"] != split["val_ids"]):
            split_mismatches.append(str(other_path))
        if other_expected != expected:
            split_mismatches.append(f"{other_path} (set mismatch)")

    files, unmatched_pool_globs = expand_globs(args.pool_glob)
    seen: dict[str, dict] = {}
    duplicate_entries = 0
    inconsistent_duplicates: list[dict[str, str]] = []
    file_records = []
    cache_load_errors = []
    declared_skipped = 0
    for path in files:
        try:
            blob = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        except Exception as exc:  # audit must still write the missing-set JSON
            cache_load_errors.append({
                "path": str(path),
                "error": f"{type(exc).__name__}: {exc}"[:500],
            })
            continue
        pools = blob.get("pools", [])
        n_skipped = int(blob.get("n_skipped", 0))
        declared_skipped += n_skipped
        file_records.append({
            "path": str(path),
            "n_entries": len(pools),
            "declared_skipped": n_skipped,
        })
        for item in pools:
            pid = item["name"]
            pose_key = item.get("pose_key")
            if pose_key is None:
                cache_load_errors.append({
                    "path": str(path),
                    "error": f"{pid} has no pose_key",
                })
                continue
            search = item["prov"] == 0
            search_positive = item["dockq"][search] >= args.dockq_threshold
            record = {
                "pose_key": pose_key,
                "n_poses": int(item["dockq"].numel()),
                "n_search_poses": int(search.sum()),
                # This is the fit_shell/nonlinear_channels estimand: only FFT
                # search-derived poses enter the ranking fit, and both classes
                # must be present. Enumerated near-native positives do not make
                # a search-failed complex eligible.
                "eligible": bool(search_positive.any().item()
                                 and not search_positive.all().item()),
                "source": str(path),
            }
            if pid in seen:
                duplicate_entries += 1
                prior = seen[pid]
                if (prior["n_poses"] != record["n_poses"]
                        or prior["n_search_poses"] != record["n_search_poses"]
                        or prior["eligible"] != record["eligible"]
                        or not torch.equal(prior["pose_key"], record["pose_key"])):
                    inconsistent_duplicates.append({
                        "id": pid,
                        "first": prior["source"],
                        "second": str(path),
                    })
            else:
                seen[pid] = record
        del blob

    skip_paths, unmatched_skip_log_globs = expand_globs(args.skip_log_glob)
    skip_records = []
    for path in skip_paths:
        for line in path.read_text().splitlines():
            if line.strip():
                skip_records.append({"log": str(path), **json.loads(line)})
    unrescued = sorted({
        row["id"] for row in skip_records if not row.get("rescued", False)
    })
    rescued = sorted({
        row["id"] for row in skip_records if row.get("rescued", False)
    })

    present = set(seen)
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    fit_set = set(split["fit_ids"])
    val_set = set(split["val_ids"])
    eligible_fit = sum(seen[pid]["eligible"] for pid in fit_set if pid in seen)
    eligible_val = sum(seen[pid]["eligible"] for pid in val_set if pid in seen)
    ok = not (
        split_mismatches
        or missing_split_files
        or unmatched_pool_globs
        or unmatched_skip_log_globs
        or cache_load_errors
        or missing
        or extra
        or inconsistent_duplicates
        or unrescued
        or declared_skipped
    )
    result = {
        "status": "complete" if ok else "incomplete",
        "split": str(args.split.resolve()),
        "n_fit_expected": len(fit_set),
        "n_val_expected": len(val_set),
        "n_total_expected": len(expected),
        "n_unique_pool_complexes": len(present),
        "n_eligible_fit": eligible_fit,
        "n_eligible_val": eligible_val,
        "n_eligible_total": eligible_fit + eligible_val,
        "eligibility_rate": (eligible_fit + eligible_val) / max(1, len(expected)),
        "missing_ids": missing,
        "extra_ids": extra,
        "split_mismatches": split_mismatches,
        "missing_split_files": missing_split_files,
        "unmatched_pool_globs": unmatched_pool_globs,
        "unmatched_skip_log_globs": unmatched_skip_log_globs,
        "n_duplicate_entries": duplicate_entries,
        "inconsistent_duplicates": inconsistent_duplicates,
        "cache_load_errors": cache_load_errors,
        "cache_declared_skipped": declared_skipped,
        "rescued_ids": rescued,
        "unrescued_ids": unrescued,
        "cache_files": file_records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"audit: status={result['status']} present={len(present)}/{len(expected)} "
        f"eligible={result['n_eligible_total']} "
        f"missing={len(missing)} extra={len(extra)} "
        f"rescued={len(rescued)} unrescued={len(unrescued)} "
        f"duplicate_entries={duplicate_entries}"
    )
    print(f"audit JSON: {args.out}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
