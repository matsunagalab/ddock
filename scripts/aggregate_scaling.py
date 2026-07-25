"""Aggregate the PINDER cluster-count scaling runs into the tables the report
needs: per (N, seed) round-0 / round-1 held-out numbers, the mining gain, and
the mean +- sd across seeds.

Also verifies the experiment's structural invariants across the runs it finds:
the fit/validation sets must be nested across N, must be disjoint from each
other, and must never intersect the fixed PINDER-S test ids.

Example
-------
    uv run python scripts/aggregate_scaling.py --runs-dir data/scaling/runs
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

KS = (1, 5, 10, 50, 100)


def _mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(v)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default="data/scaling/runs", dest="runs_dir")
    ap.add_argument("--test-ids", default="data/pinder_test_ids.txt", dest="test_ids")
    ap.add_argument("--out-prefix", default="data/scaling/scaling_summary",
                    dest="out_prefix")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    test_ids = set(ln.strip() for ln in Path(args.test_ids).read_text().splitlines()
                   if ln.strip())

    runs = {}
    splits = {}
    for d in sorted(runs_dir.glob("N*_seed*")):
        n = int(d.name.split("_")[0][1:])
        seed = int(d.name.split("seed")[1])
        rounds = {}
        for f in sorted(d.glob("round*_metrics.json")):
            r = json.loads(f.read_text())
            rounds[r["round"]] = r
        if not rounds:
            print(f"[skip] {d.name}: no completed round yet")
            continue
        base = d / "baseline_test.json"
        runs[(n, seed)] = {
            "rounds": rounds,
            "baseline": json.loads(base.read_text()) if base.exists() else None,
            "coverage": json.loads((d / "coverage.json").read_text())
            if (d / "coverage.json").exists() else None,
        }
        sp = json.loads((d / "split.json").read_text())
        splits[(n, seed)] = (set(sp["fit_ids"]), set(sp["val_ids"]))

    if not runs:
        raise SystemExit(f"no completed runs under {runs_dir}")

    # ---- structural checks ------------------------------------------------
    print("=== structural checks ===")
    ok = True
    for (n, seed), (fit, val) in sorted(splits.items()):
        if fit & val:
            print(f"  FAIL {n}/{seed}: fit ∩ val = {len(fit & val)}"); ok = False
        if (fit | val) & test_ids:
            print(f"  FAIL {n}/{seed}: TEST leak {len((fit|val) & test_ids)}"); ok = False
    ns = sorted({n for n, _ in splits})
    for seed in sorted({s for _, s in splits}):
        chain = [n for n in ns if (n, seed) in splits]
        for a, b in zip(chain, chain[1:]):
            fa, va = splits[(a, seed)]
            fb, vb = splits[(b, seed)]
            if not fa <= fb:
                print(f"  FAIL seed {seed}: fit {a} ⊄ fit {b}"); ok = False
            if not va <= vb:
                print(f"  FAIL seed {seed}: val {a} ⊄ val {b}"); ok = False
    # splits are seed-independent by construction; verify that too
    for n in ns:
        seeds = [s for (m, s) in splits if m == n]
        for s in seeds[1:]:
            if splits[(n, seeds[0])] != splits[(n, s)]:
                print(f"  FAIL N={n}: split differs between seeds"); ok = False
    print(f"  fit/val disjoint, TEST-clean, nested across N: "
          f"{'OK' if ok else 'PROBLEMS FOUND'}")

    # ---- per-run table ----------------------------------------------------
    rows = []
    for (n, seed), rec in sorted(runs.items()):
        rs = rec["rounds"]
        last = max(rs)
        r0, r1 = rs[0], rs[last]
        row = {"n_fit": n, "seed": seed, "n_val": r0["n_val"],
               "n_test": r0["n_test"], "last_round": last}
        for k in KS:
            row[f"r0_dockq@{k}"] = r0["test_success_dockq"][str(k)]
            row[f"r{last}_dockq@{k}"] = r1["test_success_dockq"][str(k)]
            row[f"gain_dockq@{k}"] = (r1["test_success_dockq"][str(k)]
                                      - r0["test_success_dockq"][str(k)])
            row[f"r0_rmsd@{k}"] = r0["test_success_rmsd"][str(k)]
            row[f"r{last}_rmsd@{k}"] = r1["test_success_rmsd"][str(k)]
        row["r0_meanbestdockq1"] = r0["test_mean_best_dockq_at1"]
        row["r1_meanbestdockq1"] = r1["test_mean_best_dockq_at1"]
        row["r0_val_loss"] = r0["val_loss"]
        row["r1_val_loss"] = r1["val_loss"]
        row["r0_fit_dockq@1"] = r0["fit_success_dockq"]["1"]
        row["r1_fit_dockq@1"] = r1["fit_success_dockq"]["1"]
        row["alpha_r1"] = r1["alpha"]
        row["d_iface_r1"] = r1["d_iface_norm"]
        row["n_skipped"] = r1["n_skipped_total"]
        row["mean_pool_r0"] = r0["pool_mean"]["n"]
        row["mean_pool_r1"] = r1["pool_mean"]["n"]
        row["n_pos"] = r1["pool_mean"]["n_pos"]
        row["n_rand_neg"] = r1["pool_mean"]["n_rand_neg"]
        row["n_hard_neg"] = r1["pool_mean"]["n_hard_neg"]
        row["r1_accepted_ckpts"] = r1["n_accepted_checkpoints"]
        row["r1_rejected_ckpts"] = r1["n_rejected_checkpoints"]
        row["wall_min"] = (r0["round_seconds"] + r1["round_seconds"]) / 60
        row["peak_gpu_gib"] = max(r0["peak_gpu_gib"], r1["peak_gpu_gib"])
        if rec["coverage"]:
            row["fit_zero_components"] = rec["coverage"]["fit"]["n_components_zero"]
            row["fit_median_coverage"] = rec["coverage"]["fit"]["median_coverage"]
        rows.append(row)

    per_run = Path(args.out_prefix + "_per_run.csv")
    cols = list(rows[0].keys())
    with open(per_run, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(f"{r.get(c, '')}" for c in cols) + "\n")

    # ---- across-seed aggregation -----------------------------------------
    agg_rows = []
    print("\n=== held-out PINDER-S TEST, DockQ success@K (mean ± sd over seeds) ===")
    hdr = f"{'N_fit':>6} {'seeds':>5} " + " ".join(f"{'r0@'+str(k):>13}" for k in KS)
    print(hdr)
    for n in ns:
        seeds = sorted(s for (m, s) in runs if m == n)
        rr = [runs[(n, s)] for s in seeds]
        last = max(rr[0]["rounds"])
        line = {"n_fit": n, "n_seeds": len(seeds)}
        cells = []
        for k in KS:
            m0, s0 = _mean_sd([r["rounds"][0]["test_success_dockq"][str(k)] for r in rr])
            m1, s1 = _mean_sd([r["rounds"][last]["test_success_dockq"][str(k)] for r in rr])
            g, gs = _mean_sd([(r["rounds"][last]["test_success_dockq"][str(k)]
                               - r["rounds"][0]["test_success_dockq"][str(k)])
                              for r in rr])
            line[f"r0_dockq@{k}_mean"] = m0
            line[f"r0_dockq@{k}_sd"] = s0
            line[f"r1_dockq@{k}_mean"] = m1
            line[f"r1_dockq@{k}_sd"] = s1
            line[f"gain_dockq@{k}_mean"] = g
            line[f"gain_dockq@{k}_sd"] = gs
            cells.append(f"{m0*100:5.1f}±{s0*100:4.1f}%")
        for k in KS:
            m0, s0 = _mean_sd([r["rounds"][0]["test_success_rmsd"][str(k)] for r in rr])
            m1, s1 = _mean_sd([r["rounds"][last]["test_success_rmsd"][str(k)] for r in rr])
            line[f"r0_rmsd@{k}_mean"] = m0
            line[f"r1_rmsd@{k}_mean"] = m1
        for key, path in (("meanbestdockq1", "test_mean_best_dockq_at1"),
                          ("val_loss", "val_loss"),
                          ("fit_dockq@1", None)):
            if path:
                m0, s0 = _mean_sd([r["rounds"][0][path] for r in rr])
                m1, s1 = _mean_sd([r["rounds"][last][path] for r in rr])
            else:
                m0, s0 = _mean_sd([r["rounds"][0]["fit_success_dockq"]["1"] for r in rr])
                m1, s1 = _mean_sd([r["rounds"][last]["fit_success_dockq"]["1"] for r in rr])
            line[f"r0_{key}_mean"], line[f"r0_{key}_sd"] = m0, s0
            line[f"r1_{key}_mean"], line[f"r1_{key}_sd"] = m1, s1
        line["mean_wall_min"], _ = _mean_sd(
            [(r["rounds"][0]["round_seconds"] + r["rounds"][last]["round_seconds"]) / 60
             for r in rr])
        line["max_peak_gpu_gib"] = max(
            max(r["rounds"][0]["peak_gpu_gib"], r["rounds"][last]["peak_gpu_gib"])
            for r in rr)
        line["total_skipped"] = sum(r["rounds"][last]["n_skipped_total"] for r in rr)
        agg_rows.append(line)
        print(f"{n:>6} {len(seeds):>5} " + " ".join(f"{c:>13}" for c in cells))

    print("\n=== hard-negative mining gain (round1 − round0), percentage points ===")
    print(f"{'N_fit':>6} " + " ".join(f"{'@'+str(k):>13}" for k in KS))
    for line in agg_rows:
        print(f"{line['n_fit']:>6} " + " ".join(
            f"{line[f'gain_dockq@{k}_mean']*100:+6.2f}±{line[f'gain_dockq@{k}_sd']*100:4.2f}"
            for k in KS))

    print("\n=== fit-set top-1 (memorisation) vs held-out top-100 ===")
    print(f"{'N_fit':>6} {'r0 fit@1':>12} {'r1 fit@1':>12} "
          f"{'r0 test@100':>12} {'r1 test@100':>12} {'val loss r0':>12} {'val loss r1':>12}")
    for line in agg_rows:
        print(f"{line['n_fit']:>6} "
              f"{line['r0_fit_dockq@1_mean']*100:11.1f}% "
              f"{line['r1_fit_dockq@1_mean']*100:11.1f}% "
              f"{line['r0_dockq@100_mean']*100:11.1f}% "
              f"{line['r1_dockq@100_mean']*100:11.1f}% "
              f"{line['r0_val_loss_mean']:12.4f} {line['r1_val_loss_mean']:12.4f}")

    agg_csv = Path(args.out_prefix + "_by_N.csv")
    cols = list(agg_rows[0].keys())
    with open(agg_csv, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in agg_rows:
            fh.write(",".join(f"{r.get(c, '')}" for c in cols) + "\n")
    print(f"\nwrote {per_run} and {agg_csv}")


if __name__ == "__main__":
    main()
