"""Re-report every completed comparison at a chosen success@K.

success@1 was the pre-registered primary endpoint and it was a poor choice. It
uses one binary observation per complex, so at n = 236 the exact McNemar test
needs a large one-directional discordant count before it can reject: several
contrasts in this report sit at "4 wins 0 losses, p = 0.125", where 0.125 is the
smallest p the test can return and significance is unreachable regardless of the
effect. success@10 is insensitive to reshuffling within the top few poses, which
is where most of the run-to-run noise lives, and it still has headroom (92%
against 98% at K = 100, which is saturated).

This re-reports the existing per-complex tables at any K, so the switch changes
the readout and not the runs.
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) * 0.5 ** n)


def load(d: str, k: int, cond: str = "trained") -> dict:
    f = glob.glob(os.path.join(d, "*_search_per_complex.csv"))
    if not f:
        return {}
    out = {}
    for r in csv.DictReader(open(f[0])):
        if r["condition"] != cond:
            continue
        out[r["name"]] = int(float(r[f"succ_dockq@{k}"]))
    return out


def paired(a: dict, b: dict):
    names = sorted(set(a) & set(b))
    w = sum(1 for n in names if not a[n] and b[n])
    l = sum(1 for n in names if a[n] and not b[n])
    return (len(names), sum(a[n] for n in names) / len(names),
            sum(b[n] for n in names) / len(names), w, l, mcnemar_exact(w, l))


CONDS = [
    ("experiment 1: dimension x margin (fixed pool)", [
        ("full 144  m5", "data/scaling/compare_fact_full_m5_seed{s}"),
        ("full 144  m0", "data/scaling/compare_fact_full_m0_seed{s}"),
        ("add   23  m5", "data/scaling/compare_fact_add_m5_seed{s}"),
        ("add   23  m0", "data/scaling/compare_fact_add_m0_seed{s}"),
        ("sym   12  m5", "data/scaling/compare_fact_sym_m5_seed{s}"),
        ("sym   12  m0", "data/scaling/compare_fact_sym_m0_seed{s}"),
    ]),
    ("experiment 4: mining round 1 (seed 0)", [
        ("round 0", "data/scaling/compare_r1_seed0_round0"),
        ("round 1 mine", "data/scaling/compare_r1_seed0_hardneg"),
        ("round 1 continue", "data/scaling/compare_r1_seed0_none"),
    ]),
    ("N scaling, matched budget", [
        ("N=220", "data/scaling/compare_nfixed_N220_seed{s}"),
        ("N=500", "data/scaling/compare_nfixed_N500_seed{s}"),
        ("N=1000", "data/scaling/compare_nfixed_N1000_seed{s}"),
    ]),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-k", type=int, default=10)
    args = ap.parse_args()
    K = args.k
    print(f"success@{K} on the frozen TEST pool, vs the published baseline\n")
    for title, conds in CONDS:
        print(f"### {title}")
        print(f"{'condition':<20}{'seeds':>6}{'baseline':>10}{'trained':>10}"
              f"{'win/lose':>11}{'McNemar p':>12}")
        for lab, pat in conds:
            rows = []
            for s in (0, 1, 2):
                d = pat.format(s=s) if "{s}" in pat else pat
                t = load(d, K)
                b = load(d, K, "baseline")
                if t and b:
                    rows.append(paired(b, t))
                if "{s}" not in pat:
                    break
            if not rows:
                print(f"{lab:<20}{'(pending)':>6}")
                continue
            n = len(rows)
            mb = sum(r[1] for r in rows) / n
            mt = sum(r[2] for r in rows) / n
            wl = f"{sum(r[3] for r in rows)/n:.0f}/{sum(r[4] for r in rows)/n:.0f}"
            print(f"{lab:<20}{n:>6}{mb*100:>9.1f}%{mt*100:>9.1f}%{wl:>11}"
                  f"{min(r[5] for r in rows):>12.4g}")
        print()


if __name__ == "__main__":
    main()
