"""Every comparison, every metric, side by side -- no composite score.

Why the primary endpoint moved from success@1 to success@10
-----------------------------------------------------------
success@1 uses one binary observation per complex, so at n = 236 the exact
McNemar test cannot reject unless the discordant count is large and
one-directional. Several contrasts in this report sit at "4 wins 0 losses,
p = 0.125" -- 0.125 being the smallest p that test can return, so significance
was unreachable there whatever the true effect. success@1 is also the metric
most exposed to reshuffling among the top few poses, which is where the
run-to-run noise lives.

success@10 keeps headroom (87.3% baseline, against 96.2% at K = 100 which is
saturated) and is insensitive to that reshuffling. It is the pre-registered
primary endpoint from 2026-07-28 onward.

Nothing is combined into a single score. A composite needs weights, and picking
weights after seeing the results is how a null becomes a win. The metrics
disagree here in a way worth seeing directly: going from N = 220 to N = 500
improves AUC and success@100 while success@1 falls.
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os

KS = (1, 5, 10, 50, 100)
CONT = ("auc", "best_dockq@1", "best_dockq@10", "first_hit_pct")
LOWER_BETTER = {"first_hit_pct"}


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) * 0.5 ** n)


def wilcoxon(diff):
    d = [x for x in diff if x != 0.0]
    n = len(d)
    if n == 0:
        return 1.0
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        r = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[order[t]] = r
        i = j + 1
    wp = sum(ranks[i] for i in range(n) if d[i] > 0)
    wm = sum(ranks[i] for i in range(n) if d[i] < 0)
    w = min(wp, wm)
    mu = n * (n + 1) / 4.0
    sig = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (w - mu + 0.5) / sig
    return min(1.0, math.erfc(abs(z) / math.sqrt(2.0)))


def read(d: str):
    f = glob.glob(os.path.join(d, "*_search_per_complex.csv"))
    if not f:
        return None
    per = {}
    for r in csv.DictReader(open(f[0])):
        per.setdefault(r["name"], {})[r["condition"]] = r
    return {n: c for n, c in per.items() if "baseline" in c and "trained" in c}


def summarise(dirs: list[str]):
    """Mean over seeds of each metric, plus the paired test against baseline."""
    accK = {k: [] for k in KS}
    accC = {m: [] for m in CONT}
    baseK = {k: [] for k in KS}
    baseC = {m: [] for m in CONT}
    wl = {k: [] for k in KS}
    pK = {k: [] for k in KS}
    pC = {m: [] for m in CONT}
    for d in dirs:
        per = read(d)
        if not per:
            continue
        names = sorted(per)
        for k in KS:
            b = [int(float(per[n]["baseline"][f"succ_dockq@{k}"])) for n in names]
            t = [int(float(per[n]["trained"][f"succ_dockq@{k}"])) for n in names]
            baseK[k].append(sum(b) / len(b))
            accK[k].append(sum(t) / len(t))
            w = sum(1 for x, y in zip(b, t) if not x and y)
            l = sum(1 for x, y in zip(b, t) if x and not y)
            wl[k].append((w, l))
            pK[k].append(mcnemar_exact(w, l))
        for m in CONT:
            b = [float(per[n]["baseline"][m]) for n in names]
            t = [float(per[n]["trained"][m]) for n in names]
            baseC[m].append(sum(b) / len(b))
            accC[m].append(sum(t) / len(t))
            pC[m].append(wilcoxon([y - x for x, y in zip(b, t)]))
    if not accK[1]:
        return None
    mean = lambda v: sum(v) / len(v)
    return dict(
        n_seeds=len(accK[1]),
        K={k: mean(accK[k]) for k in KS}, baseK={k: mean(baseK[k]) for k in KS},
        C={m: mean(accC[m]) for m in CONT}, baseC={m: mean(baseC[m]) for m in CONT},
        wl={k: (mean([x[0] for x in wl[k]]), mean([x[1] for x in wl[k]])) for k in KS},
        pK={k: max(pK[k]) for k in KS}, pC={m: max(pC[m]) for m in CONT})


def block(title: str, rows: list[tuple[str, list[str]]]) -> None:
    print(f"\n### {title}")
    head = (f"{'condition':<20}" + "".join(f"{'@'+str(k):>7}" for k in KS)
            + f"{'AUC':>8}{'bDQ@1':>8}{'bDQ@10':>8}{'fhit%':>8}")
    print(head)
    print("-" * len(head))
    printed_base = False
    for lab, dirs in rows:
        s = summarise(dirs)
        if s is None:
            print(f"{lab:<20}{'(pending)':>7}")
            continue
        if not printed_base:
            print(f"{'baseline':<20}"
                  + "".join(f"{s['baseK'][k]*100:>6.1f}%" for k in KS)
                  + f"{s['baseC']['auc']:>8.4f}{s['baseC']['best_dockq@1']:>8.4f}"
                  + f"{s['baseC']['best_dockq@10']:>8.4f}"
                  + f"{s['baseC']['first_hit_pct']*100:>7.2f}%")
            printed_base = True
        print(f"{lab:<20}" + "".join(f"{s['K'][k]*100:>6.1f}%" for k in KS)
              + f"{s['C']['auc']:>8.4f}{s['C']['best_dockq@1']:>8.4f}"
              + f"{s['C']['best_dockq@10']:>8.4f}"
              + f"{s['C']['first_hit_pct']*100:>7.2f}%")
        wls = ["%.0f/%.0f" % s["wl"][k] for k in KS]
        print(f"{'  win/lose vs base':<20}" + "".join(f"{x:>7}" for x in wls)
              + f"{'-':>8}{'-':>8}{'-':>8}{'-':>8}")
        print(f"{'  p (worst seed)':<20}"
              + "".join(f"{s['pK'][k]:>7.1g}" for k in KS)
              + "".join(f"{s['pC'][m]:>8.1g}" for m in CONT))
    print("\nprimary endpoint: success@10.  p for @K is exact McNemar, for the")
    print("continuous columns paired Wilcoxon; with several seeds the WORST p is")
    print("shown and the counts are seed means. Seeds share one TEST set, so they")
    print("are not independent tests.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()
    seeds = lambda p: [p.format(s=s) for s in (0, 1, 2)]
    block("experiment 1 -- dimension x margin, fixed pool", [
        ("full 144, margin .5", seeds("data/scaling/compare_fact_full_m5_seed{s}")),
        ("full 144, margin 0", seeds("data/scaling/compare_fact_full_m0_seed{s}")),
        ("additive 23, m .5", seeds("data/scaling/compare_fact_add_m5_seed{s}")),
        ("additive 23, m 0", seeds("data/scaling/compare_fact_add_m0_seed{s}")),
        ("symmetric 12, m .5", seeds("data/scaling/compare_fact_sym_m5_seed{s}")),
        ("symmetric 12, m 0", seeds("data/scaling/compare_fact_sym_m0_seed{s}")),
    ])
    block("experiment 4 -- mining round 1, seed 0", [
        ("round 0", ["data/scaling/compare_r1_seed0_round0"]),
        ("round 1 mine", ["data/scaling/compare_r1_seed0_hardneg"]),
        ("round 1 continue", ["data/scaling/compare_r1_seed0_none"]),
    ])
    block("N scaling -- matched budget, early stopping off", [
        ("N = 220", seeds("data/scaling/compare_nfixed_N220_seed{s}")),
        ("N = 500", seeds("data/scaling/compare_nfixed_N500_seed{s}")),
        ("N = 1000", seeds("data/scaling/compare_nfixed_N1000_seed{s}")),
    ])


if __name__ == "__main__":
    main()
