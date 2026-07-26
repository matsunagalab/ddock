"""Aggregate the 3x2x3 dimension x margin x seed factorial on the frozen TEST pool.

success@K and its McNemar test are recomputed from the per-complex table, which
compare_conditions.py writes but does not put in the JSON.
"""
import csv, json, glob, math, os, sys
from collections import defaultdict

MODES, MARGINS, SEEDS = ("full", "add", "sym"), ("m5", "m0"), (0, 1, 2)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * 0.5 ** n
    return min(1.0, 2.0 * tail)


def read_cell(d):
    f = glob.glob(os.path.join(d, "*_search.json"))
    if not f:
        return None
    j = json.load(open(f[0]))
    csvf = f[0].replace(".json", "_per_complex.csv")
    per = defaultdict(dict)
    with open(csvf) as fh:
        for r in csv.DictReader(fh):
            per[r["name"]][r["condition"]] = r
    b2t = t2b = nb = nt = n = 0
    for name, cond in per.items():
        if "baseline" not in cond or "trained" not in cond:
            continue
        n += 1
        sb, st = int(float(cond["baseline"]["succ_dockq@1"])), int(float(cond["trained"]["succ_dockq@1"]))
        nb += sb; nt += st
        b2t += (not sb) and st
        t2b += sb and (not st)
    mt = j["metrics"]
    return dict(
        n=n, s1_base=nb / n, s1=nt / n, win=b2t, lose=t2b,
        s1_p=mcnemar_exact(b2t, t2b),
        auc=mt["auc"]["trained"], auc_base=mt["auc"]["baseline"], auc_p=mt["auc"]["wilcoxon_p"],
        dq1=mt["best_dockq@1"]["trained"], dq1_base=mt["best_dockq@1"]["baseline"],
        dq1_p=mt["best_dockq@1"]["wilcoxon_p"],
        fh=mt["first_hit_pct"]["trained"], fh_base=mt["first_hit_pct"]["baseline"],
    )


rows = []
for m in MODES:
    for g in MARGINS:
        for s in SEEDS:
            r = read_cell(f"data/scaling/compare_fact_{m}_{g}_seed{s}")
            if r:
                rows.append(dict(mode=m, margin=g, seed=s, **r))
if not rows:
    sys.exit("no cells evaluated yet")

mean = lambda v: sum(v) / len(v)
def sd(v):
    if len(v) < 2: return 0.0
    mu = mean(v); return (sum((x - mu) ** 2 for x in v) / (len(v) - 1)) ** 0.5

r0 = rows[0]
print(f"{len(rows)}/18 cells   TEST pool, prov=search, n_paired={r0['n']}")
print(f"baseline (published params): success@1 {r0['s1_base']*100:.1f}%  "
      f"AUC {r0['auc_base']:.4f}  bestDQ@1 {r0['dq1_base']:.4f}  "
      f"firstHit {r0['fh_base']*100:.3f}%\n")
hdr = f"{'mode':5s} {'dof':>4s} {'mgn':4s} {'k':>2s}  {'success@1 %':>15s} {'AUC':>17s} {'bestDQ@1':>17s} {'firstHit %':>15s}"
print(hdr); print("-" * len(hdr))
DOF = {"full": 144, "add": 23, "sym": 12}
for m in MODES:
    for g in MARGINS:
        c = [r for r in rows if r["mode"] == m and r["margin"] == g]
        if not c: continue
        def col(k, sc, f):
            v = [r[k] * sc for r in c]; return f.format(mean(v), sd(v))
        print(f"{m:5s} {DOF[m]:4d} {g:4s} {len(c):2d}  "
              f"{col('s1', 100, '{:6.2f} +- {:4.2f}'):>15s} "
              f"{col('auc', 1, '{:7.5f} +- {:7.5f}'):>17s} "
              f"{col('dq1', 1, '{:7.5f} +- {:7.5f}'):>17s} "
              f"{col('fh', 100, '{:6.3f} +- {:5.3f}'):>15s}")

print("\nper-cell vs the SAME baseline (the 3 seeds share one test pool -> not 3 independent tests):")
for r in sorted(rows, key=lambda r: (MODES.index(r["mode"]), r["margin"], r["seed"])):
    print(f"  {r['mode']:5s} {r['margin']} seed{r['seed']}: "
          f"{r['s1_base']*100:5.1f}% -> {r['s1']*100:5.1f}%  {r['win']:>2}勝{r['lose']:>2}敗 "
          f"McNemar p={r['s1_p']:.3g} | AUC {r['auc']:.4f} p={r['auc_p']:.2g} | "
          f"bestDQ@1 {r['dq1']:.4f} p={r['dq1_p']:.2g}")
json.dump(rows, open("data/scaling/factorial_summary.json", "w"), indent=1)
print(f"\nwrote data/scaling/factorial_summary.json ({len(rows)} cells)")
