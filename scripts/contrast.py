"""Direct paired contrasts between two trained conditions on the frozen TEST pool.

The comparison that matters is not "does each beat the baseline" -- all six do --
but "does the higher-dimensional family beat the lower-dimensional one".
Both models are read from their per-complex tables, so the same 236 complexes are
paired.
"""
import csv, glob, math, sys
from collections import defaultdict

THR = 0.23


def load(d):
    f = glob.glob(f"{d}/*_search_per_complex.csv")[0]
    out = {}
    for r in csv.DictReader(open(f)):
        if r["condition"] != "trained":
            continue
        out[r["name"]] = (int(float(r["succ_dockq@1"])), float(r["auc"]),
                          float(r["best_dockq@1"]), float(r["first_hit_pct"]))
    return out


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) * 0.5 ** n)


def wilcoxon(diff):
    d = [x for x in diff if x != 0.0]
    n = len(d)
    if n == 0:
        return float("nan"), 1.0
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    wp = sum(ranks[i] for i in range(n) if d[i] > 0)
    wm = sum(ranks[i] for i in range(n) if d[i] < 0)
    w = min(wp, wm)
    mu = n * (n + 1) / 4.0
    sig = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (w - mu + 0.5) / sig
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return z, min(1.0, p)


def contrast(lo_dir, hi_dir, label):
    lo, hi = load(lo_dir), load(hi_dir)
    names = sorted(set(lo) & set(hi))
    b2t = sum(1 for n in names if not lo[n][0] and hi[n][0])
    t2b = sum(1 for n in names if lo[n][0] and not hi[n][0])
    s_lo = sum(lo[n][0] for n in names) / len(names)
    s_hi = sum(hi[n][0] for n in names) / len(names)
    out = [f"{label}  (n={len(names)})",
           f"  success@1 : {s_lo*100:5.2f}% -> {s_hi*100:5.2f}%  "
           f"{b2t}勝{t2b}敗  McNemar p={mcnemar_exact(b2t, t2b):.4g}"]
    for i, (nm, better) in enumerate(((("auc"), "higher"), ("best_dockq@1", "higher"),
                                      ("first_hit_pct", "lower")), start=1):
        dl = [hi[n][i] - lo[n][i] for n in names]
        m_lo = sum(lo[n][i] for n in names) / len(names)
        m_hi = sum(hi[n][i] for n in names) / len(names)
        _, p = wilcoxon(dl)
        out.append(f"  {nm:10s}: {m_lo:.4f} -> {m_hi:.4f}  "
                   f"delta={m_hi-m_lo:+.4f} ({better} better)  Wilcoxon p={p:.4g}")
    return "\n".join(out), mcnemar_exact(b2t, t2b)


D = lambda m, g, s: f"data/scaling/compare_fact_{m}_{g}_seed{s}"

print("=" * 78)
print("PRIMARY CONTRAST at the validation-selected margin (0.5): additive -> full")
print("=" * 78)
ps = []
for s in (0, 1, 2):
    t, p = contrast(D("add", "m5", s), D("full", "m5", s), f"seed {s}: additive(23) -> full(144)")
    print(t); ps.append(p)
print(f"\n  three seeds, McNemar p = {', '.join(f'{p:.4g}' for p in ps)}"
      f"   (same TEST complexes -> NOT three independent tests)")

for lab, lo, hi in (("symmetric(12) -> additive(23)", "sym", "add"),
                    ("symmetric(12) -> full(144)", "sym", "full")):
    print("\n" + "=" * 78)
    print(f"SECONDARY: {lab}  (margin 0.5)")
    print("=" * 78)
    for s in (0, 1, 2):
        print(contrast(D(lo, "m5", s), D(hi, "m5", s), f"seed {s}")[0])

print("\n" + "=" * 78)
print("MARGIN CONTRAST within each dimension: margin 0.0 -> margin 0.5")
print("=" * 78)
for m in ("full", "add", "sym"):
    raw = []
    for s in (0, 1, 2):
        t, p = contrast(D(m, "m0", s), D(m, "m5", s), f"{m:5s} seed {s}")
        print(t); raw.append(p)
    # Holm across the three dimensions is applied below on seed 0 only
    print()
print("Holm correction over the three dimensions (primary endpoint success@1, seed 0):")
raw = []
for m in ("full", "add", "sym"):
    _, p = contrast(D(m, "m0", 0), D(m, "m5", 0), m)
    raw.append((m, p))
for rank, (m, p) in enumerate(sorted(raw, key=lambda x: x[1])):
    print(f"  {m:5s} raw p={p:.4g}  Holm-adjusted p={min(1.0, p * (3 - rank)):.4g}")


def boot_ci_success(lo_dir, hi_dir, n_boot=20000, seed=0):
    """Paired complex bootstrap CI for the success@1 difference (pp).

    A p-value says the sign is unlikely to be chance; it does not bound the
    size. Non-significance is not equivalence either, so the interval is what
    a non-inferiority claim has to be read off.
    """
    import random
    lo, hi = load(lo_dir), load(hi_dir)
    names = sorted(set(lo) & set(hi))
    d = [hi[n][0] - lo[n][0] for n in names]
    n = len(d)
    rng = random.Random(seed)
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(n_boot))
    point = sum(d) / n
    return (point * 100, means[int(0.025 * n_boot)] * 100,
            means[int(0.975 * n_boot)] * 100)


if __name__ == "__main__" and "--ci" in sys.argv:
    print("\n" + "=" * 78)
    print("paired complex bootstrap CI for the success@1 difference (pp), seed 0")
    print("=" * 78)
    for lab, lo, hi in (("additive(23) -> full(144)", "add", "full"),
                        ("symmetric(12) -> additive(23)", "sym", "add"),
                        ("symmetric(12) -> full(144)", "sym", "full")):
        p, l, h = boot_ci_success(D(lo, "m5", 0), D(hi, "m5", 0))
        print(f"  {lab:<32} {p:+6.2f} pp   95% CI [{l:+.2f}, {h:+.2f}]")
    for m in ("full", "add", "sym"):
        p, l, h = boot_ci_success(D(m, "m0", 0), D(m, "m5", 0))
        print(f"  {m:5s} margin 0.0 -> 0.5              {p:+6.2f} pp   "
              f"95% CI [{l:+.2f}, {h:+.2f}]")
