"""mine vs continue: the primary contrast of the round-1 mining experiment.

Both arms branch from ONE round-0 state (parameters, Adam, minibatch stream) and
spend the same step budget under the same stopping rule. The only difference is
whether the round-1 search added new negatives. So a difference here is the
increment from hard-negative mining; no difference means round 0 -> round 1 is
explained by further optimisation alone.
"""
import csv, glob, math, sys
sys.path.insert(0, "scripts")
from contrast import mcnemar_exact, wilcoxon, boot_ci_success


def load(d, cond="trained"):
    f = glob.glob(f"{d}/*_search_per_complex.csv")[0]
    out = {}
    for r in csv.DictReader(open(f)):
        if r["condition"] != cond:
            continue
        out[r["name"]] = (int(float(r["succ_dockq@1"])), float(r["auc"]),
                          float(r["best_dockq@1"]), float(r["first_hit_pct"]))
    return out


D = "data/scaling/compare_r1_seed0_"
arms = {k: load(D + k) for k in ("round0", "hardneg", "none")}
base = load(D + "round0", "baseline")
names = sorted(set.intersection(*(set(a) for a in arms.values())))
print(f"seed 0, {len(names)} complexes paired, search-derived poses only\n")

print(f"{'condition':<28}{'success@1':>10}{'AUC':>10}{'bestDQ@1':>10}{'firstHit%':>11}")
for lab, a in (("baseline (published)", base), ("round 0 (trained)", arms["round0"]),
               ("round 1  mine", arms["hardneg"]),
               ("round 1  continue", arms["none"])):
    s1 = sum(a[n][0] for n in names) / len(names)
    au = sum(a[n][1] for n in names) / len(names)
    dq = sum(a[n][2] for n in names) / len(names)
    fh = sum(a[n][3] for n in names) / len(names)
    print(f"{lab:<28}{s1*100:>9.2f}%{au:>10.4f}{dq:>10.4f}{fh*100:>10.3f}%")


def contrast(lo, hi, label):
    b2t = sum(1 for n in names if not lo[n][0] and hi[n][0])
    t2b = sum(1 for n in names if lo[n][0] and not hi[n][0])
    s_lo = sum(lo[n][0] for n in names) / len(names)
    s_hi = sum(hi[n][0] for n in names) / len(names)
    print(f"\n{label}")
    print(f"  success@1 : {s_lo*100:6.2f}% -> {s_hi*100:6.2f}%  "
          f"{b2t}勝{t2b}敗  exact McNemar p = {mcnemar_exact(b2t, t2b):.4g}")
    for i, (nm, better) in enumerate((("AUC", "higher"), ("bestDQ@1", "higher"),
                                      ("firstHit", "lower")), start=1):
        d = [hi[n][i] - lo[n][i] for n in names]
        m_lo = sum(lo[n][i] for n in names) / len(names)
        m_hi = sum(hi[n][i] for n in names) / len(names)
        _, p = wilcoxon(d)
        print(f"  {nm:<10}: {m_lo:.4f} -> {m_hi:.4f}  delta {m_hi-m_lo:+.4f} "
              f"({better} better)  Wilcoxon p = {p:.4g}")


print("\n" + "=" * 74)
print("PRIMARY CONTRAST: continue -> mine  (the increment from mining alone)")
print("=" * 74)
contrast(arms["none"], arms["hardneg"], "round 1 continue  ->  round 1 mine")
p, lo, hi = boot_ci_success(D + "none", D + "hardneg")
print(f"  paired complex bootstrap 95% CI on success@1: {p:+.2f} pp [{lo:+.2f}, {hi:+.2f}]")

print("\n" + "=" * 74)
print("SECONDARY: what each arm did relative to where they branched from")
print("=" * 74)
contrast(arms["round0"], arms["hardneg"], "round 0  ->  round 1 mine")
contrast(arms["round0"], arms["none"], "round 0  ->  round 1 continue")
