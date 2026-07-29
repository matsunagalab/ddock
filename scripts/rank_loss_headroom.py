"""How far is the top-1 decision from being right, in the units the loss uses?

The deployed metric is Max(Top 1): the single highest-scoring pose must be
near-native. The training objective is not that. It is

    loss_basin              -log sum_{positives} p  -- listwise, all positives
  + lambda * loss_margin    hinge( margin + s_neg - MIN(s_positive) )

Both anchor on the whole positive set. `loss_margin` in particular requires
every negative to sit a margin below the WORST positive, which is strictly
harder than what Max(Top 1) needs (best positive above every negative) and
spends gradient on lifting poses that are near-native but poor.

Whether a different rank statistic could help is an empirical question with a
cheap answer, measured here on the frozen TEST pool:

* `gap_top` = (best negative - best positive) / SD, over complexes that HAVE a
  positive. Positive gap means the top-1 pose is wrong. If those gaps are a
  fraction of an SD, the ranking is losing systems it nearly has, and a
  top-focused objective has something to win. If they are several SD, the
  functional form cannot separate those poses and no reweighting of the same
  score will fix it.
* `spread_pos` = (best positive - worst positive) / SD. How far below the
  top-1 anchor the hinge's `min(positive)` anchor sits.
* `n_neg_above` = how many negatives outrank the best positive.

SD is the complex's own score standard deviation, which is the normalisation
`run_pinder_scaling.normalized_scores` applies before the margin (margin = 1.0
is in these units).

Example
-------
    uv run python scripts/rank_loss_headroom.py \
        --pool data/shards_pinder/test_pool_reachable.pt \
        --ckpt data/scaling/runs_nfixed/N220_seed0/round0_ckpt.pt --prov search
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from compare_conditions import clash_from_ckpt, score_pool  # noqa: E402
from zdock.atomtypes import iface_ij  # noqa: E402


def _q(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, int(p * len(s)))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default="data/shards_pinder/test_pool_reachable.pt")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--base-alpha", type=float, default=1.0, dest="base_alpha")
    ap.add_argument("--base-rho", type=float, default=3.5, dest="base_rho")
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--thr", type=float, default=0.23)
    ap.add_argument("--prov", default="search", choices=("search", "all"))
    args = ap.parse_args()

    blob = torch.load(args.pool, map_location="cpu", weights_only=True)
    beta = torch.tensor(args.beta, dtype=torch.float64)
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        alpha = ck["alpha"].double()
        iface = ck["iface"].double()
        rho = ck.get("rho", torch.tensor(args.base_rho)).double()
        clash = clash_from_ckpt(ck, alpha, rho)
        who = args.ckpt
    else:
        alpha = torch.tensor(args.base_alpha, dtype=torch.float64)
        iface = iface_ij(dtype=torch.float64, flat=True)
        rho = torch.tensor(args.base_rho, dtype=torch.float64)
        clash = clash_from_ckpt({}, alpha, rho)
        who = "published table"

    gaps, spreads, above, top1_ok, n_with_pos = [], [], [], 0, 0
    for d in blob:
        dd = {k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v)
              for k, v in d.items()}
        s = score_pool(dd, alpha, iface, beta, clash)
        dq = dd["dockq"]
        if args.prov == "search":
            m = dd["prov"] == 0 if "prov" in dd else torch.ones_like(dq, dtype=torch.bool)
            if m.dtype != torch.bool:
                m = m.bool()
            s, dq = s[m], dq[m]
        if s.numel() < 2:
            continue
        sd = float(s.std())
        if sd == 0:
            continue
        pos = dq >= args.thr
        if not pos.any() or pos.all():
            continue
        n_with_pos += 1
        best_pos = float(s[pos].max())
        worst_pos = float(s[pos].min())
        best_neg = float(s[~pos].max())
        gaps.append((best_neg - best_pos) / sd)
        spreads.append((best_pos - worst_pos) / sd)
        above.append(int((s[~pos] > best_pos).sum()))
        top1_ok += int(best_pos > best_neg)

    n = n_with_pos
    miss = [g for g in gaps if g > 0]
    print(f"{who}\npool {args.pool}  prov={args.prov}\n")
    print(f"complexes with both a positive and a negative : {n}")
    print(f"  top-1 pose is a positive                    : {top1_ok} "
          f"({100 * top1_ok / n:.1f}%)")
    print(f"  top-1 pose is a negative                    : {len(miss)} "
          f"({100 * len(miss) / n:.1f}%)")
    print("\nhow far the best positive is from the top, for the misses")
    print("  (best negative - best positive), in units of the complex's own SD")
    for p, name in ((0.1, "p10"), (0.25, "p25"), (0.5, "median"),
                    (0.75, "p75"), (0.9, "p90")):
        print(f"    {name:6s} {_q(miss, p):6.2f} SD")
    for t in (0.25, 0.5, 1.0, 2.0):
        k = sum(1 for g in miss if g <= t)
        print(f"    within {t:4.2f} SD of the top: {k:4d} "
              f"({100 * k / n:5.1f}% of all complexes, "
              f"{100 * k / max(1, len(miss)):5.1f}% of the misses)")
    print("\nnegatives outranking the best positive (misses only)")
    ab = sorted(a for a in above if a > 0)
    for p, name in ((0.5, "median"), (0.75, "p75"), (0.9, "p90")):
        print(f"    {name:6s} {_q(ab, p):8.0f}")

    print("\nwhere the hinge anchors: (best positive - worst positive) / SD")
    print("  the margin term requires negatives to sit below the WORST positive,")
    print("  so this is how much lower its anchor is than the top-1 decision")
    for p, name in ((0.25, "p25"), (0.5, "median"), (0.75, "p75"),
                    (0.9, "p90")):
        print(f"    {name:6s} {_q(spreads, p):6.2f} SD")
    print(f"  margin = 1.0 SD for reference")


if __name__ == "__main__":
    main()
