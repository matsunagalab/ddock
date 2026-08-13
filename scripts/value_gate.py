"""Decide whether a graded value-to-go label is worth fitting at all.

The binary version is dead: relabelling poses by the quality they reach after
refinement leaves the acceptable/not-acceptable split unchanged for 87% of
complexes and changes the top-1 verdict for none of them (report 5.14.40). The
graded version -- regress DockQ itself rather than thresholding it -- is not
refuted by that, but "the argmax moves for 14-25% of complexes" is not evidence
either, since those moves are between poses of nearly identical quality.

So the label side is checked before any fitting. Three necessary conditions,
fixed in advance (report 5.14.41, codex):

    1. refinement gain in the best pose: median >= 0.03, or p75 >= 0.05
    2. complexes whose BEST ATTAINABLE tier rises to medium or high: >= 10
    3. complexes whose CURRENT TOP-1 tier rises to medium or high: >= 5

Pass any one and graded fitting is considered; fail all three and value-to-go
is finished, graded included. The bar asks for several times more headroom in
the labels than the 1 pp it would need to be worth deploying -- a label that
barely moves cannot teach a table to move more.

Reads the pools written by `build_value_pool.py`, which carry both `dockq_now`
(what the start pose is worth) and `dockq` (what it reaches after refinement)
against identical features.

Example
-------
    uv run python scripts/value_gate.py --pools 'data/scaling/value/*.pt'
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from zdock.score import IFACE_PAIR_OFFSET, IFACE_SIGN  # noqa: E402

TIERS = (("acceptable", 0.23), ("medium", 0.49), ("high", 0.80))


def tier(q: float) -> int:
    """0 incorrect, 1 acceptable, 2 medium, 3 high."""
    return int(q >= 0.23) + int(q >= 0.49) + int(q >= 0.80)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pools", required=True, nargs="+",
                    help="glob(s) for build_value_pool.py output")
    ap.add_argument("--ckpt", default="data/scaling/runs_loss/qp/round0_ckpt.pt",
                    help="the table whose top-1 is the reference")
    ap.add_argument("--beta", type=float, default=3.0)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    alpha = ck["alpha"].double()
    clash = ck["clash_weights"].double()
    beta = torch.tensor(args.beta, dtype=torch.float64)
    e = ck["iface"].double().numpy()

    pools = []
    for pat in args.pools:
        for f in sorted(glob.glob(pat)):
            pools += torch.load(f, map_location="cpu", weights_only=True)["pools"]
    if not pools:
        raise SystemExit("no pools matched")

    gain, best_up, top1_up = [], [], []
    for p in pools:
        a = p["dockq_now"].double().numpy()
        b = p["dockq"].double().numpy()
        sc, T, el = p["sc"].double(), p["T"].double(), p["elec"].double()
        c = (alpha * sc[:, 0] - (sc[:, 1:4] * clash).sum(-1) + beta * el
             + IFACE_PAIR_OFFSET * T.sum(dim=(-2, -1))).numpy()
        A = (IFACE_SIGN * T.transpose(-2, -1).reshape(T.shape[0], -1)).numpy()
        k = int((c + A @ e).argmax())          # top-1 START pose, deployed table
        gain.append(float(b.max() - a.max()))
        best_up.append((tier(a.max()), tier(b.max())))
        top1_up.append((tier(a[k]), tier(b[k])))

    g = np.array(gain)
    n = len(pools)
    # "rises to medium or high" = crosses 0.49 or 0.80 that it did not before
    c2 = sum(1 for lo, hi in best_up if hi > lo and hi >= 2)
    c3 = sum(1 for lo, hi in top1_up if hi > lo and hi >= 2)

    print(f"{n} complexes from {len(args.pools)} pattern(s)\n")
    print("condition 1  refinement gain in the best pose")
    print(f"    median {np.median(g):+.4f}  (needs >= 0.03)")
    print(f"    p75    {np.percentile(g, 75):+.4f}  (needs >= 0.05)")
    ok1 = np.median(g) >= 0.03 or np.percentile(g, 75) >= 0.05
    print(f"    -> {'PASS' if ok1 else 'FAIL'}\n")

    print("condition 2  best attainable tier rises to medium or high")
    print(f"    {c2} complexes  (needs >= 10)")
    ok2 = c2 >= 10
    print(f"    -> {'PASS' if ok2 else 'FAIL'}\n")

    print("condition 3  current top-1 tier rises to medium or high")
    print(f"    {c3} complexes  (needs >= 5)")
    ok3 = c3 >= 5
    print(f"    -> {'PASS' if ok3 else 'FAIL'}\n")

    for name, idx in (("best attainable", best_up), ("current top-1", top1_up)):
        m = np.zeros((4, 4), dtype=int)
        for lo, hi in idx:
            m[lo, hi] += 1
        lab = ["incorrect", "acceptable", "medium", "high"]
        print(f"{name}: tier before (row) -> after (column)")
        print("    " + "".join(f"{x:>11s}" for x in lab))
        for i, r in enumerate(m):
            print(f"{lab[i]:>11s}" + "".join(f"{v:11d}" for v in r))
        print(f"    off-diagonal: {int(m.sum() - np.trace(m))}/{n}\n")

    if ok1 or ok2 or ok3:
        print("VERDICT: at least one condition passes -- graded fitting is on the table.")
    else:
        print("VERDICT: all three fail. Value-to-go is finished, graded included.")
        print("The label does not move enough to teach a table to move more.")


if __name__ == "__main__":
    main()
