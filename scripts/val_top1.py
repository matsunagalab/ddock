"""Validation top-1 on the fixed pool: the one metric that compares two losses.

`val_loss` cannot be used to choose between the arms of the loss ablation --
each arm minimises a different function, so their validation losses are not on
the same scale. What is common to all of them is the quantity the deployed
system is judged on: is the highest-scoring pose acceptable?

This computes exactly that over a run's own validation complexes, on the frozen
mining pool, restricted to search-derived poses. It is a proxy for the official
Max(Top 1): the pool is fixed, so it measures re-ranking rather than a fresh
search, and its absolute level is not comparable to the PINDER number. It is
comparable ACROSS ARMS on the same complexes, which is what selection needs.

The denominator is every validation complex, including those with no acceptable
pose in the pool -- they cannot be got right, exactly as in the official metric.

Example
-------
    uv run python scripts/val_top1.py \
        --pool-glob 'data/scaling/pool_cache/n1000_r0_*of6.pt' \
        --run data/scaling/runs_loss/toptail/N1000_seed2 \
        --run data/scaling/runs_nfixed/N1000_seed2
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from zdock.atomtypes import iface_ij  # noqa: E402
from zdock.score import iface_score_matrix  # noqa: E402

_CACHE: dict[str, dict] = {}


def pools(pattern: str) -> dict[str, dict]:
    if pattern in _CACHE:
        return _CACHE[pattern]
    seen: dict[str, dict] = {}
    for f in sorted(glob.glob(pattern)):
        for d in torch.load(f, map_location="cpu", weights_only=True)["pools"]:
            if d["name"] in seen:
                continue
            m = d["prov"] == 0
            if int(m.sum()) < 2:
                continue
            seen[d["name"]] = {"sc": d["sc"][m].double(), "T": d["T"][m].double(),
                               "elec": d["elec"][m].double(),
                               "dockq": d["dockq"][m].double()}
    _CACHE[pattern] = seen
    return seen


def params(run: Path | None, rho0: float):
    if run is None:
        alpha = torch.tensor(1.0, dtype=torch.float64)
        iface = iface_ij(dtype=torch.float64, flat=True)
        rho = torch.tensor(rho0, dtype=torch.float64)
        return alpha, iface, alpha * rho.pow(
            torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64))
    ck = torch.load(run / "round0_ckpt.pt", map_location="cpu", weights_only=True)
    alpha = ck["alpha"].double()
    iface = ck["iface"].double()
    rho = ck.get("rho", torch.tensor(rho0)).double()
    clash = ck["clash_weights"].double() if "clash_weights" in ck else \
        alpha * rho.pow(torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64))
    return alpha, iface, clash


def evaluate(ids, pool, alpha, iface, beta, clash, thr):
    M = iface_score_matrix(iface)
    n = ok = reachable = 0
    ranks = []
    for pid in ids:
        e = pool.get(pid)
        if e is None:
            continue
        n += 1
        sc = e["sc"]
        s = (alpha * sc[:, 0] - (sc[:, 1:4] * clash).sum(-1)
             + (M * e["T"]).sum(dim=(-2, -1)) + beta * e["elec"])
        pos = e["dockq"] >= thr
        if not pos.any():
            continue
        reachable += 1
        if bool(pos[int(s.argmax())]):
            ok += 1
        best = int(torch.where(pos, e["dockq"], torch.zeros_like(e["dockq"])).argmax())
        ranks.append(int((s > s[best]).sum()) + 1)
    med = sorted(ranks)[len(ranks) // 2] if ranks else float("nan")
    return {"n": n, "reachable": reachable, "ok": ok,
            "top1": 100.0 * ok / max(1, n),
            "top1_given_reachable": 100.0 * ok / max(1, reachable),
            "median_rank_of_best_positive": med}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool-glob", required=True, dest="pool_glob")
    ap.add_argument("--run", action="append", default=[],
                    help="run directory holding round0_ckpt.pt and split.json")
    ap.add_argument("--baseline", action="store_true",
                    help="also score the published table on the same complexes")
    ap.add_argument("--thr", type=float, default=0.23)
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--rho0", type=float, default=3.5)
    args = ap.parse_args()

    beta = torch.tensor(args.beta, dtype=torch.float64)
    pool = pools(args.pool_glob)
    print(f"pool: {len(pool)} complexes with >=2 search poses\n")
    hdr = f"{'run':44s} {'n':>4s} {'reach':>6s} {'top1':>7s} {'|reach':>7s} {'medrank':>8s}"
    print(hdr)
    print("-" * len(hdr))
    shown_base = False
    for r in args.run:
        run = Path(r)
        ids = json.load(open(run / "split.json"))["val_ids"]
        if args.baseline and not shown_base:
            a, i, c = params(None, args.rho0)
            m = evaluate(ids, pool, a, i, beta, c, args.thr)
            print(f"{'published table':44s} {m['n']:4d} {m['reachable']:6d} "
                  f"{m['top1']:6.2f}% {m['top1_given_reachable']:6.2f}% "
                  f"{m['median_rank_of_best_positive']:8.0f}")
            shown_base = True
        a, i, c = params(run, args.rho0)
        m = evaluate(ids, pool, a, i, beta, c, args.thr)
        label = "/".join(run.parts[-2:])
        print(f"{label:44s} {m['n']:4d} {m['reachable']:6d} {m['top1']:6.2f}% "
              f"{m['top1_given_reachable']:6.2f}% "
              f"{m['median_rank_of_best_positive']:8.0f}")


if __name__ == "__main__":
    main()
