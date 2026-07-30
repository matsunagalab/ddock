"""How many training complexes actually move the parameters?

N=220 gives three seeds that agree on every one of the 250 test systems, and
N=500 and N=1000 do not beat it. The natural explanation is that the objective's
optimum is already pinned down by 220 complexes, so extra data cannot help --
but "pinned down" has a measurable form: most complexes contribute no gradient,
because their loss is saturated.

For each complex this scores its pool with the given parameters, evaluates the
per-complex objective (`loss_basin` plus the selected negative term, exactly as
`run_pinder_scaling.mean_objective` does), and takes the gradient norm with
respect to the IFACE table. It then reports how concentrated that gradient is:
if 80% of it comes from 15% of the complexes, and the same fraction holds at
N=220 and N=1000, then the corpus is growing in the part that is already silent.

Runs on the mining pool caches, so it needs no search and no GPU.

Example
-------
    uv run python scripts/gradient_share.py \
        --pool-glob 'data/scaling/pool_cache/n1000_r0_*of6.pt' \
        --ckpt data/scaling/runs_nfixed/N1000_seed0/round0_ckpt.pt
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from zdock.atomtypes import iface_ij  # noqa: E402
from zdock.score import iface_score_matrix  # noqa: E402
from zdock.train import (loss_basin, loss_margin_hard_negatives,  # noqa: E402
                         loss_top_tail)


def load_pools(pattern: str, prov_search: bool) -> list[dict]:
    """Union of the shard caches, de-duplicated by complex name."""
    out: dict[str, dict] = {}
    for f in sorted(glob.glob(pattern)):
        blob = torch.load(f, map_location="cpu", weights_only=True)
        for d in blob["pools"]:
            out.setdefault(d["name"], d)
    pools = []
    for d in out.values():
        keep = (d["prov"] == 0) if prov_search else torch.ones_like(
            d["prov"], dtype=torch.bool)
        if int(keep.sum()) < 2:
            continue
        pools.append({"name": d["name"], "sc": d["sc"][keep].double(),
                      "T": d["T"][keep].double(), "elec": d["elec"][keep].double(),
                      "dockq": d["dockq"][keep].double()})
    return pools


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool-glob", required=True, dest="pool_glob")
    ap.add_argument("--ckpt", default="", help="omit for the published table")
    ap.add_argument("--alpha0", type=float, default=1.0)
    ap.add_argument("--rho0", type=float, default=3.5)
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--thr", type=float, default=0.23)
    ap.add_argument("--basin-temp", type=float, default=0.5, dest="basin_temp")
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--lambda-margin", type=float, default=0.5, dest="lambda_margin")
    ap.add_argument("--loss-neg", default="minanchor", dest="loss_neg",
                    choices=("minanchor", "none", "toptail"))
    ap.add_argument("--toptail-k", type=int, default=32, dest="toptail_k")
    ap.add_argument("--prov", default="search", choices=("search", "all"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        alpha = ck["alpha"].double()
        iface = ck["iface"].double().clone().requires_grad_(True)
        rho = ck.get("rho", torch.tensor(args.rho0)).double()
        clash = ck["clash_weights"].double() if "clash_weights" in ck else \
            alpha * rho.pow(torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64))
        who = args.ckpt
    else:
        alpha = torch.tensor(args.alpha0, dtype=torch.float64)
        iface = iface_ij(dtype=torch.float64, flat=True).clone().requires_grad_(True)
        rho = torch.tensor(args.rho0, dtype=torch.float64)
        clash = alpha * rho.pow(torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64))
        who = "published table"
    beta = torch.tensor(args.beta, dtype=torch.float64)

    pools = load_pools(args.pool_glob, args.prov == "search")
    if args.limit:
        pools = pools[: args.limit]

    rows = []
    for d in pools:
        sc = d["sc"]
        s_psc = alpha * sc[:, 0] - (sc[:, 1:4] * clash).sum(-1) if sc.ndim == 2 \
            else alpha * sc
        s = s_psc + (iface_score_matrix(iface) * d["T"]).sum(dim=(-2, -1)) \
            + beta * d["elec"]
        # the same per-complex normalisation the training loop applies
        s = (s - s.mean()) / s.std().clamp_min(1e-8)
        loss = loss_basin(s, d["dockq"], temperature=args.basin_temp,
                          positive_threshold=args.thr)
        if args.loss_neg == "minanchor":
            loss = loss + args.lambda_margin * loss_margin_hard_negatives(
                s, d["dockq"], margin=args.margin, positive_threshold=args.thr)
        elif args.loss_neg == "toptail":
            loss = loss + args.lambda_margin * loss_top_tail(
                s, d["dockq"], margin=args.margin, positive_threshold=args.thr,
                k=args.toptail_k)
        g, = torch.autograd.grad(loss, iface, retain_graph=False,
                                 allow_unused=True)
        rows.append({"name": d["name"], "loss": float(loss),
                     "gnorm": 0.0 if g is None else float(g.norm()),
                     "n_pos": int((d["dockq"] >= args.thr).sum()),
                     "n": int(s.numel())})
        iface.grad = None

    n = len(rows)
    gs = sorted((r["gnorm"] for r in rows), reverse=True)
    tot = sum(gs)
    print(f"{who}\npool {args.pool_glob}  prov={args.prov}  "
          f"loss-neg={args.loss_neg}\n")
    print(f"complexes scored           : {n}")
    print(f"  with no positive at all  : {sum(1 for r in rows if r['n_pos'] == 0)}")
    print(f"  gradient norm exactly 0  : {sum(1 for r in rows if r['gnorm'] == 0)}")
    print(f"  gradient norm < 1% of max: {sum(1 for r in rows if r['gnorm'] < 0.01 * gs[0])}")
    print(f"\nshare of the total IFACE gradient norm")
    for frac in (0.05, 0.10, 0.25, 0.50):
        k = max(1, int(frac * n))
        print(f"  top {100 * frac:4.0f}% of complexes ({k:4d}) : "
              f"{100 * sum(gs[:k]) / tot:5.1f}%")
    for frac in (0.5, 0.8, 0.9):
        run, k = 0.0, 0
        while run < frac * tot and k < n:
            run += gs[k]
            k += 1
        print(f"  complexes needed for {100 * frac:2.0f}% of it : {k:4d} "
              f"({100 * k / n:5.1f}%)")
    ls = sorted((r["loss"] for r in rows), reverse=True)
    print(f"\nper-complex loss: median {ls[n // 2]:.4f}  "
          f"p90 {ls[n // 10]:.4f}  min {ls[-1]:.4g}")


if __name__ == "__main__":
    main()
