"""What do rankers with NO trained parameters achieve on the same TEST pool?

A pool whose positives bury more surface than its negatives can be separated by
counting contacts, and then a "trained" model that reaches AUC 0.89 has
demonstrated nothing about interface chemistry -- it has rediscovered burial.
`compare_conditions.py` already prints these controls for AUC; this adds
success@K, which is the endpoint the report actually decides on, and puts the
published-parameter and trained scores on the same table.

Rankers compared, none of which has a fitted parameter:
  contact count   sum_ij T_ij            -- how much interface the pose makes
  PSC pair count  c_pair                 -- favourable shape-complementarity cells
  ELEC alone      the electrostatic term
  published ZDOCK alpha=1, rho=3.5, published IFACE table   (the report baseline)

CPU only.
"""
from __future__ import annotations

import argparse
import glob
import math

import torch

from zdock.atomtypes import iface_ij
from zdock.score import iface_score_matrix, SC_RHO

KS = (1, 5, 10, 50, 100)


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) * 0.5 ** n)


def mann_whitney_auc(s: torch.Tensor, pos: torch.Tensor) -> float:
    npos, nneg = int(pos.sum()), int((~pos).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = torch.argsort(s)
    ranks = torch.empty_like(s, dtype=torch.float64)
    ranks[order] = torch.arange(1, s.numel() + 1, dtype=torch.float64)
    uniq, inv, cnt = torch.unique(s, return_inverse=True, return_counts=True)
    if int((cnt > 1).sum()):
        rs = torch.zeros(uniq.numel(), dtype=torch.float64)
        rs.index_add_(0, inv, ranks)
        ranks = (rs / cnt.to(torch.float64))[inv]
    u = float(ranks[pos].sum()) - npos * (npos + 1) / 2.0
    return u / (npos * nneg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default="data/shards_pinder/test_pool_reachable.pt")
    ap.add_argument("--ckpt", default="data/scaling/runs_nfixed/N220_seed0/round0_ckpt.pt",
                    help="a trained checkpoint, for the last row")
    ap.add_argument("--dockq-threshold", type=float, default=0.23, dest="thr")
    ap.add_argument("--beta", type=float, default=3.0)
    args = ap.parse_args()

    blob = torch.load(args.pool, map_location="cpu", weights_only=True)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    e0 = iface_ij(dtype=torch.float64, flat=True)
    k = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
    cl0 = torch.tensor(1.0, dtype=torch.float64) * torch.tensor(SC_RHO, dtype=torch.float64).pow(k)

    def published(d):
        sc = d["sc"]
        return (sc[:, 0] - (sc[:, 1:4] * cl0).sum(-1)
                + (iface_score_matrix(e0) * d["T"]).sum(dim=(-2, -1))
                + args.beta * d["elec"])

    def trained(d):
        sc = d["sc"]
        return (ck["alpha"].double() * sc[:, 0]
                - (sc[:, 1:4] * ck["clash_weights"].double()).sum(-1)
                + (iface_score_matrix(ck["iface"].double()) * d["T"]).sum(dim=(-2, -1))
                + args.beta * d["elec"])

    rankers = {
        "contact count": lambda d: d["T"].sum(dim=(-2, -1)),
        "PSC pair count": lambda d: d["sc"][:, 0],
        "ELEC alone": lambda d: d["elec"],
        "published ZDOCK": published,
        "trained (N=220)": trained,
    }

    hits = {r: {k: [] for k in KS} for r in rankers}
    aucs = {r: [] for r in rankers}
    per_complex = {r: {k: [] for k in KS} for r in rankers}
    n = 0
    for d in blob:
        dd = {x: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v)
              for x, v in d.items() if torch.is_tensor(v)}
        keep = dd["prov"] == 0
        dd = {x: v[keep] for x, v in dd.items()}
        pos = dd["dockq"] >= args.thr
        if not bool(pos.any()) or not bool((~pos).any()):
            continue
        n += 1
        for name, fn in rankers.items():
            s = fn(dd)
            order = torch.argsort(s, descending=True)
            dq = dd["dockq"][order]
            for kk in KS:
                h = int((dq[:kk] >= args.thr).any())
                hits[name][kk].append(h)
                per_complex[name][kk].append(h)
            aucs[name].append(mann_whitney_auc(s, pos))

    print(f"parameter-free rankers on {n} TEST complexes, search-derived poses\n")
    head = f"{'ranker':<20}" + "".join(f"{'@'+str(kk):>8}" for kk in KS) + f"{'AUC':>9}"
    print(head); print("-" * len(head))
    for name in rankers:
        print(f"{name:<20}"
              + "".join(f"{100*sum(hits[name][kk])/n:>7.1f}%" for kk in KS)
              + f"{sum(aucs[name])/n:>9.4f}")

    print(f"\npaired against the parameter-free contact counter (exact McNemar)\n")
    print(f"{'ranker':<20}" + "".join(f"{'@'+str(kk):>16}" for kk in KS))
    ref = per_complex["contact count"]
    for name in ("published ZDOCK", "trained (N=220)"):
        cells = []
        for kk in KS:
            a, b = ref[kk], per_complex[name][kk]
            w = sum(1 for x, y in zip(a, b) if not x and y)
            l = sum(1 for x, y in zip(a, b) if x and not y)
            cells.append(f"{w}/{l} p={mcnemar_exact(w, l):.2g}")
        print(f"{name:<20}" + "".join(f"{c:>16}" for c in cells))
    print("\nA trained model that cannot beat the contact counter has learned burial,")
    print("not chemistry. The counter needs no parameters at all.")


if __name__ == "__main__":
    main()
