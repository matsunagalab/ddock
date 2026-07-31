"""Before spending a GPU: does the shaping signal break the complexes that work?

`--loss-shape pairwise` wakes up the complexes with no acceptable pose (461 of
1250 at N=1000). The risk is that what it teaches drags down the complexes that
already rank a near-native pose first: the two groups share 144 parameters.

This walks the parameters along the shaping gradient ALONE -- computed only from
the zero-positive complexes -- and reports, at each step, what happens to the
complexes that DO have a positive. Pass `--split-json` to take the gradient from
FIT complexes and read the effect off VALIDATION ones; without it both groups
come from the whole pool and the result says nothing about generalisation:

* how many flip at top-1 in each direction,
* the decision gap max(s_positive) - max(s_negative), per complex and in
  aggregate,
* whether the near-native pose is still inside the top 1500 (the pool is fixed
  here, so this is retention within the pool, not a re-search),

against what the zero-positive group gains:

* the rank of their best pose, and its DockQ.

An aggregate gradient cosine would not do: a handful of borderline complexes can
break while the average looks fine, so the per-complex flips are reported.

Runs on the mining pool caches with plain Adam on CPU. No search, no GPU.

Example
-------
    uv run python scripts/shaping_precheck.py \
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
from zdock.train import loss_shape_pairwise  # noqa: E402


def load(pattern: str, thr: float) -> tuple[list[dict], list[dict]]:
    """(complexes with an acceptable search pose, complexes without)."""
    seen: dict[str, dict] = {}
    for f in sorted(glob.glob(pattern)):
        for d in torch.load(f, map_location="cpu", weights_only=True)["pools"]:
            seen.setdefault(d["name"], d)
    live, dead = [], []
    for d in seen.values():
        m = d["prov"] == 0
        if int(m.sum()) < 2:
            continue
        e = {"name": d["name"], "sc": d["sc"][m].double(), "T": d["T"][m].double(),
             "elec": d["elec"][m].double(), "dockq": d["dockq"][m].double()}
        (live if float(e["dockq"].max()) >= thr else dead).append(e)
    return live, dead


def scores(e: dict, alpha, iface, beta, clash) -> torch.Tensor:
    sc = e["sc"]
    s_psc = alpha * sc[:, 0] - (sc[:, 1:4] * clash).sum(-1)
    s = s_psc + (iface_score_matrix(iface) * e["T"]).sum(dim=(-2, -1)) \
        + beta * e["elec"]
    return (s - s.mean()) / s.std().clamp_min(1e-8)


def report(live: list[dict], dead: list[dict], alpha, iface, beta, clash,
           thr: float, base=None) -> dict:
    gaps, top1 = [], []
    for e in live:
        s = scores(e, alpha, iface, beta, clash)
        pos = e["dockq"] >= thr
        gaps.append(float(s[pos].max() - s[~pos].max()))
        top1.append(bool(pos[int(s.argmax())]))
    ranks, bestdq = [], []
    for e in dead:
        s = scores(e, alpha, iface, beta, clash)
        best = int(e["dockq"].argmax())
        ranks.append(int((s > s[best]).sum()) + 1)
        bestdq.append(float(e["dockq"][best]))
    out = {"n_top1": sum(top1), "gap_mean": sum(gaps) / len(gaps),
           "top1": top1, "gaps": gaps, "dead_rank": ranks}
    if base is not None:
        out["flip_lost"] = sum(1 for a, b in zip(base["top1"], top1) if a and not b)
        out["flip_won"] = sum(1 for a, b in zip(base["top1"], top1) if b and not a)
        out["dead_rank_better"] = sum(1 for a, b in zip(base["dead_rank"], ranks)
                                      if b < a)
        out["dead_rank_worse"] = sum(1 for a, b in zip(base["dead_rank"], ranks)
                                     if b > a)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool-glob", required=True, dest="pool_glob")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--thr", type=float, default=0.23)
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--rho0", type=float, default=3.5)
    ap.add_argument("--iface-lr", type=float, default=5e-4, dest="iface_lr")
    ap.add_argument("--steps", default="1,10,100")
    ap.add_argument("--lambda-shape", type=float, default=1.0, dest="lambda_shape")
    ap.add_argument("--shape-anchors", type=int, default=16, dest="shape_anchors")
    ap.add_argument("--shape-k", type=int, default=32, dest="shape_k")
    ap.add_argument("--shape-delta-q", type=float, default=0.02, dest="shape_delta_q")
    ap.add_argument("--shape-tau", type=float, default=1.0, dest="shape_tau")
    ap.add_argument("--batch", type=int, default=16,
                    help="zero-positive complexes per virtual step")
    # The pool cache holds fit AND validation complexes. Walking the parameters
    # using fit complexes and then reading the effect off fit complexes measures
    # nothing about generalisation, so the two are reported separately.
    ap.add_argument("--split-json", default="", dest="split_json",
                    help="run<N>/split.json, to separate fit from validation")
    args = ap.parse_args()

    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        alpha = ck["alpha"].double()
        iface = ck["iface"].double().clone()
        rho = ck.get("rho", torch.tensor(args.rho0)).double()
        clash = ck["clash_weights"].double() if "clash_weights" in ck else \
            alpha * rho.pow(torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64))
    else:
        alpha = torch.tensor(1.0, dtype=torch.float64)
        iface = iface_ij(dtype=torch.float64, flat=True).clone()
        rho = torch.tensor(args.rho0, dtype=torch.float64)
        clash = alpha * rho.pow(torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64))
    beta = torch.tensor(args.beta, dtype=torch.float64)

    live, dead = load(args.pool_glob, args.thr)
    if args.split_json:
        import json
        sp = json.load(open(args.split_json))
        fit, val = set(sp["fit_ids"]), set(sp["val_ids"])
        live_val = [e for e in live if e["name"] in val]
        dead = [e for e in dead if e["name"] in fit]     # walk on fit only
        print(f"split {args.split_json}: shaping gradient from FIT complexes "
              f"only, effect read off VALIDATION complexes")
        live = live_val
    print(f"pool {args.pool_glob}")
    print(f"  with an acceptable search pose : {len(live)}")
    print(f"  without (the shaping group)    : {len(dead)}\n")

    base = report(live, dead, alpha, iface, beta, clash, args.thr)
    print(f"start: top-1 correct {base['n_top1']}/{len(live)} "
          f"({100 * base['n_top1'] / len(live):.1f}%), "
          f"mean decision gap {base['gap_mean']:+.3f} SD")

    iface = iface.requires_grad_(True)
    opt = torch.optim.Adam([iface], lr=args.iface_lr)
    want = [int(x) for x in args.steps.split(",")]
    gen = torch.Generator().manual_seed(0)
    for step in range(1, max(want) + 1):
        idx = torch.randint(0, len(dead), (min(args.batch, len(dead)),),
                            generator=gen)
        loss = torch.zeros((), dtype=torch.float64)
        for i in idx.tolist():
            s = scores(dead[i], alpha, iface, beta, clash)
            loss = loss + args.lambda_shape * loss_shape_pairwise(
                s, dead[i]["dockq"], positive_threshold=args.thr,
                n_anchor=args.shape_anchors, k_neg=args.shape_k,
                delta_q=args.shape_delta_q, tau=args.shape_tau)
        loss = loss / len(idx)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in want:
            with torch.no_grad():
                r = report(live, dead, alpha, iface.detach(), beta, clash,
                           args.thr, base)
            print(f"\nafter {step:4d} shaping-only step(s)  "
                  f"(||d iface|| = {float((iface.detach() - ck['iface'].double() if args.ckpt else iface.detach()).norm()):.4f})")
            print(f"  complexes WITH a positive: top-1 correct "
                  f"{r['n_top1']}/{len(live)} "
                  f"({100 * r['n_top1'] / len(live):.1f}%)   "
                  f"flips: {r['flip_won']} won, {r['flip_lost']} LOST")
            print(f"     mean decision gap {r['gap_mean']:+.3f} SD "
                  f"(was {base['gap_mean']:+.3f})")
            print(f"  the shaping group: best pose ranks better in "
                  f"{r['dead_rank_better']}/{len(dead)}, worse in "
                  f"{r['dead_rank_worse']}")
            med = sorted(r["dead_rank"])[len(dead) // 2]
            med0 = sorted(base["dead_rank"])[len(dead) // 2]
            print(f"     median rank of their best pose {med0} -> {med}")


if __name__ == "__main__":
    main()
