"""Did round-0 training itself make the hard negatives less hard?

`loss_margin_hard_negatives` exists to push negatives that outrank a positive
back down. So mining with the trained parameters returns poses the model has
already been trained to score low -- "the second round finds no new hard
negative" is a restatement of the first round having worked, not an independent
mechanism. This measures that directly: the same round-0 pool, scored once with
the published table and once with the trained one.

The pool also accumulates (`absorb` appends; the cap evicted nothing here), so
the hinge stays anchored on the hardest negative ever seen. Mining can only add
signal by finding something harder than that -- which is the failure mode the
training is suppressing. That is why a scheme like this self-limits as it
succeeds.

CPU only.
"""
from __future__ import annotations

import argparse
import glob
import json

import torch

from zdock.atomtypes import iface_ij
from zdock.score import iface_score_matrix


def load(pattern: str) -> dict:
    out = {}
    for f in sorted(glob.glob(pattern)):
        for d in torch.load(f, map_location="cpu", weights_only=True)["pools"]:
            out[d["name"]] = d
    return out


def raw_score(d, alpha, iface, clash, beta):
    sc = d["sc"].double()
    return (alpha * sc[:, 0] - (sc[:, 1:4] * clash).sum(-1)
            + (iface_score_matrix(iface) * d["T"].double()).sum(dim=(-2, -1))
            + beta * d["elec"].double())


def normalize(s: torch.Tensor) -> torch.Tensor:
    return (s - s.mean()) / s.std(unbiased=False).clamp_min(1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--round0", default="data/scaling/pool_cache/*_r0_*_pk2.pt")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--dockq-threshold", type=float, default=0.23, dest="thr")
    ap.add_argument("--alpha0", type=float, default=1.0)
    ap.add_argument("--rho0", type=float, default=3.5)
    ap.add_argument("--beta", type=float, default=3.0)
    args = ap.parse_args()

    pools = load(args.round0)
    fit = json.loads(open(args.split).read())["fit_ids"]
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    beta = torch.tensor(args.beta, dtype=torch.float64)
    a0 = torch.tensor(args.alpha0, dtype=torch.float64)
    k = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
    conds = {
        "baseline e0": (a0, iface_ij(dtype=torch.float64, flat=True),
                        a0 * torch.tensor(args.rho0, dtype=torch.float64).pow(k)),
        "trained": (ck["alpha"].double(), ck["iface"].double(),
                    ck["clash_weights"].double()),
    }

    print(f"the SAME round-0 pool, scored with each parameter set")
    print(f"margin band = negatives above min(positive) - {args.margin}, "
          f"normalized units\n")
    print(f"{'':<20}{'neg in band':>14}{'per complex':>13}"
          f"{'hardest neg':>13}{'min(pos)-max(neg)':>19}")
    for lab, (alpha, iface, clash) in conds.items():
        tot = n = 0
        hardest, gaps = [], []
        for pid in fit:
            d = pools.get(pid)
            if d is None:
                continue
            keep = d["prov"] == 0
            pos = d["dockq"][keep] >= args.thr
            if not bool(pos.any()):
                continue
            s = normalize(raw_score({x: v[keep] for x, v in d.items()
                                     if torch.is_tensor(v)},
                                    alpha, iface, clash, beta))
            thr = float(s[pos].min()) - args.margin
            tot += int((s[~pos] > thr).sum())
            n += 1
            hardest.append(float(s[~pos].max()))
            gaps.append(float(s[pos].min()) - float(s[~pos].max()))
        med = lambda v: sorted(v)[len(v) // 2]
        print(f"{lab:<20}{tot:>14,}{tot/max(1,n):>13.0f}"
              f"{med(hardest):>13.3f}{med(gaps):>19.3f}")
    print(f"\n({n} complexes with a search-derived positive)")
    print("A drop in the band count and in the hardest negative is the margin")
    print("loss doing its job. It is also why a second mining round, which")
    print("searches with those same trained parameters, surfaces poses the")
    print("model already ranks low.")


if __name__ == "__main__":
    main()
