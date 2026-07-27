"""Did the round-1 negatives actually drive the loss, or merely enlarge the pool?

A 69% duplicate rate does not by itself prove "the search offers nothing new" --
102,433 genuinely new negatives entered, growing the fit pool by 27%. What
decides whether they can move the parameters is not how many there are but
whether the loss can see them.

The loss is top-heavy in two ways. `loss_margin_hard_negatives` penalises only
negatives scoring above `min(positive) - margin`; below that the gradient is
exactly zero. `loss_basin` is a softmax at temperature 0.5, so a pose far below
the top carries a vanishing weight.

Both are applied to `normalized_scores`, NOT to raw scores: the pool is centred
and divided by its own standard deviation (raw score std is 5e2-2e3 while the
margin is 1.0), and that standardisation is recomputed over whichever pool the
loss is given. So the round-1 pool renormalises -- the same pose sits at a
different place once 27% more poses are added. Measuring the margin band on raw
scores compares nothing the optimiser ever saw; this reproduces the training
path instead, `loss_view` subset and all.

Run after Phase B/C, against the round-0 pool cache and the round-1 raw shards.
CPU only.
"""
from __future__ import annotations

import argparse
import glob
import json

import torch

from zdock.score import iface_score_matrix


def load(pattern: str) -> dict:
    out = {}
    for f in sorted(glob.glob(pattern)):
        for d in torch.load(f, map_location="cpu", weights_only=True)["pools"]:
            out[d["name"]] = d
    return out


def raw_score(d, alpha, iface, clash, beta):
    sc = d["sc"].double()
    s_psc = alpha * sc[:, 0] - (sc[:, 1:4] * clash).sum(-1)
    return (s_psc + (iface_score_matrix(iface) * d["T"].double()).sum(dim=(-2, -1))
            + beta * d["elec"].double())


def normalize(s: torch.Tensor) -> torch.Tensor:
    """`run_pinder_scaling.normalized_scores`, without the Params plumbing."""
    scale = s.std(unbiased=False).clamp_min(1.0)
    return (s - s.mean()) / scale


def band_stats(s: torch.Tensor, pos: torch.Tensor, margin: float):
    """Which poses the margin hinge can actually push on, in normalized units."""
    thr = float(s[pos].min()) - margin
    neg = ~pos
    return thr, (s[neg] > thr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--round0", default="data/scaling/pool_cache/*_r0_*_pk2.pt")
    ap.add_argument("--round1", required=True,
                    help="glob for the round-1 raw shards of ONE checkpoint")
    ap.add_argument("--ckpt", required=True, help="the round-0 checkpoint")
    ap.add_argument("--split", required=True, help="split.json, for fit_ids")
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--pool-cap", type=int, default=4000, dest="pool_cap")
    ap.add_argument("--dockq-threshold", type=float, default=0.23, dest="thr")
    ap.add_argument("--beta", type=float, default=3.0)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    alpha, iface = ck["alpha"].double(), ck["iface"].double()
    clash = ck["clash_weights"].double()
    beta = torch.tensor(args.beta, dtype=torch.float64)

    r0, r1 = load(args.round0), load(args.round1)
    fit = json.loads(open(args.split).read())["fit_ids"]
    have = [p for p in fit if p in r0 and p in r1]
    if len(have) != len(fit):
        raise SystemExit(
            f"only {len(have)} of {len(fit)} fit complexes have both a round-0 "
            f"pool and round-1 candidates; a partial set would bias every "
            f"figure below. Check the --round0/--round1 globs.")
    print(f"{len(have)} fit complexes, both rounds present")

    n_active = 0
    tot_new = tot_new_in_band = tot_old_in_band_r1 = 0
    n_new_hardest = 0
    d_thr, d_top, shares = [], [], []
    for pid in have:
        a, b = r0[pid], r1[pid]
        ka = a["prov"] == 0                       # what loss_view keeps
        if not ka.any():
            continue
        dq_a = a["dockq"][ka]
        pos_a = dq_a >= args.thr
        if not bool(pos_a.any()):
            continue                              # no gradient under either arm
        n_active += 1

        s_a = normalize(raw_score({k: v[ka] for k, v in a.items()
                                   if torch.is_tensor(v)},
                                  alpha, iface, clash, beta))
        thr_a, in_a = band_stats(s_a, pos_a, args.margin)

        kb = (b["prov"] == 0) & (b["dockq"] < args.thr)
        seen = set(map(tuple, a["pose_key"][ka].tolist()))
        fresh_idx = [i for i, k in enumerate(b["pose_key"][kb].tolist())
                     if tuple(k) not in seen]
        if not fresh_idx:
            continue
        sel = torch.tensor(fresh_idx, dtype=torch.long)
        cand = {k: v[kb][sel] for k, v in b.items() if torch.is_tensor(v)}
        tot_new += len(fresh_idx)

        # the round-1 pool the loss actually saw: round-0 view + fresh negatives
        merged = {k: torch.cat([a[k][ka], cand[k]]) for k in
                  ("sc", "T", "elec", "rmsd", "dockq")}
        s_b = normalize(raw_score(merged, alpha, iface, clash, beta))
        dq_b = merged["dockq"]
        pos_b = dq_b >= args.thr
        thr_b, in_b = band_stats(s_b, pos_b, args.margin)

        n_old = int(ka.sum()) - int(pos_a.sum())
        is_new = torch.zeros(int((~pos_b).sum()), dtype=torch.bool)
        is_new[n_old:] = True
        new_in = int((in_b & is_new).sum())
        old_in = int((in_b & ~is_new).sum())
        tot_new_in_band += new_in
        tot_old_in_band_r1 += old_in
        if new_in + old_in:
            shares.append(new_in / (new_in + old_in))
        d_thr.append(thr_b - thr_a)
        s_new = s_b[(~pos_b).nonzero(as_tuple=True)[0]][n_old:]
        s_old = s_b[(~pos_b).nonzero(as_tuple=True)[0]][:n_old]
        if s_new.numel() and s_old.numel():
            n_new_hardest += float(s_new.max()) > float(s_old.max())
            d_top.append(float(s_new.max()) - float(s_old.max()))

    med = lambda v: sorted(v)[len(v) // 2] if v else float("nan")
    print(f"\n{n_active} complexes have a search-derived positive "
          f"(the rest give no gradient under either arm)\n")
    print("all figures in NORMALIZED score units, the units the loss uses")
    print(f"  new negatives added                   : {tot_new:,}")
    print(f"  ... inside the margin band            : {tot_new_in_band:,} "
          f"({100.0*tot_new_in_band/max(1,tot_new):.2f}%)")
    print(f"  old negatives inside the band (round1): {tot_old_in_band_r1:,}")
    if tot_new_in_band + tot_old_in_band_r1:
        print(f"  new share of the band                 : "
              f"{100.0*tot_new_in_band/(tot_new_in_band+tot_old_in_band_r1):.2f}%")
        print(f"  per-complex new share: median {med(shares)*100:.1f}%")
    print(f"\n  complexes where a NEW negative is the hardest: "
          f"{n_new_hardest}/{n_active}")
    print(f"  median shift of the hardest negative  : {med(d_top):+.4f}")
    print(f"  median shift of the margin threshold  : {med(d_thr):+.4f}")
    print("\nReading: the hinge is driven by the highest-scoring negative. New")
    print("poses drawn from the same search can fill the band without moving")
    print("its top, and then the gradient they add is redundant with what the")
    print("old negatives already supplied.")


if __name__ == "__main__":
    main()
