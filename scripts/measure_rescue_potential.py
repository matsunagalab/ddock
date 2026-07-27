"""How many zero-gradient complexes would an on-policy positive refresh rescue?

Under contract A the positive set is frozen at round 0, so a complex whose
round-0 search returned no near-native pose contributes nothing to the loss for
ever: 72 of 220 in the current fit set. Since mining negatives turned out to add
nothing (report section 5.14.21), those 72 are the largest remaining structural
inefficiency -- but "largest" is not "the cause", and contract B (taking new
positives too) is a different intervention with its own self-training risk.

This decides whether contract B is even worth running, using the round-1
candidates ALREADY mined: it asks how many of the 72 the trained parameters'
search actually found a near-native pose for. No new FFT.

A complex that stays empty here cannot be rescued by any positive-refresh
contract, and the bottleneck is the search's reach, not the training contract.

CPU only.
"""
from __future__ import annotations

import argparse
import glob
import json

import torch


def load(pattern: str) -> dict:
    out = {}
    for f in sorted(glob.glob(pattern)):
        for d in torch.load(f, map_location="cpu", weights_only=True)["pools"]:
            out[d["name"]] = d
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--round0", default="data/scaling/pool_cache/*_r0_*_pk2.pt")
    ap.add_argument("--round1", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--dockq-threshold", type=float, default=0.23, dest="thr")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    r0, r1 = load(args.round0), load(args.round1)
    fit = json.loads(open(args.split).read())["fit_ids"]
    missing = [p for p in fit if p not in r0 or p not in r1]
    if missing:
        raise SystemExit(f"{len(missing)} fit complexes lack a pool "
                         f"(e.g. {missing[0]}); fix the globs")

    dead, alive = [], []
    for pid in fit:
        a = r0[pid]
        k = a["prov"] == 0
        (alive if bool(((a["dockq"][k] >= args.thr)).any()) else dead).append(pid)
    print(f"fit {len(fit)}: {len(alive)} contribute a gradient, "
          f"{len(dead)} are silent under contract A")

    rescued, rows = [], []
    for pid in dead:
        b = r1[pid]
        k = b["prov"] == 0
        dq = b["dockq"][k]
        pos = dq >= args.thr
        n = int(pos.sum())
        rows.append({"id": pid, "n_new_positive": n,
                     "best_dockq_round1": float(dq.max()) if dq.numel() else 0.0,
                     "best_dockq_round0": float(r0[pid]["dockq"][
                         r0[pid]["prov"] == 0].max())})
        if n:
            rescued.append(pid)

    print(f"\nof the {len(dead)} silent complexes, the TRAINED parameters' "
          f"search found a near-native pose for {len(rescued)}")
    if dead:
        print(f"  rescue rate: {100.0*len(rescued)/len(dead):.1f}%")
        print(f"  effective fit set would go {len(alive)} -> "
              f"{len(alive)+len(rescued)} of {len(fit)}")
    b0 = sorted(r["best_dockq_round0"] for r in rows)
    b1 = sorted(r["best_dockq_round1"] for r in rows)
    if b0:
        q = lambda v, f: v[min(len(v) - 1, int(f * len(v)))]
        print(f"\n  best DockQ among the silent complexes' search poses:")
        print(f"    round 0 : median {q(b0,.5):.3f}  p90 {q(b0,.9):.3f}  "
              f"max {b0[-1]:.3f}")
        print(f"    round 1 : median {q(b1,.5):.3f}  p90 {q(b1,.9):.3f}  "
              f"max {b1[-1]:.3f}")
        print(f"  (threshold is {args.thr}; a complex whose best stays far below "
              f"it is out of\n   the search's reach, not of the training "
              f"contract's)")
    if rescued:
        print(f"\n  rescued, with how many positives each:")
        for r in sorted((r for r in rows if r["n_new_positive"]),
                        key=lambda r: -r["n_new_positive"])[:15]:
            print(f"    {r['id'][:44]:<46} {r['n_new_positive']:>4} pos, "
                  f"best DockQ {r['best_dockq_round0']:.3f} -> "
                  f"{r['best_dockq_round1']:.3f}")
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
