"""Does the trained table transfer to unbound structures, or only to bound ones?

The IFACE table was fitted on holo interfaces. If its gain shrinks on apo
monomers, two explanations fit equally well: the table leans on bound side-chain
packing, or apo is simply harder for every method. The contrast that separates
them is the interaction, on the SAME systems:

    G_holo = M(trained, holo) - M(baseline, holo)
    G_apo  = M(trained, apo)  - M(baseline, apo)
    D_int  = G_apo - G_holo

D_int ~ 0 with both levels down means apo is harder for everyone. D_int < 0 means
the trained table specifically loses its advantage when the side chains move.

Only 93 of the 250 PINDER-S systems have both apo monomers, so the holo side is
restricted to those 93 -- comparing a 250-system holo rate against a 93-system
apo rate would confound the contrast with which systems have apo structures at
all. Confidence intervals come from resampling systems (paired, both monomer
settings move together).

Example
-------
    uv run python scripts/apo_holo_interaction.py \
        --metrics data/pinder_eval/leaderboard_per_decoy.csv \
        --base published --test trained_N220
"""

from __future__ import annotations

import argparse

import pandas as pd
import torch


def rate(j: pd.DataFrame, col: str, lvl: int) -> float:
    return 100.0 * (j[col] >= lvl).mean()


def top(m: pd.DataFrame, method: str, monomer: str, k: int) -> pd.Series:
    d = m[(m.method_name == method) & (m.monomer_name == monomer)]
    d = d[d["rank"] <= k]
    return d.groupby("id")["CAPRI_rank"].max()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics", default="data/pinder_eval/leaderboard_per_decoy.csv")
    ap.add_argument("--base", default="published")
    ap.add_argument("--test", default="trained_N220")
    ap.add_argument("--top", type=int, default=1)
    ap.add_argument("--level", type=int, default=1,
                    help="1 acceptable, 2 medium, 3 high")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    m = pd.read_csv(args.metrics)
    m = m[m.pinder_s.astype(bool)]
    if "rank" not in m.columns:
        m["rank"] = m["model_name"].astype(str).str.extract(r"(\d+)$").astype(float)
    m = m[m["rank"].notna()]
    m["rank"] = m["rank"].astype(int)

    cols = {}
    for who, method in (("base", args.base), ("test", args.test)):
        for mono in ("holo", "apo"):
            cols[f"{who}_{mono}"] = top(m, method, mono, args.top)
    j = pd.DataFrame(cols)
    # the 93 systems that have apo at all; a system missing from a column is a
    # PINDER penalty row and is already CAPRI 0, so drop only true absences
    j = j.dropna()
    n = len(j)
    lvl = args.level
    name = {1: "acceptable", 2: "medium", 3: "high"}[lvl]
    print(f"{n} systems with both apo monomers, Max(Top {args.top}), {name}")
    print(f"{args.base} -> {args.test}\n")

    g = {}
    for mono in ("holo", "apo"):
        b = rate(j, f"base_{mono}", lvl)
        t = rate(j, f"test_{mono}", lvl)
        g[mono] = t - b
        print(f"  {mono:4s}  {b:6.2f}% -> {t:6.2f}%   gain {t - b:+6.2f} pp")
    d_int = g["apo"] - g["holo"]
    print(f"\n  D_int = G_apo - G_holo = {d_int:+.2f} pp")

    gen = torch.Generator().manual_seed(args.seed)
    arr = torch.tensor(j.values, dtype=torch.float64)   # (n, 4) base/test x holo/apo
    idx = torch.randint(0, n, (args.boot, n), generator=gen)
    hit = (arr >= lvl).double()
    s = hit[idx]                                        # (boot, n, 4)
    r = 100.0 * s.mean(dim=1)
    gh = r[:, 1] - r[:, 0]                              # base_holo, test_holo order below
    ga = r[:, 3] - r[:, 2]
    boot = ga - gh
    lo, hi = torch.quantile(boot, torch.tensor([0.025, 0.975], dtype=torch.float64))
    print(f"  paired system bootstrap 95% CI: [{float(lo):+.2f}, {float(hi):+.2f}] pp "
          f"({args.boot} resamples)")
    print(f"  P(D_int < 0) = {float((boot < 0).double().mean()):.3f}")

    # discordant systems, the McNemar view
    for mono in ("holo", "apo"):
        b = j[f"base_{mono}"] >= lvl
        t = j[f"test_{mono}"] >= lvl
        print(f"  {mono:4s}  won {int((t & ~b).sum())}, lost {int((~t & b).sum())}")


if __name__ == "__main__":
    main()
