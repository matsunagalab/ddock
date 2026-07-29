"""Which systems moved, and by how much, between two PINDER submissions?

A hit rate is an aggregate. "+8.40 pp acceptable at Max(Top 1)" is compatible
with 21 systems improving decisively and with 21 systems whose DockQ crept from
0.229 to 0.231. The second would be a threshold artefact, and it is the most
likely way the headline is overstated -- especially since the High rate moved
only +1.60 pp.

Reads the per-decoy metrics frame written by
`scripts/score_decoys_with_pinder.py --metrics-out`, aggregates it exactly as
PINDER does (max over the submitted decoys per system), and reports every
system whose CAPRI class changed, with the DockQ on both sides.

Also splits the gain by whether the system is a same-UniProt homodimer. Our
scoring treats the symmetry-equivalent solution of a homodimer as a negative,
which is a known limitation, so the two cohorts belong in separate columns.

Example
-------
    uv run python scripts/capri_transitions.py \
        --metrics data/pinder_eval/leaderboard_per_decoy.csv \
        --base published --test trained_N220
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

CLASS = {0: "Incorrect", 1: "Acceptable", 2: "Medium", 3: "High"}


def _is_homodimer(system_id: str) -> bool:
    """PINDER ids are {pdb}__{chain}{copy}_{uniprot}--{pdb}__{chain}{copy}_{uniprot}."""
    try:
        left, right = system_id.split("--")
        return left.split("_")[-1] == right.split("_")[-1]
    except ValueError:
        return False


def _top(df: pd.DataFrame, k: int | None) -> pd.DataFrame:
    """PINDER's Max(Top k): best decoy among the k lowest-ranked models."""
    d = df if k is None else df[df["rank"] <= k]
    return d.groupby("id").agg(DockQ=("DockQ", "max"),
                               CAPRI=("CAPRI_rank", "max"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--base", required=True, help="method_name of the baseline")
    ap.add_argument("--test", required=True, help="method_name to compare")
    ap.add_argument("--top", type=int, default=1)
    # MethodMetrics returns EVERY subset it knows, with a DockQ=0 penalty row
    # for each system that was not submitted. Left unfiltered, the 1705
    # pinder_xl systems we never docked would dominate every cohort split.
    ap.add_argument("--subset", default="pinder_s",
                    choices=("pinder_s", "pinder_xl", "pinder_af2", "all"))
    ap.add_argument("--monomer", default="holo")
    ap.add_argument("--margin", type=float, default=0.02,
                    help="a DockQ within this of the 0.23 acceptable "
                         "threshold counts as threshold-adjacent")
    args = ap.parse_args()

    m = pd.read_csv(args.metrics)
    if args.subset != "all":
        m = m[m[args.subset].astype(bool)]
    if args.monomer and "monomer_name" in m.columns:
        m = m[m.monomer_name == args.monomer]
    if "rank" not in m.columns:
        # PINDER names it model_name (…_1.pdb); recover the trailing integer
        m["rank"] = (m["model_name"].astype(str)
                     .str.extract(r"(\d+)$").astype(float))
    m = m[m["rank"].notna()]
    m["rank"] = m["rank"].astype(int)

    a = _top(m[m.method_name == args.base], args.top)
    b = _top(m[m.method_name == args.test], args.top)
    j = a.join(b, lsuffix="_base", rsuffix="_test", how="outer").fillna(0.0)
    j["homodimer"] = [ _is_homodimer(i) for i in j.index ]
    n = len(j)
    print(f"{n} systems, Max(Top {args.top}), {args.base} -> {args.test}\n")

    for lvl, name in ((1, "acceptable"), (2, "medium"), (3, "high")):
        pa = (j.CAPRI_base >= lvl).sum()
        pb = (j.CAPRI_test >= lvl).sum()
        gain = ((j.CAPRI_test >= lvl) & (j.CAPRI_base < lvl)).sum()
        loss = ((j.CAPRI_test < lvl) & (j.CAPRI_base >= lvl)).sum()
        print(f"  {name:11s} {100 * pa / n:6.2f}% -> {100 * pb / n:6.2f}%  "
              f"({100 * (pb - pa) / n:+.2f} pp)   won {gain}, lost {loss}")

    moved = j[j.CAPRI_base != j.CAPRI_test].sort_values(
        "DockQ_test", ascending=False)
    print(f"\n{len(moved)} systems changed CAPRI class:")
    print(f"{'system':50s} {'base':>22s}  {'test':>22s}")
    for i, r in moved.iterrows():
        tag = " [homodimer]" if r.homodimer else ""
        print(f"  {i[:48]:48s} "
              f"{CLASS[int(r.CAPRI_base)]:>10s} {r.DockQ_base:6.3f}  -> "
              f"{CLASS[int(r.CAPRI_test)]:>10s} {r.DockQ_test:6.3f}{tag}")

    # threshold artefact check: of the systems that newly became acceptable,
    # how many only just cleared 0.23?
    won = j[(j.CAPRI_test >= 1) & (j.CAPRI_base < 1)]
    near = won[won.DockQ_test < 0.23 + args.margin]
    print(f"\nnewly acceptable: {len(won)}; of those within "
          f"{args.margin} of the 0.23 threshold: {len(near)}")
    if len(won):
        print(f"  their DockQ: min {won.DockQ_test.min():.3f} "
              f"median {won.DockQ_test.median():.3f} "
              f"max {won.DockQ_test.max():.3f}")

    print("\nby cohort (acceptable rate at this Top-k):")
    for flag, name in ((True, "same-UniProt homodimer"), (False, "other")):
        c = j[j.homodimer == flag]
        if not len(c):
            continue
        pa = 100 * (c.CAPRI_base >= 1).mean()
        pb = 100 * (c.CAPRI_test >= 1).mean()
        print(f"  {name:24s} n={len(c):4d}  {pa:6.2f}% -> {pb:6.2f}%  "
              f"({pb - pa:+.2f} pp)")


if __name__ == "__main__":
    main()
