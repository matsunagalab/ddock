"""Did fixing the pose-identity overflow change what the loss and the fixed-pool
evaluation actually saw?

Background
----------
`generate_pool_reachable` drops an enumerated candidate the FFT search already
returned. The identity it compared on used to be packed into one int64 as
``((qi*2**21 + cx)*2**21 + cy)*2**21 + cz``, which puts the rotation index at
bit 63 and overflows: measured, all 1944 Hopf rotations at one cell produced
TWO distinct keys, so an enumerated pose was dropped whenever a search pose
shared its translation cell and its rotation *parity*. Measured over 12
complexes, that wrongly removed 0-56% of the 216 enumerated candidates.

What this audits
----------------
The claim that the headline results are unaffected. They are computed under
``--loss-prov search`` / ``--prov search``, i.e. from poses with ``prov == 0``,
and the de-duplication only ever masked the enumerated side. So the search-derived
rows should be identical, tensor for tensor, between the old cache and the new
``_pk2`` one. This checks that rather than asserting it from the code.

What it cannot check: the all-provenance quantities (pool size, positive count,
IFACE coverage, reachability, pooled metrics) *are* expected to change, and this
prints the size of that change so the report can be corrected.

Example
-------
    uv run python scripts/audit_pose_key_fix.py \
        --old 'data/scaling/pool_cache/n220_r0_*_bg1.*of3.pt' \
        --new 'data/scaling/pool_cache/n220_r0_*_bg1_pk2.*of5.pt'
"""

from __future__ import annotations

import argparse
import glob
import json

import torch

FEATURES = ("sc", "T", "elec", "rmsd", "dockq")


def load(patterns: list[str]) -> dict:
    pools = {}
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            blob = torch.load(f, map_location="cpu", weights_only=True)
            for d in blob["pools"]:
                pools[d["name"]] = d
    return pools


def search_rows(d: dict) -> dict:
    keep = (d["prov"] == 0).nonzero(as_tuple=True)[0]
    return {k: d[k][keep] for k in FEATURES}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old", nargs="+", required=True)
    ap.add_argument("--new", nargs="+", required=True)
    ap.add_argument("--dockq-threshold", type=float, default=0.23, dest="thr")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    old, new = load(args.old), load(args.new)
    shared = sorted(set(old) & set(new))
    print(f"old {len(old)} pools, new {len(new)} pools, {len(shared)} shared")
    if not shared:
        raise SystemExit("no complex appears in both caches; check the globs")

    rows, n_ident, n_diff = [], 0, 0
    for name in shared:
        o, n = search_rows(old[name]), search_rows(new[name])
        same_n = o["rmsd"].shape[0] == n["rmsd"].shape[0]
        worst = {}
        identical = same_n
        if same_n:
            for k in FEATURES:
                a, b = o[k].double(), n[k].double()
                if not torch.equal(o[k], n[k]):
                    identical = False
                worst[k] = float((a - b).abs().max()) if a.numel() else 0.0
        n_ident += identical
        n_diff += not identical
        # all-provenance quantities that ARE expected to move
        po = old[name]["dockq"] >= args.thr
        pn = new[name]["dockq"] >= args.thr
        rows.append({
            "name": name,
            "search_rows_old": int(o["rmsd"].shape[0]),
            "search_rows_new": int(n["rmsd"].shape[0]),
            "search_identical": bool(identical),
            "search_max_abs_diff": worst,
            "pool_old": int(old[name]["dockq"].shape[0]),
            "pool_new": int(new[name]["dockq"].shape[0]),
            "pos_old": int(po.sum()), "pos_new": int(pn.sum()),
            "pos_enum_old": int((po & (old[name]["prov"] != 0)).sum()),
            "pos_enum_new": int((pn & (new[name]["prov"] != 0)).sum()),
            "pos_search_old": int((po & (old[name]["prov"] == 0)).sum()),
            "pos_search_new": int((pn & (new[name]["prov"] == 0)).sum()),
        })

    print(f"\nsearch-derived rows identical (tensor for tensor): "
          f"{n_ident}/{len(shared)}")
    if n_diff:
        print(f"  DIFFERING: {n_diff} -- the headline results are NOT immune")
        for r in rows:
            if not r["search_identical"]:
                print(f"    {r['name'][:44]:<46} "
                      f"{r['search_rows_old']} -> {r['search_rows_new']} rows, "
                      f"max|d| {r['search_max_abs_diff']}")
    else:
        print("  every --prov search tensor the loss and the fixed-pool "
              "evaluation see is unchanged")

    def tot(k):
        return sum(r[k] for r in rows)
    print(f"\nall-provenance quantities (these DO change):")
    print(f"  pool size          {tot('pool_old'):>8} -> {tot('pool_new'):>8}  "
          f"({100.0*(tot('pool_new')-tot('pool_old'))/max(1,tot('pool_old')):+.1f}%)")
    print(f"  positives          {tot('pos_old'):>8} -> {tot('pos_new'):>8}  "
          f"({100.0*(tot('pos_new')-tot('pos_old'))/max(1,tot('pos_old')):+.1f}%)")
    print(f"    from the search  {tot('pos_search_old'):>8} -> "
          f"{tot('pos_search_new'):>8}")
    print(f"    enumerated       {tot('pos_enum_old'):>8} -> "
          f"{tot('pos_enum_new'):>8}")
    z_old = sum(1 for r in rows if r["pos_old"] == 0)
    z_new = sum(1 for r in rows if r["pos_new"] == 0)
    print(f"  complexes with NO positive at all: {z_old} -> {z_new}")
    zs_old = sum(1 for r in rows if r["pos_search_old"] == 0)
    zs_new = sum(1 for r in rows if r["pos_search_new"] == 0)
    print(f"  complexes with no SEARCH positive: {zs_old} -> {zs_new}")

    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")
    if n_diff:
        raise SystemExit("search-derived features changed; the report's "
                         "--prov search results need re-checking")


if __name__ == "__main__":
    main()
