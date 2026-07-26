"""Build the frozen TEST candidate pool with the *same* recipe as the fit pools.

Why this has to be rebuilt
--------------------------
``data/shards_pinder/test_feats.pt`` was produced before the PSC / ELEC / grid
-binning fixes and with the old candidate recipe, so it is invalid twice over:

* Its ``(S_SC, T, S_ELEC)`` were computed by a scorer whose ELEC term had 80% of
  its magnitude deleted and the wrong sign, whose PSC gave core precedence over
  surface, and whose ligand atoms were floor-binned half a cell low. Those are
  the cached *features*, not just parameters, so they cannot be re-weighted.
* Its poses came from a search seeded with a 25 deg cone around the native
  orientation, plus positives injected at the exact native translation. Measured
  on that file: 95.4% of the 241 complexes had no positive at all among the 2000
  search poses, and for 72.6% *every* search pose outranked *every* injected
  positive. A near-zero baseline success rate on such a pool is close to a
  tautology rather than a measurement.

This script rebuilds it through ``mine_complex``, so the TEST pool and the fit
pools are built by one code path with one set of flags. Anything that differs
between them is then a deliberate argument, not a drift.

The pool is built once with the **baseline** parameters and frozen: it must not
depend on the parameters being evaluated, or conditions would not be comparable.

Example
-------
    export PINDER_BASE_DIR=$PWD/external/pinder
    uv run python scripts/build_test_pool.py \
        --ids data/pinder_test_ids.txt \
        --prep-cache data/scaling/prep_cache \
        --out data/shards_pinder/test_feats_reachable.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

from zdock.atomtypes import charge_score as default_charge_score  # noqa: E402
from zdock.prep_cache import load_prepared  # noqa: E402
from zdock.score import SC_REFERENCE_SPACING, SC_RHO  # noqa: E402

from run_pinder_scaling import Params, mine_complex  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ids", default="data/pinder_test_ids.txt")
    ap.add_argument("--prep-cache", default="data/scaling/prep_cache",
                    dest="prep_cache")
    ap.add_argument("--out", default="data/shards_pinder/test_feats_reachable.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1",
                    help="i/n -- take every n-th id starting at i, so several "
                         "GPUs can build disjoint parts of one pool")
    ap.add_argument("--grid-voxels", default="", dest="grid_voxels",
                    help="voxel table; with --max-grid-voxels, drop the tail")
    ap.add_argument("--max-grid-voxels", type=int, default=0,
                    dest="max_grid_voxels",
                    help="apply the SAME size cutoff as the training corpus, so "
                         "the two cohorts are not drawn from different size "
                         "distributions")
    # These must mirror the training run's pool flags.
    ap.add_argument("--pool", choices=("reachable", "legacy"), default="reachable")
    ap.add_argument("--rot-set", choices=("hopf", "uniform"), default="hopf",
                    dest="rot_set")
    ap.add_argument("--hopf-nside", type=int, default=3, dest="hopf_nside")
    ap.add_argument("--near-rot", type=int, default=8, dest="near_rot")
    ap.add_argument("--trans-cells", type=int, default=1, dest="trans_cells")
    ap.add_argument("--spacing", type=float, default=SC_REFERENCE_SPACING)
    ap.add_argument("--mine-ntop", type=int, default=1500, dest="mine_ntop")
    ap.add_argument("--mine-random-rot", type=int, default=1500,
                    dest="mine_random_rot")
    ap.add_argument("--mine-cone", type=int, default=400, dest="mine_cone")
    ap.add_argument("--psc-decompose", dest="psc_decompose",
                    action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--alpha0", type=float, default=1.0)
    ap.add_argument("--rho0", type=float, default=SC_RHO)
    ap.add_argument("--dockq-threshold", type=float, default=0.23,
                    dest="dockq_thr")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rot-chunk", type=int, default=2, dest="rot_chunk")
    ap.add_argument("--frame-chunk", type=int, default=100, dest="frame_chunk")
    ap.add_argument("--feature-budget", type=int, default=1_000_000_000,
                    dest="feature_budget")
    ap.add_argument("--dockq-budget", type=int, default=50_000_000,
                    dest="dockq_budget")
    ap.add_argument("--oom-retries", type=int, default=3, dest="oom_retries")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32
    ids = [ln.strip() for ln in Path(args.ids).read_text().splitlines() if ln.strip()]
    n_all = len(ids)
    n_oversized = 0
    if args.max_grid_voxels:
        if not args.grid_voxels:
            raise SystemExit("--max-grid-voxels needs --grid-voxels")
        vox = json.loads(Path(args.grid_voxels).read_text())
        kept = [i for i in ids if vox.get(i, 0) <= args.max_grid_voxels]
        n_oversized = len([i for i in ids if vox.get(i, 0) > args.max_grid_voxels])
        ids = kept
    if args.limit:
        ids = ids[: args.limit]
    si, sn = (int(x) for x in args.shard.split("/"))
    ids = ids[si::sn]
    print(f"{len(ids)} TEST ids (shard {si}/{sn} of {n_all} listed, "
          f"{n_oversized} over the size cutoff) from {args.ids}", flush=True)

    p0 = Params.initial(args, device, dtype)
    beta0 = torch.tensor(3.0, device=device, dtype=dtype)
    charge0 = default_charge_score(device=device, dtype=dtype)

    blob, skipped, t0 = [], [], time.time()
    for i, pid in enumerate(ids):
        prot_cpu = load_prepared(args.prep_cache, pid)
        if prot_cpu is None:
            skipped.append({"id": pid, "reason": "not in prep cache"})
            continue
        prot = None
        try:
            prot = prot_cpu.to(device, dtype=dtype)
            f = mine_complex(prot, p0, beta0, charge0, args, 0)
            blob.append({"name": f.name, "sc": f.sc, "T": f.T, "elec": f.elec,
                         "rmsd": f.rmsd, "dockq": f.dockq, "prov": f.prov})
            c = f.counts(args.dockq_thr)
            if (i + 1) % 20 == 0 or c["n_pos"] == 0:
                el = time.time() - t0
                print(f"  [{i+1}/{len(ids)}] {pid[:44]:<46} n={c['n']} "
                      f"pos={c['n_pos']} (search {c['n_pos_from_search']}) "
                      f"{el/(i+1):.1f}s/complex", flush=True)
        except Exception as exc:  # noqa: BLE001
            skipped.append({"id": pid, "reason": f"{type(exc).__name__}: {exc}"[:200]})
            print(f"  [{i+1}/{len(ids)}] SKIP {pid}: {type(exc).__name__}", flush=True)
        finally:
            del prot, prot_cpu
            if device.type == "cuda":
                torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, out)
    n_nopos = sum(1 for d in blob
                  if int((d["dockq"] >= args.dockq_thr).sum()) == 0)
    n_search_pos = sum(1 for d in blob if int(
        ((d["dockq"] >= args.dockq_thr) & (d["prov"] == 0)).sum()) > 0)
    meta = {"n_listed": n_all, "n_oversized_excluded": n_oversized,
            "n_requested": len(ids), "n_built": len(blob),
            "n_skipped": len(skipped),
            "n_without_any_positive": n_nopos,
            "n_with_a_search_found_positive": n_search_pos,
            "config": vars(args), "skipped": skipped}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"\nwrote {out}  ({len(blob)} complexes, {len(skipped)} skipped)")
    print(f"  complexes with NO reachable positive: {n_nopos}/{len(blob)} "
          f"-- these bound what any parameter fit can achieve")
    print(f"  complexes where the SEARCH itself found a positive: "
          f"{n_search_pos}/{len(blob)}")


if __name__ == "__main__":
    main()
