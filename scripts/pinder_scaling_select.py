"""Build the deterministic, nested PINDER cluster master list for the
interface-cluster **scaling-law** experiment (EXPERIMENT_REPORT §5.6).

Selection rules (all enforced here, asserted again at run time):

* ``split == "train"`` in the PINDER 2024-02 index (PINDER's own
  interface-deleaked split, so no FoldSeek/MMseqs/iAlign leakage against the
  held-out PINDER-S test set).
* ``holo_R and holo_L`` — we redock bound monomers, so both must exist.
* **one system per ``cluster_id``** (the statistical unit is the interface
  cluster, not the PDB entry), choosing the lexicographically smallest system
  id inside the cluster for determinism.
* ``cluster_id`` disjoint from every cluster represented in the fixed
  PINDER-S test id list.
* systems whose chain names are ``UNDEFINED`` are dropped (these are the
  unmapped-chain entries that most often fail our plain-PDB parser).

The output is a single **ordered** master list. Every experiment size takes a
*prefix* of the usable (successfully prepared) portion of this list, which is
what makes the 500 / 1,000 / 2,000 subsets exactly nested. The order is a
deterministic shuffle (fixed seed) of the eligible clusters, so every prefix is
an unbiased sample of the same cluster population — the property a scaling
curve needs. The earlier 220-cluster run used a *different* sample and a
different id-selection rule, so it is kept as a separate historical anchor
rather than pinned into this list; ``--pin-previous`` can force the old
clusters to the front if a like-for-like subset is wanted instead.

Example
-------
    PINDER_BASE_DIR=$PWD/external/pinder uv run python \
        scripts/pinder_scaling_select.py --n-master 4000
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent


def _index_path(base_dir: str | None) -> Path:
    base = Path(base_dir or os.environ.get("PINDER_BASE_DIR", "external/pinder"))
    cands = sorted(base.glob("pinder/*/index.parquet"))
    if not cands:
        cands = sorted(base.glob("*/index.parquet"))
    if not cands:
        raise SystemExit(f"no index.parquet found under {base}")
    return cands[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pinder-base", default=None, dest="pinder_base")
    ap.add_argument("--test-ids", default="data/pinder_test_ids.txt", dest="test_ids")
    ap.add_argument("--prev-train-ids", default="data/pinder_train_ids.txt",
                    dest="prev_train_ids",
                    help="ids of the earlier 220-cluster run (reported; only "
                         "pinned to the front when --pin-previous is given)")
    ap.add_argument("--pin-previous", action="store_true", dest="pin_previous")
    ap.add_argument("--n-master", type=int, default=4000, dest="n_master",
                    help="how many ordered candidate clusters to emit")
    ap.add_argument("--shuffle-seed", type=int, default=20260725, dest="shuffle_seed")
    ap.add_argument("--out-dir", default="data/scaling", dest="out_dir")
    args = ap.parse_args()

    idx_path = _index_path(args.pinder_base)
    print(f"index: {idx_path}", flush=True)
    cols = ["split", "id", "cluster_id", "cluster_id_R", "cluster_id_L",
            "holo_R", "holo_L", "chain_R", "chain_L", "pinder_s"]
    df = pd.read_parquet(idx_path, columns=cols)

    test_ids = [ln.strip() for ln in Path(args.test_ids).read_text().splitlines()
                if ln.strip()]
    test_rows = df[df["id"].isin(set(test_ids))]
    test_clusters = set(test_rows["cluster_id"].astype(str))
    print(f"test ids {len(test_ids)} -> matched {len(test_rows)} rows, "
          f"{len(test_clusters)} clusters", flush=True)
    if len(test_rows) != len(test_ids):
        raise SystemExit("some test ids are missing from the PINDER index")

    tr = df[(df["split"] == "train") & df["holo_R"] & df["holo_L"]].copy()
    n0 = len(tr)
    tr = tr[(tr["chain_R"].astype(str) != "UNDEFINED")
            & (tr["chain_L"].astype(str) != "UNDEFINED")]
    tr = tr[~tr["id"].str.contains("UNDEFINED", regex=False)]
    print(f"train holo rows {n0} -> {len(tr)} after dropping UNDEFINED chains",
          flush=True)

    tr["cluster_id"] = tr["cluster_id"].astype(str)
    overlap = set(tr["cluster_id"]) & test_clusters
    print(f"train/test cluster_id overlap (must be 0): {len(overlap)}", flush=True)
    if overlap:
        tr = tr[~tr["cluster_id"].isin(overlap)]

    # One representative system per cluster: smallest id, deterministic.
    tr = tr.sort_values("id", kind="mergesort")
    rep = tr.groupby("cluster_id", as_index=False, observed=True).first()
    rep = rep.sort_values("cluster_id", kind="mergesort").reset_index(drop=True)
    print(f"eligible unique clusters: {len(rep)}", flush=True)

    # Pin the clusters used by the earlier 220-cluster run to the front so the
    # old run's complexes are contained in every new (nested) subset.
    prev_clusters: list[str] = []
    prev_path = Path(args.prev_train_ids)
    if prev_path.exists():
        prev_ids = [ln.strip() for ln in prev_path.read_text().splitlines() if ln.strip()]
        prev_rows = df[df["id"].isin(set(prev_ids))]
        prev_clusters = sorted(set(prev_rows["cluster_id"].astype(str)))
        print(f"previous-run ids {len(prev_ids)} -> {len(prev_clusters)} clusters",
              flush=True)

    eligible = set(rep["cluster_id"])
    front = [c for c in prev_clusters if c in eligible] if args.pin_previous else []
    rest = [c for c in rep["cluster_id"] if c not in set(front)]
    random.Random(args.shuffle_seed).shuffle(rest)
    ordered = (front + rest)[: args.n_master]
    print(f"master order: {len(front)} pinned + {len(ordered) - len(front)} "
          f"shuffled = {len(ordered)}", flush=True)
    n_prev_in_master = len(set(prev_clusters) & set(ordered))
    print(f"[info] clusters shared with the previous 220-cluster run: "
          f"{n_prev_in_master}", flush=True)

    out = rep.set_index("cluster_id").loc[ordered].reset_index()
    out.insert(0, "rank", range(len(out)))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / "master_clusters.csv"
    txt = out_dir / "master_ids.txt"
    out[["rank", "id", "cluster_id", "cluster_id_R", "cluster_id_L",
         "chain_R", "chain_L"]].to_csv(csv, index=False)
    txt.write_text("\n".join(out["id"].tolist()) + "\n")

    # Sanity: uniqueness + disjointness.
    assert out["id"].is_unique, "duplicate system id in master list"
    assert out["cluster_id"].is_unique, "duplicate cluster_id in master list"
    assert not (set(out["cluster_id"]) & test_clusters), "test cluster leaked into master"
    assert not (set(out["id"]) & set(test_ids)), "test id leaked into master"

    # Informational only: PINDER deleaks at the interface-cluster level, so a
    # shared *single-chain* cluster between train and test is allowed by the
    # benchmark. We report it rather than filtering on it.
    rl = set(out["cluster_id_R"].astype(str)) | set(out["cluster_id_L"].astype(str))
    rl_test = (set(test_rows["cluster_id_R"].astype(str))
               | set(test_rows["cluster_id_L"].astype(str)))
    print(f"[info] shared single-chain clusters train∩test: {len(rl & rl_test)} "
          f"(allowed by PINDER's interface-level deleaking)", flush=True)

    print(f"wrote {csv} and {txt}", flush=True)


if __name__ == "__main__":
    main()
