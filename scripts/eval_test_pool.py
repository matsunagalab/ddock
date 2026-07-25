"""Post-hoc re-scoring of the fixed deleaked TEST pool, with metrics that do
not sit on the floor.

Why this exists
---------------
``success@K`` on 241 test complexes moves in steps of 1/241 = 0.41 pp, and
every number measured so far lives between 0.0% and 1.7%. A metric quantised at
the size of the effect cannot show a scaling trend: it is at the floor. This
script recomputes the same fixed pool with **continuous, per-complex** readouts
that use the whole ranking rather than a threshold crossing:

* ``auc`` — Mann-Whitney AUC between the complex's positives (DockQ >= 0.23)
  and its negatives. 0.5 = the scorer orders near-native and decoy poses no
  better than chance, 1.0 = every positive outranks every negative. This is the
  sensitive primary readout.
* ``first_hit_rank`` / ``first_hit_pct`` — how deep the ranking must be read
  before the first positive appears, absolutely and as a fraction of the pool.
* ``best_dockq_at_K``, ``min_rmsd_at_K`` — quality of the best pose in top-K.
* ``success@K`` — kept for continuity with §5.4/§5.5, floor effects and all.

Per-complex rows are written out so that two conditions can be compared with a
**paired** test over the same 241 complexes, which is far more powerful than
comparing two success rates.

What this does and does not measure
-----------------------------------
The pool was built once (§5.4) with default ZDOCK parameters and contains
(a) the top-2,000 poses of an FFT search whose rotation set was seeded with a
25 deg cone around the native orientation, and (b) 400 near-native poses
injected at the native translation. So a positive is *guaranteed* to be present
and was not found by the search. These numbers therefore measure **re-ranking
of a fixed candidate set**, not end-to-end docking — for that see
``scripts/eval_search_test.py``. The pool is identical for every N, seed and
round, so comparisons between them are valid.

Example
-------
    uv run python scripts/eval_test_pool.py --runs-dir data/scaling/runs
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from zdock.atomtypes import iface_ij
from zdock.evaluate import evaluate_ranking
from zdock.score import iface_score_matrix

KS = (1, 5, 10, 50, 100)


def score_from_feats(sc, T, elec, alpha, iface_flat, beta):
    imat = iface_score_matrix(iface_flat)
    return alpha * sc + (imat * T).sum(dim=(-2, -1)) + beta * elec


def mann_whitney_auc(scores: torch.Tensor, pos: torch.Tensor) -> float:
    """P(score of a random positive > score of a random negative), ties = 0.5.

    Computed from the rank sum, so it is O(n log n) and uses every pose rather
    than a top-K cut."""
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # average ranks (1-based, ascending) with tie correction
    order = torch.argsort(scores)
    ranks = torch.empty_like(scores, dtype=torch.float64)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float64)
    uniq, inv, counts = torch.unique(scores, return_inverse=True, return_counts=True)
    if int((counts > 1).sum()):
        rank_sum = torch.zeros(uniq.numel(), dtype=torch.float64)
        rank_sum.index_add_(0, inv, ranks)
        ranks = (rank_sum / counts.to(torch.float64))[inv]
    r_pos = float(ranks[pos].sum())
    u = r_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def per_complex_metrics(f, alpha, iface, beta, rmsd_thr, dockq_thr):
    s = score_from_feats(f["sc"], f["T"], f["elec"], alpha, iface, beta)
    dockq, rmsd = f["dockq"], f["rmsd"]
    pos = dockq >= dockq_thr
    rep = evaluate_ranking(s, rmsd, dockq, ks=KS, rmsd_threshold=rmsd_thr,
                           dockq_threshold=dockq_thr)
    order = torch.argsort(s, descending=True)
    pos_sorted = pos[order]
    hit_idx = pos_sorted.nonzero(as_tuple=True)[0]
    n = s.numel()
    first_hit = int(hit_idx[0]) + 1 if hit_idx.numel() else n + 1
    row = {
        "name": f["name"], "n_poses": n, "n_pos": int(pos.sum()),
        "auc": mann_whitney_auc(s, pos),
        "first_hit_rank": first_hit,
        "first_hit_pct": first_hit / n,
        "best_dockq_overall": float(dockq.max()),
        "min_rmsd_overall": float(rmsd.min()),
    }
    for k in KS:
        row[f"succ_dockq@{k}"] = int(rep.success_dockq[k])
        row[f"succ_rmsd@{k}"] = int(rep.success_rmsd[k])
        row[f"best_dockq@{k}"] = rep.best_dockq_at[k]
        row[f"min_rmsd@{k}"] = rep.min_rmsd_at[k]
    return row


def _agg(rows, key):
    xs = [r[key] for r in rows if not (isinstance(r[key], float)
                                       and math.isnan(r[key]))]
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, sd


def summarise(rows, label, extra=None):
    out = {"label": label, "n_complexes": len(rows)}
    out.update(extra or {})
    for key in ("auc", "first_hit_pct", "first_hit_rank"):
        m, sd = _agg(rows, key)
        out[f"mean_{key}"] = m
        out[f"sd_{key}"] = sd
        vals = sorted(r[key] for r in rows
                      if not (isinstance(r[key], float) and math.isnan(r[key])))
        out[f"median_{key}"] = vals[len(vals) // 2] if vals else float("nan")
    for k in KS:
        out[f"succ_dockq@{k}"] = sum(r[f"succ_dockq@{k}"] for r in rows) / len(rows)
        out[f"succ_rmsd@{k}"] = sum(r[f"succ_rmsd@{k}"] for r in rows) / len(rows)
        out[f"mean_best_dockq@{k}"], _ = _agg(rows, f"best_dockq@{k}")
    out["frac_complexes_with_positive"] = sum(r["n_pos"] > 0 for r in rows) / len(rows)
    out["mean_n_pos"], _ = _agg(rows, "n_pos")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default="data/scaling/runs", dest="runs_dir")
    ap.add_argument("--test-cache", default="data/shards_pinder/test_feats.pt",
                    dest="test_cache")
    ap.add_argument("--out-dir", default="data/scaling/eval_pool", dest="out_dir")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rmsd-threshold", type=float, default=5.0, dest="rmsd_thr")
    ap.add_argument("--dockq-threshold", type=float, default=0.23, dest="dockq_thr")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    blob = torch.load(args.test_cache, map_location="cpu", weights_only=True)
    feats = [{k: (v.to(device=device, dtype=dtype) if torch.is_tensor(v) else v)
              for k, v in d.items()} for d in blob]
    print(f"test complexes: {len(feats)}", flush=True)

    beta0 = torch.tensor(3.0, device=device, dtype=dtype)
    alpha0 = torch.tensor(0.01, device=device, dtype=dtype)
    iface0 = iface_ij(device=device, dtype=dtype, flat=True)

    jobs = [("baseline", None, {"n_fit": 0, "seed": -1, "round": -1})]
    for d in sorted(Path(args.runs_dir).glob("N*_seed*")):
        n = int(d.name.split("_")[0][1:])
        seed = int(d.name.split("seed")[1])
        for ck in sorted(d.glob("round*_ckpt.pt")):
            rnd = int(ck.stem.replace("round", "").replace("_ckpt", ""))
            jobs.append((f"{d.name}_round{rnd}", ck,
                         {"n_fit": n, "seed": seed, "round": rnd}))

    summaries = []
    for label, ck, meta in jobs:
        if ck is None:
            alpha, iface = alpha0, iface0
        else:
            blob_ck = torch.load(ck, map_location="cpu", weights_only=True)
            alpha = blob_ck["alpha"].to(device=device, dtype=dtype)
            iface = blob_ck["iface"].to(device=device, dtype=dtype)
        rows = [per_complex_metrics(f, alpha, iface, beta0, args.rmsd_thr,
                                    args.dockq_thr) for f in feats]
        cols = list(rows[0].keys())
        with open(out_dir / f"{label}_per_complex.csv", "w") as fh:
            fh.write(",".join(cols) + "\n")
            for r in rows:
                fh.write(",".join(str(r[c]) for c in cols) + "\n")
        s = summarise(rows, label, meta)
        s["alpha"] = float(alpha)
        s["d_iface_norm"] = float((iface - iface0).norm())
        summaries.append(s)
        print(f"{label:<26} AUC={s['mean_auc']:.4f}±{s['sd_auc']:.4f}  "
              f"median first-hit rank={s['median_first_hit_rank']:>5}  "
              f"succ@1={s['succ_dockq@1']*100:4.1f}%  "
              f"succ@100={s['succ_dockq@100']*100:4.1f}%", flush=True)

    cols = list(summaries[0].keys())
    with open(out_dir / "summary.csv", "w") as fh:
        fh.write(",".join(cols) + "\n")
        for s in summaries:
            fh.write(",".join(str(s.get(c, "")) for c in cols) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=1))
    print(f"\nwrote {out_dir}/summary.csv and per-complex tables")


if __name__ == "__main__":
    main()
