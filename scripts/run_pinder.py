"""PINDER interface-deleaked generalization experiment.

Unlike :mod:`scripts.run_db55`, which pools all complexes and takes a *random*
complex-level split (so train and test can share near-identical interfaces),
this script keeps PINDER's own **interface-deleaked** split: ``--train-shards``
come from sampled ``train`` clusters and ``--test-shards`` are the held-out
``pinder_s`` test cluster representatives. PINDER guarantees no FoldSeek/MMseqs
interface-cluster (and iAlign) leakage between the two, so success rates here
are the honest "unseen interface" number — directly comparable to the optimistic
random-split figure to quantify how much of it was leakage.

Same fast path as DB5.5: the score is linear in ``(alpha, iface)`` given fixed
per-pose features, so we extract ``(S_SC, T[12,12], S_ELEC)`` once and train on
the cached features.

Example
-------
    uv run python scripts/run_pinder.py \
        --train-shards 'data/shards_pinder/train_gpu*.h5' \
        --test-shards  'data/shards_pinder/test_gpu*.h5' \
        --device cuda --epochs 3000
"""

from __future__ import annotations

import argparse
import glob
import os

import torch

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from zdock.atomtypes import charge_score as default_charge_score
from zdock.atomtypes import iface_ij
from zdock.data import list_proteins, load_training_dataset
from zdock.evaluate import evaluate_ranking
from zdock.score import docking_score_elec, iface_score_matrix
from zdock.train import loss_basin, loss_margin_hard_negatives, loss_param_prior


class Feats:
    """Cached per-pose features + labels for one complex."""

    __slots__ = ("name", "sc", "T", "elec", "rmsd", "dockq")

    def __init__(self, name, sc, T, elec, rmsd, dockq):
        self.name = name
        self.sc = sc
        self.T = T
        self.elec = elec
        self.rmsd = rmsd
        self.dockq = dockq


def score_from_feats(f: Feats, alpha, iface_flat, beta) -> torch.Tensor:
    imat = iface_score_matrix(iface_flat)
    return alpha * f.sc + (imat * f.T).sum(dim=(-2, -1)) + beta * f.elec


@torch.no_grad()
def featurize(prot, name, alpha0, iface0, beta0, charge0, chunk) -> Feats:
    sc, T, elec = docking_score_elec(
        prot.rec_xyz, prot.rec_radius, prot.rec_sasa,
        prot.rec_atomtype_id, prot.rec_charge_id,
        prot.lig_xyz, prot.lig_radius, prot.lig_sasa,
        prot.lig_atomtype_id, prot.lig_charge_id,
        alpha0, iface0, beta0, charge0,
        frame_chunk_size=chunk, return_components=True,
    )
    return Feats(name, sc, T, elec, prot.rmsd, prot.dockq)


def _featurize_shards(patterns, alpha0, iface0, beta0, charge0, device, dtype, chunk):
    files = []
    for pat in patterns:
        files.extend(sorted(glob.glob(pat)))
    if not files:
        raise SystemExit(f"no shard files matched {patterns}")
    feats: list[Feats] = []
    for path in files:
        names = list_proteins(path)
        prots = load_training_dataset(path, device=device, dtype=dtype)
        for name, prot in zip(names, prots):
            if prot.dockq is None:
                continue
            try:
                feats.append(featurize(prot, name, alpha0, iface0, beta0, charge0, chunk))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  [{name}] SKIP featurize OOM", flush=True)
                continue
            del prot
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return feats


def load_feats(patterns, alpha0, iface0, beta0, charge0, device, dtype, chunk,
               cache=None):
    """Featurize shards, caching the (parameter-independent) per-pose features
    to ``cache`` so hyper-parameter sweeps skip the expensive grid pass."""
    if cache and os.path.exists(cache):
        blob = torch.load(cache, map_location=device)
        return [Feats(d["name"], d["sc"].to(dtype), d["T"].to(dtype),
                      d["elec"].to(dtype), d["rmsd"].to(dtype), d["dockq"].to(dtype))
                for d in blob]
    feats = _featurize_shards(patterns, alpha0, iface0, beta0, charge0,
                              device, dtype, chunk)
    if cache:
        torch.save([{"name": f.name, "sc": f.sc, "T": f.T, "elec": f.elec,
                     "rmsd": f.rmsd, "dockq": f.dockq} for f in feats], cache)
    return feats


def aggregate(feats_list, alpha, iface_flat, beta, ks, rmsd_thr, dockq_thr):
    n = len(feats_list)
    succ_r = {k: 0 for k in ks}
    succ_d = {k: 0 for k in ks}
    top1_dockq = 0.0
    for f in feats_list:
        s = score_from_feats(f, alpha, iface_flat, beta)
        rep = evaluate_ranking(s, f.rmsd, f.dockq, ks=ks,
                               rmsd_threshold=rmsd_thr, dockq_threshold=dockq_thr)
        for k in ks:
            succ_r[k] += int(rep.success_rmsd[k])
            succ_d[k] += int(rep.success_dockq[k])
        top1_dockq += rep.best_dockq_at[1]
    return (
        {k: succ_r[k] / n for k in ks},
        {k: succ_d[k] / n for k in ks},
        top1_dockq / n,
    )


def _fmt(tag, sr, sd, t1, ks):
    line = [f"  [{tag}]  mean best-DockQ@top1 = {t1:.3f}"]
    line.append("     K   | success@K (RMSD<=thr) | success@K (DockQ>=thr)")
    for k in ks:
        line.append(f"    {k:>4} |        {sr[k]*100:5.1f}%        |        {sd[k]*100:5.1f}%")
    return "\n".join(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-shards", nargs="+",
                    default=["data/shards_pinder/train_gpu*.h5"], dest="train_shards")
    ap.add_argument("--test-shards", nargs="+",
                    default=["data/shards_pinder/test_gpu*.h5"], dest="test_shards")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--rmsd-threshold", type=float, default=5.0, dest="rmsd_thr")
    ap.add_argument("--dockq-threshold", type=float, default=0.23, dest="dockq_thr")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frame-chunk", type=int, default=400, dest="frame_chunk")
    ap.add_argument("--train-cache", default="data/shards_pinder/train_feats.pt",
                    dest="train_cache")
    ap.add_argument("--test-cache", default="data/shards_pinder/test_feats.pt",
                    dest="test_cache")
    ap.add_argument("--batch-size", type=int, default=16, dest="batch_size")
    ap.add_argument("--lambda-margin", type=float, default=0.5, dest="lambda_margin")
    ap.add_argument("--lambda-prior", type=float, default=0.02, dest="lambda_prior")
    ap.add_argument("--basin-temperature", type=float, default=0.5, dest="basin_temp")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float64 if device.type == "cpu" else torch.float32
    ks = (1, 5, 10, 50, 100)

    alpha0 = torch.tensor(0.01, device=device, dtype=dtype)
    beta0 = torch.tensor(3.0, device=device, dtype=dtype)
    iface0 = iface_ij(device=device, dtype=dtype, flat=True)
    charge0 = default_charge_score(device=device, dtype=dtype)

    print("featurizing TRAIN complexes ...", flush=True)
    train_feats = load_feats(args.train_shards, alpha0, iface0, beta0, charge0,
                             device, dtype, args.frame_chunk, cache=args.train_cache)
    print(f"  train complexes: {len(train_feats)}", flush=True)
    print("featurizing TEST complexes ...", flush=True)
    test_feats = load_feats(args.test_shards, alpha0, iface0, beta0, charge0,
                            device, dtype, args.frame_chunk, cache=args.test_cache)
    print(f"  test complexes:  {len(test_feats)}\n", flush=True)

    sr, sd, t1 = aggregate(test_feats, alpha0, iface0, beta0, ks,
                           args.rmsd_thr, args.dockq_thr)
    srb, sdb, t1b = aggregate(train_feats, alpha0, iface0, beta0, ks,
                              args.rmsd_thr, args.dockq_thr)
    print("=" * 62)
    print("BASELINE (default ZDOCK)")
    print("=" * 62)
    print(_fmt("baseline/TEST(deleaked)", sr, sd, t1, ks))
    print(_fmt("baseline/TRAIN", srb, sdb, t1b, ks))
    print()

    alpha = alpha0.clone().detach().requires_grad_(True)
    iface = iface0.clone().detach().requires_grad_(True)
    alpha_init = alpha0.clone()
    iface_init = iface0.clone()
    charge_dummy = torch.zeros(0, device=device, dtype=dtype)
    opt = torch.optim.Adam([alpha, iface], lr=args.lr)

    g = torch.Generator().manual_seed(args.seed)
    bs = min(args.batch_size, len(train_feats))
    print("=" * 62)
    print(f"TRAINING  epochs={args.epochs}  lr={args.lr}  bs={bs}  "
          f"(basin + {args.lambda_margin}*margin + {args.lambda_prior}*prior)")
    print("=" * 62)
    for epoch in range(args.epochs):
        opt.zero_grad()
        batch = torch.randperm(len(train_feats), generator=g)[:bs].tolist()
        total = torch.zeros((), device=device, dtype=dtype)
        for i in batch:
            f = train_feats[i]
            s = score_from_feats(f, alpha, iface, beta0)
            total = total + loss_basin(s, f.dockq, temperature=args.basin_temp)
            total = total + args.lambda_margin * loss_margin_hard_negatives(s, f.dockq)
        total = total / len(batch)
        total = total + args.lambda_prior * loss_param_prior(
            alpha, iface, charge_dummy, alpha_init, iface_init, charge_dummy,
        )
        total.backward()
        opt.step()
        if epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:5d}  loss={float(total):.4f}", flush=True)

    print(f"\ntrained alpha={float(alpha):.4f}  "
          f"||Δiface||={float((iface - iface0).norm()):.3f}\n")

    sr, sd, t1 = aggregate(test_feats, alpha.detach(), iface.detach(), beta0, ks,
                           args.rmsd_thr, args.dockq_thr)
    print("=" * 62)
    print("TRAINED — held-out DELEAKED TEST (PINDER-S)")
    print("=" * 62)
    print(_fmt("trained", sr, sd, t1, ks))
    print()

    sr, sd, t1 = aggregate(train_feats, alpha.detach(), iface.detach(), beta0, ks,
                           args.rmsd_thr, args.dockq_thr)
    print(_fmt("trained/TRAIN-set", sr, sd, t1, ks))


if __name__ == "__main__":
    main()
