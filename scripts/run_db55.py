"""DB5.5 cross-complex generalization experiment.

Loads decoy datasets (one or more shard h5 files), splits by *complex*
into train / test, trains the SC weight (alpha) and the 12x12 IFACE
matrix (charge_score frozen at the ZDOCK default => the ELEC term is a
fixed per-pose feature), and reports top-K success rates on the held-out
*test complexes* — the honest generalization measure.

Speed: the score is linear in (alpha, iface) with fixed per-pose
features (score = alpha*S_SC + <iface, T> + beta*S_ELEC). We extract
(S_SC, T[12,12], S_ELEC) once per pose with a single grid pass, then
train on the cached features, so thousands of epochs over all complexes
run in seconds.

Example
-------
    uv run python scripts/run_db55.py --shards data/shards/shard*.h5 \
        --device cuda --epochs 2000
"""

from __future__ import annotations

import argparse
import glob

import torch

from zdock.atomtypes import charge_score as default_charge_score
from zdock.atomtypes import iface_ij
from zdock.data import list_proteins, load_training_dataset
from zdock.evaluate import evaluate_ranking
from zdock.score import docking_score_elec, iface_score_matrix
from zdock.train import (
    loss_basin,
    loss_margin_hard_negatives,
    loss_param_prior,
)


class Feats:
    """Cached per-pose features + labels for one complex."""

    __slots__ = ("name", "sc", "T", "elec", "rmsd", "dockq")

    def __init__(self, name, sc, T, elec, rmsd, dockq):
        self.name = name
        self.sc = sc          # (F,)
        self.T = T            # (F, 12, 12)
        self.elec = elec      # (F,)
        self.rmsd = rmsd      # (F,)
        self.dockq = dockq    # (F,)


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


def aggregate(feats_list, alpha, iface_flat, beta, ks, rmsd_thr, dockq_thr):
    """Return dict k -> (success_rmsd_rate, success_dockq_rate) over complexes,
    plus mean best-DockQ@1."""
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
    ap.add_argument("--shards", nargs="+", default=["data/shards/shard*.h5"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--test-frac", type=float, default=0.35, dest="test_frac")
    ap.add_argument("--rmsd-threshold", type=float, default=5.0, dest="rmsd_thr")
    ap.add_argument("--dockq-threshold", type=float, default=0.23, dest="dockq_thr")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frame-chunk", type=int, default=400, dest="frame_chunk")
    ap.add_argument("--lambda-margin", type=float, default=0.5, dest="lambda_margin")
    ap.add_argument("--lambda-prior", type=float, default=0.02, dest="lambda_prior")
    ap.add_argument("--basin-temperature", type=float, default=0.5, dest="basin_temp")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float64 if device.type == "cpu" else torch.float32
    ks = (1, 5, 10, 50, 100)

    files = []
    for pat in args.shards:
        files.extend(sorted(glob.glob(pat)))
    if not files:
        raise SystemExit(f"no shard files matched {args.shards}")

    alpha0 = torch.tensor(0.01, device=device, dtype=dtype)
    beta0 = torch.tensor(3.0, device=device, dtype=dtype)
    iface0 = iface_ij(device=device, dtype=dtype, flat=True)
    charge0 = default_charge_score(device=device, dtype=dtype)

    print(f"featurizing complexes from {len(files)} shard file(s) ...", flush=True)
    feats: list[Feats] = []
    from zdock.train import ProteinInputs  # noqa: F401
    for path in files:
        names = list_proteins(path)
        prots = load_training_dataset(path, device=device, dtype=dtype)
        for name, prot in zip(names, prots):
            if prot.dockq is None:
                continue
            try:
                f = featurize(prot, name, alpha0, iface0, beta0, charge0, args.frame_chunk)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  [{name}] SKIP featurize OOM", flush=True)
                continue
            feats.append(f)
            del prot
            if device.type == "cuda":
                torch.cuda.empty_cache()
    print(f"featurized {len(feats)} complexes\n", flush=True)

    # Complex-level train/test split.
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(feats), generator=g).tolist()
    n_test = int(len(feats) * args.test_frac)
    test_idx = set(perm[:n_test])
    train_feats = [f for i, f in enumerate(feats) if i not in test_idx]
    test_feats = [f for i, f in enumerate(feats) if i in test_idx]
    print(f"train complexes: {len(train_feats)}   test complexes: {len(test_feats)}\n")

    # Baseline (default ZDOCK) on test.
    sr, sd, t1 = aggregate(test_feats, alpha0, iface0, beta0, ks,
                           args.rmsd_thr, args.dockq_thr)
    print("=" * 60)
    print("BASELINE (default ZDOCK) — held-out TEST complexes")
    print("=" * 60)
    print(_fmt("baseline", sr, sd, t1, ks))
    print()

    # Train alpha + iface on cached features (charge frozen).
    alpha = alpha0.clone().detach().requires_grad_(True)
    iface = iface0.clone().detach().requires_grad_(True)
    alpha_init = alpha0.clone()
    iface_init = iface0.clone()
    charge_dummy = torch.zeros(0, device=device, dtype=dtype)
    opt = torch.optim.Adam([alpha, iface], lr=args.lr)

    bs = 16
    print("=" * 60)
    print(f"TRAINING  epochs={args.epochs}  lr={args.lr}  "
          f"(basin + {args.lambda_margin}*margin + {args.lambda_prior}*prior)")
    print("=" * 60)
    idx_all = list(range(len(train_feats)))
    for epoch in range(args.epochs):
        opt.zero_grad()
        batch = torch.randperm(len(train_feats), generator=g)[:bs].tolist()
        total = torch.zeros((), device=device, dtype=dtype)
        for i in batch:
            f = train_feats[i]
            s = score_from_feats(f, alpha, iface, beta0)
            lb = loss_basin(s, f.dockq, temperature=args.basin_temp)
            lm = loss_margin_hard_negatives(s, f.dockq)
            total = total + lb + args.lambda_margin * lm
        total = total / len(batch)
        total = total + args.lambda_prior * loss_param_prior(
            alpha, iface, charge_dummy, alpha_init, iface_init, charge_dummy,
        )
        total.backward()
        opt.step()
        if epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:5d}  loss={float(total):.4f}", flush=True)

    print(f"\ntrained alpha={float(alpha):.4f}  "
          f"||Δiface||={float((iface-iface0).norm()):.3f}\n")

    # Eval trained params on test.
    sr, sd, t1 = aggregate(test_feats, alpha.detach(), iface.detach(), beta0, ks,
                           args.rmsd_thr, args.dockq_thr)
    print("=" * 60)
    print("TRAINED — held-out TEST complexes")
    print("=" * 60)
    print(_fmt("trained", sr, sd, t1, ks))
    print()

    # Also report train-set success for reference (overfitting check).
    sr, sd, t1 = aggregate(train_feats, alpha.detach(), iface.detach(), beta0, ks,
                           args.rmsd_thr, args.dockq_thr)
    print(_fmt("trained/TRAIN-set", sr, sd, t1, ks))


if __name__ == "__main__":
    main()
