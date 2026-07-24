"""Train the 156 ZDOCK parameters on FFT-generated decoys and measure the
improvement in top-K ranking quality.

For a single complex we split the candidate poses into train / val sets,
train on the train poses, and evaluate ranking on the **held-out** val
poses. This shows the learned parameters produce a better ranking that
generalizes to poses not seen during training (a valid sanity result;
cross-complex generalization needs a multi-complex benchmark — see the
report at the end of the run).

Example
-------
    uv run python scripts/run_experiment.py --dataset data/decoys.h5 \
        --loss combined --epochs 300 --device cuda
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from zdock.atomtypes import charge_score as default_charge_score
from zdock.atomtypes import iface_ij
from zdock.data import list_proteins, load_training_dataset
from zdock.evaluate import evaluate_ranking, format_report, score_poses
from zdock.train import ProteinInputs, train


def _slice_poses(p: ProteinInputs, idx: torch.Tensor) -> ProteinInputs:
    return replace(
        p,
        lig_xyz=p.lig_xyz[idx],
        hit_mask=p.hit_mask[idx],
        rmsd=None if p.rmsd is None else p.rmsd[idx],
        dockq=None if p.dockq is None else p.dockq[idx],
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="data/decoys.h5")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--loss", default="combined")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--val-frac", type=float, default=0.4, dest="val_frac")
    ap.add_argument("--rmsd-threshold", type=float, default=5.0, dest="rmsd_threshold")
    ap.add_argument("--dockq-threshold", type=float, default=0.23, dest="dockq_threshold")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frame-chunk-size", type=int, default=256, dest="frame_chunk")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float64 if device.type == "cpu" else torch.float32
    torch.manual_seed(args.seed)

    names = list_proteins(args.dataset)
    proteins = load_training_dataset(args.dataset, device=device, dtype=dtype)
    print(f"loaded {len(proteins)} protein(s): {names}\n")

    # Default ZDOCK params (baseline scorer).
    alpha0 = torch.tensor(0.01, device=device, dtype=dtype)
    beta0 = torch.tensor(3.0, device=device, dtype=dtype)
    iface0 = iface_ij(device=device, dtype=dtype, flat=True)
    charge0 = default_charge_score(device=device, dtype=dtype)

    train_set: list[ProteinInputs] = []
    val_set: list[tuple[str, ProteinInputs]] = []
    for name, p in zip(names, proteins):
        F = p.lig_xyz.shape[0]
        perm = torch.randperm(F, device=device)
        n_val = int(F * args.val_frac)
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        train_set.append(_slice_poses(p, train_idx))
        val_set.append((name, _slice_poses(p, val_idx)))

    print("=" * 68)
    print("BASELINE (default ZDOCK parameters) — held-out val poses")
    print("=" * 68)
    for name, pv in val_set:
        s = score_poses(pv, alpha0, iface0, beta0, charge0,
                        frame_chunk_size=args.frame_chunk)
        rep = evaluate_ranking(s, pv.rmsd, pv.dockq,
                               rmsd_threshold=args.rmsd_threshold,
                               dockq_threshold=args.dockq_threshold)
        print(format_report(name + " [baseline]", rep))
        print()

    print("=" * 68)
    print(f"TRAINING  loss={args.loss}  epochs={args.epochs}  lr={args.lr}")
    print("=" * 68)
    out = train(
        train_set, n_epoch=args.epochs, lr=args.lr, device=device, dtype=dtype,
        progress_every=max(1, args.epochs // 10), loss=args.loss,
        frame_chunk_size=args.frame_chunk,
    )
    alpha_t = out["alpha"]
    iface_t = out["iface"]
    charge_t = out["charge"]
    print(f"\ntrained α={float(alpha_t):.4f}  "
          f"‖Δiface‖={float((iface_t - iface0).norm()):.3f}  "
          f"‖Δcharge‖={float((charge_t - charge0).norm()):.3f}\n")

    print("=" * 68)
    print("TRAINED parameters — held-out val poses")
    print("=" * 68)
    for name, pv in val_set:
        s = score_poses(pv, alpha_t, iface_t, beta0, charge_t,
                        frame_chunk_size=args.frame_chunk)
        rep = evaluate_ranking(s, pv.rmsd, pv.dockq,
                               rmsd_threshold=args.rmsd_threshold,
                               dockq_threshold=args.dockq_threshold)
        print(format_report(name + " [trained]", rep))
        print()


if __name__ == "__main__":
    main()
