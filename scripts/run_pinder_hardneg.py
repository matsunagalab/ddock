"""PINDER deleaked experiment WITH iterative hard-negative mining.

Question: does hard-negative mining raise held-out (interface-deleaked)
top-K success over the single-shot decoy pool used in ``run_pinder.py``?

Idea. The score is an energy-based model; the negatives that matter are the
ones the *current* parameters rank highly but are not native. Each mining
round we re-run the FFT search with the current parameters over every TRAIN
complex — its high-scoring output *is* the current model's hard negatives —
label + featurize those poses, append them to that complex's pool (capping by
keeping all positives + the hardest current negatives), and retrain. The TEST
pool is fixed (single-shot, default-param decoys, reused from run_pinder's
cache) so the round-over-round comparison isolates the training effect.

Only per-pose features (S_SC, T[12,12], S_ELEC) + (RMSD, DockQ) are kept in
memory across rounds; pose coordinates are discarded right after featurizing,
so memory stays bounded even as the pool grows.

Example
-------
    PINDER_BASE_DIR=$PWD/external/pinder uv run python scripts/run_pinder_hardneg.py \
        --rounds 4 --epochs-per-round 1500 --device cuda
"""

from __future__ import annotations

import argparse
import copy
import glob
import os

import torch

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from zdock.atomtypes import charge_score as default_charge_score
from zdock.atomtypes import iface_ij
from zdock.data import list_proteins, load_training_dataset
from zdock.dataset import generate_decoys, label_decoys, prepare_protein_from_pdb
from zdock.evaluate import evaluate_ranking
from zdock.score import (docking_score_elec, iface_score_matrix,
                         SC_REFERENCE_SPACING)
from zdock.train import loss_basin, loss_margin_hard_negatives, loss_param_prior


class Feats:
    __slots__ = ("name", "sc", "T", "elec", "rmsd", "dockq")

    def __init__(self, name, sc, T, elec, rmsd, dockq):
        self.name = name
        self.sc = sc
        self.T = T
        self.elec = elec
        self.rmsd = rmsd
        self.dockq = dockq

    def cat(self, other):
        self.sc = torch.cat([self.sc, other.sc])
        self.T = torch.cat([self.T, other.T])
        self.elec = torch.cat([self.elec, other.elec])
        self.rmsd = torch.cat([self.rmsd, other.rmsd])
        self.dockq = torch.cat([self.dockq, other.dockq])

    def index(self, idx):
        return Feats(self.name, self.sc[idx], self.T[idx], self.elec[idx],
                     self.rmsd[idx], self.dockq[idx])


def score_from_feats(f, alpha, iface_flat, beta):
    imat = iface_score_matrix(iface_flat)
    return alpha * f.sc + (imat * f.T).sum(dim=(-2, -1)) + beta * f.elec


@torch.no_grad()
def featurize_poses(prot, poses, name, beta0, charge0, chunk,
                    spacing=SC_REFERENCE_SPACING):
    """Compute (parameter-independent) features + labels for arbitrary poses."""
    alpha_d = torch.zeros((), device=poses.device, dtype=poses.dtype)
    iface_d = iface_ij(device=poses.device, dtype=poses.dtype, flat=True)
    sc, T, elec = docking_score_elec(
        prot.rec_xyz, prot.rec_radius, prot.rec_sasa,
        prot.rec_atomtype_id, prot.rec_charge_id,
        poses, prot.lig_radius, prot.lig_sasa,
        prot.lig_atomtype_id, prot.lig_charge_id,
        alpha_d, iface_d, beta0, charge0,
        # PreparedProtein carries lig_ref; the shard-loaded ProteinInputs does
        # not, and its poses are already in the oriented frame.
        lig_xyz_for_grid=getattr(prot, "lig_ref", None), spacing=spacing,
        frame_chunk_size=chunk, return_components=True,
    )
    rmsd, dockq = label_decoys(prot, poses)
    return Feats(name, sc, T, elec, rmsd, dockq)


def cap_pool(f, cap, alpha, iface, beta, dockq_thr):
    """Keep the original positives + hardest current negatives.

    Mining rounds append *negatives only*, so the positive set remains fixed
    instead of growing by ~400 near-native cone poses every round.  This keeps
    both the class ratio and the multi-positive basin target stationary.
    """
    if f.sc.shape[0] <= cap:
        return f
    s = score_from_feats(f, alpha, iface, beta)
    pos = (f.dockq >= dockq_thr).nonzero(as_tuple=True)[0]
    neg = (f.dockq < dockq_thr).nonzero(as_tuple=True)[0]
    n_keep = max(0, cap - pos.numel())
    if n_keep and neg.numel():
        top = torch.topk(s[neg], min(n_keep, neg.numel())).indices
        neg = neg[top]
    else:
        neg = neg[:0]
    keep = torch.cat([pos, neg])
    return f.index(keep)


def normalized_scores(f, alpha, iface, beta):
    """Ranking-preserving per-complex score normalization for stable losses.

    Raw score standard deviations are typically 5e2--2e3 while the original
    basin temperature was 0.5 and margin was 1.0.  That makes InfoNCE nearly a
    discontinuous argmax and the margin scale negligible.  Centering and
    dividing by a detached positive standard deviation does not alter pose
    order, hence parameters trained here produce exactly the same raw ranking
    used by FFT search and evaluation.
    """
    s = score_from_feats(f, alpha, iface, beta)
    # `std()` of a 1-element tensor is NaN and `clamp_min` propagates it, so a
    # single one-pose complex would NaN the loss and Adam would permanently
    # poison alpha and iface. `unbiased=False` returns 0 there instead.
    scale = s.detach().std(unbiased=False).clamp_min(1.0)
    return (s - s.detach().mean()) / scale


def aggregate(feats_list, alpha, iface_flat, beta, ks, rmsd_thr, dockq_thr):
    n = len(feats_list)
    succ_r = {k: 0 for k in ks}
    succ_d = {k: 0 for k in ks}
    top1 = 0.0
    for f in feats_list:
        s = score_from_feats(f, alpha, iface_flat, beta)
        rep = evaluate_ranking(s, f.rmsd, f.dockq, ks=ks,
                               rmsd_threshold=rmsd_thr, dockq_threshold=dockq_thr)
        for k in ks:
            succ_r[k] += int(rep.success_rmsd[k])
            succ_d[k] += int(rep.success_dockq[k])
        top1 += rep.best_dockq_at[1]
    return ({k: succ_r[k] / n for k in ks},
            {k: succ_d[k] / n for k in ks}, top1 / n)


def _fmt(tag, sr, sd, t1, ks):
    out = [f"  [{tag}]  mean best-DockQ@top1 = {t1:.3f}",
           "     K   | success@K (RMSD<=thr) | success@K (DockQ>=thr)"]
    for k in ks:
        out.append(f"    {k:>4} |        {sr[k]*100:5.1f}%        |        {sd[k]*100:5.1f}%")
    return "\n".join(out)


def mean_objective(feats, alpha, iface, alpha0, iface0, beta0, args,
                   charge_dummy):
    total = torch.zeros((), device=alpha.device, dtype=alpha.dtype)
    for f in feats:
        s = normalized_scores(f, alpha, iface, beta0)
        # Forward the pool's own threshold (the losses default to 0.23).
        total = total + loss_basin(s, f.dockq, temperature=args.basin_temp,
                                   positive_threshold=args.dockq_thr)
        total = total + args.lambda_margin * loss_margin_hard_negatives(
            s, f.dockq, margin=args.margin, positive_threshold=args.dockq_thr)
    total = total / max(1, len(feats))
    return total + args.lambda_prior * loss_param_prior(
        alpha, iface, charge_dummy, alpha0, iface0, charge_dummy)


def train_params(train_feats, val_feats, alpha, iface, alpha0, iface0, beta0,
                 args, device, dtype, gen, optimizer_state=None):
    """Continue from current parameters, selecting a checkpoint on TRAIN-val."""
    alpha.requires_grad_(True)
    iface.requires_grad_(True)
    charge_dummy = torch.zeros(0, device=device, dtype=dtype)
    opt = torch.optim.Adam([
        {"params": [alpha], "lr": args.alpha_lr},
        {"params": [iface], "lr": args.iface_lr},
    ])
    if optimizer_state is not None:
        opt.load_state_dict(optimizer_state)
    bs = min(args.batch_size, len(train_feats))
    with torch.no_grad():
        best_val = float(mean_objective(
            val_feats, alpha, iface, alpha0, iface0, beta0, args,
            charge_dummy))
    best_params = (alpha.detach().clone(), iface.detach().clone())
    best_opt = copy.deepcopy(opt.state_dict())
    stale = 0
    for epoch in range(args.epochs_per_round):
        opt.zero_grad()
        batch = torch.randperm(len(train_feats), generator=gen)[:bs].tolist()
        total = mean_objective(
            [train_feats[i] for i in batch], alpha, iface, alpha0, iface0,
            beta0, args, charge_dummy)
        total.backward()
        torch.nn.utils.clip_grad_norm_([alpha, iface], args.grad_clip)
        opt.step()
        with torch.no_grad():
            alpha.clamp_(min=0.0, max=args.alpha_max)

        if epoch % args.val_every == 0 or epoch == args.epochs_per_round - 1:
            with torch.no_grad():
                val = float(mean_objective(
                    val_feats, alpha, iface, alpha0, iface0, beta0, args,
                    charge_dummy))
            if val < best_val - args.min_delta:
                best_val = val
                best_params = (alpha.detach().clone(), iface.detach().clone())
                best_opt = copy.deepcopy(opt.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= args.patience:
                    break

    with torch.no_grad():
        alpha.copy_(best_params[0])
        iface.copy_(best_params[1])
    if best_opt is not None:
        opt.load_state_dict(best_opt)
    return alpha.detach(), iface.detach(), best_val, epoch + 1, opt.state_dict()


def load_test_feats(patterns, beta0, charge0, device, dtype, chunk, cache):
    if cache and os.path.exists(cache):
        blob = torch.load(cache, map_location=device, weights_only=True)
        return [Feats(d["name"], d["sc"].to(dtype), d["T"].to(dtype),
                      d["elec"].to(dtype), d["rmsd"].to(dtype), d["dockq"].to(dtype))
                for d in blob]
    files = []
    for pat in patterns:
        files.extend(sorted(glob.glob(pat)))
    feats = []
    for path in files:
        names = list_proteins(path)
        prots = load_training_dataset(path, device=device, dtype=dtype)
        for name, prot in zip(names, prots):
            if prot.dockq is None:
                continue
            try:
                feats.append(featurize_poses(prot, prot.lig_xyz, name, beta0, charge0, chunk))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if cache:
        torch.save([{"name": f.name, "sc": f.sc, "T": f.T, "elec": f.elec,
                     "rmsd": f.rmsd, "dockq": f.dockq} for f in feats], cache)
    return feats


def prepare_train_prots(ids_file, device, dtype):
    """Re-prepare TRAIN complexes from PINDER (holo monomers) so we can re-run
    the FFT search each mining round."""
    from pinder.core import PinderSystem

    ids = [ln.strip() for ln in open(ids_file) if ln.strip()]
    prots = []
    for pid in ids:
        try:
            ps = PinderSystem(pid, pdb_engine="biotite")
            prot = prepare_protein_from_pdb(
                pid, str(ps.holo_receptor.filepath), str(ps.holo_ligand.filepath),
                device=device, dtype=dtype)
            prots.append(prot)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  [{pid}] SKIP prepare OOM", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{pid}] SKIP prepare ({type(exc).__name__})", flush=True)
    return prots


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-ids", default="data/pinder_train_ids.txt", dest="train_ids")
    ap.add_argument("--test-shards", nargs="+",
                    default=["data/shards_pinder/test_gpu*.h5"], dest="test_shards")
    ap.add_argument("--test-cache", default="data/shards_pinder/test_feats.pt",
                    dest="test_cache")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--epochs-per-round", type=int, default=1500, dest="epochs_per_round")
    ap.add_argument("--alpha-lr", type=float, default=1e-5, dest="alpha_lr")
    ap.add_argument("--iface-lr", type=float, default=5e-4, dest="iface_lr")
    # alpha0 is a CLI knob and alpha_max defaults to 10*alpha0 so the box
    # constraint can never sit below the initial value. The old hardcoded
    # (0.01, max 0.1) pair capped alpha a factor of 10 BELOW the 1.0 that
    # Chen et al. 2003 Eq. (2) implies, i.e. the optimum was outside the
    # feasible set and "training did not help" was confounded with it.
    ap.add_argument("--alpha0", type=float, default=0.01, dest="alpha0")
    ap.add_argument("--alpha-max", type=float, default=0.0, dest="alpha_max",
                    help="0 = 10 * alpha0")
    ap.add_argument("--batch-size", type=int, default=16, dest="batch_size")
    ap.add_argument("--lambda-margin", type=float, default=0.5, dest="lambda_margin")
    ap.add_argument("--lambda-prior", type=float, default=0.1, dest="lambda_prior")
    ap.add_argument("--basin-temperature", type=float, default=0.5, dest="basin_temp")
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--grad-clip", type=float, default=5.0, dest="grad_clip")
    ap.add_argument("--val-frac", type=float, default=0.2, dest="val_frac")
    ap.add_argument("--val-every", type=int, default=50, dest="val_every")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--min-delta", type=float, default=1e-4, dest="min_delta")
    ap.add_argument("--frame-chunk", type=int, default=400, dest="frame_chunk")
    # Must be the same for the search and the featuriser — see run_pinder_scaling.
    ap.add_argument("--spacing", type=float, default=SC_REFERENCE_SPACING)
    ap.add_argument("--rmsd-threshold", type=float, default=5.0, dest="rmsd_thr")
    ap.add_argument("--dockq-threshold", type=float, default=0.23, dest="dockq_thr")
    ap.add_argument("--pool-cap", type=int, default=4000, dest="pool_cap")
    ap.add_argument("--mine-random-rot", type=int, default=1500, dest="mine_random_rot")
    ap.add_argument("--mine-cone", type=int, default=400, dest="mine_cone")
    ap.add_argument("--mine-ntop", type=int, default=1500, dest="mine_ntop")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint-dir", default="data/shards_pinder/hardneg_checkpoints",
                    dest="checkpoint_dir")
    args = ap.parse_args()
    if args.alpha_max <= 0:
        args.alpha_max = 10.0 * args.alpha0
    assert args.alpha_max >= args.alpha0, (
        f"--alpha-max {args.alpha_max} is below --alpha0 {args.alpha0}: "
        "the initial value would be outside the feasible set")

    device = torch.device(args.device)
    dtype = torch.float64 if device.type == "cpu" else torch.float32
    ks = (1, 5, 10, 50, 100)
    gen = torch.Generator().manual_seed(args.seed)

    alpha0 = torch.tensor(args.alpha0, device=device, dtype=dtype)
    beta0 = torch.tensor(3.0, device=device, dtype=dtype)
    iface0 = iface_ij(device=device, dtype=dtype, flat=True)
    charge0 = default_charge_score(device=device, dtype=dtype)

    print("loading FIXED test pool ...", flush=True)
    test_feats = load_test_feats(args.test_shards, beta0, charge0, device, dtype,
                                 args.frame_chunk, args.test_cache)
    print(f"  test complexes: {len(test_feats)}", flush=True)

    print("preparing TRAIN complexes from PINDER ...", flush=True)
    train_prots = prepare_train_prots(args.train_ids, device, dtype)
    print(f"  train complexes: {len(train_prots)}\n", flush=True)
    ordered_names = sorted(p.name for p in train_prots)
    val_stride = max(2, round(1 / args.val_frac))
    val_names = set(ordered_names[::val_stride])
    print(f"  fixed split: fit={len(train_prots)-len(val_names)} "
          f"validation={len(val_names)}", flush=True)

    # Per-complex accumulated feature pools.
    pools: list[Feats] = []
    alpha, iface = alpha0.clone(), iface0.clone()
    history = []
    optimizer_state = None

    for rnd in range(args.rounds + 1):
        # Mine poses with the CURRENT parameters (round 0 == default params).
        n_skipped = 0
        new_pools = []
        for ci, prot in enumerate(train_prots):
            # Validation complexes get one fixed candidate set in round 0 and
            # are never mined or optimized. This gives a round-comparable,
            # test-independent checkpoint criterion.
            if rnd > 0 and prot.name in val_names:
                new_pools.append(None)
                continue
            try:
                poses, _ = generate_decoys(
                    prot, alpha=alpha, iface_ij_flat=iface, beta=beta0,
                    charge_score_lut=charge0,
                    n_random_rot=args.mine_random_rot, n_cone=args.mine_cone,
                    ntop=args.mine_ntop, seed=args.seed + rnd,
                    spacing=args.spacing,
                )
                nf = featurize_poses(prot, poses, prot.name, beta0, charge0,
                                     args.frame_chunk, spacing=args.spacing)
                del poses
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                n_skipped += 1
                new_pools.append(None)
                continue
            if device.type == "cuda":
                torch.cuda.empty_cache()
            new_pools.append(nf)

        if rnd == 0:
            pools = [p for p in new_pools if p is not None]
        else:
            existing = {p.name: p for p in pools}
            for nf in new_pools:
                if nf is None:
                    continue
                # This is hard-*negative* mining: do not repeatedly add the
                # explicit near-native cone positives generated for search.
                neg = (nf.dockq < args.dockq_thr).nonzero(as_tuple=True)[0]
                if neg.numel() == 0:
                    continue
                nf = nf.index(neg)
                if nf.name in existing:
                    existing[nf.name].cat(nf)
                    existing[nf.name] = cap_pool(existing[nf.name], args.pool_cap,
                                                 alpha, iface, beta0, args.dockq_thr)
                else:
                    existing[nf.name] = nf
            pools = list(existing.values())

        # Fixed TRAIN/validation split. TEST is never used for checkpoint or
        # round selection, and validation candidate pools never change.
        fit_pools = [f for f in pools if f.name not in val_names]
        val_pools = [f for f in pools if f.name in val_names]

        # Continue from the previous round's parameters (the old code reset to
        # ZDOCK defaults here, which made rounds independent random restarts).
        alpha, iface, val_loss, steps, optimizer_state = train_params(
            fit_pools, val_pools, alpha, iface, alpha0, iface0, beta0, args,
            device, dtype, gen, optimizer_state)

        sr, sd, t1 = aggregate(test_feats, alpha, iface, beta0, ks, args.rmsd_thr, args.dockq_thr)
        srt, sdt, t1t = aggregate(pools, alpha, iface, beta0, ks, args.rmsd_thr, args.dockq_thr)
        tag = "round 0 (no mining)" if rnd == 0 else f"round {rnd} (mined)"
        mean_pool = sum(p.sc.shape[0] for p in pools) / max(1, len(pools))
        print("=" * 62)
        print(f"{tag}  | mean pool size = {mean_pool:.0f}  | alpha={float(alpha):.4f} "
              f"||dIface||={float((iface-iface0).norm()):.2f}  "
              f"val_loss={val_loss:.4f} steps={steps}")
        print("=" * 62)
        print(_fmt("TEST(deleaked)", sr, sd, t1, ks))
        print(_fmt("TRAIN", srt, sdt, t1t, ks))
        print(flush=True)
        history.append((tag, sd[1], sd[10], sd[100]))
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        torch.save({
            "round": rnd,
            "seed": args.seed,
            "alpha": alpha.cpu(),
            "iface": iface.cpu(),
            "val_loss": val_loss,
            "history": history,
            "config": vars(args),
        }, os.path.join(args.checkpoint_dir, f"seed{args.seed}_round{rnd}.pt"))

    print("\n=== summary: DockQ success@K on held-out DELEAKED TEST ===")
    print(f"{'round':<22} {'top1':>7} {'top10':>7} {'top100':>7}")
    for tag, s1, s10, s100 in history:
        print(f"{tag:<22} {s1*100:6.1f}% {s10*100:6.1f}% {s100*100:6.1f}%")


if __name__ == "__main__":
    main()
