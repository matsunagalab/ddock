"""Interface-cluster **scaling law** for hard-negative mining on the PINDER
interface-deleaked split (EXPERIMENT_REPORT §5.6).

Question
--------
§5.5 found that hard-negative mining with 220 fit interface clusters produced
*zero* improvement on the held-out PINDER-S test set, and §5.5.1 listed
"not enough independent interfaces" as the leading (but unproven) explanation:
145 learnable parameters (alpha + 12x12 IFACE) against 220 clusters. This
script measures held-out performance and mining gain as a function of the
number of independent fit interface clusters N (220 / 500 / 1,000 / 2,000) at
fixed everything-else, so that a rising curve supports the data-limited
hypothesis and an early plateau supports the capacity/loss-limited one.

Streaming design (what changed vs ``run_pinder_hardneg.py``)
-----------------------------------------------------------
The old script held every ``PreparedProtein`` *and* every per-pose feature pool
on the GPU, which does not fit past a few hundred complexes. Here:

* prepared complexes live in a CPU **disk cache** (:mod:`zdock.prep_cache`),
  built once by ``scripts/prep_pinder_cache.py`` and shared by every seed;
* mining pages **one complex at a time** onto the GPU, runs the FFT search,
  labels + featurizes the poses, moves the ``(S_SC, T, S_ELEC, RMSD, DockQ)``
  features to CPU and frees the protein and the pose coordinates immediately;
* the feature pool therefore lives in **host** memory (~1-2 MB per complex);
* training moves only the current mini-batch to the GPU; validation and test
  evaluation stream one complex at a time as well;
* CUDA OOM is caught per stage, retried with halved chunk sizes, and finally
  recorded (id, atom counts, stage) and skipped rather than killing the run.

Scientific conditions are inherited unchanged from the stabilized §5.5 run:
parameters and Adam state carry over between rounds, scores are standardized
per complex *inside the loss only* (a positive affine map, so pose ranking and
therefore every reported metric is unchanged), ``alpha_lr=1e-5``,
``iface_lr=5e-4``, gradient clipping, ``0 <= alpha <= 0.1``, the positive set is
frozen at round 0, mining appends only ``DockQ < 0.23`` negatives, the
validation candidate pools are frozen at round 0 and never mined, early
stopping uses the fixed validation loss, and TEST is never used to pick a
checkpoint, a round, or a hyper-parameter.

Example
-------
    PINDER_BASE_DIR=$PWD/external/pinder CUDA_VISIBLE_DEVICES=0 \\
    uv run python -u scripts/run_pinder_scaling.py --n-fit 500 --seed 0 --rounds 1
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
# Must be set before the CUDA context is created. The FFT search allocates a
# few large, short-lived grids per complex while the run keeps a long-lived
# pool, which fragments the default caching allocator badly enough that a
# 1.4 GiB request failed on a 47 GiB card. Expandable segments let the
# allocator grow a single region instead of hoarding fixed-size blocks.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

from zdock.atomtypes import charge_score as default_charge_score
from zdock.atomtypes import iface_ij
from zdock.dataset import generate_decoys, label_decoys
from zdock.evaluate import evaluate_ranking
from zdock.prep_cache import load_prepared
from zdock.score import docking_score_elec
from zdock.train import loss_basin, loss_margin_hard_negatives, loss_param_prior

KS = (1, 5, 10, 50, 100)


# --------------------------------------------------------------------------
# per-complex feature pool (host-resident)
# --------------------------------------------------------------------------
class Feats:
    """Per-pose features + labels for one complex, normally on CPU.

    ``origin`` records the mining round each pose came from (0 = the round-0
    default-parameter candidate set), which is what lets us report pool
    composition (positives / random negatives / hard negatives) and assert that
    mining never grows the positive set.
    """

    __slots__ = ("name", "sc", "T", "elec", "rmsd", "dockq", "origin")

    def __init__(self, name, sc, T, elec, rmsd, dockq, origin):
        self.name = name
        self.sc = sc
        self.T = T
        self.elec = elec
        self.rmsd = rmsd
        self.dockq = dockq
        self.origin = origin

    @property
    def n(self) -> int:
        return int(self.sc.shape[0])

    def to(self, device, non_blocking: bool = False) -> "Feats":
        if self.sc.device == torch.device(device):
            return self
        return Feats(self.name,
                     self.sc.to(device, non_blocking=non_blocking),
                     self.T.to(device, non_blocking=non_blocking),
                     self.elec.to(device, non_blocking=non_blocking),
                     self.rmsd.to(device, non_blocking=non_blocking),
                     self.dockq.to(device, non_blocking=non_blocking),
                     self.origin.to(device, non_blocking=non_blocking))

    def cat(self, other: "Feats") -> None:
        self.sc = torch.cat([self.sc, other.sc])
        self.T = torch.cat([self.T, other.T])
        self.elec = torch.cat([self.elec, other.elec])
        self.rmsd = torch.cat([self.rmsd, other.rmsd])
        self.dockq = torch.cat([self.dockq, other.dockq])
        self.origin = torch.cat([self.origin, other.origin])

    def index(self, idx) -> "Feats":
        return Feats(self.name, self.sc[idx], self.T[idx], self.elec[idx],
                     self.rmsd[idx], self.dockq[idx], self.origin[idx])

    def counts(self, dockq_thr: float) -> dict:
        pos = self.dockq >= dockq_thr
        r0 = self.origin == 0
        return {"n": self.n,
                "n_pos": int(pos.sum()),
                "n_rand_neg": int((~pos & r0).sum()),
                "n_hard_neg": int((~pos & ~r0).sum())}


def score_from_feats(f: Feats, alpha, iface_flat, beta) -> torch.Tensor:
    imat = iface_flat.view(12, 12).T
    return alpha * f.sc + (imat * f.T).sum(dim=(-2, -1)) + beta * f.elec


def normalized_scores(f: Feats, alpha, iface, beta) -> torch.Tensor:
    """Ranking-preserving per-complex standardization, used *only* inside the
    loss (see §5.5: raw score std is 5e2-2e3 while the basin temperature is
    0.5). Centering and dividing by a detached positive scalar cannot change
    pose order, so the trained parameters rank exactly as the raw score does."""
    s = score_from_feats(f, alpha, iface, beta)
    scale = s.detach().std().clamp_min(1.0)
    return (s - s.detach().mean()) / scale


# --------------------------------------------------------------------------
# streaming mining: one complex at a time on the GPU
# --------------------------------------------------------------------------
def _adaptive_pose_chunk(n_rec: int, n_lig: int, budget_elems: int) -> int:
    """DockQ builds a dense (chunk, N_rec, N_lig) tensor; keep it bounded."""
    per = max(1, n_rec * n_lig)
    return int(max(1, min(64, budget_elems // per)))


@torch.no_grad()
def mine_complex(prot, alpha, iface, beta0, charge0, args, round_idx: int,
                 *, rot_chunk: int | None = None, frame_chunk: int | None = None,
                 pose_chunk: int | None = None):
    """FFT-search + label + featurize one complex; return host-side ``Feats``.

    Raises ``torch.cuda.OutOfMemoryError`` if even the smallest chunk sizes do
    not fit — the caller records it for the round's rescue pass and, if that
    also fails, skips the complex.
    """
    device = prot.rec_xyz.device
    dtype = prot.rec_xyz.dtype
    rot_chunk = args.rot_chunk if rot_chunk is None else rot_chunk
    frame_chunk = args.frame_chunk if frame_chunk is None else frame_chunk
    if pose_chunk is None:
        pose_chunk = _adaptive_pose_chunk(prot.n_rec, prot.n_lig, args.dockq_budget)

    last_exc = None
    for attempt in range(args.oom_retries + 1):
        try:
            poses, _ = generate_decoys(
                prot, alpha=alpha, iface_ij_flat=iface, beta=beta0,
                charge_score_lut=charge0,
                n_random_rot=args.mine_random_rot, n_cone=args.mine_cone,
                ntop=args.mine_ntop, seed=args.seed + 1000 * round_idx,
                rot_chunk_size=rot_chunk,
            )
            alpha_d = torch.zeros((), device=device, dtype=dtype)
            iface_d = iface_ij(device=device, dtype=dtype, flat=True)
            sc, T, elec = docking_score_elec(
                prot.rec_xyz, prot.rec_radius, prot.rec_sasa,
                prot.rec_atomtype_id, prot.rec_charge_id,
                poses, prot.lig_radius, prot.lig_sasa,
                prot.lig_atomtype_id, prot.lig_charge_id,
                alpha_d, iface_d, beta0, charge0,
                frame_chunk_size=frame_chunk, return_components=True,
            )
            rmsd, dockq = label_decoys(prot, poses, pose_chunk=pose_chunk)
            out = Feats(prot.name, sc.cpu(), T.cpu(), elec.cpu(),
                        rmsd.cpu(), dockq.cpu(),
                        torch.full((sc.shape[0],), round_idx, dtype=torch.int16))
            del poses, sc, T, elec, rmsd, dockq
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return out
        except torch.cuda.OutOfMemoryError as exc:
            last_exc = exc
            if device.type == "cuda":
                torch.cuda.empty_cache()
            rot_chunk = max(1, rot_chunk // 2)
            frame_chunk = max(25, frame_chunk // 2)
            pose_chunk = max(1, pose_chunk // 2)
            args_retry = attempt + 1
            print(f"    [{prot.name}] OOM (attempt {args_retry}) -> retry with "
                  f"rot_chunk={rot_chunk} frame_chunk={frame_chunk} "
                  f"pose_chunk={pose_chunk}", flush=True)
    raise last_exc


# --------------------------------------------------------------------------
# pool bookkeeping
# --------------------------------------------------------------------------
def cap_pool(f: Feats, cap: int, alpha, iface, beta, dockq_thr: float) -> Feats:
    """Keep every positive + the hardest (highest-scoring) current negatives."""
    if f.n <= cap:
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
    return f.index(torch.cat([pos, neg]))


# --------------------------------------------------------------------------
# evaluation (streams complexes to the GPU one at a time)
# --------------------------------------------------------------------------
@torch.no_grad()
def aggregate(feats_list, alpha, iface_flat, beta, device, rmsd_thr, dockq_thr):
    n = max(1, len(feats_list))
    succ_r = {k: 0 for k in KS}
    succ_d = {k: 0 for k in KS}
    top1 = 0.0
    for f in feats_list:
        g = f.to(device)
        s = score_from_feats(g, alpha, iface_flat, beta)
        rep = evaluate_ranking(s, g.rmsd, g.dockq, ks=KS,
                               rmsd_threshold=rmsd_thr, dockq_threshold=dockq_thr)
        for k in KS:
            succ_r[k] += int(rep.success_rmsd[k])
            succ_d[k] += int(rep.success_dockq[k])
        top1 += float(rep.best_dockq_at[1])
        del g, s
    return ({k: succ_r[k] / n for k in KS},
            {k: succ_d[k] / n for k in KS}, top1 / n)


def _fmt(tag, sr, sd, t1):
    out = [f"  [{tag}]  mean best-DockQ@top1 = {t1:.3f}",
           "     K   | success@K (RMSD<=thr) | success@K (DockQ>=thr)"]
    for k in KS:
        out.append(f"    {k:>4} |        {sr[k]*100:5.1f}%        |        {sd[k]*100:5.1f}%")
    return "\n".join(out)


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def mean_objective(feats, alpha, iface, alpha0, iface0, beta0, args,
                   charge_dummy, device):
    total = torch.zeros((), device=alpha.device, dtype=alpha.dtype)
    for f in feats:
        g = f.to(device)
        s = normalized_scores(g, alpha, iface, beta0)
        total = total + loss_basin(s, g.dockq, temperature=args.basin_temp)
        total = total + args.lambda_margin * loss_margin_hard_negatives(
            s, g.dockq, margin=args.margin)
    total = total / max(1, len(feats))
    return total + args.lambda_prior * loss_param_prior(
        alpha, iface, charge_dummy, alpha0, iface0, charge_dummy)


@torch.no_grad()
def _val_loss(val_feats, alpha, iface, alpha0, iface0, beta0, args,
              charge_dummy, device):
    return float(mean_objective(val_feats, alpha, iface, alpha0, iface0, beta0,
                                args, charge_dummy, device))


def train_params(fit_feats, val_feats, alpha, iface, alpha0, iface0, beta0,
                 args, device, dtype, gen, n_steps, val_every,
                 optimizer_state=None):
    """Continue from the current parameters; select a checkpoint on the fixed
    validation loss only. Returns the trajectory for the report."""
    alpha.requires_grad_(True)
    iface.requires_grad_(True)
    charge_dummy = torch.zeros(0, device=device, dtype=dtype)
    opt = torch.optim.Adam([
        {"params": [alpha], "lr": args.alpha_lr},
        {"params": [iface], "lr": args.iface_lr},
    ])
    if optimizer_state is not None:
        opt.load_state_dict(optimizer_state)
    bs = min(args.batch_size, len(fit_feats))

    best_val = _val_loss(val_feats, alpha, iface, alpha0, iface0, beta0, args,
                         charge_dummy, device)
    best_params = (alpha.detach().clone(), iface.detach().clone())
    best_opt = copy.deepcopy(opt.state_dict())
    traj = [{"step": 0, "fit_loss": float("nan"), "val_loss": best_val,
             "grad_norm": float("nan"), "accepted": 1,
             "alpha": float(alpha), "d_iface": float((iface - iface0).norm())}]
    stale = 0
    step = 0
    grad_norms = []
    for step in range(n_steps):
        opt.zero_grad()
        batch = torch.randperm(len(fit_feats), generator=gen)[:bs].tolist()
        total = mean_objective([fit_feats[i] for i in batch], alpha, iface,
                               alpha0, iface0, beta0, args, charge_dummy, device)
        total.backward()
        gn = float(torch.nn.utils.clip_grad_norm_([alpha, iface], args.grad_clip))
        grad_norms.append(gn)
        opt.step()
        with torch.no_grad():
            alpha.clamp_(min=0.0, max=args.alpha_max)

        if step % val_every == 0 or step == n_steps - 1:
            val = _val_loss(val_feats, alpha, iface, alpha0, iface0, beta0,
                            args, charge_dummy, device)
            accepted = val < best_val - args.min_delta
            traj.append({"step": step + 1, "fit_loss": float(total.detach()),
                         "val_loss": val, "grad_norm": gn,
                         "accepted": int(accepted), "alpha": float(alpha),
                         "d_iface": float((iface - iface0).norm())})
            if accepted:
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
    opt.load_state_dict(best_opt)
    stats = {"best_val_loss": best_val, "steps_run": step + 1,
             "mean_grad_norm": sum(grad_norms) / max(1, len(grad_norms)),
             "max_grad_norm": max(grad_norms) if grad_norms else float("nan")}
    return alpha.detach(), iface.detach(), stats, traj, opt.state_dict()


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------
def iface_coverage(feats_list) -> dict:
    """How many complexes actually observe each of the 144 IFACE components.

    ``T[f, i, j]`` is the pose-f contact statistic for atom-type pair (i, j);
    a component with zero mass in a complex contributes no gradient there, so
    the effective sample size per parameter is this count, not the number of
    complexes.
    """
    cov = torch.zeros(12, 12, dtype=torch.int64)
    mass = torch.zeros(12, 12, dtype=torch.float64)
    for f in feats_list:
        m = f.T.abs().sum(dim=0).double().cpu()
        cov += (m > 0).long()
        mass += m
    n = max(1, len(feats_list))
    flat = cov.flatten()
    return {
        "n_complexes": len(feats_list),
        "coverage_matrix": cov.tolist(),
        "coverage_frac_matrix": (cov.double() / n).tolist(),
        "total_mass_matrix": mass.tolist(),
        "n_components_zero": int((flat == 0).sum()),
        "n_components_lt10": int((flat < 10).sum()),
        "n_components_lt_1pct": int((flat.double() / n < 0.01).sum()),
        "min_coverage": int(flat.min()),
        "median_coverage": int(flat.median()),
        "mean_coverage_frac": float((cov.double() / n).mean()),
    }


# --------------------------------------------------------------------------
# dataset selection (deterministic, nested)
# --------------------------------------------------------------------------
def select_split(args):
    """First ``n_total`` *usable* master-list clusters, split 80/20 by stride.

    Because both the usable list and the stride rule are prefix-stable, the fit
    and validation sets for N=500 are exact subsets of those for N=1,000 and
    N=2,000.
    """
    order = [ln.strip() for ln in Path(args.master_ids).read_text().splitlines()
             if ln.strip()]
    status = {}
    for line in Path(args.prep_manifest).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            status[r["id"]] = r

    # Deterministic grid-volume cutoff (see scripts/compute_grid_sizes.py).
    # A structural property, so identical across seeds and across N — unlike
    # excluding on observed OOM, which would depend on transient GPU pressure
    # and silently break the nested-subset design.
    max_vox = getattr(args, "max_grid_voxels", 0)
    voxels: dict[str, int] = {}
    if max_vox:
        vpath = Path(args.grid_voxels)
        if not vpath.exists():
            raise SystemExit(
                f"--max-grid-voxels is set but {vpath} is missing; run "
                f"scripts/compute_grid_sizes.py first")
        voxels = json.loads(vpath.read_text())

    def eligible(pid: str) -> bool:
        if status.get(pid, {}).get("status") != "ok":
            return False
        if max_vox and voxels.get(pid, 0) > max_vox:
            return False
        return True

    usable = [pid for pid in order if eligible(pid)]
    n_oversized = sum(1 for pid in order
                      if status.get(pid, {}).get("status") == "ok"
                      and max_vox and voxels.get(pid, 0) > max_vox)

    val_stride = int(round(1.0 / args.val_frac))
    n_total = args.n_total or int(round(args.n_fit / (1.0 - args.val_frac)))
    if len(usable) < n_total:
        raise SystemExit(
            f"need {n_total} usable clusters but the prep cache only has "
            f"{len(usable)} (attempted {len(status)} of {len(order)}); run "
            f"scripts/prep_pinder_cache.py with a larger --limit")
    sel = usable[:n_total]
    val_ids = [pid for i, pid in enumerate(sel) if i % val_stride == 0]
    fit_ids = [pid for i, pid in enumerate(sel) if i % val_stride != 0]
    info = {"n_master_scanned": len(order), "n_prepared_ok": len(status),
            "n_excluded_oversized": n_oversized, "n_usable": len(usable),
            "max_grid_voxels": max_vox}
    return sel, fit_ids, val_ids, status, info


def load_test_feats(path, dtype):
    blob = torch.load(path, map_location="cpu", weights_only=True)
    out = []
    for d in blob:
        n = d["sc"].shape[0]
        out.append(Feats(d["name"], d["sc"].to(dtype), d["T"].to(dtype),
                         d["elec"].to(dtype), d["rmsd"].to(dtype),
                         d["dockq"].to(dtype),
                         torch.zeros(n, dtype=torch.int16)))
    return out


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-fit", type=int, required=True, dest="n_fit",
                    help="number of FIT interface clusters (validation is extra)")
    ap.add_argument("--n-total", type=int, default=0, dest="n_total",
                    help="override total (fit+validation); default n_fit/(1-val_frac)")
    ap.add_argument("--val-frac", type=float, default=0.2, dest="val_frac")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=1,
                    help="number of hard-negative mining rounds after round 0")
    ap.add_argument("--master-ids", default="data/scaling/master_ids.txt",
                    dest="master_ids")
    ap.add_argument("--prep-cache", default="data/scaling/prep_cache", dest="prep_cache")
    ap.add_argument("--prep-manifest", default="data/scaling/prep_manifest.jsonl",
                    dest="prep_manifest")
    ap.add_argument("--grid-voxels", default="data/scaling/grid_voxels.json",
                    dest="grid_voxels")
    # p95 of the corpus. Cost and peak VRAM scale with the FFT lattice volume;
    # the tail reaches 1.2e10 voxels, and a single 4.1e6-voxel complex stalled
    # four jobs for >20 min even after the OOM ladder dropped to rot_chunk=1.
    ap.add_argument("--max-grid-voxels", type=int, default=2_000_000,
                    dest="max_grid_voxels",
                    help="drop complexes whose FFT lattice exceeds this many "
                         "voxels (0 disables the filter)")
    ap.add_argument("--test-cache", default="data/shards_pinder/test_feats.pt",
                    dest="test_cache")
    ap.add_argument("--test-ids", default="data/pinder_test_ids.txt", dest="test_ids")
    ap.add_argument("--out-dir", default="data/scaling/runs", dest="out_dir")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # optimization (identical to the stabilized §5.5 configuration)
    ap.add_argument("--epoch-passes", type=int, default=100, dest="epoch_passes",
                    help="optimizer steps = max(min_steps, passes * ceil(N/bs))")
    ap.add_argument("--min-steps", type=int, default=1500, dest="min_steps")
    ap.add_argument("--alpha-lr", type=float, default=1e-5, dest="alpha_lr")
    ap.add_argument("--iface-lr", type=float, default=5e-4, dest="iface_lr")
    ap.add_argument("--alpha-max", type=float, default=0.1, dest="alpha_max")
    ap.add_argument("--batch-size", type=int, default=16, dest="batch_size")
    ap.add_argument("--lambda-margin", type=float, default=0.5, dest="lambda_margin")
    ap.add_argument("--lambda-prior", type=float, default=0.1, dest="lambda_prior")
    ap.add_argument("--basin-temperature", type=float, default=0.5, dest="basin_temp")
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--grad-clip", type=float, default=5.0, dest="grad_clip")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--min-delta", type=float, default=1e-4, dest="min_delta")
    # search / labelling
    # Chunk defaults are set from a 12-complex profile (see report §5.6): the
    # FFT rotation batch dominates peak VRAM. rot_chunk 32 -> 8 cuts the worst
    # complex from 35.6 to 24.7 GiB at identical wall time (the search is
    # launch-bound, not bandwidth-bound, at this size); rot_chunk 4 halves
    # memory again but costs ~29% more time.
    ap.add_argument("--frame-chunk", type=int, default=200, dest="frame_chunk")
    ap.add_argument("--rot-chunk", type=int, default=8, dest="rot_chunk")
    ap.add_argument("--dockq-budget", type=int, default=50_000_000,
                    dest="dockq_budget",
                    help="max elements in the dense DockQ (chunk,N_rec,N_lig) tensor")
    ap.add_argument("--oom-retries", type=int, default=3, dest="oom_retries")
    ap.add_argument("--rmsd-threshold", type=float, default=5.0, dest="rmsd_thr")
    ap.add_argument("--dockq-threshold", type=float, default=0.23, dest="dockq_thr")
    ap.add_argument("--pool-cap", type=int, default=4000, dest="pool_cap")
    ap.add_argument("--mine-random-rot", type=int, default=1500, dest="mine_random_rot")
    ap.add_argument("--mine-cone", type=int, default=400, dest="mine_cone")
    ap.add_argument("--mine-ntop", type=int, default=1500, dest="mine_ntop")
    args = ap.parse_args()

    t_start = time.time()
    device = torch.device(args.device)
    dtype = torch.float64 if device.type == "cpu" else torch.float32
    gen = torch.Generator().manual_seed(args.seed)
    torch.manual_seed(args.seed)

    run_dir = Path(args.out_dir) / f"N{args.n_fit}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    skip_log = open(run_dir / "skipped.jsonl", "w", buffering=1)

    alpha0 = torch.tensor(0.01, device=device, dtype=dtype)
    beta0 = torch.tensor(3.0, device=device, dtype=dtype)
    iface0 = iface_ij(device=device, dtype=dtype, flat=True)
    charge0 = default_charge_score(device=device, dtype=dtype)

    # ---- dataset selection + leakage assertions --------------------------
    sel, fit_ids, val_ids, prep_status, sel_info = select_split(args)
    test_ids = set(ln.strip() for ln in Path(args.test_ids).read_text().splitlines()
                   if ln.strip())
    assert len(set(sel)) == len(sel), "duplicate id in selection"
    assert not (set(fit_ids) & set(val_ids)), "fit/validation overlap"
    assert set(fit_ids) | set(val_ids) == set(sel), "split does not cover selection"
    assert not (set(sel) & test_ids), "TEST id leaked into fit/validation"
    assert len(fit_ids) == args.n_fit, (
        f"expected {args.n_fit} fit clusters, got {len(fit_ids)}")
    print(f"selection: total={len(sel)} fit={len(fit_ids)} val={len(val_ids)} "
          f"(usable={sel_info['n_usable']} of {sel_info['n_prepared_ok']} prepared; "
          f"{sel_info['n_excluded_oversized']} excluded as oversized "
          f"> {sel_info['max_grid_voxels']} voxels)", flush=True)

    (run_dir / "split.json").write_text(json.dumps(
        {"n_fit": len(fit_ids), "n_val": len(val_ids), "n_total": len(sel),
         "selection": sel_info,
         "fit_ids": fit_ids, "val_ids": val_ids, "config": vars(args)}, indent=1))

    # ---- fixed TEST pool (built once with default parameters, §5.4) ------
    print("loading FIXED deleaked TEST pool ...", flush=True)
    test_feats = load_test_feats(args.test_cache, dtype)
    assert set(f.name for f in test_feats) <= test_ids, "TEST cache holds non-test ids"
    print(f"  test complexes: {len(test_feats)}", flush=True)

    val_set = set(val_ids)
    pools: dict[str, Feats] = {}
    alpha, iface = alpha0.clone(), iface0.clone()
    optimizer_state = None
    history = []
    n_skipped_total = 0

    def absorb(pid: str, nf: Feats, rnd: int) -> None:
        """Install a freshly mined candidate set into the complex's pool."""
        if rnd == 0:
            pools[pid] = nf
            return
        # hard-*negative* mining: never re-add the near-native cone positives,
        # so the positive set stays frozen at round 0.
        neg = (nf.dockq < args.dockq_thr).nonzero(as_tuple=True)[0]
        if neg.numel() == 0 or pid not in pools:
            return
        before_pos = int((pools[pid].dockq >= args.dockq_thr).sum())
        pools[pid].cat(nf.index(neg))
        pools[pid] = cap_pool(pools[pid], args.pool_cap,
                              alpha.detach().cpu(), iface.detach().cpu(),
                              beta0.detach().cpu(), args.dockq_thr)
        after_pos = int((pools[pid].dockq >= args.dockq_thr).sum())
        assert after_pos == before_pos, (
            f"{pid}: positives changed {before_pos} -> {after_pos} "
            f"during mining round {rnd}")

    def try_mine(pid: str, rnd: int, **chunks):
        """Return ``(Feats, meta)`` on success or ``(None, meta)`` on failure."""
        prot_cpu = load_prepared(args.prep_cache, pid)
        if prot_cpu is None:
            return None, {"stage": "cache_load", "reason": "missing",
                          "n_rec": -1, "n_lig": -1}
        meta = {"stage": "mine", "n_rec": prot_cpu.n_rec, "n_lig": prot_cpu.n_lig}
        prot = None
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            prot = prot_cpu.to(device, dtype=dtype)
            nf = mine_complex(prot, alpha, iface, beta0, charge0, args, rnd,
                              **chunks)
            return nf, meta
        except torch.cuda.OutOfMemoryError as exc:
            meta["reason"] = f"OOM: {str(exc)[:160]}"
            return None, meta
        except Exception as exc:  # noqa: BLE001
            meta["reason"] = f"{type(exc).__name__}: {exc}"[:200]
            return None, meta
        finally:
            del prot, prot_cpu
            if device.type == "cuda":
                torch.cuda.empty_cache()

    for rnd in range(args.rounds + 1):
        t_round = time.time()
        targets = sel if rnd == 0 else fit_ids
        t_mine = time.time()
        failed: list[tuple[str, dict]] = []
        for ci, pid in enumerate(targets):
            nf, meta = try_mine(pid, rnd)
            if nf is None:
                failed.append((pid, meta))
            else:
                absorb(pid, nf, rnd)
            if (ci + 1) % 50 == 0:
                el = time.time() - t_mine
                print(f"  round {rnd}: mined {ci+1}/{len(targets)} "
                      f"({el/(ci+1):.2f}s/complex, {el/60:.1f} min, "
                      f"{len(failed)} pending rescue)", flush=True)

        # Rescue pass. Whether a complex OOMs depends on how fragmented the
        # allocator happens to be when its turn comes, which would make set
        # membership depend on transient GPU pressure — i.e. differ between
        # seeds and between N for reasons unrelated to the science. Re-running
        # the failures at the minimum chunk sizes, with the pool already built
        # and the cache emptied, makes membership deterministic in practice.
        n_skipped = 0
        if failed:
            print(f"  round {rnd}: rescue pass over {len(failed)} complex(es) "
                  f"at rot_chunk=1", flush=True)
            for pid, meta in failed:
                if meta["stage"] != "mine":
                    skip_log.write(json.dumps({"round": rnd, "id": pid, **meta})
                                   + "\n")
                    n_skipped += 1
                    continue
                nf, meta2 = try_mine(pid, rnd, rot_chunk=1, frame_chunk=25,
                                     pose_chunk=1)
                if nf is None:
                    skip_log.write(json.dumps(
                        {"round": rnd, "id": pid, "rescued": False,
                         "first_reason": meta.get("reason", ""), **meta2}) + "\n")
                    n_skipped += 1
                    print(f"    [{pid}] SKIP after rescue "
                          f"(n_rec={meta2.get('n_rec')} n_lig={meta2.get('n_lig')})",
                          flush=True)
                else:
                    absorb(pid, nf, rnd)
                    skip_log.write(json.dumps(
                        {"round": rnd, "id": pid, "rescued": True,
                         "first_reason": meta.get("reason", ""),
                         "stage": "mine", "n_rec": meta["n_rec"],
                         "n_lig": meta["n_lig"]}) + "\n")
                    print(f"    [{pid}] rescued at rot_chunk=1", flush=True)
        n_skipped_total += n_skipped
        mine_seconds = time.time() - t_mine

        fit_pools = [pools[p] for p in fit_ids if p in pools]
        val_pools = [pools[p] for p in val_ids if p in pools]
        assert not (set(f.name for f in fit_pools) & set(f.name for f in val_pools))
        assert not (set(f.name for f in fit_pools) & test_ids)
        assert not (set(f.name for f in val_pools) & test_ids)

        if rnd == 0:
            # Baseline (default ZDOCK parameters) on the same fixed pools.
            b_sr, b_sd, b_t1 = aggregate(test_feats, alpha0, iface0, beta0,
                                         device, args.rmsd_thr, args.dockq_thr)
            print("=" * 62)
            print("BASELINE (default ZDOCK params) on fixed deleaked TEST")
            print("=" * 62)
            print(_fmt("baseline/TEST", b_sr, b_sd, b_t1), flush=True)
            cov = {"fit": iface_coverage(fit_pools),
                   "val": iface_coverage(val_pools),
                   "test": iface_coverage(test_feats)}
            (run_dir / "coverage.json").write_text(json.dumps(cov, indent=1))
            print(f"  IFACE coverage: fit zero-components="
                  f"{cov['fit']['n_components_zero']}/144  "
                  f"median complexes/component={cov['fit']['median_coverage']}"
                  f" of {cov['fit']['n_complexes']}", flush=True)
            (run_dir / "baseline_test.json").write_text(json.dumps(
                {"success_rmsd": {str(k): b_sr[k] for k in KS},
                 "success_dockq": {str(k): b_sd[k] for k in KS},
                 "mean_best_dockq_at1": b_t1}, indent=1))

        n_steps = max(args.min_steps,
                      args.epoch_passes * -(-len(fit_pools) // args.batch_size))
        val_every = max(50, n_steps // 100)
        t_train = time.time()
        alpha, iface, stats, traj, optimizer_state = train_params(
            fit_pools, val_pools, alpha, iface, alpha0, iface0, beta0, args,
            device, dtype, gen, n_steps, val_every, optimizer_state)
        train_seconds = time.time() - t_train

        with open(run_dir / f"round{rnd}_trajectory.csv", "w") as fh:
            fh.write("step,fit_loss,val_loss,grad_norm,accepted,alpha,d_iface\n")
            for r in traj:
                fh.write(f"{r['step']},{r['fit_loss']:.6f},{r['val_loss']:.6f},"
                         f"{r['grad_norm']:.6f},{r['accepted']},{r['alpha']:.6f},"
                         f"{r['d_iface']:.6f}\n")

        sr, sd, t1 = aggregate(test_feats, alpha, iface, beta0, device,
                               args.rmsd_thr, args.dockq_thr)
        srt, sdt, t1t = aggregate(fit_pools, alpha, iface, beta0, device,
                                  args.rmsd_thr, args.dockq_thr)
        comp = [p.counts(args.dockq_thr) for p in fit_pools]
        pool_stats = {k: sum(c[k] for c in comp) / max(1, len(comp))
                      for k in ("n", "n_pos", "n_rand_neg", "n_hard_neg")}
        rejected = [r for r in traj if r["step"] > 0 and not r["accepted"]]
        accepted = [r for r in traj if r["step"] > 0 and r["accepted"]]
        peak_mem = (torch.cuda.max_memory_allocated() / 2**30
                    if device.type == "cuda" else 0.0)

        rec = {
            "round": rnd, "seed": args.seed, "n_fit": len(fit_pools),
            "n_val": len(val_pools), "n_test": len(test_feats),
            "n_skipped_this_round": n_skipped, "n_skipped_total": n_skipped_total,
            "alpha": float(alpha), "d_iface_norm": float((iface - iface0).norm()),
            "val_loss": stats["best_val_loss"], "steps_run": stats["steps_run"],
            "steps_budget": n_steps, "val_every": val_every,
            "mean_grad_norm": stats["mean_grad_norm"],
            "max_grad_norm": stats["max_grad_norm"],
            "n_accepted_checkpoints": len(accepted),
            "n_rejected_checkpoints": len(rejected),
            "rejected_mean_fit_loss": (sum(r["fit_loss"] for r in rejected)
                                       / len(rejected)) if rejected else None,
            "rejected_mean_val_loss": (sum(r["val_loss"] for r in rejected)
                                       / len(rejected)) if rejected else None,
            "pool_mean": pool_stats,
            "test_success_rmsd": {str(k): sr[k] for k in KS},
            "test_success_dockq": {str(k): sd[k] for k in KS},
            "test_mean_best_dockq_at1": t1,
            "fit_success_rmsd": {str(k): srt[k] for k in KS},
            "fit_success_dockq": {str(k): sdt[k] for k in KS},
            "fit_mean_best_dockq_at1": t1t,
            "mine_seconds": mine_seconds, "train_seconds": train_seconds,
            "round_seconds": time.time() - t_round,
            "peak_gpu_gib": peak_mem,
        }
        (run_dir / f"round{rnd}_metrics.json").write_text(json.dumps(rec, indent=1))
        torch.save({"round": rnd, "seed": args.seed, "n_fit": len(fit_pools),
                    "alpha": alpha.cpu(), "iface": iface.cpu(),
                    "val_loss": stats["best_val_loss"], "config": vars(args)},
                   run_dir / f"round{rnd}_ckpt.pt")

        tag = "round 0 (no mining)" if rnd == 0 else f"round {rnd} (mined)"
        print("=" * 62)
        print(f"{tag} | N_fit={len(fit_pools)} | mean pool={pool_stats['n']:.0f} "
              f"(pos {pool_stats['n_pos']:.0f} / rand-neg "
              f"{pool_stats['n_rand_neg']:.0f} / hard-neg {pool_stats['n_hard_neg']:.0f})")
        print(f"  alpha={float(alpha):.4f} ||dIface||="
              f"{float((iface-iface0).norm()):.3f} val_loss={stats['best_val_loss']:.4f} "
              f"steps={stats['steps_run']}/{n_steps} skipped={n_skipped} "
              f"peakGPU={peak_mem:.1f} GiB")
        print("=" * 62)
        print(_fmt("TEST(deleaked)", sr, sd, t1))
        print(_fmt("FIT", srt, sdt, t1t))
        print(flush=True)
        history.append(rec)

    summary_csv = run_dir / "summary.csv"
    with open(summary_csv, "w") as fh:
        fh.write("round,seed,n_fit,n_val,n_test,"
                 + ",".join(f"test_dockq@{k}" for k in KS) + ","
                 + ",".join(f"test_rmsd@{k}" for k in KS)
                 + ",test_mean_best_dockq1,val_loss,alpha,d_iface,"
                   "mean_pool,n_pos,n_rand_neg,n_hard_neg,n_skipped,"
                   "mine_s,train_s,peak_gpu_gib\n")
        for r in history:
            fh.write(",".join([
                str(r["round"]), str(r["seed"]), str(r["n_fit"]), str(r["n_val"]),
                str(r["n_test"]),
                *[f"{r['test_success_dockq'][str(k)]:.6f}" for k in KS],
                *[f"{r['test_success_rmsd'][str(k)]:.6f}" for k in KS],
                f"{r['test_mean_best_dockq_at1']:.6f}", f"{r['val_loss']:.6f}",
                f"{r['alpha']:.6f}", f"{r['d_iface_norm']:.6f}",
                f"{r['pool_mean']['n']:.1f}", f"{r['pool_mean']['n_pos']:.1f}",
                f"{r['pool_mean']['n_rand_neg']:.1f}",
                f"{r['pool_mean']['n_hard_neg']:.1f}",
                str(r["n_skipped_this_round"]), f"{r['mine_seconds']:.1f}",
                f"{r['train_seconds']:.1f}", f"{r['peak_gpu_gib']:.2f}"]) + "\n")

    skip_log.close()
    print("\n=== DockQ success@K on held-out DELEAKED TEST (PINDER-S) ===")
    print(f"{'round':<20} {'top1':>7} {'top5':>7} {'top10':>7} {'top50':>7} {'top100':>7}")
    for r in history:
        d = r["test_success_dockq"]
        print(f"{'round '+str(r['round']):<20} "
              + " ".join(f"{d[str(k)]*100:6.1f}%" for k in KS))
    if len(history) > 1:
        g = {k: (history[-1]["test_success_dockq"][str(k)]
                 - history[0]["test_success_dockq"][str(k)]) * 100 for k in KS}
        print("mining gain (pp)     " + " ".join(f"{g[k]:6.1f} " for k in KS))
    print(f"\ntotal wall time: {(time.time()-t_start)/60:.1f} min  "
          f"-> {run_dir}")


if __name__ == "__main__":
    main()
