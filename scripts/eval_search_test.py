"""End-to-end docking evaluation on the deleaked TEST set: search with the
*trained* parameters, then ask whether the search itself found a near-native
pose.

Why this exists
---------------
Every number in §5.4-§5.6 so far comes from re-ranking a **fixed** candidate
pool built once with default ZDOCK parameters, in which

* the FFT rotation set was seeded with a 25 deg cone around the native
  orientation ``q*`` — information a real docking run does not have, and
* 400 near-native poses were *injected* at the native translation, so a
  positive is guaranteed to be present regardless of what the search found.

That makes it a re-ranking benchmark with the answer planted in the candidate
set. It is internally consistent (the same pool for every N, seed and round, so
comparisons are valid) but it cannot tell us whether the learned parameters
actually dock anything.

This script removes both crutches: rotations are drawn **uniformly at random
with no reference to** ``q*``, nothing is injected, and the candidate set is
exactly what ``docking_search`` returns under the parameters being evaluated.
The rotation set is generated from a fixed seed, so it is identical for every
parameter setting and any difference between conditions is attributable to
scoring rather than to sampling luck.

It also separates the two ways this can fail, which the pooled metric conflates:

* **ceiling** — with a finite uniform rotation set, is a near-native pose even
  reachable? Reported as the DockQ of the closest sampled rotation placed at
  the native translation ``t*``. Parameter-independent.
* **recall** — does the returned top-N contain any pose with DockQ >= 0.23?
* **success@K** — given the returned set, is a near-native pose ranked top-K?
  Reported unconditionally and conditioned on recall.

Example
-------
    PINDER_BASE_DIR=$PWD/external/pinder CUDA_VISIBLE_DEVICES=6 \\
    uv run python scripts/eval_search_test.py \\
        --ckpt data/scaling/runs/N220_seed2/round1_ckpt.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

from zdock.atomtypes import charge_score as default_charge_score  # noqa: E402
from zdock.atomtypes import iface_ij  # noqa: E402
from zdock.dockq import dockq_batch, ligand_rmsd_to_native  # noqa: E402
from zdock.prep_cache import load_prepared  # noqa: E402
from zdock.rotation_grid import random_quaternions  # noqa: E402
from zdock.search import _rotate_batch, docking_search  # noqa: E402

KS = (1, 5, 10, 50, 100)


def _quat_angle_deg(q: torch.Tensor, q_star: torch.Tensor) -> torch.Tensor:
    """Geodesic angle (deg) between each quaternion in ``q`` and ``q_star``,
    accounting for the double cover (q and -q are the same rotation)."""
    d = (q * q_star.unsqueeze(0)).sum(dim=-1).abs().clamp(max=1.0)
    return torch.rad2deg(2.0 * torch.acos(d))


@torch.no_grad()
def _label(prot, poses, budget):
    rmsd = ligand_rmsd_to_native(prot.native_lig, poses)
    per = max(1, prot.n_rec * prot.n_lig)
    chunk = int(max(1, min(64, budget // per)))
    parts = []
    for s in range(0, poses.shape[0], chunk):
        parts.append(dockq_batch(prot.rec_xyz, poses[s:s + chunk],
                                 prot.native_lig).dockq)
    return rmsd, torch.cat(parts, dim=0)


@torch.no_grad()
def evaluate_complex(prot, alpha, iface, beta, charge, args):
    device = prot.rec_xyz.device
    dtype = prot.rec_xyz.dtype
    # Uniform rotations, fixed seed, no native-orientation cone.
    quats = random_quaternions(args.n_rot, seed=args.rot_seed, device=device,
                               dtype=dtype)

    # Parameter-independent ceiling: best sampled rotation at the ideal
    # translation. Bounds what any scorer could achieve with this rotation set.
    ang = _quat_angle_deg(quats, prot.q_star)
    best_rot = int(ang.argmin())
    ceil_pose = (_rotate_batch(prot.lig_ref, quats[best_rot:best_rot + 1])
                 + prot.t_star.unsqueeze(0).unsqueeze(0))
    ceil_rmsd, ceil_dockq = _label(prot, ceil_pose, args.dockq_budget)

    rot_chunk = args.rot_chunk
    for attempt in range(args.oom_retries + 1):
        try:
            res = docking_search(
                prot.rec_xyz, prot.rec_radius, prot.rec_sasa,
                prot.rec_atomtype_id, prot.rec_charge_id,
                prot.lig_ref, prot.lig_radius, prot.lig_sasa,
                prot.lig_atomtype_id, prot.lig_charge_id,
                quats, alpha=alpha, iface_ij_flat=iface, beta=beta,
                charge_score_lut=charge, spacing=args.spacing,
                ntop=args.ntop, rot_chunk_size=rot_chunk)
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            rot_chunk = max(1, rot_chunk // 2)
            if attempt == args.oom_retries:
                raise
    poses = (_rotate_batch(prot.lig_ref, quats[res.quat_indices])
             + res.translations.unsqueeze(1))
    rmsd, dockq = _label(prot, poses, args.dockq_budget)

    # docking_search returns poses ordered by its own score; keep that order.
    scores = res.scores
    order = torch.argsort(scores, descending=True)
    dq_s, rm_s = dockq[order], rmsd[order]
    pos = dq_s >= args.dockq_thr
    hit = pos.nonzero(as_tuple=True)[0]
    n = dq_s.numel()
    row = {
        "name": prot.name,
        "n_out": n,
        "ceiling_dockq": float(ceil_dockq[0]),
        "ceiling_rmsd": float(ceil_rmsd[0]),
        "min_rot_angle_deg": float(ang.min()),
        "recall": int(bool(pos.any())),
        "n_pos_out": int(pos.sum()),
        "first_hit_rank": int(hit[0]) + 1 if hit.numel() else n + 1,
        "best_dockq_out": float(dq_s.max()),
        "min_rmsd_out": float(rm_s.min()),
    }
    for k in KS:
        kk = min(k, n)
        row[f"succ_dockq@{k}"] = int(bool((dq_s[:kk] >= args.dockq_thr).any()))
        row[f"succ_rmsd@{k}"] = int(bool((rm_s[:kk] <= args.rmsd_thr).any()))
        row[f"best_dockq@{k}"] = float(dq_s[:kk].max())
    return row


def _mean(rows, key, mask=None):
    xs = [r[key] for r in rows if (mask is None or mask(r))]
    xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="", help="checkpoint; omit for default ZDOCK")
    ap.add_argument("--label", default="")
    ap.add_argument("--test-ids", default="data/pinder_test_ids.txt", dest="test_ids")
    ap.add_argument("--prep-cache", default="data/scaling/prep_cache_test",
                    dest="prep_cache")
    ap.add_argument("--out-dir", default="data/scaling/eval_search", dest="out_dir")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-rot", type=int, default=1900, dest="n_rot",
                    help="uniform rotations (matches the mining budget of "
                         "1500 random + 400 cone, but with no cone)")
    ap.add_argument("--rot-seed", type=int, default=12345, dest="rot_seed",
                    help="fixed so every condition searches the same rotations")
    ap.add_argument("--ntop", type=int, default=1500)
    ap.add_argument("--spacing", type=float, default=3.0)
    ap.add_argument("--rot-chunk", type=int, default=8, dest="rot_chunk")
    ap.add_argument("--dockq-budget", type=int, default=50_000_000, dest="dockq_budget")
    ap.add_argument("--oom-retries", type=int, default=3, dest="oom_retries")
    ap.add_argument("--rmsd-threshold", type=float, default=5.0, dest="rmsd_thr")
    ap.add_argument("--dockq-threshold", type=float, default=0.23, dest="dockq_thr")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float64 if device.type == "cpu" else torch.float32
    label = args.label or (Path(args.ckpt).parent.name + "_" + Path(args.ckpt).stem
                           if args.ckpt else "baseline")

    beta = torch.tensor(3.0, device=device, dtype=dtype)
    charge = default_charge_score(device=device, dtype=dtype)
    iface0 = iface_ij(device=device, dtype=dtype, flat=True)
    if args.ckpt:
        blob = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        alpha = blob["alpha"].to(device=device, dtype=dtype)
        iface = blob["iface"].to(device=device, dtype=dtype)
    else:
        alpha = torch.tensor(0.01, device=device, dtype=dtype)
        iface = iface0.clone()
    print(f"[{label}] alpha={float(alpha):.4f} "
          f"||dIface||={float((iface - iface0).norm()):.3f}", flush=True)

    ids = [ln.strip() for ln in Path(args.test_ids).read_text().splitlines()
           if ln.strip()]
    if args.limit:
        ids = ids[: args.limit]

    rows, skipped = [], []
    t0 = time.time()
    for i, pid in enumerate(ids):
        prot_cpu = load_prepared(args.prep_cache, pid)
        if prot_cpu is None:
            skipped.append({"id": pid, "reason": "not in prep cache"})
            continue
        try:
            prot = prot_cpu.to(device, dtype=dtype)
            rows.append(evaluate_complex(prot, alpha, iface, beta, charge, args))
        except torch.cuda.OutOfMemoryError as exc:
            skipped.append({"id": pid, "n_rec": prot_cpu.n_rec,
                            "n_lig": prot_cpu.n_lig,
                            "reason": f"OOM: {str(exc)[:120]}"})
        except Exception as exc:  # noqa: BLE001
            skipped.append({"id": pid, "reason": f"{type(exc).__name__}: {exc}"[:160]})
        finally:
            del prot_cpu
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(ids)}  ({(time.time()-t0)/(i+1):.1f}s/complex)",
                  flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with open(out_dir / f"{label}_per_complex.csv", "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r[c]) for c in cols) + "\n")

    n = len(rows)
    rec = [r for r in rows if r["recall"]]
    summary = {
        "label": label, "n_evaluated": n, "n_skipped": len(skipped),
        "alpha": float(alpha), "d_iface_norm": float((iface - iface0).norm()),
        "n_rot": args.n_rot, "ntop": args.ntop, "rot_seed": args.rot_seed,
        # parameter-independent: what the rotation sampling permits at all
        "ceiling_frac_dockq_ok": sum(r["ceiling_dockq"] >= args.dockq_thr
                                     for r in rows) / max(1, n),
        "ceiling_mean_dockq": _mean(rows, "ceiling_dockq"),
        "mean_min_rot_angle_deg": _mean(rows, "min_rot_angle_deg"),
        # did the search return anything near-native at all
        "recall": len(rec) / max(1, n),
        "mean_n_pos_out": _mean(rows, "n_pos_out"),
        "mean_best_dockq_out": _mean(rows, "best_dockq_out"),
        "seconds": time.time() - t0,
    }
    for k in KS:
        summary[f"succ_dockq@{k}"] = _mean(rows, f"succ_dockq@{k}")
        summary[f"succ_rmsd@{k}"] = _mean(rows, f"succ_rmsd@{k}")
        summary[f"succ_dockq@{k}|recall"] = _mean(rec, f"succ_dockq@{k}") if rec else float("nan")
        summary[f"mean_best_dockq@{k}"] = _mean(rows, f"best_dockq@{k}")
    (out_dir / f"{label}_summary.json").write_text(json.dumps(summary, indent=1))
    if skipped:
        (out_dir / f"{label}_skipped.json").write_text(json.dumps(skipped, indent=1))

    print("=" * 66)
    print(f"[{label}]  end-to-end search on {n} deleaked TEST complexes "
          f"({len(skipped)} skipped)")
    print(f"  rotation-sampling ceiling: {summary['ceiling_frac_dockq_ok']*100:.1f}% "
          f"of complexes could reach DockQ>=0.23 at the ideal translation "
          f"(mean nearest-rotation angle {summary['mean_min_rot_angle_deg']:.1f} deg)")
    print(f"  search recall (any near-native in returned top-{args.ntop}): "
          f"{summary['recall']*100:.1f}%")
    print(f"     K   | succ@K (DockQ) | succ@K | recall | succ@K (RMSD)")
    for k in KS:
        print(f"    {k:>4} |     {summary[f'succ_dockq@{k}']*100:5.1f}%     |"
              f"     {summary[f'succ_dockq@{k}|recall']*100:5.1f}%    |"
              f"     {summary[f'succ_rmsd@{k}']*100:5.1f}%")
    print(f"  mean best DockQ in returned set = {summary['mean_best_dockq_out']:.3f}")
    print(f"  wall {summary['seconds']/60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
