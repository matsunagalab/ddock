"""How good a pose can the search *possibly* return, on the grid we actually use?

This is the ceiling on the whole parameter-learning stage. The FFT search can
only return poses of the form ``rotate(lig_ref, q_i) + k * spacing`` for a
quaternion ``q_i`` in the rotation grid and an integer cell vector ``k``. If no
such pose clears the DockQ threshold for a complex, then no assignment of
(alpha, rho, e_ij) can make the search succeed on it: the correct answer is not
in the set being ranked.

That is a different question from "does the current scorer rank it first", and
it is much cheaper to answer -- no FFT search is needed, only pose construction
and labelling. We enumerate the ``K`` grid rotations closest to the native
orientation ``q*`` (SO(3) geodesic) crossed with every lattice translation
within ``R`` cells of the snapped ``t*``, which is exactly the set
``generate_pool_reachable`` draws its positives from.

Reporting the sweep over ``(K, R)`` also says whether the pool builder's
defaults (8, 2) are generous enough, or whether they are the thing limiting the
positives rather than the geometry.

Example
-------
    uv run python scripts/reachable_ceiling.py \
        --ids-file data/scaling/master_ids.txt --limit 300 \
        --out data/scaling/reachable_ceiling.csv
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

from zdock.dockq import dockq_batch  # noqa: E402
from zdock.prep_cache import load_prepared  # noqa: E402
from zdock.rotation_grid import (hopf_quaternions, random_quaternions,  # noqa: E402
                                 so3_geodesic_deg)
from zdock.score import SC_REFERENCE_SPACING  # noqa: E402
from zdock.search import _rotate_batch  # noqa: E402


def best_reachable(prot, quats, *, spacing, near_rot, trans_cells,
                   pose_chunk, contact_cutoff=5.0, iface_cutoff=8.0):
    """Max DockQ over the K nearest grid rotations x the +/-R cell lattice."""
    device, dtype = prot.rec_xyz.device, prot.rec_xyz.dtype
    d = so3_geodesic_deg(quats, prot.q_star.unsqueeze(0))[:, 0]
    order = torch.argsort(d)
    k = min(int(near_rot), quats.shape[0])
    near = order[:k]

    r = int(trans_cells)
    off = torch.arange(-r, r + 1, device=device, dtype=torch.long)
    oz, oy, ox = torch.meshgrid(off, off, off, indexing="ij")
    off3 = torch.stack([ox.reshape(-1), oy.reshape(-1), oz.reshape(-1)], dim=-1)
    t_cell = torch.round(prot.t_star / spacing).to(torch.long)
    cells = (t_cell.unsqueeze(0) + off3).to(dtype) * spacing      # (C, 3)

    best = -1.0
    for qi in near.tolist():
        base = _rotate_batch(prot.lig_ref, quats[qi:qi + 1])[0]   # (N, 3)
        poses = base.unsqueeze(0) + cells.unsqueeze(1)            # (C, N, 3)
        for s in range(0, poses.shape[0], pose_chunk):
            comp = dockq_batch(prot.rec_xyz, poses[s:s + pose_chunk],
                               prot.native_lig, contact_cutoff=contact_cutoff,
                               iface_cutoff=iface_cutoff)
            best = max(best, float(comp.dockq.max()))
        del poses
    return best, float(d[order[0]])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ids-file", default="data/scaling/master_ids.txt",
                    dest="ids_file")
    ap.add_argument("--prep-cache", default="data/scaling/prep_cache",
                    dest="prep_cache")
    ap.add_argument("--out", default="data/scaling/reachable_ceiling.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--rot-set", choices=("hopf", "uniform"), default="hopf",
                    dest="rot_set")
    ap.add_argument("--hopf-nside", type=int, default=3, dest="hopf_nside")
    ap.add_argument("--n-rot", type=int, default=1944, dest="n_rot",
                    help="only for --rot-set uniform")
    ap.add_argument("--spacing", type=float, default=SC_REFERENCE_SPACING)
    ap.add_argument("--dockq-threshold", type=float, default=0.23, dest="dockq_thr")
    ap.add_argument("--sweep", default="1x0,8x1,8x2",
                    help="comma-separated K x R settings to compare")
    ap.add_argument("--dockq-budget", type=int, default=50_000_000,
                    dest="dockq_budget")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if device.type == "cuda" else torch.float64
    if args.rot_set == "hopf":
        quats = hopf_quaternions(args.hopf_nside, device=device, dtype=dtype)
        quats = quats / quats.norm(dim=-1, keepdim=True)
    else:
        quats = random_quaternions(args.n_rot, seed=12345, device=device,
                                   dtype=dtype)
    settings = []
    for tok in args.sweep.split(","):
        k, r = tok.strip().split("x")
        settings.append((int(k), int(r)))
    print(f"rotation set: {args.rot_set} N={quats.shape[0]}  spacing={args.spacing}")
    print(f"settings (K rotations x +/-R cells): {settings}", flush=True)

    ids = [ln.strip() for ln in Path(args.ids_file).read_text().splitlines()
           if ln.strip()][: args.limit or None]
    rows, missing, t0 = [], 0, time.time()
    for i, pid in enumerate(ids):
        prot_cpu = load_prepared(args.prep_cache, pid)
        if prot_cpu is None:
            missing += 1
            continue
        prot = None
        try:
            prot = prot_cpu.to(device, dtype=dtype)
            pose_chunk = max(1, min(64, args.dockq_budget
                                    // max(1, prot.n_rec * prot.n_lig)))
            row = {"id": pid, "n_rec": prot.n_rec, "n_lig": prot.n_lig}
            for (k, r) in settings:
                b, near_deg = best_reachable(
                    prot, quats, spacing=args.spacing, near_rot=k,
                    trans_cells=r, pose_chunk=pose_chunk)
                row[f"best_dockq_{k}x{r}"] = b
                row["nearest_rot_deg"] = near_deg
            rows.append(row)
        except torch.cuda.OutOfMemoryError:
            missing += 1
            print(f"  [{i+1}] OOM {pid}", flush=True)
        finally:
            del prot, prot_cpu
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  [{i+1}/{len(ids)}] {el/(i+1):.2f}s/complex "
                  f"({el/60:.1f} min)", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys()) if rows else []
    with open(out, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

    print(f"\n{len(rows)} complexes usable ({missing} missing/OOM)")
    print(f"{'K x R':>8}{'frac >= thr':>13}{'mean best':>12}{'median best':>13}")
    summary = {"n": len(rows), "n_missing": missing, "settings": {}}
    for (k, r) in settings:
        v = sorted(x[f"best_dockq_{k}x{r}"] for x in rows)
        frac = sum(1 for x in v if x >= args.dockq_thr) / max(1, len(v))
        m = sum(v) / max(1, len(v))
        med = v[len(v) // 2] if v else float("nan")
        print(f"{f'{k}x{r}':>8}{frac*100:12.1f}%{m:12.3f}{med:13.3f}")
        summary["settings"][f"{k}x{r}"] = {
            "frac_above_threshold": frac, "mean_best_dockq": m,
            "median_best_dockq": med}
    nr = sorted(x["nearest_rot_deg"] for x in rows)
    if nr:
        print(f"\nnearest grid rotation to q*: median {nr[len(nr)//2]:.2f} deg, "
              f"max {nr[-1]:.2f} deg")
        summary["nearest_rot_deg"] = {"median": nr[len(nr) // 2], "max": nr[-1]}
    summary["config"] = vars(args)
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1,
                                                           default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
