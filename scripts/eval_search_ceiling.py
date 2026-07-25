"""Is a near-native pose even *reachable* by the FFT search?

This is the question that has to be answered before any success@K number from
§5.3-§5.6 can be interpreted, and it needs no scoring at all — it is a property
of the discretisation.

``docking_search`` can only return poses of the form (sampled rotation, grid
translation). Two approximations bound what it could ever produce:

* **rotation** — the candidate set is a finite quaternion sample, so the best
  available orientation differs from the native ``q*`` by some angle;
* **translation** — the FFT translation lattice has ``spacing`` (3 Å default),
  so the best available offset differs from the native ``t*`` by up to
  ``spacing*sqrt(3)/2`` ≈ 2.6 Å.

The upper bound on DockQ for the search is therefore the DockQ of
``(nearest sampled rotation, grid-snapped t*)``. If that is already below the
0.23 acceptance threshold, no choice of the 145 scoring parameters can make the
search succeed, and success@K on the pooled benchmark is measuring only the
ability to re-rank *injected* near-native poses that the search itself would
never return.

Reported for three rotation sets:

* ``uniform`` — what a real docking run has: N uniform random quaternions.
* ``cone`` — the native-informed 25 deg cone used to build the training and
  test pools (an upper bound that leaks ``q*``).
* ``exact`` — ``q*`` itself with the grid-snapped translation, isolating the
  translation lattice alone.

Example
-------
    uv run python scripts/eval_search_ceiling.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import torch  # noqa: E402

from zdock.dockq import dockq_batch, ligand_rmsd_to_native  # noqa: E402
from zdock.prep_cache import load_prepared  # noqa: E402
from zdock.score import SC_REFERENCE_SPACING  # noqa: E402
from zdock.rotation_grid import random_quaternions, rotation_cone  # noqa: E402
from zdock.search import _rotate_batch  # noqa: E402


def _angle_deg(q, q_star):
    d = (q * q_star.unsqueeze(0)).sum(dim=-1).abs().clamp(max=1.0)
    return torch.rad2deg(2.0 * torch.acos(d))


@torch.no_grad()
def _dockq_of(prot, poses, budget):
    per = max(1, prot.n_rec * prot.n_lig)
    chunk = int(max(1, min(64, budget // per)))
    rmsd = ligand_rmsd_to_native(prot.native_lig, poses)
    parts = []
    for s in range(0, poses.shape[0], chunk):
        parts.append(dockq_batch(prot.rec_xyz, poses[s:s + chunk],
                                 prot.native_lig).dockq)
    return rmsd, torch.cat(parts, dim=0)


@torch.no_grad()
def ceiling_for(prot, args):
    device, dtype = prot.rec_xyz.device, prot.rec_xyz.dtype
    q_uni = random_quaternions(args.n_rot, seed=args.rot_seed, device=device,
                               dtype=dtype)
    q_cone = rotation_cone(prot.q_star, args.n_cone, cone_deg=args.cone_deg,
                           seed=args.rot_seed, device=device, dtype=dtype)

    t_exact = prot.t_star
    t_grid = torch.round(t_exact / args.spacing) * args.spacing

    row = {"name": prot.name, "n_rec": prot.n_rec, "n_lig": prot.n_lig,
           "t_snap_error_ang": float((t_grid - t_exact).norm())}
    for tag, quats in (("uniform", q_uni), ("cone", q_cone),
                       ("exact", prot.q_star.unsqueeze(0))):
        ang = _angle_deg(quats, prot.q_star)
        best = int(ang.argmin())
        q = quats[best:best + 1]
        row[f"{tag}_min_angle_deg"] = float(ang.min())
        for tname, t in (("gridT", t_grid), ("exactT", t_exact)):
            pose = _rotate_batch(prot.lig_ref, q) + t.unsqueeze(0).unsqueeze(0)
            rmsd, dq = _dockq_of(prot, pose, args.dockq_budget)
            row[f"{tag}_{tname}_dockq"] = float(dq[0])
            row[f"{tag}_{tname}_rmsd"] = float(rmsd[0])
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ids-file", default="data/pinder_test_ids.txt", dest="ids_file")
    ap.add_argument("--prep-cache", default="data/scaling/prep_cache_test",
                    dest="prep_cache")
    ap.add_argument("--out", default="data/scaling/eval_search/ceiling.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-rot", type=int, default=1900, dest="n_rot")
    # 0 = the honest condition. A non-zero cone seeds the rotation set with
    # orientations near q*, which leaks the answer into the candidate set.
    ap.add_argument("--n-cone", type=int, default=0, dest="n_cone")
    ap.add_argument("--cone-deg", type=float, default=25.0, dest="cone_deg")
    ap.add_argument("--rot-seed", type=int, default=12345, dest="rot_seed")
    ap.add_argument("--spacing", type=float, default=SC_REFERENCE_SPACING)
    ap.add_argument("--dockq-budget", type=int, default=50_000_000, dest="dockq_budget")
    ap.add_argument("--dockq-threshold", type=float, default=0.23, dest="dockq_thr")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float64 if device.type == "cpu" else torch.float32
    ids = [ln.strip() for ln in Path(args.ids_file).read_text().splitlines()
           if ln.strip()]
    if args.limit:
        ids = ids[: args.limit]

    rows, missing = [], 0
    for i, pid in enumerate(ids):
        prot_cpu = load_prepared(args.prep_cache, pid)
        if prot_cpu is None:
            missing += 1
            continue
        try:
            rows.append(ceiling_for(prot_cpu.to(device, dtype=dtype), args))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            missing += 1
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(ids)}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with open(out, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r[c]) for c in cols) + "\n")

    n = len(rows)
    def frac(key):
        return sum(r[key] >= args.dockq_thr for r in rows) / n
    def mean(key):
        return sum(r[key] for r in rows) / n

    print("=" * 72)
    print(f"reachability ceiling over {n} complexes ({missing} unavailable)")
    print(f"  translation lattice {args.spacing} Å -> mean snap error "
          f"{mean('t_snap_error_ang'):.2f} Å")
    print(f"{'rotation set':<10} {'min angle':>10} | "
          f"{'DockQ (grid T)':>15} {'>=0.23':>8} | {'DockQ (exact T)':>16} {'>=0.23':>8}")
    for tag in ("uniform", "cone", "exact"):
        print(f"{tag:<10} {mean(f'{tag}_min_angle_deg'):9.1f}° | "
              f"{mean(f'{tag}_gridT_dockq'):15.3f} {frac(f'{tag}_gridT_dockq')*100:7.1f}% | "
              f"{mean(f'{tag}_exactT_dockq'):16.3f} {frac(f'{tag}_exactT_dockq')*100:7.1f}%")
    print(f"\nwrote {out}")
    summary = {"n": n, "missing": missing, "spacing": args.spacing,
               "mean_t_snap_error": mean("t_snap_error_ang")}
    for tag in ("uniform", "cone", "exact"):
        for tname in ("gridT", "exactT"):
            summary[f"{tag}_{tname}_mean_dockq"] = mean(f"{tag}_{tname}_dockq")
            summary[f"{tag}_{tname}_frac_ok"] = frac(f"{tag}_{tname}_dockq")
        summary[f"{tag}_mean_min_angle_deg"] = mean(f"{tag}_min_angle_deg")
    Path(str(out).replace(".csv", "_summary.json")).write_text(
        json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
