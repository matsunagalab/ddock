"""Sanity check that should have preceded §5.3-§5.6: does the FFT search find
near-native poses at all, with **default ZDOCK parameters**, and how much of
the answer depends on the grid spacing?

Background
----------
ZDOCK's published grid spacing is 1.2 Å, and the scoring function's constants
are absolute distances that presume a lattice fine enough to resolve them:
a 3.4 Å surface shell (``radius[surf] + 3.4``), core/surface radii of
``r*sqrt(1.5)`` / ``r*sqrt(0.8)`` (~1.3-2.4 Å), ``rcut_iface = 6 Å`` and
``rcut_elec = 8 Å`` — with ``scatter_mode="nearest"``, so each atom lands in a
single cell. The Julia notebook this repository ports, however, ran at
**3.0 Å** (``tests/data/refs/1KXQ/phase2_grid.h5`` stores ``spacing = 3.0``),
and `docking_search` / `docking_score_elec` inherited that default, even though
``geom.generate_grid``'s own default is 1.2. At 3.0 Å the surface shell is
~1 voxel and the atomic radii are sub-voxel, so the shape-complementarity term
— the core of ZDOCK — is heavily aliased, and the 144 IFACE potentials are
being used far from the discretisation they were derived for.

Two consequences are confounded in every number reported so far and are
separated here:

* **scoring** — at a given spacing, does the search *rank* a near-native pose
  into its top-K?
* **reachability** — could it, even in principle? The search can only emit
  (sampled rotation, lattice translation) pairs, so the ceiling is the DockQ of
  the nearest sampled rotation at the lattice-snapped native translation. This
  is parameter-free and is reported alongside.

Nothing is injected into the candidate set: the poses evaluated are exactly
what ``docking_search`` returns. The rotation set is identical across spacings,
so the comparison isolates the lattice.

Example
-------
    PINDER_BASE_DIR=$PWD/external/pinder CUDA_VISIBLE_DEVICES=6 \\
    uv run python scripts/validate_grid_spacing.py --n-complexes 20
"""

from __future__ import annotations

import argparse
import json
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
from zdock.rotation_grid import random_quaternions, rotation_cone  # noqa: E402
from zdock.search import _rotate_batch, docking_search  # noqa: E402

KS = (1, 5, 10, 50, 100, 500)


@torch.no_grad()
def _label(prot, poses, budget):
    per = max(1, prot.n_rec * prot.n_lig)
    chunk = int(max(1, min(64, budget // per)))
    rmsd = ligand_rmsd_to_native(prot.native_lig, poses)
    parts = []
    for s in range(0, poses.shape[0], chunk):
        parts.append(dockq_batch(prot.rec_xyz, poses[s:s + chunk],
                                 prot.native_lig).dockq)
    return rmsd, torch.cat(parts, dim=0)


def _angle_deg(q, q_star):
    d = (q * q_star.unsqueeze(0)).sum(dim=-1).abs().clamp(max=1.0)
    return torch.rad2deg(2.0 * torch.acos(d))


@torch.no_grad()
def run_one(prot, quats, spacing, alpha, iface, beta, charge, args):
    device, dtype = prot.rec_xyz.device, prot.rec_xyz.dtype

    # Parameter-free reachability ceiling at this lattice.
    ang = _angle_deg(quats, prot.q_star)
    best = int(ang.argmin())
    t_snap = torch.round(prot.t_star / spacing) * spacing
    ceil_pose = (_rotate_batch(prot.lig_ref, quats[best:best + 1])
                 + t_snap.unsqueeze(0).unsqueeze(0))
    ceil_rmsd, ceil_dockq = _label(prot, ceil_pose, args.dockq_budget)

    rot_chunk = args.rot_chunk
    t0 = time.time()
    while True:
        try:
            res = docking_search(
                prot.rec_xyz, prot.rec_radius, prot.rec_sasa,
                prot.rec_atomtype_id, prot.rec_charge_id,
                prot.lig_ref, prot.lig_radius, prot.lig_sasa,
                prot.lig_atomtype_id, prot.lig_charge_id,
                quats, alpha=alpha, iface_ij_flat=iface, beta=beta,
                charge_score_lut=charge, spacing=spacing, ntop=args.ntop,
                rot_chunk_size=rot_chunk,
                **({} if args.sc_rho is None else {"sc_rho": args.sc_rho}))
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if rot_chunk == 1:
                raise
            rot_chunk = max(1, rot_chunk // 2)
    poses = (_rotate_batch(prot.lig_ref, quats[res.quat_indices])
             + res.translations.unsqueeze(1))
    rmsd, dockq = _label(prot, poses, args.dockq_budget)
    order = torch.argsort(res.scores, descending=True)
    dq_s, rm_s = dockq[order], rmsd[order]

    row = {"name": prot.name, "spacing": spacing, "n_out": int(dq_s.numel()),
           "seconds": time.time() - t0, "rot_chunk": rot_chunk,
           "ceiling_dockq": float(ceil_dockq[0]),
           "ceiling_rmsd": float(ceil_rmsd[0]),
           "t_snap_error": float((t_snap - prot.t_star).norm()),
           "min_rot_angle_deg": float(ang.min()),
           "n_pos_out": int((dq_s >= args.dockq_thr).sum()),
           "best_dockq_out": float(dq_s.max()),
           "min_rmsd_out": float(rm_s.min())}
    for k in KS:
        kk = min(k, dq_s.numel())
        row[f"succ_dockq@{k}"] = int(bool((dq_s[:kk] >= args.dockq_thr).any()))
        row[f"succ_rmsd@{k}"] = int(bool((rm_s[:kk] <= args.rmsd_thr).any()))
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prep-cache", default="data/scaling/prep_cache", dest="prep_cache")
    ap.add_argument("--prep-manifest", default="data/scaling/prep_manifest.jsonl",
                    dest="prep_manifest")
    ap.add_argument("--grid-voxels", default="data/scaling/grid_voxels.json",
                    dest="grid_voxels")
    ap.add_argument("--max-voxels-at-3a", type=int, default=150_000,
                    dest="max_vox",
                    help="pick small complexes: at 1.2 Å the lattice is "
                         "(3.0/1.2)^3 = 15.6x larger")
    ap.add_argument("--n-complexes", type=int, default=20, dest="n_complexes")
    ap.add_argument("--spacings", default="3.0,1.2")
    ap.add_argument("--n-random-rot", type=int, default=1500, dest="n_random_rot")
    ap.add_argument("--n-cone", type=int, default=400, dest="n_cone")
    ap.add_argument("--cone-deg", type=float, default=25.0, dest="cone_deg")
    ap.add_argument("--rot-seed", type=int, default=12345, dest="rot_seed")
    ap.add_argument("--ntop", type=int, default=2000)
    ap.add_argument("--rot-chunk", type=int, default=4, dest="rot_chunk")
    ap.add_argument("--dockq-budget", type=int, default=50_000_000, dest="dockq_budget")
    ap.add_argument("--rmsd-threshold", type=float, default=5.0, dest="rmsd_thr")
    ap.add_argument("--dockq-threshold", type=float, default=0.23, dest="dockq_thr")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="weight on S_SC. Chen et al. 2003 Eq.(2) has no alpha "
                         "at all — the PSC scale is set by rho — so 1.0 is the "
                         "newer paper's implied value; 0.01 came from the older "
                         "GSC formulation of Chen & Weng 2002 Eq.(6).")
    ap.add_argument("--sc-rho", type=float, default=None, dest="sc_rho")
    ap.add_argument("--n-rot-override", type=int, default=0, dest="n_rot_override")
    ap.add_argument("--out", default="data/scaling/grid_spacing_validation.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float64 if device.type == "cpu" else torch.float32
    spacings = [float(s) for s in args.spacings.split(",") if s.strip()]

    vox = json.loads(Path(args.grid_voxels).read_text())
    ok = [json.loads(l)["id"] for l in Path(args.prep_manifest).read_text().splitlines()
          if l.strip() and json.loads(l)["status"] == "ok"]
    picks = [p for p in ok if vox.get(p, 1 << 60) <= args.max_vox][: args.n_complexes]
    print(f"{len(picks)} complexes with <= {args.max_vox} voxels at 3.0 Å", flush=True)

    beta = torch.tensor(3.0, device=device, dtype=dtype)
    alpha = torch.tensor(args.alpha, device=device, dtype=dtype)
    iface = iface_ij(device=device, dtype=dtype, flat=True)
    charge = default_charge_score(device=device, dtype=dtype)

    rows = []
    for i, pid in enumerate(picks):
        prot_cpu = load_prepared(args.prep_cache, pid)
        if prot_cpu is None:
            continue
        prot = prot_cpu.to(device, dtype=dtype)
        # Identical rotation set for every spacing: 1500 uniform + 400 cone,
        # exactly what generate_decoys uses to build the pools.
        n_rand = args.n_rot_override or args.n_random_rot
        q_rand = random_quaternions(n_rand, seed=args.rot_seed,
                                    device=device, dtype=dtype)
        q_cone = rotation_cone(prot.q_star, args.n_cone, cone_deg=args.cone_deg,
                               seed=args.rot_seed, device=device, dtype=dtype)
        quats = torch.cat([q_rand, q_cone], dim=0)
        for sp in spacings:
            try:
                rows.append(run_one(prot, quats, sp, alpha, iface, beta, charge, args))
            except torch.cuda.OutOfMemoryError as exc:
                print(f"  [{pid}] spacing {sp}: OOM ({str(exc)[:80]})", flush=True)
            finally:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        del prot, prot_cpu
        r3 = [r for r in rows if r["name"] == pid]
        print(f"  [{i+1}/{len(picks)}] {pid[:40]:<40} " +
              "  ".join(f"sp={r['spacing']}: ceil={r['ceiling_dockq']:.2f} "
                        f"best={r['best_dockq_out']:.2f} "
                        f"succ@100={r['succ_dockq@100']} ({r['seconds']:.0f}s)"
                        for r in r3), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with open(out, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r[c]) for c in cols) + "\n")

    print("\n" + "=" * 78)
    print("DEFAULT ZDOCK PARAMETERS — search output only, nothing injected")
    print("=" * 78)
    hdr = (f"{'spacing':>8} {'n':>4} {'ceil DockQ':>11} {'ceil ok%':>9} "
           f"{'recall%':>8} {'best DockQ':>11} " +
           " ".join(f"{'s@'+str(k):>6}" for k in KS) + f" {'sec':>6}")
    print(hdr)
    summary = []
    for sp in spacings:
        rs = [r for r in rows if r["spacing"] == sp]
        if not rs:
            continue
        n = len(rs)
        mean = lambda k: sum(r[k] for r in rs) / n  # noqa: E731
        s = {"spacing": sp, "n": n,
             "ceiling_mean_dockq": mean("ceiling_dockq"),
             "ceiling_frac_ok": sum(r["ceiling_dockq"] >= args.dockq_thr
                                    for r in rs) / n,
             "recall": sum(r["n_pos_out"] > 0 for r in rs) / n,
             "mean_best_dockq_out": mean("best_dockq_out"),
             "mean_t_snap_error": mean("t_snap_error"),
             "mean_seconds": mean("seconds")}
        for k in KS:
            s[f"succ_dockq@{k}"] = mean(f"succ_dockq@{k}")
            s[f"succ_rmsd@{k}"] = mean(f"succ_rmsd@{k}")
        summary.append(s)
        print(f"{sp:>8} {n:>4} {s['ceiling_mean_dockq']:>11.3f} "
              f"{s['ceiling_frac_ok']*100:>8.1f}% {s['recall']*100:>7.1f}% "
              f"{s['mean_best_dockq_out']:>11.3f} " +
              " ".join(f"{s[f'succ_dockq@{k}']*100:>5.0f}%" for k in KS) +
              f" {s['mean_seconds']:>6.1f}")
    Path(str(out).replace(".csv", "_summary.json")).write_text(
        json.dumps(summary, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
