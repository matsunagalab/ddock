"""How far does Top-1 move if we refine far more than 50 poses?

Section 5.14.49 established what HDOCK actually does, from the HDOCK2.0 paper:
the FFT carries the shape term only, the top **10 translations per rotation**
are kept, and *all* of them -- 4392 x 10 = 43,920 poses -- are optimised by
SIMPLEX under the knowledge-based score before anything is ranked. This
repository refines **50**. Section 5.14.46 independently found that the refiner
already sits at a local optimum out to 4 deg / 1 A, so the only way left to
improve it is to start it from more places. Those two facts point at the same
experiment.

The search and the refinement run in one process and poses stay in tensors, so
nothing is written to disk per pose (250 complexes x 1000 poses would be a
quarter of a million PDB files). The refinement is vectorised across poses:
`refine_batch` proposes the same twelve axis-aligned moves for every live pose
and scores them in a single batched call, so the pattern search is identical to
`refine_poses.refine` while the Python loop runs once per iteration instead of
once per pose.

The K-sweep is free. Refining N poses once and then asking "if we had only
refined the first K by search score, which would we have submitted?" gives the
whole curve from one run, because the refined score of a pose does not depend
on how many other poses were refined.

    CUDA_VISIBLE_DEVICES=0 uv run python scripts/refine_scaling.py \
        --ckpt data/scaling/runs_sym/N1000/round0_ckpt.pt \
        --tpr 10 --n-refine 1000 --limit 20 --out data/scaling/refscale_0.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_search_test import _usable_atom_lines, _write_decoy  # noqa: E402
from refine_poses import Scorer  # noqa: E402

from zdock.atomtypes import charge_score as default_charge_score  # noqa: E402
from zdock.atomtypes import iface_ij  # noqa: E402
from zdock.dockq import ligand_rmsd_to_native  # noqa: E402
from zdock.prep_cache import load_prepared  # noqa: E402
from zdock.rotation_grid import hopf_quaternions  # noqa: E402
from zdock.score import SC_REFERENCE_SPACING, SC_RHO  # noqa: E402
from zdock.search import _rotate_batch, docking_search  # noqa: E402

#: prefixes of the search ranking at which Top-1 is reported
KS = (1, 5, 10, 25, 50, 100, 200, 500, 1000, 2000, 5000)


def _rotvec_matrices(v: torch.Tensor) -> torch.Tensor:
    """Rodrigues for a batch of 3-vectors whose norm is the angle in degrees."""
    ang = torch.linalg.norm(v, dim=-1, keepdim=True)
    axis = torch.where(ang > 0, v / ang.clamp_min(1e-12), v.new_tensor([1.0, 0.0, 0.0]))
    a = torch.deg2rad(ang.squeeze(-1))
    x, y, z = axis.unbind(-1)
    c, s, C = torch.cos(a), torch.sin(a), 1.0 - torch.cos(a)
    return torch.stack([
        torch.stack([c + x * x * C, x * y * C - z * s, x * z * C + y * s], -1),
        torch.stack([y * x * C + z * s, c + y * y * C, y * z * C - x * s], -1),
        torch.stack([z * x * C - y * s, z * y * C + x * s, c + z * z * C], -1),
    ], dim=-2)


def _apply(base: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Rigid move of each pose in `base` (P, n, 3) by `x` (P, 6), about its own
    centroid -- matching `refine_poses.apply`."""
    rot = _rotvec_matrices(x[:, :3])
    cen = base.mean(dim=1, keepdim=True)
    return torch.einsum("pij,pnj->pni", rot, base - cen) + cen + x[:, None, 3:]


@torch.no_grad()
def refine_batch(base: torch.Tensor, scorer, *, iters: int, deg: float,
                 shift: float, shrink: float = 0.5, tol_deg: float = 0.05,
                 chunk: int = 64):
    """`refine_poses.refine`, run on (P, n_lig, 3) at once.

    Each pose keeps its own step size and shrinks it independently whenever a
    round of twelve probes fails to improve it, which is what makes the search
    terminate at different times for different poses.
    """
    P = base.shape[0]
    dev, dt = base.device, base.dtype
    x = torch.zeros((P, 6), device=dev, dtype=dt)
    best = torch.from_numpy(np.asarray(scorer(base))).to(dev, dt)
    step = torch.zeros((P, 6), device=dev, dtype=dt)
    step[:, :3], step[:, 3:] = deg, shift
    live = torch.ones(P, dtype=torch.bool, device=dev)

    for _ in range(iters):
        idx = torch.nonzero(live).flatten()
        if idx.numel() == 0:
            break
        cand = x[idx].unsqueeze(1).repeat(1, 12, 1)               # (L,12,6)
        eye = torch.eye(6, device=dev, dtype=dt)
        cand[:, 0::2] += step[idx].unsqueeze(1) * eye
        cand[:, 1::2] -= step[idx].unsqueeze(1) * eye

        scores = torch.empty((idx.numel(), 12), device=dev, dtype=dt)
        flat_base = base[idx].unsqueeze(1).expand(-1, 12, -1, -1).reshape(-1, base.shape[1], 3)
        flat_cand = cand.reshape(-1, 6)
        for s in range(0, flat_cand.shape[0], chunk):
            poses = _apply(flat_base[s:s + chunk], flat_cand[s:s + chunk])
            scores.view(-1)[s:s + chunk] = torch.from_numpy(
                np.asarray(scorer(poses))).to(dev, dt)

        bs, bi = scores.max(dim=1)
        better = bs > best[idx]
        gi = idx[better]
        if gi.numel():
            x[gi] = cand[better, bi[better]]
            best[gi] = bs[better]
        worse = idx[~better]
        if worse.numel():
            step[worse] *= shrink
        live[idx] = step[idx, 0] > tol_deg
    return x, best


@torch.no_grad()
def run_complex(prot, alpha, iface, beta, charge, clash, args, rho):
    dev, dt = prot.rec_xyz.device, prot.rec_xyz.dtype
    quats = hopf_quaternions(args.hopf_nside, device=dev, dtype=dt)
    quats = quats / quats.norm(dim=-1, keepdim=True)

    chunk = args.rot_chunk
    for attempt in range(args.oom_retries + 1):
        try:
            res = docking_search(
                prot.rec_xyz, prot.rec_radius, prot.rec_sasa,
                prot.rec_atomtype_id, prot.rec_charge_id,
                prot.lig_ref, prot.lig_radius, prot.lig_sasa,
                prot.lig_atomtype_id, prot.lig_charge_id,
                quats, alpha=alpha, iface_ij_flat=iface, beta=beta, sc_rho=rho,
                charge_score_lut=charge, spacing=args.spacing,
                ntop=args.ntop, rot_chunk_size=chunk,
                trans_per_rotation=args.tpr)
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            chunk = max(1, chunk // 2)
            if attempt == args.oom_retries:
                return None

    order = torch.argsort(res.scores, descending=True)[: args.n_refine]
    poses = (_rotate_batch(prot.lig_ref, quats[res.quat_indices[order]])
             + res.translations[order].unsqueeze(1))
    score0 = res.scores[order]

    # `L_count` is (frame_chunk * 12, nx, ny, nz), which is what has OOM'd
    # three times in this project (5.10.5, 5.14.33, 5.14.39). Scale the chunk
    # to the grid volume up front and still halve it on a miss, so one large
    # complex cannot take the shard down.
    from zdock.geom import grid_shape
    nx, ny, nz = grid_shape(prot.rec_xyz, prot.lig_ref, spacing=args.spacing)
    vox = nx * ny * nz
    fc = int(max(1, min(args.frame_chunk, args.chunk_elems // max(1, vox * 12))))
    t0 = time.time()
    for attempt in range(args.oom_retries + 1):
        try:
            scorer = Scorer(prot, alpha, iface, beta, charge, clash,
                            args.spacing, fc)
            x, score1 = refine_batch(poses, scorer, iters=args.iters,
                                     deg=args.deg, shift=args.shift,
                                     chunk=max(fc, args.batch))
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            fc = max(1, fc // 2)
            if attempt == args.oom_retries:
                return None
    refined = _apply(poses, x)
    dt_refine = time.time() - t0

    def labels(p):
        rmsd = ligand_rmsd_to_native(prot.native_lig, p)
        dq = torch.zeros_like(rmsd)
        near = torch.nonzero(rmsd <= args.dockq_rmsd_cut).flatten()
        if near.numel():
            from zdock.dockq import dockq_batch
            per = max(1, prot.n_rec * prot.n_lig)
            step = int(max(1, min(64, args.dockq_budget // per)))
            parts = [dockq_batch(prot.rec_xyz, p[near[s:s + step]],
                                 prot.native_lig).dockq
                     for s in range(0, near.numel(), step)]
            dq[near] = torch.cat(parts)
        return dq

    # Own DockQ is systematically ~0.117 low (5.14.31), so it cannot speak
    # about medium/high at all. Write the arms out as PINDER decoys and let the
    # official harness decide. For arm K, the submission is the poses ranked by
    # *refined* score among the first K of the search ranking -- exactly what
    # "we refined K poses" would have produced.
    if args.export_dir:
        s1 = score1.cpu().numpy()
        src = Path(args.pdb_dir) / f"{prot.name}.pdb"
        rec_lines = _usable_atom_lines(src, "R")
        lig_lines = _usable_atom_lines(src, "L")
        rec_cpu = prot.rec_xyz.detach().cpu()
        ref_cpu = refined.detach().cpu()
        for k in args.export_ks:
            kk = min(k, s1.shape[0])
            order = np.argsort(-s1[:kk])[: args.export_top_k]
            out = (Path(args.export_dir) / f"K{k}" / prot.name / args.monomer
                   / "models")
            out.mkdir(parents=True, exist_ok=True)
            for stale in out.glob("*.pdb"):
                stale.unlink()
            for rank, j in enumerate(order, start=1):
                tmp = out / f".model_{rank}.pdb.tmp"
                _write_decoy(tmp, rec_lines, lig_lines, rec_cpu, ref_cpu[int(j)])
                tmp.rename(out / f"model_{rank}.pdb")

    return {
        "name": prot.name,
        "n_refined": int(poses.shape[0]),
        "refine_s": dt_refine,
        "score0": score0.cpu().numpy(),
        "score1": score1.cpu().numpy(),
        "dockq0": labels(poses).cpu().numpy(),
        "dockq1": labels(refined).cpu().numpy(),
    }


def summarise(rows, thr):
    """Top-1 if only the first K search-ranked poses had been refined.

    Oracle@K is reported next to it because the two answer different
    questions: Oracle says whether refining more poses put a better answer in
    reach, Top-1 says whether the score then picked it. Every null in this
    project so far has been the first rising while the second stayed put.
    """
    out = []
    for k in KS:
        acc = hi = orc = n = 0
        for r in rows:
            kk = min(k, r["n_refined"])
            if kk == 0:
                continue
            j = int(np.argmax(r["score1"][:kk]))
            acc += int(r["dockq1"][j] >= thr)
            hi += int(r["dockq1"][j] >= 0.80)
            orc += int(r["dockq1"][:kk].max() >= thr)
            n += 1
        if n:
            out.append((k, acc / n * 100, hi / n * 100, orc / n * 100, n))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--prep-cache", default="data/scaling/prep_cache_test",
                    dest="prep_cache")
    ap.add_argument("--ids-file", default="data/scaling/pinder_s_all_ids.txt",
                    dest="ids_file")
    ap.add_argument("--tpr", type=int, default=10)
    ap.add_argument("--ntop", type=int, default=20000)
    ap.add_argument("--n-refine", type=int, default=1000, dest="n_refine")
    ap.add_argument("--hopf-nside", type=int, default=3, dest="hopf_nside")
    ap.add_argument("--spacing", type=float, default=SC_REFERENCE_SPACING)
    ap.add_argument("--rot-chunk", type=int, default=8, dest="rot_chunk")
    ap.add_argument("--frame-chunk", type=int, default=32, dest="frame_chunk")
    ap.add_argument("--chunk-elems", type=int, default=800_000_000,
                    dest="chunk_elems",
                    help="cap on frame_chunk*12*voxels for the ligand grid")
    ap.add_argument("--batch", type=int, default=64,
                    help="poses scored per scorer call during refinement")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--deg", type=float, default=4.0)
    ap.add_argument("--shift", type=float, default=1.0)
    ap.add_argument("--oom-retries", type=int, default=4, dest="oom_retries")
    ap.add_argument("--dockq-thr", type=float, default=0.23, dest="dockq_thr")
    ap.add_argument("--dockq-budget", type=int, default=200_000_000,
                    dest="dockq_budget")
    ap.add_argument("--dockq-rmsd-cut", type=float, default=30.0,
                    dest="dockq_rmsd_cut")
    ap.add_argument("--rho0", type=float, default=SC_RHO)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--export-dir", default="", dest="export_dir",
                    help="write PINDER-shaped decoys as <dir>/K<k>/<system>/...")
    ap.add_argument("--export-ks", default="50,200,1000", dest="export_ks",
                    type=lambda s: [int(x) for x in s.split(",")])
    ap.add_argument("--export-top-k", type=int, default=50, dest="export_top_k")
    ap.add_argument("--monomer", default="holo")
    ap.add_argument("--pdb-dir", default="external/pinder/pinder/2024-02/pdbs",
                    dest="pdb_dir")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dev, dt = torch.device(args.device), torch.float32
    beta = torch.tensor(3.0, device=dev, dtype=dt)
    charge = default_charge_score(device=dev, dtype=dt)
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        alpha = ck["alpha"].to(device=dev, dtype=dt)
        iface = ck["iface"].to(device=dev, dtype=dt)
        rho = ck.get("rho", torch.tensor(args.rho0)).to(device=dev, dtype=dt)
        clash = ck["clash_weights"].to(device=dev, dtype=dt)
    else:
        alpha = torch.tensor(1.0, device=dev, dtype=dt)
        iface = iface_ij(device=dev, dtype=dt, flat=True)
        rho = torch.tensor(args.rho0, device=dev, dtype=dt)
        clash = alpha * rho.pow(torch.tensor([2.0, 3.0, 4.0], device=dev, dtype=dt))

    ids = [i.strip() for i in Path(args.ids_file).read_text().split() if i.strip()]
    si, sn = (int(x) for x in args.shard.split("/"))
    ids = ids[si::sn]
    if args.limit:
        ids = ids[: args.limit]

    rows, t0 = [], time.time()
    for pid in ids:
        prot = load_prepared(args.prep_cache, pid)
        if prot is None:
            continue
        r = run_complex(prot.to(dev, dtype=dt), alpha, iface, beta, charge,
                        clash, args, rho)
        if r is None:
            print(f"  {pid}: OOM, skipped", flush=True)
            continue
        rows.append(r)
        j0 = int(np.argmax(r["score0"]))
        j1 = int(np.argmax(r["score1"]))
        print(f"  {r['name'][:32]:34s} n={r['n_refined']:5d} "
              f"{r['refine_s']:6.0f}s  dockq top1 {r['dockq0'][j0]:.3f} "
              f"-> {r['dockq1'][j1]:.3f}", flush=True)

    if not rows:
        raise SystemExit("nothing refined")
    tot = time.time() - t0
    npose = sum(r["n_refined"] for r in rows)
    print(f"\n{len(rows)} complexes, {npose} poses refined, {tot:.0f} s "
          f"({tot / max(1, npose) * 1000:.0f} ms/pose)\n")

    print(f"{'refined K':>10} | {'Top-1 acceptable':>17} | {'Top-1 high':>11} | "
          f"{'Oracle@K acc':>13}")
    for k, acc, hi, orc, n in summarise(rows, args.dockq_thr):
        print(f"{k:>10} | {acc:16.2f}% | {hi:10.2f}% | {orc:12.2f}%")
    print("\n(own DockQ, threshold "
          f"{args.dockq_thr}; systematically ~0.117 below the official "
          "harness, so `high` here is not comparable to PINDER's)")

    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["name", "rank", "score0", "score1", "dockq0", "dockq1"])
            for r in rows:
                for i in range(r["n_refined"]):
                    w.writerow([r["name"], i + 1, r["score0"][i], r["score1"][i],
                                r["dockq0"][i], r["dockq1"][i]])
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
