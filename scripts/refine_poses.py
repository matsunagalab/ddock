"""Polish the submitted poses off the rotation grid, without re-running search.

The search can only return orientations from a fixed grid, and at Hopf nside=3
the nearest one is a median 9.5 degrees from the truth. High quality needs the
error under about 5 degrees, so the grid alone caps High at 39.6% -- which the
method has essentially reached, 38.0% (report section 5.14.31).

Sampling around the native showed where the score actually peaks: a median 1.0
degrees and 0.84 A from the native pose for the trained table, inside the range
High needs. So climbing the score off-grid should help, and the cheapest way to
find out is to climb it.

Derivative-free on purpose. The score is differentiable with respect to ligand
coordinates only in `scatter_mode="trilinear"`, and even then the PSC term --
which carries the clash penalty -- keeps its non-differentiable neighbour path.
Gradient refinement would therefore optimise a score that cannot see clashes.
Six parameters is small enough that a pattern search needs no gradient at all.

The search is batched rather than sequential: each iteration proposes both
directions along all six axes at once and scores them in a single call, so one
complex costs a few dozen batched evaluations rather than a few hundred single
ones. Steps shrink when no proposal improves.

Poses come from an existing submission directory, so nothing is re-docked. They
are written in the prepared receptor-centred frame, which is the frame the
scorer wants.

Example
-------
    uv run python scripts/refine_poses.py \
        --in-dir data/pinder_eval/trained_QP \
        --out-dir data/pinder_eval/trained_QP_refined \
        --ckpt data/scaling/runs_convex/N1000/round0_ckpt.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval_search_test import _usable_atom_lines, _write_decoy  # noqa: E402
from zdock.atomtypes import charge_score as default_charge_score  # noqa: E402
from zdock.atomtypes import iface_ij  # noqa: E402
from zdock.dockq import dockq_batch  # noqa: E402
from zdock.prep_cache import load_prepared  # noqa: E402
from zdock.score import SC_REFERENCE_SPACING, SC_RHO, docking_score_elec  # noqa: E402


def read_ligand(path: Path) -> np.ndarray:
    """Chain L coordinates of a written decoy, in the order they were written."""
    out = []
    for line in open(path):
        if line.startswith("ATOM") and line[21] == "L":
            out.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return np.asarray(out)


def rot(axis_angle: np.ndarray) -> np.ndarray:
    """Rodrigues rotation from a 3-vector whose norm is the angle in degrees."""
    a = np.deg2rad(np.linalg.norm(axis_angle))
    if a < 1e-12:
        return np.eye(3)
    u = axis_angle / np.linalg.norm(axis_angle)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


def apply(base: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Rigid move of `base` by the 6-vector x = (rotvec_deg[3], shift_A[3])."""
    cen = base.mean(axis=0)
    return (base - cen) @ rot(x[:3]).T + cen + x[3:]


class Scorer:
    def __init__(self, prot, alpha, iface, beta, charge, clash, spacing,
                 frame_chunk):
        self.p, self.alpha, self.iface = prot, alpha, iface
        self.beta, self.charge, self.clash = beta, charge, clash
        self.spacing, self.frame_chunk = spacing, frame_chunk
        self.M = -1.0 * iface.view(12, 12).T
        self.calls = 0

    def __call__(self, poses: np.ndarray) -> np.ndarray:
        """(F, N_lig, 3) -> (F,) scores, in one batched call."""
        p = self.p
        x = torch.as_tensor(poses, device=p.rec_xyz.device, dtype=p.rec_xyz.dtype)
        sc, T, elec = docking_score_elec(
            p.rec_xyz, p.rec_radius, p.rec_sasa, p.rec_atomtype_id, p.rec_charge_id,
            x, p.lig_radius, p.lig_sasa, p.lig_atomtype_id, p.lig_charge_id,
            torch.zeros((), device=x.device, dtype=x.dtype),
            iface_ij(device=x.device, dtype=x.dtype, flat=True),
            self.beta, self.charge, lig_xyz_for_grid=p.lig_ref,
            spacing=self.spacing, frame_chunk_size=self.frame_chunk,
            return_components=True, psc_decompose=True)
        self.calls += x.shape[0]
        s = (self.alpha * sc[:, 0] - (sc[:, 1:4] * self.clash).sum(-1)
             + (self.M * T).sum(dim=(-2, -1)) + self.beta * elec)
        return s.detach().cpu().numpy()


def refine(base: np.ndarray, scorer: Scorer, *, iters: int, deg: float,
           shift: float, shrink: float = 0.5, tol_deg: float = 0.05):
    """Pattern search over the six rigid-body parameters.

    Both directions along all six axes are proposed together and scored in one
    batch. The step shrinks whenever nothing improves, which is what makes the
    search converge rather than oscillate around the optimum.
    """
    x = np.zeros(6)
    best = float(scorer(base[None])[0])
    start = best
    while deg > tol_deg and iters > 0:
        step = np.concatenate([np.full(3, deg), np.full(3, shift)])
        cand = np.repeat(x[None], 12, axis=0)
        for k in range(6):
            cand[2 * k, k] += step[k]
            cand[2 * k + 1, k] -= step[k]
        poses = np.stack([apply(base, c) for c in cand])
        s = scorer(poses)
        iters -= 1
        k = int(np.argmax(s))
        if s[k] > best:
            best, x = float(s[k]), cand[k]
        else:
            deg *= shrink
            shift *= shrink
    return x, best, start


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", required=True, dest="in_dir")
    ap.add_argument("--out-dir", required=True, dest="out_dir")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--prep-cache", default="data/scaling/prep_cache_test",
                    dest="prep_cache")
    ap.add_argument("--pdb-dir", default="external/pinder/pinder/2024-02/pdbs",
                    dest="pdb_dir")
    ap.add_argument("--monomer", default="holo")
    ap.add_argument("--top-k", type=int, default=5, dest="top_k")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--deg", type=float, default=4.0, help="initial angle step")
    ap.add_argument("--shift", type=float, default=1.0, help="initial shift step, A")
    ap.add_argument("--spacing", type=float, default=SC_REFERENCE_SPACING)
    ap.add_argument("--frame-chunk", type=int, default=12, dest="frame_chunk")
    ap.add_argument("--rho0", type=float, default=SC_RHO)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32
    beta = torch.tensor(3.0, device=device, dtype=dtype)
    charge = default_charge_score(device=device, dtype=dtype)
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        alpha = ck["alpha"].to(device=device, dtype=dtype)
        iface = ck["iface"].to(device=device, dtype=dtype)
        rho = ck.get("rho", torch.tensor(args.rho0)).to(device=device, dtype=dtype)
        clash = ck["clash_weights"].to(device=device, dtype=dtype)
    else:
        alpha = torch.tensor(1.0, device=device, dtype=dtype)
        iface = iface_ij(device=device, dtype=dtype, flat=True)
        rho = torch.tensor(args.rho0, device=device, dtype=dtype)
        clash = alpha * rho.pow(torch.tensor([2.0, 3.0, 4.0], device=device,
                                             dtype=dtype))

    src_root, dst_root = Path(args.in_dir), Path(args.out_dir)
    pdb_dir = Path(args.pdb_dir)
    ids = sorted(p.name for p in src_root.iterdir() if p.is_dir())
    if args.limit:
        ids = ids[: args.limit]
    print(f"{len(ids)} systems from {src_root}\n")

    moved, dq_before, dq_after, t0 = [], [], [], time.time()
    for i, pid in enumerate(ids):
        prot = load_prepared(args.prep_cache, pid)
        if prot is None:
            print(f"  {pid}: not in prep cache, skipped")
            continue
        prot = prot.to(device, dtype=dtype)
        scorer = Scorer(prot, alpha, iface, beta, charge, clash, args.spacing,
                        args.frame_chunk)
        models = sorted((src_root / pid / args.monomer / "models").glob("model_*.pdb"),
                        key=lambda p: int(p.stem.split("_")[1]))[: args.top_k]
        src = pdb_dir / f"{pid}.pdb"
        rec_lines = _usable_atom_lines(src, "R")
        lig_lines = _usable_atom_lines(src, "L")
        out_poses, out_scores = [], []
        for m in models:
            base = read_ligand(m)
            if base.shape[0] != prot.n_lig:
                raise SystemExit(f"{pid}/{m.name}: {base.shape[0]} ligand atoms "
                                 f"against {prot.n_lig} prepared")
            x, best, start = refine(base, scorer, iters=args.iters, deg=args.deg,
                                    shift=args.shift)
            out_poses.append(apply(base, x))
            out_scores.append(best)
            moved.append((float(np.linalg.norm(x[:3])), float(np.linalg.norm(x[3:]))))
        order = np.argsort(-np.asarray(out_scores))       # re-rank on the new score
        pt = torch.as_tensor(np.stack(out_poses), device=device, dtype=dtype)
        base_t = torch.as_tensor(np.stack([read_ligand(m) for m in models]),
                                 device=device, dtype=dtype)
        dq_before.append(float(dockq_batch(prot.rec_xyz, base_t, prot.native_lig).dockq[0]))
        dq_after.append(float(dockq_batch(prot.rec_xyz, pt, prot.native_lig).dockq[int(order[0])]))

        out = dst_root / pid / args.monomer / "models"
        out.mkdir(parents=True, exist_ok=True)
        for stale in out.glob("*.pdb"):
            stale.unlink()
        for rank, j in enumerate(order, start=1):
            tmp = out / f".model_{rank}.pdb.tmp"
            _write_decoy(tmp, rec_lines, lig_lines, prot.rec_xyz.detach().cpu(),
                         torch.as_tensor(out_poses[int(j)], dtype=torch.float32))
            tmp.rename(out / f"model_{rank}.pdb")
        del prot
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(ids)}  ({(time.time() - t0) / (i + 1):.1f}s each)",
                  flush=True)

    a = np.array(moved)
    b, c = np.array(dq_before), np.array(dq_after)
    print(f"\n{len(b)} systems refined, wall {(time.time() - t0) / 60:.1f} min")
    print(f"how far the poses moved: rotation median {np.median(a[:, 0]):.1f} deg, "
          f"translation median {np.median(a[:, 1]):.2f} A")
    print(f"\nrank-1 DockQ (this repository's own, biased about 0.117 low)")
    print(f"  before {b.mean():.3f}   after {c.mean():.3f}   change {c.mean() - b.mean():+.3f}")
    print(f"  improved {int((c > b + 0.01).sum())}, unchanged "
          f"{int((abs(c - b) <= 0.01).sum())}, worsened {int((c < b - 0.01).sum())}")
    for t in (0.23, 0.49, 0.80):
        print(f"  DockQ >= {t:.2f}: {100 * (b >= t).mean():5.1f}% -> "
              f"{100 * (c >= t).mean():5.1f}%")


if __name__ == "__main__":
    main()
