"""Does the score have its local maximum at the native pose, or somewhere else?

The rotation grid caps quality: at Hopf nside=3 the nearest orientation is a
median 9.5 degrees away, and High needs the error under about 5 degrees (report
section 5.14.31). Two ways out -- an eight-times finer grid, or refining the best
poses off-grid after the search. Refinement is nearly free, but it only helps if
climbing the score leads TOWARDS the native. If the score peaks somewhere else,
refinement makes the pose worse while making the score better.

That is measurable without any search. Around each complex's native pose this
samples orientations up to `--max-deg` away and translations inside
`--max-shift` Angstrom, scores them with the real scoring function, and asks
where the best-scoring sample sits:

* how far from native, in degrees and Angstrom
* what DockQ it has, against 1.0 at the native itself
* whether the native beats everything sampled

A score whose maximum is at the native would refine well. One whose maximum sits
9 degrees away has nothing to gain from refinement -- it would walk there.

Sampling, not optimisation: a local optimiser would report where it converged,
which confounds the landscape with the optimiser. The DockQ here is this
repository's own, which tracks the official one at Pearson 0.992 but sits a mean
0.117 low, so absolute DockQ values are not comparable to PINDER's -- the
comparison here is between poses of the same complex, where the bias cancels.

Example
-------
    uv run python scripts/score_landscape_near_native.py \
        --ckpt data/scaling/runs_convex/N1000/round0_ckpt.pt --limit 50
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from zdock.atomtypes import charge_score as default_charge_score  # noqa: E402
from zdock.atomtypes import iface_ij  # noqa: E402
from zdock.dockq import dockq_batch  # noqa: E402
from zdock.prep_cache import load_prepared  # noqa: E402
from zdock.score import SC_REFERENCE_SPACING, SC_RHO, docking_score_elec  # noqa: E402


def perturbations(n: int, max_deg: float, max_shift: float, rng):
    """n random (rotation matrix, translation) pairs, plus the identity first."""
    R = np.zeros((n, 3, 3))
    t = np.zeros((n, 3))
    deg = np.zeros(n)
    R[0] = np.eye(3)
    for i in range(1, n):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        # uniform in angle, not in rotation measure: the question is how the
        # score behaves at a given displacement, so the displacement is the
        # variable that should be swept evenly
        a = rng.uniform(0.0, max_deg)
        deg[i] = a
        th = np.deg2rad(a)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R[i] = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
        v = rng.normal(size=3)
        t[i] = v / np.linalg.norm(v) * rng.uniform(0.0, max_shift)
    return R, t, deg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="", help="omit for the published table")
    ap.add_argument("--test-ids", default="data/pinder_test_ids.txt", dest="test_ids")
    ap.add_argument("--prep-cache", default="data/scaling/prep_cache_test",
                    dest="prep_cache")
    ap.add_argument("--n-samples", type=int, default=400, dest="n_samples")
    ap.add_argument("--max-deg", type=float, default=12.0, dest="max_deg")
    ap.add_argument("--max-shift", type=float, default=2.5, dest="max_shift")
    ap.add_argument("--spacing", type=float, default=SC_REFERENCE_SPACING)
    ap.add_argument("--frame-chunk", type=int, default=25, dest="frame_chunk")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32
    beta = torch.tensor(3.0, device=device, dtype=dtype)
    charge = default_charge_score(device=device, dtype=dtype)
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        alpha = ck["alpha"].to(device=device, dtype=dtype)
        iface = ck["iface"].to(device=device, dtype=dtype)
        rho = ck.get("rho", torch.tensor(SC_RHO)).to(device=device, dtype=dtype)
        clash = ck["clash_weights"].to(device=device, dtype=dtype) \
            if "clash_weights" in ck else alpha * rho.pow(
                torch.tensor([2.0, 3.0, 4.0], device=device, dtype=dtype))
        who = args.ckpt
    else:
        alpha = torch.tensor(1.0, device=device, dtype=dtype)
        iface = iface_ij(device=device, dtype=dtype, flat=True)
        rho = torch.tensor(SC_RHO, device=device, dtype=dtype)
        clash = alpha * rho.pow(torch.tensor([2.0, 3.0, 4.0], device=device,
                                             dtype=dtype))
        who = "published table"

    ids = [ln.strip() for ln in Path(args.test_ids).read_text().splitlines()
           if ln.strip()][: args.limit]
    print(f"{who}\n{len(ids)} complexes, {args.n_samples} samples each within "
          f"{args.max_deg:.0f} deg and {args.max_shift:.1f} A\n")

    rows = []
    t0 = time.time()
    for i, pid in enumerate(ids):
        prot = load_prepared(args.prep_cache, pid)
        if prot is None:
            continue
        prot = prot.to(device, dtype=dtype)
        rng = np.random.default_rng(args.seed + i)
        R, t, deg = perturbations(args.n_samples, args.max_deg, args.max_shift, rng)
        nat = prot.native_lig                                  # (N_lig, 3)
        cen = nat.mean(dim=0, keepdim=True)
        Rt = torch.tensor(R, device=device, dtype=dtype)
        tt = torch.tensor(t, device=device, dtype=dtype)
        poses = torch.einsum("fij,nj->fni", Rt, (nat - cen)) + cen + tt[:, None, :]
        try:
            sc, T, elec = docking_score_elec(
                prot.rec_xyz, prot.rec_radius, prot.rec_sasa,
                prot.rec_atomtype_id, prot.rec_charge_id,
                poses, prot.lig_radius, prot.lig_sasa,
                prot.lig_atomtype_id, prot.lig_charge_id,
                torch.zeros((), device=device, dtype=dtype),
                iface_ij(device=device, dtype=dtype, flat=True), beta, charge,
                lig_xyz_for_grid=prot.lig_ref, spacing=args.spacing,
                frame_chunk_size=args.frame_chunk, return_components=True,
                psc_decompose=True)
            s = (alpha * sc[:, 0] - (sc[:, 1:4] * clash).sum(-1)
                 + (iface.view(12, 12).T * -1.0 * T).sum(dim=(-2, -1))
                 + beta * elec)
            dq = dockq_batch(prot.rec_xyz, poses, prot.native_lig).dockq
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue
        k = int(s.argmax())
        rows.append({"id": pid, "deg": float(deg[k]),
                     "shift": float(np.linalg.norm(t[k])),
                     "dockq": float(dq[k]), "dockq_native": float(dq[0]),
                     "native_is_best": bool(k == 0),
                     "s_gap": float(s[k] - s[0])})
        del prot, poses, sc, T, elec, s, dq
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(ids)}  ({(time.time() - t0) / (i + 1):.1f}s each)",
                  flush=True)

    n = len(rows)
    q = lambda k, p: float(np.quantile([r[k] for r in rows], p))    # noqa: E731
    print(f"\n{n} complexes measured\n")
    print(f"the native pose is the best-scoring sample : "
          f"{sum(r['native_is_best'] for r in rows)}/{n}")
    print("\nwhere the best-scoring sample sits, relative to native")
    print(f"  rotation offset  p25 {q('deg', .25):5.1f}  median {q('deg', .5):5.1f}"
          f"  p75 {q('deg', .75):5.1f} deg")
    print(f"  translation      p25 {q('shift', .25):5.2f}  median {q('shift', .5):5.2f}"
          f"  p75 {q('shift', .75):5.2f} A")
    print(f"\nquality there, against 1.0 at the native (repo DockQ)")
    print(f"  DockQ            p25 {q('dockq', .25):5.3f}  median {q('dockq', .5):5.3f}"
          f"  p75 {q('dockq', .75):5.3f}")
    worse = sum(1 for r in rows if r["dockq"] < r["dockq_native"] - 0.02)
    print(f"  worse than the native by more than 0.02: {worse}/{n}")
    print(f"\nscore gained by leaving the native: median "
          f"{q('s_gap', .5):.1f} raw units")


if __name__ == "__main__":
    main()
