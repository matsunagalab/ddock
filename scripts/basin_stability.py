"""Measure how big and how stable each refined pose's basin is.

Ranking is where the remaining acceptable gap lives: the top 50 contain an
acceptable pose for 94.8% of the test set, and the score picks the right one for
84.0% (report section 5.14.33). Refitting the 144-number table on the refined
distribution recovers one complex out of 123 (section 5.14.34), so the score
itself has no more to give -- what is needed is information the score never sees.

Refinement supplies some. A pose sitting in a real energy basin has somewhere to
climb to and neighbours that climb to the same place; a pose sitting on a
coincidental score bump does not. Two quantities capture that:

  basin_size   how many of the 50 starting poses converge to the same endpoint
  stability    of twelve standard perturbations applied to the endpoint, how
               many refine back to it

The second is the important one. Counting how many of the top 50 land in a basin
mostly measures how densely the FFT sampled that region, which is a property of
the grid rather than of the energy surface. Perturbing by a FIXED amount and
asking what comes back measures the basin's actual width, and every candidate is
probed identically.

The twelve probes are +-4 degrees about each axis and +-1 Angstrom along each,
matching the scale of the grid error the refinement has to undo.

Writes one row per candidate with the trace of its refinement -- start and end
score, what it gained, how far it moved, how many iterations it took -- so a
ranker can be fitted on these without re-running anything.

Example
-------
    uv run python scripts/basin_stability.py \
        --in-dir data/scaling/q1_top50 --prep-cache data/scaling/prep_cache \
        --ckpt data/scaling/runs_loss/qp/round0_ckpt.pt \
        --out data/scaling/basin/pilot.csv --limit 40
"""

from __future__ import annotations

import argparse
import csv
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

from refine_poses import Scorer, apply, read_ligand, refine  # noqa: E402
from zdock.atomtypes import charge_score as default_charge_score  # noqa: E402
from zdock.atomtypes import iface_ij  # noqa: E402
from zdock.dockq import dockq_batch  # noqa: E402
from zdock.prep_cache import load_prepared  # noqa: E402
from zdock.score import SC_REFERENCE_SPACING, SC_RHO  # noqa: E402

def probes(deg: float, shift: float) -> np.ndarray:
    """Twelve equal displacements: +-deg about each axis, +-shift along each.

    The magnitude has to be calibrated, not assumed. At +-4 deg / +-1 A almost
    every candidate returns to where it started (stability 1.0 for the median
    candidate over a pilot), so the probe is smaller than every basin and
    separates nothing. Probing has to happen at the scale where basins actually
    differ in width.
    """
    return np.array(
        [[s * (deg if j < 3 else shift) if k == j else 0.0 for k in range(6)]
         for j in range(6) for s in (1.0, -1.0)], dtype=float)


def cluster(endpoints: np.ndarray, tol: float) -> np.ndarray:
    """Single-linkage on ligand RMSD. Endpoints of the same basin coincide to
    well under an Angstrom, so the threshold is not delicate."""
    n = endpoints.shape[0]
    lab = -np.ones(n, dtype=int)
    nxt = 0
    for i in range(n):
        if lab[i] >= 0:
            continue
        lab[i] = nxt
        stack = [i]
        while stack:
            a = stack.pop()
            d = np.sqrt(((endpoints - endpoints[a]) ** 2).sum(-1).mean(-1))
            for b in np.flatnonzero((d < tol) & (lab < 0)):
                lab[b] = nxt
                stack.append(int(b))
        nxt += 1
    return lab


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", required=True, dest="in_dir",
                    help="directory of UNREFINED top-K submissions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--prep-cache", default="data/scaling/prep_cache",
                    dest="prep_cache")
    ap.add_argument("--monomer", default="holo")
    ap.add_argument("--top-k", type=int, default=50, dest="top_k")
    ap.add_argument("--max-clusters", type=int, default=10, dest="max_clusters",
                    help="how many basins get the perturbation probes")
    ap.add_argument("--cluster-tol", type=float, default=1.5, dest="cluster_tol",
                    help="ligand RMSD in Angstrom for two endpoints to be one basin")
    ap.add_argument("--probe-deg", type=float, default=8.0, dest="probe_deg")
    ap.add_argument("--probe-shift", type=float, default=2.0, dest="probe_shift")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--deg", type=float, default=4.0)
    ap.add_argument("--shift", type=float, default=1.0)
    ap.add_argument("--spacing", type=float, default=SC_REFERENCE_SPACING)
    ap.add_argument("--frame-chunk", type=int, default=4, dest="frame_chunk")
    ap.add_argument("--rho0", type=float, default=SC_RHO)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ids-file", default="", dest="ids_file")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32
    beta = torch.tensor(3.0, device=device, dtype=dtype)
    charge = default_charge_score(device=device, dtype=dtype)
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        alpha = ck["alpha"].to(device=device, dtype=dtype)
        iface = ck["iface"].to(device=device, dtype=dtype)
        clash = ck["clash_weights"].to(device=device, dtype=dtype)
    else:
        alpha = torch.tensor(1.0, device=device, dtype=dtype)
        iface = iface_ij(device=device, dtype=dtype, flat=True)
        rho = torch.tensor(args.rho0, device=device, dtype=dtype)
        clash = alpha * rho.pow(torch.tensor([2.0, 3.0, 4.0], device=device,
                                             dtype=dtype))

    src = Path(args.in_dir)
    ids = sorted(p.name for p in src.iterdir() if p.is_dir())
    if args.ids_file:
        want = {ln.strip() for ln in Path(args.ids_file).read_text().splitlines()
                if ln.strip()}
        ids = [i for i in ids if i in want]
    if args.limit:
        ids = ids[: args.limit]
    si, sn = (int(x) for x in args.shard.split("/"))
    ids = ids[si::sn]
    P = probes(args.probe_deg, args.probe_shift)
    print(f"{len(ids)} complexes, {args.top_k} candidates each, {len(P)} probes "
          f"at +-{args.probe_deg} deg / +-{args.probe_shift} A per basin "
          f"(max {args.max_clusters})", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out, "w", newline="")
    w = csv.writer(fh)
    w.writerow(["id", "rank", "score0", "score1", "gain", "rot_deg", "shift_A",
                "iters", "cluster", "basin_size", "stability", "dispersion",
                "dockq0", "dockq1"])
    t0, done, skipped = time.time(), 0, []
    for i, pid in enumerate(ids):
        prot = load_prepared(args.prep_cache, pid)
        if prot is None:
            skipped.append((pid, "not in prep cache"))
            continue
        prot = prot.to(device, dtype=dtype)
        sc = Scorer(prot, alpha, iface, beta, charge, clash, args.spacing,
                    args.frame_chunk)
        try:
            models = sorted((src / pid / args.monomer / "models").glob("model_*.pdb"),
                            key=lambda p: int(p.stem.split("_")[1]))[: args.top_k]
            base = [read_ligand(m) for m in models]
            ends, traces = [], []
            for b in base:
                x, s1, s0 = refine(b, sc, iters=args.iters, deg=args.deg,
                                   shift=args.shift)
                ends.append(apply(b, x))
                traces.append((s0, s1, float(np.linalg.norm(x[:3])),
                               float(np.linalg.norm(x[3:]))))
            E = np.stack(ends)
            lab = cluster(E, args.cluster_tol)
            sizes = np.bincount(lab)
            order = np.argsort(-sizes)[: args.max_clusters]
            # one probe run per basin, from its highest-scoring member
            stab = {}
            disp = {}
            for cl in order:
                members = np.flatnonzero(lab == cl)
                rep = members[int(np.argmax([traces[m][1] for m in members]))]
                back = 0
                for p in P:
                    y, _, _ = refine(apply(E[rep], p), sc, iters=args.iters,
                                     deg=args.deg, shift=args.shift)
                    z = apply(apply(E[rep], p), y)
                    if np.sqrt(((z - E[rep]) ** 2).sum(-1).mean()) < args.cluster_tol:
                        back += 1
                stab[cl] = back / len(P)
                d = np.sqrt(((E[members] - E[members].mean(0)) ** 2).sum(-1).mean(-1))
                disp[cl] = float(d.mean())
            pair = torch.as_tensor(np.concatenate([np.stack(base), E]),
                                   device=device, dtype=dtype)
            dq = dockq_batch(prot.rec_xyz, pair, prot.native_lig).dockq.cpu().numpy()
            k = len(base)
            for r in range(k):
                cl = int(lab[r])
                w.writerow([pid, r + 1, f"{traces[r][0]:.3f}", f"{traces[r][1]:.3f}",
                            f"{traces[r][1] - traces[r][0]:.3f}",
                            f"{traces[r][2]:.2f}", f"{traces[r][3]:.3f}",
                            args.iters, cl, int(sizes[cl]),
                            f"{stab.get(cl, float('nan')):.3f}",
                            f"{disp.get(cl, float('nan')):.3f}",
                            f"{dq[r]:.4f}", f"{dq[k + r]:.4f}"])
            fh.flush()
            done += 1
        except torch.cuda.OutOfMemoryError as exc:
            skipped.append((pid, f"OOM: {str(exc)[:90]}"))
        except Exception as exc:                                # noqa: BLE001
            skipped.append((pid, f"{type(exc).__name__}: {exc}"[:110]))
        finally:
            del prot
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(ids)}  ({(time.time() - t0) / (i + 1):.1f}s each)",
                  flush=True)
    fh.close()
    print(f"\n{done} complexes written, {len(skipped)} skipped -> {out}")
    for pid, why in skipped[:10]:
        print(f"  skipped {pid}: {why}")
    print(f"wall {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
