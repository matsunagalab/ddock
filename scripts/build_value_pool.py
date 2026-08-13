"""Label each search pose with the quality it reaches AFTER refinement.

The deployed pipeline is search, then refine, then pick. The table has always
been fitted to rank poses by the quality they have *now* -- but what the search
stage is actually being asked for is a pose that *polishes well*. Those are not
the same thing. A pose sitting on a coincidental score bump and a pose sitting
at the edge of a real basin can look identical to the current objective, and
only one of them survives refinement.

So the label here is the value-to-go

    y_p = DockQ(R(p))

where R is a fixed refiner, attached to the features of the UNREFINED pose. A
table fitted against this ranks basins by how far they can be polished, and it
is still one table used for the FFT search -- no second discriminator is
stacked on top, so this is not a reranker.

Why the existing refined directories cannot be reused: `refine_poses.py`
re-ranks its output on the new score before writing, so `model_k` in the
refined directory is not the refinement of `model_k` in the source directory,
and the correspondence is not recorded anywhere. Recovering it by nearest
neighbour would be ambiguous exactly where several starts converge to one
basin, which is the case the label is supposed to distinguish. The refinement
is therefore redone here with the mapping kept.

Both labels are written, so the value-to-go table and the current-quality table
can be fitted on identical features and compared with nothing else moving.

Example
-------
    uv run python scripts/build_value_pool.py \
        --in-dir data/scaling/q1_top50 \
        --ckpt data/scaling/runs_loss/qp/round0_ckpt.pt \
        --prep-cache data/scaling/prep_cache \
        --out data/scaling/pool_cache/value_top50.pt --shard 0/3
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

from refine_poses import Scorer, apply, read_ligand, refine  # noqa: E402
from zdock.atomtypes import charge_score as default_charge_score  # noqa: E402
from zdock.atomtypes import iface_ij  # noqa: E402
from zdock.dataset import POSE_IDENTITY_MISSING  # noqa: E402
from zdock.dockq import dockq_batch, ligand_rmsd_to_native  # noqa: E402
from zdock.prep_cache import load_prepared  # noqa: E402
from zdock.score import SC_REFERENCE_SPACING, docking_score_elec  # noqa: E402


def dockq_chunked(rec, poses, native, chunk):
    """(K,) DockQ. Chunked because the (K, N_lig, N_rec) distance matrix is
    40 GB at K=50 on a large complex -- the mistake that OOMed section 5.14.33."""
    return torch.cat([dockq_batch(rec, poses[a:a + chunk], native).dockq
                      for a in range(0, poses.shape[0], chunk)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", required=True, dest="in_dir",
                    help="directory of UNREFINED top-K submissions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True,
                    help="the table the refiner uses; must be held fixed")
    ap.add_argument("--prep-cache", default="data/scaling/prep_cache",
                    dest="prep_cache")
    ap.add_argument("--monomer", default="holo")
    ap.add_argument("--top-k", type=int, default=50, dest="top_k")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--deg", type=float, default=4.0)
    ap.add_argument("--shift", type=float, default=1.0)
    ap.add_argument("--spacing", type=float, default=SC_REFERENCE_SPACING)
    ap.add_argument("--frame-chunk", type=int, default=4, dest="frame_chunk",
                    help="4, not the usual 12: the default OOMs on big complexes")
    ap.add_argument("--pose-chunk", type=int, default=8, dest="pose_chunk")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ids-file", default="", dest="ids_file")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32
    beta = torch.tensor(3.0, device=device, dtype=dtype)
    charge = default_charge_score(device=device, dtype=dtype)
    iface0 = iface_ij(device=device, dtype=dtype, flat=True)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    alpha = ck["alpha"].to(device=device, dtype=dtype)
    iface = ck["iface"].to(device=device, dtype=dtype)
    clash = ck["clash_weights"].to(device=device, dtype=dtype)

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
    print(f"{len(ids)} complexes, {args.top_k} poses each, refiner = {args.ckpt}",
          flush=True)

    pools, skipped, t0 = [], [], time.time()
    for i, pid in enumerate(ids):
        prot = load_prepared(args.prep_cache, pid)
        if prot is None:
            skipped.append((pid, "not in prep cache"))
            continue
        prot = prot.to(device, dtype=dtype)
        try:
            models = sorted((src / pid / args.monomer / "models").glob("model_*.pdb"),
                            key=lambda p: int(p.stem.split("_")[1]))[: args.top_k]
            base = [read_ligand(m) for m in models]
            if base[0].shape[0] != prot.n_lig:
                raise ValueError(f"{base[0].shape[0]} ligand atoms against {prot.n_lig}")
            start = torch.as_tensor(np.stack(base), device=device, dtype=dtype)

            # features of the START poses -- these are what the table will see
            sc, T, elec = docking_score_elec(
                prot.rec_xyz, prot.rec_radius, prot.rec_sasa,
                prot.rec_atomtype_id, prot.rec_charge_id,
                start, prot.lig_radius, prot.lig_sasa,
                prot.lig_atomtype_id, prot.lig_charge_id,
                torch.zeros((), device=device, dtype=dtype), iface0, beta, charge,
                lig_xyz_for_grid=prot.lig_ref, spacing=args.spacing,
                frame_chunk_size=args.frame_chunk, return_components=True,
                psc_decompose=True)

            # refine each start pose, keeping the correspondence
            scorer = Scorer(prot, alpha, iface, beta, charge, clash, args.spacing,
                            args.frame_chunk)
            ends = []
            for b in base:
                x, _, _ = refine(b, scorer, iters=args.iters, deg=args.deg,
                                 shift=args.shift)
                ends.append(apply(b, x))
            end = torch.as_tensor(np.stack(ends), device=device, dtype=dtype)

            dq_now = dockq_chunked(prot.rec_xyz, start, prot.native_lig,
                                   args.pose_chunk)
            dq_next = dockq_chunked(prot.rec_xyz, end, prot.native_lig,
                                    args.pose_chunk)
            rm_now = torch.cat([ligand_rmsd_to_native(prot.native_lig,
                                                      start[a:a + args.pose_chunk])
                                for a in range(0, start.shape[0], args.pose_chunk)])
            n = start.shape[0]
            pools.append({"name": pid, "sc": sc.cpu(), "T": T.cpu(),
                          "elec": elec.cpu(), "rmsd": rm_now.cpu(),
                          "dockq": dq_next.cpu(),        # the value-to-go label
                          "dockq_now": dq_now.cpu(),     # what the pose is worth now
                          "origin": torch.zeros(n, dtype=torch.int16),
                          "prov": torch.zeros(n, dtype=torch.int16),
                          "pose_key": torch.full((n, 4), POSE_IDENTITY_MISSING,
                                                 dtype=torch.int64)})
        except torch.cuda.OutOfMemoryError as exc:
            skipped.append((pid, f"OOM: {str(exc)[:100]}"))
        except Exception as exc:                                # noqa: BLE001
            skipped.append((pid, f"{type(exc).__name__}: {exc}"[:120]))
        finally:
            del prot
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(ids)}  ({(time.time() - t0) / (i + 1):.1f}s each)",
                  flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"n_skipped": len(skipped), "refiner_ckpt": args.ckpt,
                "pools": pools}, out)

    thr = 0.23
    live_now = sum(1 for p in pools if bool((p["dockq_now"] >= thr).any()))
    live_next = sum(1 for p in pools if bool((p["dockq"] >= thr).any()))
    n_now = sum(int((p["dockq_now"] >= thr).sum()) for p in pools)
    n_next = sum(int((p["dockq"] >= thr).sum()) for p in pools)
    print(f"\n{len(pools)} complexes written, {len(skipped)} skipped -> {out}")
    print(f"  complexes with an acceptable pose   now {live_now} / after {live_next}")
    print(f"  acceptable poses in total           now {n_now} / after {n_next}")
    print("  the gap between the two counts is the signal this pool exists to expose:")
    print("  poses that are not acceptable yet but refine into something that is.")
    for pid, why in skipped[:10]:
        print(f"  skipped {pid}: {why}")
    print(f"wall {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
