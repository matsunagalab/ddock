"""Turn refined poses back into a training pool.

The table in use was fitted on poses that sit on the rotation grid, because that
is all the search returns. Refinement moves every candidate off the grid to a
local optimum of the score, so the poses being ranked at deployment are drawn
from a different distribution than the poses the table was fitted on -- and
nobody has checked whether that matters.

This reads a refined submission directory, recomputes the score's own features
for each pose, labels them with DockQ, and writes a pool file in the same shape
as the mining cache. `qp_path.py` and `capacity_series.py` then fit on refined
poses with no further changes.

Why the features have to be recomputed rather than reused: the mining cache
stores (S_PSC, T, S_ELEC) for the grid poses. A refined pose is a different
geometry, so its contact counts are different; nothing about it survives from
the cached entry except the complex it belongs to.

Provenance is written as 0 (search-derived) throughout, since every pose here
descends from one the search returned. `pose_key` is left as the missing marker:
refined poses are off-grid, so a rotation index and translation cell do not
exist for them, and pretending otherwise would let the de-duplication in
`run_pinder_scaling` compare things that are not comparable.

Example
-------
    uv run python scripts/build_refined_pool.py \
        --in-dir data/scaling/q1_top50_refined \
        --prep-cache data/scaling/prep_cache \
        --out data/scaling/pool_cache/refined_top50_fit.pt
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

from refine_poses import read_ligand  # noqa: E402
from zdock.atomtypes import charge_score as default_charge_score  # noqa: E402
from zdock.atomtypes import iface_ij  # noqa: E402
from zdock.dataset import POSE_IDENTITY_MISSING  # noqa: E402
from zdock.dockq import dockq_batch, ligand_rmsd_to_native  # noqa: E402
from zdock.prep_cache import load_prepared  # noqa: E402
from zdock.score import SC_REFERENCE_SPACING, docking_score_elec  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", required=True, dest="in_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prep-cache", default="data/scaling/prep_cache",
                    dest="prep_cache")
    ap.add_argument("--monomer", default="holo")
    ap.add_argument("--ids-file", default="", dest="ids_file",
                    help="restrict to these complexes (default: everything in --in-dir)")
    ap.add_argument("--spacing", type=float, default=SC_REFERENCE_SPACING)
    ap.add_argument("--frame-chunk", type=int, default=8, dest="frame_chunk")
    ap.add_argument("--pose-chunk", type=int, default=8, dest="pose_chunk")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32
    beta = torch.tensor(3.0, device=device, dtype=dtype)
    charge = default_charge_score(device=device, dtype=dtype)
    iface0 = iface_ij(device=device, dtype=dtype, flat=True)

    src = Path(args.in_dir)
    ids = sorted(p.name for p in src.iterdir() if p.is_dir())
    if args.ids_file:
        want = {ln.strip() for ln in Path(args.ids_file).read_text().splitlines()
                if ln.strip()}
        ids = [i for i in ids if i in want]
    si, sn = (int(x) for x in args.shard.split("/"))
    ids = ids[si::sn]
    print(f"{len(ids)} complexes from {src}", flush=True)

    pools, skipped, t0 = [], [], time.time()
    for i, pid in enumerate(ids):
        prot = load_prepared(args.prep_cache, pid)
        if prot is None:
            skipped.append((pid, "not in prep cache"))
            continue
        prot = prot.to(device, dtype=dtype)
        try:
            models = sorted((src / pid / args.monomer / "models").glob("model_*.pdb"),
                            key=lambda p: int(p.stem.split("_")[1]))
            xyz = np.stack([read_ligand(m) for m in models])
            if xyz.shape[1] != prot.n_lig:
                raise ValueError(f"{xyz.shape[1]} ligand atoms against {prot.n_lig}")
            poses = torch.as_tensor(xyz, device=device, dtype=dtype)
            sc, T, elec = docking_score_elec(
                prot.rec_xyz, prot.rec_radius, prot.rec_sasa,
                prot.rec_atomtype_id, prot.rec_charge_id,
                poses, prot.lig_radius, prot.lig_sasa,
                prot.lig_atomtype_id, prot.lig_charge_id,
                torch.zeros((), device=device, dtype=dtype), iface0, beta, charge,
                lig_xyz_for_grid=prot.lig_ref, spacing=args.spacing,
                frame_chunk_size=args.frame_chunk, return_components=True,
                psc_decompose=True)
            dq = torch.cat([dockq_batch(prot.rec_xyz, poses[a:a + args.pose_chunk],
                                        prot.native_lig).dockq
                            for a in range(0, poses.shape[0], args.pose_chunk)])
            rmsd = torch.cat([ligand_rmsd_to_native(prot.native_lig,
                                                    poses[a:a + args.pose_chunk])
                              for a in range(0, poses.shape[0], args.pose_chunk)])
            n = poses.shape[0]
            pools.append({"name": pid, "sc": sc.cpu(), "T": T.cpu(),
                          "elec": elec.cpu(), "rmsd": rmsd.cpu(), "dockq": dq.cpu(),
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
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(ids)}  ({(time.time() - t0) / (i + 1):.1f}s each)",
                  flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"n_skipped": len(skipped), "pools": pools}, out)
    npos = sum(int((p["dockq"] >= 0.23).sum()) for p in pools)
    live = sum(1 for p in pools if bool((p["dockq"] >= 0.23).any()))
    print(f"\n{len(pools)} complexes written, {len(skipped)} skipped")
    print(f"  with at least one acceptable pose: {live}/{len(pools)}")
    print(f"  acceptable poses in total        : {npos}")
    for pid, why in skipped[:10]:
        print(f"  skipped {pid}: {why}")
    print(f"wall {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
