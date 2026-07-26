"""Tabulate the FFT grid volume of every prepared complex.

``docking_search`` builds ~15 grids per rotation chunk (SC+, SC-, 12 atom-type
IFACE grids, ELEC) on a lattice whose size is fixed by the receptor+ligand
bounding box at 3 Å spacing, so both the run time and the peak VRAM of a
complex scale with this voxel count — not with its atom count. The corpus has
an extreme tail (a 200 Å coiled-coil ligand produces a 153x159x167 grid, ~20x
the median, and the worst entry is ~1.2e10 voxels), and those complexes stall a
run for tens of minutes even after the OOM ladder drops to ``rot_chunk=1``.

The scaling experiment therefore applies a **deterministic** grid-volume cutoff
at selection time: it is a property of the structure alone, identical for every
seed and every N, so the nested-prefix property of the subsets is preserved.
Excluding on measured OOM instead would make set membership depend on transient
GPU pressure.

Example
-------
    uv run python scripts/compute_grid_sizes.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from zdock.geom import grid_shape
from zdock.prep_cache import load_prepared
from zdock.score import SC_REFERENCE_SPACING


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prep-manifest", default="data/scaling/prep_manifest.jsonl",
                    dest="prep_manifest")
    ap.add_argument("--prep-cache", default="data/scaling/prep_cache", dest="prep_cache")
    ap.add_argument("--spacing", type=float, default=SC_REFERENCE_SPACING)
    ap.add_argument("--out", default="data/scaling/grid_voxels.json")
    args = ap.parse_args()

    ids = []
    for line in Path(args.prep_manifest).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["status"] == "ok":
                ids.append(r["id"])

    dev = torch.device("cpu")
    out: dict[str, int] = {}
    for pid in ids:
        prot = load_prepared(args.prep_cache, pid)
        if prot is None:
            continue
        # `grid_shape`, not `generate_grid`: the tail of this corpus reaches
        # ~2e11 voxels at 1.2 A, and allocating those two zero grids just to
        # read their shape gets the process SIGKILLed (measured).
        nx, ny, nz = grid_shape(prot.rec_xyz, prot.lig_ref, spacing=args.spacing)
        out[pid] = nx * ny * nz

    Path(args.out).write_text(json.dumps(out))
    vals = sorted(out.values())
    n = len(vals)
    def pct(q):
        return vals[min(n - 1, int(q / 100 * n))]
    print(f"n={n}  median={pct(50)}  p90={pct(90)}  p95={pct(95)}  "
          f"p99={pct(99)}  max={vals[-1]}")
    for thr in (1_000_000, 2_000_000, 3_000_000):
        k = sum(v > thr for v in vals)
        print(f"  > {thr:>10,}: {k} complexes ({100*k/n:.2f}%) excluded")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
