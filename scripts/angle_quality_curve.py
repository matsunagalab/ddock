"""How much orientation error can a pose carry and still count as High?

The search can only return orientations from a fixed grid, so every returned
pose carries a residual rotation error -- a median 9.5 degrees at Hopf nside=3.
Measured officially, that grid tops out at 39.6% High while the method already
reaches 38.0%: quality is limited by the grid, not by the scoring function
(report section 5.14.31).

To decide what a finer grid would buy, the quality has to be measured as a
function of the error itself. For each complex this takes the NATIVE pose,
rotates the ligand by exactly `theta` degrees about a random axis, re-centres it
on the native centroid, and submits that. Everything else is ideal: the position
is the best one, and the orientation error is the only defect.

theta = 0 therefore reproduces the native pose and must score DockQ = 1. Each
angle becomes its own PINDER submission, so the official harness measures the
curve in the same units as every other number in the report.

Random axis with a fixed angle is the model for "the nearest grid point is theta
away"; the grid's own axis distribution is not uniform, so this is an
approximation, but the angle is what varies by orders of magnitude between grid
resolutions.

Example
-------
    uv run python scripts/angle_quality_curve.py \
        --angles 0,2,4,6,8,10,12 --export-root data/pinder_eval/angle_curve
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_search_test import (ExportError, _usable_atom_lines,  # noqa: E402
                              _write_decoy)
from zdock.prep_cache import load_prepared  # noqa: E402


def rotation_matrix(axis: np.ndarray, deg: float) -> np.ndarray:
    """Rodrigues: rotate by `deg` about a unit `axis`."""
    a = axis / np.linalg.norm(axis)
    t = np.deg2rad(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--angles", default="0,2,4,6,8,10,12")
    ap.add_argument("--test-ids", default="data/pinder_test_ids.txt", dest="test_ids")
    ap.add_argument("--prep-cache", default="data/scaling/prep_cache_test",
                    dest="prep_cache")
    ap.add_argument("--pdb-dir", default="external/pinder/pinder/2024-02/pdbs",
                    dest="pdb_dir")
    ap.add_argument("--export-root", required=True, dest="export_root")
    ap.add_argument("--monomer", default="holo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    angles = [float(x) for x in args.angles.split(",")]
    ids = [ln.strip() for ln in Path(args.test_ids).read_text().splitlines() if ln.strip()]
    if args.limit:
        ids = ids[: args.limit]
    root = Path(args.export_root)
    pdb_dir = Path(args.pdb_dir)
    print(f"{len(ids)} complexes x {len(angles)} angles -> {root}")

    t0 = time.time()
    written = {a: 0 for a in angles}
    skipped = []
    for i, pid in enumerate(ids):
        prot = load_prepared(args.prep_cache, pid)
        if prot is None:
            skipped.append((pid, "not in prep cache"))
            continue
        try:
            nat = prot.native_lig.double().numpy()
            rec = prot.rec_xyz.detach().cpu()
            cen = nat.mean(axis=0)
            src = pdb_dir / f"{pid}.pdb"
            rec_lines = _usable_atom_lines(src, "R")
            lig_lines = _usable_atom_lines(src, "L")
            if len(rec_lines) != prot.n_rec or len(lig_lines) != prot.n_lig:
                raise ExportError(f"{pid}: PDB/prep atom mismatch")
            # one axis per complex, shared across angles, so the curve is not
            # confounded by which direction the error happens to point
            rng = np.random.default_rng(args.seed + i)
            axis = rng.normal(size=3)
            for a in angles:
                R = rotation_matrix(axis, a)
                pose = (nat - cen) @ R.T + cen
                out = root / f"err{int(a):02d}deg" / pid / args.monomer / "models"
                out.mkdir(parents=True, exist_ok=True)
                for stale in out.glob("*.pdb"):
                    stale.unlink()
                tmp = out / ".model_1.pdb.tmp"
                _write_decoy(tmp, rec_lines, lig_lines, rec,
                             torch.tensor(pose, dtype=torch.float32))
                tmp.rename(out / "model_1.pdb")
                written[a] += 1
        except ExportError:
            raise
        except Exception as exc:                                # noqa: BLE001
            skipped.append((pid, f"{type(exc).__name__}: {exc}"[:120]))
        finally:
            del prot
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(ids)}", flush=True)

    for a in angles:
        print(f"  {a:5.1f} deg: {written[a]} poses")
    for pid, why in skipped:
        print(f"  skipped {pid}: {why}")
    print(f"wall {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
