"""What is the best pose the rotation grid can express, measured officially?

The search only ever returns orientations from a fixed set -- 1944 of them at
Hopf nside=3 -- so the true orientation is never on the grid. The nearest grid
point is a median 9.5 degrees away, and that residual caps how good any returned
pose can be, whatever the scoring function does.

`eval_search_test.py` already reports this ceiling, but with the repository's own
DockQ, which tracks the official one closely (Pearson 0.992) while sitting a mean
0.117 below it. Near the acceptable threshold that hardly matters (75.6% against
76.8% on the same poses); at the High threshold it changes everything (6.0%
against 39.6%). So a ceiling for High computed that way is meaningless.

This writes the ceiling pose itself -- the ligand placed at its nearest grid
rotation and the translation that best matches the native -- as a PINDER
submission, so the official harness can score it. The result is an upper bound
on Max(Top 1) for this rotation grid, in the same units as every other number in
the report.

No search is involved: the rotation is chosen by angle to the native quaternion
and the translation is the analytic one, so this costs a rotation per complex.

Example
-------
    uv run python scripts/export_rotation_ceiling.py \
        --export-pdb-dir data/pinder_eval/rotation_ceiling
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_search_test import (ExportError, _quat_angle_deg,  # noqa: E402
                              _usable_atom_lines, _write_decoy)
from zdock.prep_cache import load_prepared  # noqa: E402
from zdock.rotation_grid import hopf_quaternions  # noqa: E402
from zdock.search import _rotate_batch  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-ids", default="data/pinder_test_ids.txt", dest="test_ids")
    ap.add_argument("--prep-cache", default="data/scaling/prep_cache_test",
                    dest="prep_cache")
    ap.add_argument("--pdb-dir", default="external/pinder/pinder/2024-02/pdbs",
                    dest="pdb_dir")
    ap.add_argument("--export-pdb-dir", required=True, dest="export_pdb_dir")
    ap.add_argument("--hopf-nside", type=int, default=3, dest="hopf_nside")
    ap.add_argument("--monomer", default="holo")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32
    quats = hopf_quaternions(args.hopf_nside, device=device, dtype=dtype)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    print(f"rotation grid: nside={args.hopf_nside} -> {quats.shape[0]} orientations")

    ids = [ln.strip() for ln in Path(args.test_ids).read_text().splitlines() if ln.strip()]
    if args.limit:
        ids = ids[: args.limit]
    root = Path(args.export_pdb_dir)
    pdb_dir = Path(args.pdb_dir)
    t0 = time.time()
    done, skipped, angles = 0, [], []
    for i, pid in enumerate(ids):
        prot = load_prepared(args.prep_cache, pid)
        if prot is None:
            skipped.append((pid, "not in prep cache"))
            continue
        prot = prot.to(device, dtype=dtype)
        try:
            ang = _quat_angle_deg(quats, prot.q_star)
            k = int(ang.argmin())
            angles.append(float(ang[k]))
            # best grid rotation, then the translation that matches the native
            # centroid -- the best this grid can do for this complex
            rot = _rotate_batch(prot.lig_ref, quats[k:k + 1])[0]
            pose = rot + (prot.native_lig - rot).mean(dim=0)

            src = pdb_dir / f"{pid}.pdb"
            rec_lines = _usable_atom_lines(src, "R")
            lig_lines = _usable_atom_lines(src, "L")
            if len(rec_lines) != prot.n_rec or len(lig_lines) != prot.n_lig:
                raise ExportError(f"{pid}: PDB/prep atom mismatch")
            out = root / pid / args.monomer / "models"
            out.mkdir(parents=True, exist_ok=True)
            for stale in out.glob("*.pdb"):
                stale.unlink()
            tmp = out / ".model_1.pdb.tmp"
            _write_decoy(tmp, rec_lines, lig_lines,
                         prot.rec_xyz.detach().cpu(), pose.detach().cpu())
            tmp.rename(out / "model_1.pdb")
            done += 1
        except ExportError:
            raise
        except Exception as exc:                                # noqa: BLE001
            skipped.append((pid, f"{type(exc).__name__}: {exc}"[:120]))
        finally:
            del prot
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(ids)}", flush=True)

    a = sorted(angles)
    print(f"\nwrote {done} ceiling poses, skipped {len(skipped)}")
    if a:
        print(f"angle to the nearest grid rotation: median {a[len(a) // 2]:.1f} deg, "
              f"max {a[-1]:.1f} deg")
    for pid, why in skipped:
        print(f"  skipped {pid}: {why}")
    print(f"wall {(time.time() - t0) / 60:.1f} min -> {root}")


if __name__ == "__main__":
    main()
