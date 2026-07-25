"""Populate the PreparedProtein disk cache for the PINDER scaling-law runs.

Parses the holo receptor/ligand monomers of each PINDER system, derives the
atom-type / radius / SASA features and the native reference frame once, and
writes the result to a CPU disk cache (:mod:`zdock.prep_cache`). All seeds and
all cluster-count conditions then share the same prepared inputs, and a run
never re-downloads or re-parses anything.

Preparation is done on a GPU (SASA builds an ``(N, N)`` neighbour matrix, which
is slow on CPU) and the result is immediately moved to CPU before saving, so
GPU memory never holds more than one complex.

Failures are recorded, not raised: the manifest keeps ``status`` (``ok`` /
``fail`` / ``oom``), the exception type, and the atom counts, so the run script
can take "the first N *usable* clusters" and report exactly what was dropped.

Example
-------
    PINDER_BASE_DIR=$PWD/external/pinder uv run python scripts/prep_pinder_cache.py \\
        --ids-file data/scaling/master_ids.txt --limit 2800 --gpus 0,1,2,3,4,5,6
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


def _worker(args) -> None:
    import torch

    from zdock.dataset import prepare_protein_from_pdb
    from zdock.prep_cache import has_prepared, save_prepared

    device = torch.device(args.device)
    dtype = torch.float32
    ids = [ln.strip() for ln in Path(args.ids_file).read_text().splitlines() if ln.strip()]
    ids = ids[: args.limit] if args.limit else ids
    mine = [(r, i) for r, i in enumerate(ids) if r % args.n_workers == args.worker_id]

    out = open(args.manifest, "a", buffering=1)
    from pinder.core import PinderSystem

    for rank, pid in mine:
        if has_prepared(args.cache_dir, pid) and not args.force:
            continue
        t0 = time.time()
        rec = {"rank": rank, "id": pid, "worker": args.worker_id}
        try:
            ps = PinderSystem(pid, pdb_engine="biotite")
            prot = prepare_protein_from_pdb(
                pid, str(ps.holo_receptor.filepath), str(ps.holo_ligand.filepath),
                device=device, dtype=dtype)
            rec.update(status="ok", n_rec=prot.n_rec, n_lig=prot.n_lig,
                       reason="", stage="")
            save_prepared(args.cache_dir, prot)
            del prot
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            rec.update(status="oom", n_rec=-1, n_lig=-1,
                       reason=str(exc)[:200], stage="prepare")
        except Exception as exc:  # noqa: BLE001 — parse / download / LUT failures
            rec.update(status="fail", n_rec=-1, n_lig=-1,
                       reason=f"{type(exc).__name__}: {exc}"[:200], stage="prepare")
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()
        rec["seconds"] = round(time.time() - t0, 2)
        out.write(json.dumps(rec) + "\n")
    out.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ids-file", default="data/scaling/master_ids.txt", dest="ids_file")
    ap.add_argument("--cache-dir", default="data/scaling/prep_cache", dest="cache_dir")
    ap.add_argument("--manifest-dir", default="data/scaling/prep_manifest",
                    dest="manifest_dir")
    # Explicit: deriving this from --manifest-dir's parent once made a TEST-set
    # prep run overwrite the main corpus manifest.
    ap.add_argument("--merged-manifest", default="", dest="merged_manifest",
                    help="where to write the merged manifest "
                         "(default: <manifest-dir>.jsonl)")
    ap.add_argument("--limit", type=int, default=0,
                    help="only attempt the first N ids of the master list (0 = all)")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6")
    ap.add_argument("--jobs-per-gpu", type=int, default=2, dest="jobs_per_gpu")
    ap.add_argument("--force", action="store_true", help="re-prepare cached entries")
    # internal (worker mode)
    ap.add_argument("--worker-id", type=int, default=-1, dest="worker_id")
    ap.add_argument("--n-workers", type=int, default=1, dest="n_workers")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.worker_id >= 0:
        _worker(args)
        return

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    cpu_only = gpus == ["cpu"]
    if not cpu_only and any(not g.isdigit() for g in gpus):
        raise SystemExit(f"--gpus must be numeric indices or the word 'cpu'; got {gpus}")
    n_workers = len(gpus) * args.jobs_per_gpu
    mdir = Path(args.manifest_dir)
    mdir.mkdir(parents=True, exist_ok=True)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    procs = []
    for w in range(n_workers):
        gpu = gpus[w % len(gpus)]
        env = dict(os.environ,
                   CUDA_VISIBLE_DEVICES="" if cpu_only else gpu,
                   HDF5_USE_FILE_LOCKING="FALSE")
        cmd = [sys.executable, __file__,
               "--ids-file", args.ids_file, "--cache-dir", args.cache_dir,
               "--limit", str(args.limit),
               "--worker-id", str(w), "--n-workers", str(n_workers),
               "--manifest", str(mdir / f"worker{w:02d}.jsonl"),
               "--device", "cpu" if cpu_only else "cuda"]
        if args.force:
            cmd.append("--force")
        log = open(mdir / f"worker{w:02d}.log", "a")
        procs.append((w, subprocess.Popen(cmd, cwd=_REPO, env=env,
                                          stdout=log, stderr=subprocess.STDOUT)))
        print(f"  worker {w} -> GPU {gpu}", flush=True)

    fail = 0
    for w, p in procs:
        rc = p.wait()
        fail += int(rc != 0)
        print(f"  worker {w} rc={rc}", flush=True)

    # Merge manifests, newest record per id wins.
    recs: dict[str, dict] = {}
    for f in sorted(mdir.glob("worker*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                recs[r["id"]] = r
    merged = Path(args.merged_manifest) if args.merged_manifest \
        else mdir.with_suffix(".jsonl")
    if merged.resolve() == mdir.resolve():
        raise SystemExit("merged manifest would overwrite the manifest dir")
    with open(merged, "w") as fh:
        for r in sorted(recs.values(), key=lambda x: x["rank"]):
            fh.write(json.dumps(r) + "\n")
    n_ok = sum(r["status"] == "ok" for r in recs.values())
    print(f"attempted {len(recs)}  ok {n_ok}  "
          f"fail {sum(r['status']=='fail' for r in recs.values())}  "
          f"oom {sum(r['status']=='oom' for r in recs.values())}  "
          f"({fail} workers with nonzero rc)")
    print(f"manifest -> {merged}")


if __name__ == "__main__":
    main()
