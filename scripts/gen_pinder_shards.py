"""Generate PINDER decoy shards in parallel across multiple GPUs.

Splits a PINDER id list (one system id per line) into one chunk per GPU and
launches :mod:`scripts.build_decoy_dataset` (``--format pinder``) per chunk,
each writing its own HDF5 shard. Because the score is linear in the learnable
parameters we only need the per-pose features + (RMSD, DockQ) labels; the
shards are the exact schema :mod:`zdock.data` consumes.

The train/test membership is *not* chosen here — it is PINDER's own
interface-deleaked split (``pinder_s`` test vs sampled ``train`` clusters), so
shards tagged ``test`` are genuinely held out from ``train`` by FoldSeek/MMseqs
interface clustering + iAlign deleaking.

Example
-------
    PINDER_BASE_DIR=$PWD/external/pinder uv run python scripts/gen_pinder_shards.py \
        --ids-file data/pinder_test_ids.txt --tag test \
        --out-dir data/shards_pinder --gpus 0,1,2,3,4,5,6
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def _chunk(items, n):
    """Split ``items`` into ``n`` near-even contiguous chunks."""
    k, r = divmod(len(items), n)
    out, s = [], 0
    for i in range(n):
        e = s + k + (1 if i < r else 0)
        out.append(items[s:e])
        s = e
    return [c for c in out if c]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ids-file", required=True)
    ap.add_argument("--tag", required=True, help="shard name prefix, e.g. train/test")
    ap.add_argument("--out-dir", default=str(_REPO / "data" / "shards_pinder"))
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6")
    ap.add_argument("--n-random-rot", type=int, default=2000)
    ap.add_argument("--n-cone", type=int, default=400)
    ap.add_argument("--cone-deg", type=float, default=25.0)
    ap.add_argument("--ntop", type=int, default=2000)
    args = ap.parse_args()

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = out_dir / "_chunks"
    chunk_dir.mkdir(exist_ok=True)

    ids = [ln.strip() for ln in Path(args.ids_file).read_text().splitlines() if ln.strip()]
    chunks = _chunk(ids, len(gpus))
    print(f"{len(ids)} ids -> {len(chunks)} chunks over GPUs {gpus}", flush=True)

    procs = []
    log_dir = out_dir / "_logs"
    log_dir.mkdir(exist_ok=True)
    for i, (gpu, chunk) in enumerate(zip(gpus, chunks)):
        cf = chunk_dir / f"{args.tag}_gpu{gpu}.txt"
        cf.write_text("\n".join(chunk) + "\n")
        shard = out_dir / f"{args.tag}_gpu{gpu}.h5"
        # HDF5 flock() over networked filesystems raises errno 11 (EAGAIN)
        # under concurrent writers; each worker owns a distinct shard so
        # disabling the lock is safe here.
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
                   HDF5_USE_FILE_LOCKING="FALSE")
        cmd = [
            "uv", "run", "python", "scripts/build_decoy_dataset.py",
            "--format", "pinder", "--device", "cuda",
            "--ids-file", str(cf), "--output", str(shard),
            "--n-random-rot", str(args.n_random_rot),
            "--n-cone", str(args.n_cone),
            "--cone-deg", str(args.cone_deg),
            "--ntop", str(args.ntop),
        ]
        log = open(log_dir / f"{args.tag}_gpu{gpu}.log", "w")
        print(f"  GPU {gpu}: {len(chunk)} ids -> {shard.name} (log {log.name})", flush=True)
        procs.append((gpu, subprocess.Popen(cmd, cwd=_REPO, env=env,
                                            stdout=log, stderr=subprocess.STDOUT)))

    fail = 0
    for gpu, p in procs:
        rc = p.wait()
        print(f"  GPU {gpu} finished rc={rc}", flush=True)
        fail += int(rc != 0)
    print(f"done ({fail} worker(s) with nonzero rc)")


if __name__ == "__main__":
    main()
