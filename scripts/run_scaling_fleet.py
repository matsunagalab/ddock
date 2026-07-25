"""Queue runner for the PINDER cluster-count scaling experiment.

Each (N_fit, seed) job is an independent process pinned to one GPU with its own
output directory and checkpoints — no two jobs ever write the same file. Jobs
are dispatched longest-first onto a pool of GPU slots so the long N=2,000 runs
start immediately and the short ones backfill.

Peak VRAM is ~25 GiB for the worst complex at the default ``--rot-chunk 8``, so
the default is **one job per GPU**. ``--jobs-per-gpu 2`` works only if the
per-complex OOM-retry ladder is acceptable (it halves the chunk sizes and
retries, which costs wall time but does not lose the complex).

Example
-------
    PINDER_BASE_DIR=$PWD/external/pinder uv run python scripts/run_scaling_fleet.py \\
        --jobs 500:0,500:1,500:2 --gpus 0,1,2
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", required=True,
                    help="comma-separated N:seed pairs, e.g. 500:0,500:1,1000:0")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6")
    ap.add_argument("--jobs-per-gpu", type=int, default=1, dest="jobs_per_gpu")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--out-dir", default="data/scaling/runs", dest="out_dir")
    ap.add_argument("--log-dir", default="data/scaling/logs", dest="log_dir")
    ap.add_argument("--extra", default="", help="extra args forwarded verbatim")
    args = ap.parse_args()

    jobs = []
    for tok in args.jobs.split(","):
        tok = tok.strip()
        if not tok:
            continue
        n, s = tok.split(":")
        jobs.append((int(n), int(s)))
    # longest first
    jobs.sort(key=lambda t: (-t[0], t[1]))

    slots = [g.strip() for g in args.gpus.split(",") if g.strip()] * args.jobs_per_gpu
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    pending = list(jobs)
    running: list[tuple] = []
    done: list[tuple] = []
    free = list(slots)
    print(f"{len(jobs)} jobs over {len(slots)} slots (GPUs {args.gpus}, "
          f"{args.jobs_per_gpu}/GPU)", flush=True)

    while pending or running:
        while pending and free:
            n, seed = pending.pop(0)
            gpu = free.pop(0)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
                       HDF5_USE_FILE_LOCKING="FALSE")
            cmd = [sys.executable, "-u", str(_REPO / "scripts" / "run_pinder_scaling.py"),
                   "--n-fit", str(n), "--seed", str(seed),
                   "--rounds", str(args.rounds), "--out-dir", args.out_dir]
            if args.extra:
                cmd += args.extra.split()
            log = open(log_dir / f"N{n}_seed{seed}.log", "w")
            p = subprocess.Popen(cmd, cwd=_REPO, env=env, stdout=log,
                                 stderr=subprocess.STDOUT)
            running.append((n, seed, gpu, p, time.time()))
            print(f"[{time.strftime('%H:%M:%S')}] start N={n} seed={seed} "
                  f"GPU={gpu} -> {log.name}", flush=True)

        time.sleep(20)
        still = []
        for n, seed, gpu, p, t0 in running:
            rc = p.poll()
            if rc is None:
                still.append((n, seed, gpu, p, t0))
            else:
                el = (time.time() - t0) / 60
                print(f"[{time.strftime('%H:%M:%S')}] done  N={n} seed={seed} "
                      f"GPU={gpu} rc={rc} ({el:.1f} min)", flush=True)
                done.append((n, seed, rc, el))
                free.append(gpu)
        running = still

    bad = [d for d in done if d[2] != 0]
    print(f"\nall finished: {len(done)} jobs, {len(bad)} nonzero rc")
    for n, seed, rc, el in sorted(done):
        print(f"  N={n:<5} seed={seed}  rc={rc}  {el:.1f} min")


if __name__ == "__main__":
    main()
