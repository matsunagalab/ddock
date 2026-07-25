"""Weight-aware dispatcher for the remaining scaling-law jobs.

The GPUs on this host are shared with unrelated long-running MD simulations, so
throughput is very uneven: ``nvidia-smi pmon`` shows this experiment getting
80-95% of the SMs on an uncontended GPU but only 10-36% on one already running
an MD job. A naive round-robin puts a 4,500-complex N=2,000 job on a contended
GPU and the whole experiment waits ~5x longer than necessary for it.

This dispatcher polls which GPUs currently have no scaling job of ours, ranks
them by *measured* contention (other processes' memory use is the proxy —
an idle GPU has none), and hands the heaviest pending job to the least
contended free GPU. It never starts a second job on a GPU that already runs
one, and never launches the same (N, seed) twice.

Example
-------
    uv run python scripts/scaling_dispatcher.py \\
        --jobs 2000:0,2000:1,2000:2,1000:1,1000:2 --gpus 0,1,2,3,4,5,6
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


_MARKER = "run_pinder_scaling.py"


def _nvsmi(query: str) -> list[list[str]]:
    out = subprocess.run(["nvidia-smi", f"--query-{query.split(':')[0]}={query.split(':')[1]}",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return [[t.strip() for t in line.split(",")]
            for line in out.strip().splitlines() if line.strip()]


def _uuid_to_index() -> dict[str, str]:
    return {u: i for i, u in _nvsmi("gpu:index,gpu_uuid")}


def _is_ours(pid: str) -> bool:
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="ignore")
    except OSError:
        return False
    return _MARKER in cmd


def _gpu_state() -> tuple[dict[str, int], set[str]]:
    """Return (MiB used per GPU, set of GPUs already running one of our jobs).

    Any GPU carrying a scaling job — whether this dispatcher started it or an
    earlier fleet did — is off limits, so a card never gets two of them.
    """
    used = {i: int(m) for i, m in _nvsmi("gpu:index,memory.used")}
    idx_of = _uuid_to_index()
    ours: set[str] = set()
    for row in _nvsmi("compute-apps:pid,gpu_uuid"):
        pid, uuid = row[0], row[1]
        if _is_ours(pid) and uuid in idx_of:
            ours.add(idx_of[uuid])
    return used, ours


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", required=True, help="comma-separated N:seed pairs")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--out-dir", default="data/scaling/runs", dest="out_dir")
    ap.add_argument("--log-dir", default="data/scaling/logs", dest="log_dir")
    ap.add_argument("--poll", type=int, default=60)
    # A heavy job on a GPU shared with an MD run measured ~19 s/complex vs
    # ~6 s/complex on a free one. For N=2,000 (4,500 searches) that is 24 h
    # versus 7.5 h — worse than leaving the job queued until a free GPU opens.
    ap.add_argument("--heavy-threshold", type=int, default=1000,
                    dest="heavy_threshold",
                    help="jobs with n_fit >= this only run on uncontended GPUs")
    ap.add_argument("--contended-mib", type=int, default=400,
                    dest="contended_mib",
                    help="a GPU with more than this much *foreign* memory in "
                         "use counts as contended")
    args = ap.parse_args()

    pending = []
    for tok in args.jobs.split(","):
        if tok.strip():
            n, s = tok.strip().split(":")
            pending.append((int(n), int(s)))
    pending.sort(key=lambda t: (-t[0], t[1]))      # heaviest first
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    running: dict[str, tuple] = {}                  # gpu -> (n, seed, proc, t0)
    print(f"dispatcher: {len(pending)} pending job(s) over GPUs {gpus}", flush=True)

    while pending or running:
        for gpu in list(running):
            n, seed, p, t0 = running[gpu]
            rc = p.poll()
            if rc is not None:
                print(f"[{time.strftime('%H:%M:%S')}] done N={n} seed={seed} "
                      f"GPU={gpu} rc={rc} ({(time.time()-t0)/60:.1f} min)", flush=True)
                del running[gpu]

        if pending:
            used, busy_with_ours = _gpu_state()
            busy_with_ours |= set(running)
            free = sorted((used.get(g, 0), g) for g in gpus
                          if g not in busy_with_ours)
            for mib, gpu in free:
                # Take the heaviest job this GPU is allowed to run; a heavy job
                # stays queued rather than crawl on a contended card.
                pick = next((i for i, (n, _) in enumerate(pending)
                             if n < args.heavy_threshold
                             or mib <= args.contended_mib), None)
                if pick is None:
                    continue
                n, seed = pending.pop(pick)
                run_dir = Path(args.out_dir) / f"N{n}_seed{seed}"
                if (run_dir / "summary.csv").exists():
                    print(f"  skip N={n} seed={seed}: already complete", flush=True)
                    continue
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
                           HDF5_USE_FILE_LOCKING="FALSE")
                cmd = [sys.executable, "-u",
                       str(_REPO / "scripts" / "run_pinder_scaling.py"),
                       "--n-fit", str(n), "--seed", str(seed),
                       "--rounds", str(args.rounds), "--out-dir", args.out_dir]
                log = open(log_dir / f"N{n}_seed{seed}.log", "w")
                p = subprocess.Popen(cmd, cwd=_REPO, env=env, stdout=log,
                                     stderr=subprocess.STDOUT)
                running[gpu] = (n, seed, p, time.time())
                print(f"[{time.strftime('%H:%M:%S')}] start N={n} seed={seed} "
                      f"GPU={gpu} (foreign memory {mib} MiB)",
                      flush=True)
        time.sleep(args.poll)

    print("dispatcher: all jobs finished")


if __name__ == "__main__":
    main()
