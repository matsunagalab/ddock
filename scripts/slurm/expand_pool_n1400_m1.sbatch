#!/usr/bin/env bash
#SBATCH --job-name=ddock_n1400_m1_s0
#SBATCH --partition=all
#SBATCH --nodelist=m1
#SBATCH --exclude=floyd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:8000:1
#SBATCH --mem=12G
#SBATCH --time=14:00:00
#SBATCH --output=/home/yasu/tmp/ddock/logs/pool_n1400_m1_%j.out
#SBATCH --error=/home/yasu/tmp/ddock/logs/pool_n1400_m1_%j.err

set -euo pipefail

REPO=/home/yasu/tmp/ddock
OLD_CACHE="$REPO/data/scaling/pool_cache"
NEW_CACHE="$REPO/data/scaling/pool_cache_n1400_20260814"
RUN_ROOT="$REPO/data/scaling/runs_n1400_expand_20260814"

cd "$REPO" || exit 1
mkdir -p "$REPO/logs" "$NEW_CACHE" "$RUN_ROOT"

export PINDER_BASE_DIR="$REPO/external/pinder"
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export TMPDIR="/tmp/ddock-pool-${SLURM_JOB_ID}"
export UV_CACHE_DIR="$TMPDIR/uv-cache"
mkdir -p "$TMPDIR" "$UV_CACHE_DIR"

COMMON=(
  uv run python -u scripts/run_pinder_scaling.py
  --n-fit 1400
  --n-total 1750
  --val-frac 0.2
  --seed 0
  --rounds 0
  --mine-only
  --fail-on-skip
  --device cuda
  --alpha0 1.0
  --rho0 3.5
  --pool reachable
  --rot-set hopf
  --hopf-nside 3
  --spacing 1.2
  --mine-ntop 1500
  --near-rot 8
  --trans-cells 1
  --psc-decompose
  --frame-chunk 100
  --rot-chunk 2
  --feature-budget 1000000000
  --dockq-budget 50000000
  --oom-retries 3
  --grid-voxels data/scaling/grid_voxels_1.2.json
  --max-grid-voxels 31250000
  --exclude-bad-geometry data/scaling/excluded_bad_geometry.txt
  --exclude-homodimer
  --prep-cache data/scaling/prep_cache
  --prep-manifest data/scaling/prep_manifest.jsonl
  --master-ids data/scaling/master_ids.txt
  --test-cache data/shards_pinder/test_pool_reachable.pt
  --test-ids data/pinder_test_ids.txt
  --pool-cache "$NEW_CACHE"
  --pool-cache-read "$OLD_CACHE"
  --out-dir "$RUN_ROOT"
)

shard_log="$REPO/logs/pool_n1400_${SLURM_JOB_ID}_shard0of3.log"
echo "launching m1 shard 0/3 -> $shard_log"
srun --nodes=1 --ntasks=1 --cpus-per-task=1 --gres=gpu:8000:1 \
  "${COMMON[@]}" --mine-shard 0/3 >"$shard_log" 2>&1
