#!/usr/bin/env bash
#SBATCH --job-name=ddock_n1400_m2_s12
#SBATCH --partition=all
#SBATCH --nodelist=m2
#SBATCH --exclude=floyd
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --mem=24G
#SBATCH --time=18:00:00
#SBATCH --output=/home/yasu/tmp/ddock/logs/pool_n1400_m2_%j.out
#SBATCH --error=/home/yasu/tmp/ddock/logs/pool_n1400_m2_%j.err

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
  --frame-chunk 50
  --rot-chunk 1
  --feature-budget 400000000
  --dockq-budget 25000000
  --oom-retries 5
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

pids=()
for shard in 1 2; do
  shard_log="$REPO/logs/pool_n1400_${SLURM_JOB_ID}_shard${shard}of3.log"
  echo "launching m2 shard ${shard}/3 -> $shard_log"
  srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=1 --gres=gpu:1 \
    "${COMMON[@]}" --mine-shard "${shard}/3" >"$shard_log" 2>&1 &
  pids+=("$!")
done

set +e
wait "${pids[0]}"
status1=$?
wait "${pids[1]}"
status2=$?
set -e
echo "worker status: shard1=$status1 shard2=$status2"
if (( status1 != 0 || status2 != 0 )); then
  echo "one or more m2 shards failed; dependent afterany audit will enumerate missing IDs" >&2
  exit 1
fi
