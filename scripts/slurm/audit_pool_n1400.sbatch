#!/usr/bin/env bash
#SBATCH --job-name=ddock_n1400_audit
#SBATCH --partition=all
#SBATCH --nodelist=m2
#SBATCH --exclude=floyd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/home/yasu/tmp/ddock/logs/pool_n1400_audit_%j.out
#SBATCH --error=/home/yasu/tmp/ddock/logs/pool_n1400_audit_%j.err

set -euo pipefail

REPO=/home/yasu/tmp/ddock
OLD_CACHE="$REPO/data/scaling/pool_cache"
NEW_CACHE="$REPO/data/scaling/pool_cache_n1400_20260814"
RUN_ROOT="$REPO/data/scaling/runs_n1400_expand_20260814"
AUDIT_JSON="$RUN_ROOT/pool_audit.json"

cd "$REPO" || exit 1
mkdir -p "$REPO/logs" "$RUN_ROOT"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export TMPDIR="/tmp/ddock-pool-audit-${SLURM_JOB_ID}"
export UV_CACHE_DIR="$TMPDIR/uv-cache"
mkdir -p "$TMPDIR" "$UV_CACHE_DIR"

echo "dependency: ${SLURM_JOB_DEPENDENCY:-not exported by Slurm}"
uv run python scripts/audit_pool_expansion.py \
  --n-fit 1400 --n-total 1750 --val-frac 0.2 \
  --spacing 1.2 \
  --grid-voxels data/scaling/grid_voxels_1.2.json \
  --max-grid-voxels 31250000 \
  --exclude-bad-geometry data/scaling/excluded_bad_geometry.txt \
  --exclude-homodimer \
  --master-ids data/scaling/master_ids.txt \
  --prep-manifest data/scaling/prep_manifest.jsonl \
  --split "$RUN_ROOT/N1400_seed0_mine0of3/split.json" \
  --split-check "$RUN_ROOT/N1400_seed0_mine1of3/split.json" \
  --split-check "$RUN_ROOT/N1400_seed0_mine2of3/split.json" \
  --pool-glob "$OLD_CACHE/n1000_r0_sp1.2_reachable_hopf3_ntop1500_nr8_tc1_a1.0_rho3.5_hd1_mv31250000_bg1_pk2.*of6.pt" \
  --pool-glob "$NEW_CACHE/n1400_r0_sp1.2_reachable_hopf3_ntop1500_nr8_tc1_a1.0_rho3.5_hd1_mv31250000_bg1_pk2.*of3.pt" \
  --skip-log-glob "$RUN_ROOT/N1400_seed0_mine0of3/skipped.jsonl" \
  --skip-log-glob "$RUN_ROOT/N1400_seed0_mine1of3/skipped.jsonl" \
  --skip-log-glob "$RUN_ROOT/N1400_seed0_mine2of3/skipped.jsonl" \
  --dockq-threshold 0.23 \
  --out "$AUDIT_JSON"
