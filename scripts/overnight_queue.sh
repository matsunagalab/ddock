#!/usr/bin/env bash
# Chain the three overnight jobs behind the official N-scaling searches.
#
#   A  PINDER `apo` docking of the 93 PINDER-S systems that have both apo
#      monomers, for the published table and the N=220 seed 0 table, then one
#      harness pass that scores every method and both monomer settings.
#   C  N=1000 hard-negative mining, shards 1/6 2/6 4/6 5/6. Shards 0 and 3 are
#      already cached; these four were interrupted, and `--mine-only` writes its
#      cache only when a shard completes, so each restarts from its first
#      complex. At ~150 s/complex and 208 complexes per shard that is ~8.7 h per
#      shard, two waves on three GPUs -- it will NOT be done by morning.
#
# Runs unattended. Each stage logs to data/scaling/logs/.
#
#   bash scripts/overnight_queue.sh [gpu_list]     # default 0,3,6
set -u -o pipefail

cd "$(dirname "$0")/.."
GPUS=${1:-0,3,6}
IFS=',' read -r -a GPU <<< "$GPUS"
NG=${#GPU[@]}
LOG=data/scaling/logs
mkdir -p "$LOG"
say() { echo "[$(date +%m-%d\ %H:%M)] $*"; }

# ---- wait for the official N-scaling searches -----------------------------
say "waiting for run_official_nscaling.sh"
while pgrep -f "run_official_nscaling.sh" > /dev/null; do sleep 120; done
while pgrep -f "scripts/eval_search_test.py" > /dev/null; do sleep 120; done
say "N-scaling stage clear"

# ---- A: apo ---------------------------------------------------------------
run_apo() {                                  # $1 = method dir, $2 = ckpt or ""
    local method=$1 ck=$2 name
    name=$(basename "$method")
    for ((i = 0; i < NG; i++)); do
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES=${GPU[$i]} setsid nohup uv run python \
            scripts/eval_search_apo.py \
            --shard "$i/$NG" ${ck:+--ckpt "$ck"} --label "${name}_sh$i" \
            --export-pdb-dir "$method" --export-top-k 5 \
            > "$LOG/apo_${name}_sh$i.log" 2>&1 < /dev/null &
    done
    wait
}

say "=== A: apo, published table ==="
run_apo data/pinder_eval/published ""
say "=== A: apo, trained_N220 ==="
run_apo data/pinder_eval/trained_N220 data/scaling/runs_nfixed/N220_seed0/round0_ckpt.pt

say "=== A: scoring every method (holo + apo) ==="
PINDER_BASE_DIR=$PWD/external/pinder uv run python \
    scripts/score_decoys_with_pinder.py --eval-dir data/pinder_eval \
    --out data/pinder_eval/leaderboard.csv --max-workers 16 \
    > "$LOG/apo_harness.log" 2>&1
say "A done"

# ---- C: N=1000 mining -----------------------------------------------------
mine() {                                     # $1.. = shard indices
    local n=0 s
    for s in "$@"; do
        CUDA_VISIBLE_DEVICES=${GPU[$((n % NG))]} setsid nohup uv run python \
            scripts/run_pinder_scaling.py \
            --n-fit 1000 --rounds 0 --seed 0 --alpha0 1.0 \
            --mine-shard "$s/6" --mine-only \
            --grid-voxels data/scaling/grid_voxels_1.2.json \
            --max-grid-voxels 31250000 \
            --test-cache data/shards_pinder/test_pool_reachable.pt \
            --out-dir data/scaling/runs_nscale_mine \
            > "$LOG/nscale_mine_N1000_shard$s.log" 2>&1 < /dev/null &
        n=$((n + 1))
    done
    wait
}

say "=== C: N=1000 mining, wave 1 (shards 1 2 4) ==="
mine 1 2 4
say "=== C: N=1000 mining, wave 2 (shard 5) ==="
mine 5
say "C done -- all six shards cached; training is minutes from here"
