#!/usr/bin/env bash
# Re-mine the N=1000 pools keeping every candidate the search generates.
#
# --mine-ntop has been 1500 since the DB5.5 configuration, where the search drew
# 1500 random rotations and produced exactly 1500 candidates -- so "keep 1500"
# meant "keep everything". The rotation set later became the Hopf nside=3 grid,
# 1944 orientations with one translation each, and the 1500 was carried over. It
# now discards 444 candidates per complex (22.8%) for no reason anyone recorded.
#
# It does not change what the deployed system answers: the search is re-run with
# the trained table and only the top 5 poses are submitted. What it changes is
# the set of wrong poses the LOSS is shown. The pool is ranked by the PUBLISHED
# table, so a pose sitting at rank 1600 there may rank first under a trained one
# -- and the loss has never seen it.
#
# The cache key carries ntop, so this writes new files and leaves the ntop=1500
# pools intact for the comparison.
#
#   bash scripts/remine_full_ntop.sh [gpu_list]      # default 4,5,6
set -u -o pipefail

cd "$(dirname "$0")/.."
GPUS=${1:-4,5,6}
IFS=',' read -r -a GPU <<< "$GPUS"
NG=${#GPU[@]}
LOG=data/scaling/logs
mkdir -p "$LOG"
say() { echo "[$(date +%m-%d\ %H:%M)] $*"; }

say "###### re-mining N=1000 with --mine-ntop 1944 on GPUs $GPUS ######"
for ((i = 0; i < NG; i++)); do
    CUDA_VISIBLE_DEVICES=${GPU[$i]} setsid nohup uv run python \
        scripts/run_pinder_scaling.py \
        --n-fit 1000 --rounds 0 --seed 0 --alpha0 1.0 \
        --mine-ntop 1944 --mine-shard "$i/$NG" --mine-only \
        --grid-voxels data/scaling/grid_voxels_1.2.json \
        --max-grid-voxels 31250000 \
        --test-cache data/shards_pinder/test_pool_reachable.pt \
        --out-dir data/scaling/runs_ntop1944 \
        > "$LOG/remine_ntop1944_shard$i.log" 2>&1 < /dev/null &
done
wait
say "###### done ######"
ls -la data/scaling/pool_cache/n1000_*ntop1944* 2>/dev/null
