#!/usr/bin/env bash
# Score the N-scaling checkpoints with PINDER's own harness.
#
# The +8.40 pp in section 5.14.22 is one checkpoint (N=220, seed 0). The N
# scaling question was settled on the fixed TEST pool with our own DockQ, before
# the evaluation moved to PINDER, so nothing is known about how N behaves at the
# official endpoint -- nor how much of the +8.40 pp is seed noise.
#
# Each condition is a separate PINDER "method" directory, so the harness scores
# them side by side over the same official 250 systems. Training is not re-run:
# these checkpoints already exist.
#
#   bash scripts/run_official_nscaling.sh [gpu_list]      # default 0,3,6
set -u -o pipefail

cd "$(dirname "$0")/.."
GPUS=${1:-0,3,6}
IFS=',' read -r -a GPU <<< "$GPUS"
NG=${#GPU[@]}
LOG=data/scaling/logs
mkdir -p "$LOG"

CONDS=(N220_seed1 N220_seed2 N500_seed0 N500_seed1 N500_seed2)

echo "###### official N-scaling started $(date +%H:%M) on GPUs $GPUS ######"
for c in "${CONDS[@]}"; do
    ck=data/scaling/runs_nfixed/$c/round0_ckpt.pt
    if [ ! -f "$ck" ]; then echo "MISSING $ck -- skipping $c"; continue; fi
    echo "=== [$(date +%H:%M)] $c ==="
    for ((i = 0; i < NG; i++)); do
        CUDA_VISIBLE_DEVICES=${GPU[$i]} setsid nohup uv run python \
            scripts/eval_search_test.py \
            --shard "$i/$NG" --ckpt "$ck" --label "${c}_sh$i" \
            --export-pdb-dir "data/pinder_eval/trained_$c" \
            --export-top-k 5 --monomer holo \
            --prep-cache data/scaling/prep_cache_test \
            --out-dir data/scaling/eval_search_pinder \
            > "$LOG/official_${c}_sh$i.log" 2>&1 < /dev/null &
    done
    wait
    echo "=== [$(date +%H:%M)] $c done ==="
done

echo "###### searches done $(date +%H:%M); scoring ######"
PINDER_BASE_DIR=$PWD/external/pinder uv run python \
    scripts/score_decoys_with_pinder.py \
    --eval-dir data/pinder_eval --out data/pinder_eval/leaderboard.csv \
    --max-workers 16 > "$LOG/official_nscaling_harness.log" 2>&1
echo "###### harness done $(date +%H:%M) ######"
