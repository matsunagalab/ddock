#!/usr/bin/env bash
# Wait for the N=1000 training runs, then score their checkpoints with PINDER's
# harness on the same official 250 systems as every other condition.
#
#   bash scripts/queue_n1000_eval.sh [gpu_list]      # default 0,3,6
set -u -o pipefail

cd "$(dirname "$0")/.."
GPUS=${1:-0,3,6}
IFS=',' read -r -a GPU <<< "$GPUS"
NG=${#GPU[@]}
LOG=data/scaling/logs
say() { echo "[$(date +%m-%d\ %H:%M)] $*"; }

say "waiting for the N=1000 training runs"
while pgrep -f "run_pinder_scaling.py --n-fit 1000" > /dev/null; do sleep 60; done
for sd in 0 1 2; do
    ck=data/scaling/runs_nfixed/N1000_seed$sd/round0_ckpt.pt
    [ -f "$ck" ] || { say "MISSING $ck -- training failed, stopping"; exit 1; }
done
say "checkpoints present"

for sd in 0 1 2; do
    c=N1000_seed$sd
    say "=== $c ==="
    for ((i = 0; i < NG; i++)); do
        CUDA_VISIBLE_DEVICES=${GPU[$i]} setsid nohup uv run python \
            scripts/eval_search_test.py \
            --shard "$i/$NG" --ckpt "data/scaling/runs_nfixed/$c/round0_ckpt.pt" \
            --label "${c}_sh$i" --export-pdb-dir "data/pinder_eval/trained_$c" \
            --export-top-k 5 --monomer holo \
            --prep-cache data/scaling/prep_cache_test \
            --out-dir data/scaling/eval_search_pinder \
            > "$LOG/official_${c}_sh$i.log" 2>&1 < /dev/null &
    done
    wait
    say "=== $c done ==="
done

say "scoring"
PINDER_BASE_DIR=$PWD/external/pinder uv run python \
    scripts/score_decoys_with_pinder.py --eval-dir data/pinder_eval \
    --out data/pinder_eval/leaderboard.csv --max-workers 16 \
    > "$LOG/n1000_harness.log" 2>&1
say "done"
