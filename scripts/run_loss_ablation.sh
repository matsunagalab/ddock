#!/usr/bin/env bash
# Train the loss ablation of report sections 5.14.26-28, and re-run N=1000 with
# the stopping rule the other N conditions used.
#
# Every arm here uses --patience 100000 --epoch-passes 1 --min-steps 6818, which
# is what N=220 and N=500 were trained with. The first N=1000 runs did not: they
# took the defaults (patience 8, epoch_passes 100) and stopped at a quarter of
# the budget, so `matched` below is both the corrected N scaling point and the
# control arm of the ablation.
#
#   matched   basin + min-anchor hinge          the recipe behind every result
#   toptail   basin + soft top-tail             fixes the anchor (5.14.26)
#   shape     matched + pairwise shaping        wakes the 461 silent complexes
#   both      toptail + pairwise shaping        (5.14.28)
#
# N=1000 is the testbed because its validation set has 250 complexes; N=220's 55
# cannot resolve a top-1 difference. Selection is on validation only. TEST is
# touched once, by whichever arm wins.
#
#   bash scripts/run_loss_ablation.sh [gpu_list]      # default 0,1,2,3,4,5,6
set -u -o pipefail

cd "$(dirname "$0")/.."
GPUS=${1:-0,1,2,3,4,5,6}
IFS=',' read -r -a GPU <<< "$GPUS"
NG=${#GPU[@]}
LOG=data/scaling/logs
OUT=data/scaling/runs_loss
mkdir -p "$LOG" "$OUT"
say() { echo "[$(date +%m-%d\ %H:%M)] $*"; }

COMMON="--n-fit 1000 --rounds 0 --alpha0 1.0 --loss-prov search --freeze-psc
        --iface-mode full --lambda-margin 0.5 --margin 1.0 --lambda-prior 0.1
        --min-steps 6818 --patience 100000 --epoch-passes 1
        --grid-voxels data/scaling/grid_voxels_1.2.json
        --max-grid-voxels 31250000
        --test-cache data/shards_pinder/test_pool_reachable.pt"

arm_flags() {
    case "$1" in
        matched) echo "--loss-neg minanchor --loss-shape none" ;;
        toptail) echo "--loss-neg toptail   --loss-shape none" ;;
        shape)   echo "--loss-neg minanchor --loss-shape pairwise" ;;
        both)    echo "--loss-neg toptail   --loss-shape pairwise" ;;
        *) echo "unknown arm $1" >&2; exit 1 ;;
    esac
}

say "###### loss ablation on GPUs $GPUS ######"
n=0
for arm in matched toptail shape both; do
    for sd in 0 1 2; do
        g=${GPU[$((n % NG))]}
        CUDA_VISIBLE_DEVICES=$g setsid nohup uv run python \
            scripts/run_pinder_scaling.py $COMMON $(arm_flags "$arm") \
            --seed "$sd" --out-dir "$OUT/$arm" \
            > "$LOG/loss_${arm}_seed$sd.log" 2>&1 < /dev/null &
        n=$((n + 1))
        if [ $((n % NG)) -eq 0 ]; then
            say "wave of $NG launched, waiting"
            wait
        fi
    done
done
wait
say "###### all arms trained ######"
for arm in matched toptail shape both; do
    for sd in 0 1 2; do
        ck="$OUT/$arm/N1000_seed$sd/round0_ckpt.pt"
        [ -f "$ck" ] && say "ok   $arm seed$sd" || say "MISSING $ck"
    done
done
