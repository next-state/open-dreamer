#!/bin/bash
# Unified Scaling Experiment Runner
# Based on Karpathy's nanochat methodology
#
# Stage 1 (Iso-FLOPs): Find optimal tokens_per_param ratio
#   ./scripts/run_scaling.sh isoflop tokenizer
#
# Stage 2 (Compute-Optimal): Train at discovered ratio
#   TOKENS_PER_PARAM=20 ./scripts/run_scaling.sh optimal tokenizer
#
# Options:
#   --dry-run    Print commands without executing
#   CUDA_DEVICES=0,1 ./scripts/run_scaling.sh ...  # GPU selection

set -e

MODE=${1:-isoflop}
MODEL=${2:-tokenizer}
DRY_RUN=""
[[ "$3" == "--dry-run" || "$4" == "--dry-run" ]] && DRY_RUN=1

# GPU selection (only set if explicitly provided, otherwise use system default)
if [ -n "$CUDA_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
fi

# Architecture scaling: d_model = depth × 64
D_MODEL_MULT=64

# Configuration by mode
if [ "$MODE" == "isoflop" ]; then
    # Iso-FLOPs: multiple depths × multiple FLOPs budgets
    DEPTHS=(6 5 4 3 2 1)
    FLOPS_BUDGETS=(1e16 3e16 6e16 1e17 3e17 6e17)
elif [ "$MODE" == "optimal" ]; then
    # Compute-optimal: multiple depths × fixed tokens_per_param
    DEPTHS=(7)
    TOKENS_PER_PARAM=${TOKENS_PER_PARAM:-20}
else
    echo "Unknown mode: $MODE. Use 'isoflop' or 'optimal'."
    exit 1
fi

# Output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="logs/scaling_${MODE}_${MODEL}_${TIMESTAMP}"
mkdir -p "$OUT_DIR"

# CSV header (training script appends rows)
echo "run_name,params,data_tokens_per_step,total_tokens_per_step,flops_per_step,flops_budget,total_steps,data_tokens_trained,total_tokens_trained,hours,final_loss,final_psnr" > "$OUT_DIR/results.csv"

# Log experiment config
echo "========================================"
echo "Scaling Experiment: $MODE"
echo "========================================"
echo "Model: $MODEL"
echo "Depths: ${DEPTHS[*]}"
[ "$MODE" == "isoflop" ] && echo "FLOPs budgets: ${FLOPS_BUDGETS[*]}"
[ "$MODE" == "optimal" ] && echo "Tokens per param: $TOKENS_PER_PARAM"
echo "Output: $OUT_DIR"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set, using all>}"
echo "========================================"

# Count total runs
if [ "$MODE" == "isoflop" ]; then
    TOTAL=$((${#DEPTHS[@]} * ${#FLOPS_BUDGETS[@]}))
else
    TOTAL=${#DEPTHS[@]}
fi
RUN=0

# Main loop
for DEPTH in "${DEPTHS[@]}"; do
    D_MODEL=$((DEPTH * D_MODEL_MULT))
    N_HEADS=$((D_MODEL / 64))

    if [ "$MODE" == "isoflop" ]; then
        for FLOPS in "${FLOPS_BUDGETS[@]}"; do
            RUN=$((RUN + 1))
            RUN_NAME="${MODE}_${MODEL}_F${FLOPS}_d${DEPTH}"

            echo ""
            echo "[$RUN/$TOTAL] $RUN_NAME (depth=$DEPTH, FLOPs=$FLOPS)"

            CMD="uv run scripts/train_${MODEL}.py \
                run_name=${RUN_NAME} \
                use_wandb=true \
                ckpt.max_to_keep=null \
                encoder.depth=${DEPTH} encoder.d_model=${D_MODEL} \
                encoder.n_heads=${N_HEADS} encoder.n_kv_heads=${N_HEADS} \
                decoder.depth=${DEPTH} decoder.d_model=${D_MODEL} \
                decoder.n_heads=${N_HEADS} decoder.n_kv_heads=${N_HEADS} \
                scaling_flops_budget=${FLOPS} \
                hydra.run.dir=${OUT_DIR}/${RUN_NAME}"

            if [ "$DRY_RUN" ]; then
                echo "[DRY] $CMD"
            else
                eval "$CMD" || echo "FAILED: $RUN_NAME"
            fi
        done
    else
        RUN=$((RUN + 1))
        RUN_NAME="${MODE}_${MODEL}_d${DEPTH}"

        echo ""
        echo "[$RUN/$TOTAL] $RUN_NAME (depth=$DEPTH)"

        CMD="uv run scripts/train_${MODEL}.py \
            run_name=${RUN_NAME} \
            use_wandb=true \
            ckpt.max_to_keep=null \
            encoder.depth=${DEPTH} encoder.d_model=${D_MODEL} \
            encoder.n_heads=${N_HEADS} encoder.n_kv_heads=${N_HEADS} \
            decoder.depth=${DEPTH} decoder.d_model=${D_MODEL} \
            decoder.n_heads=${N_HEADS} decoder.n_kv_heads=${N_HEADS} \
            scaling_tokens_per_param=${TOKENS_PER_PARAM} \
            hydra.run.dir=${OUT_DIR}/${RUN_NAME}"

        if [ "$DRY_RUN" ]; then
            echo "[DRY] $CMD"
        else
            eval "$CMD" || echo "FAILED: $RUN_NAME"
        fi
    fi
done

echo ""
echo "========================================"
echo "Completed $RUN runs"
echo "Results: $OUT_DIR/results.csv"
echo ""
if [ "$MODE" == "isoflop" ]; then
    echo "Next: python scripts/analyze_isoflop.py $OUT_DIR"
else
    echo "Next: python scripts/analyze_scaling.py $OUT_DIR"
fi
echo "========================================"
