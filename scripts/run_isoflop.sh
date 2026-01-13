#!/bin/bash
# Iso-FLOPs Scaling Law Experiment Runner
# Usage: ./scripts/run_isoflop.sh [tokenizer|dynamics] [--dry-run]
#        CUDA_DEVICES=0,1 ./scripts/run_isoflop.sh tokenizer
#
# Trains models at fixed compute budgets to discover the optimal tokens_per_param ratio.
# Based on Chinchilla methodology: for each compute budget, train multiple model sizes
# and find which achieves the best loss.

set -e

MODEL_TYPE=${1:-tokenizer}
DRY_RUN=${2:-}

# GPU selection (set CUDA_DEVICES env var to override, e.g., CUDA_DEVICES=0,1)
export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES}}

# Compute budgets (FLOPs) - adjust based on your hardware
# These are example values; you may need to scale up/down
FLOPS_BUDGETS=(1e16 3e16 6e16 1e17 3e17 6e17)

# Model depths to try at each compute budget
DEPTHS=(8)

# Architecture scaling
D_MODEL_MULT=64

# Output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ISOFLOP_DIR="logs/isoflop_${MODEL_TYPE}_${TIMESTAMP}"
mkdir -p "$ISOFLOP_DIR"

# Log file
LOG_FILE="${ISOFLOP_DIR}/experiment.log"
echo "========================================" | tee "$LOG_FILE"
echo "Iso-FLOPs Scaling Law Experiment" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "Model type: ${MODEL_TYPE}" | tee -a "$LOG_FILE"
echo "FLOPs budgets: ${FLOPS_BUDGETS[*]}" | tee -a "$LOG_FILE"
echo "Depths: ${DEPTHS[*]}" | tee -a "$LOG_FILE"
echo "Output dir: ${ISOFLOP_DIR}" | tee -a "$LOG_FILE"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Results CSV
RESULTS_FILE="${ISOFLOP_DIR}/results.csv"
echo "flops_budget,depth,d_model,n_heads,status,duration_sec" > "$RESULTS_FILE"

# Total experiments
TOTAL_RUNS=$((${#FLOPS_BUDGETS[@]} * ${#DEPTHS[@]}))
CURRENT_RUN=0

# Run experiments: nested loop over (flops_budget, depth)
for FLOPS in "${FLOPS_BUDGETS[@]}"; do
    for DEPTH in "${DEPTHS[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))
        D_MODEL=$((DEPTH * D_MODEL_MULT))
        N_HEADS=$((D_MODEL / 64))

        # Create unique run name encoding both flops budget and depth
        # Use scientific notation without special chars for wandb compatibility
        FLOPS_SHORT=$(echo "$FLOPS" | sed 's/e/e/g')
        RUN_NAME="isoflop_${MODEL_TYPE}_F${FLOPS_SHORT}_d${DEPTH}"

        echo "" | tee -a "$LOG_FILE"
        echo "==================================================" | tee -a "$LOG_FILE"
        echo "Run ${CURRENT_RUN}/${TOTAL_RUNS}" | tee -a "$LOG_FILE"
        echo "FLOPs budget: ${FLOPS}, depth: ${DEPTH}, d_model: ${D_MODEL}" | tee -a "$LOG_FILE"
        echo "Run name: ${RUN_NAME}" | tee -a "$LOG_FILE"
        echo "==================================================" | tee -a "$LOG_FILE"

        if [ "$MODEL_TYPE" == "tokenizer" ]; then
            CMD="uv run scripts/train_tokenizer.py \
                run_name=${RUN_NAME} \
                use_wandb=true \
                encoder.depth=${DEPTH} \
                encoder.d_model=${D_MODEL} \
                encoder.n_heads=${N_HEADS} \
                encoder.n_kv_heads=${N_HEADS} \
                decoder.depth=${DEPTH} \
                decoder.d_model=${D_MODEL} \
                decoder.n_heads=${N_HEADS} \
                decoder.n_kv_heads=${N_HEADS} \
                scaling_flops_budget=${FLOPS} \
                hydra.run.dir=${ISOFLOP_DIR}/${RUN_NAME}"
        elif [ "$MODEL_TYPE" == "dynamics" ]; then
            CMD="uv run scripts/train_dynamics.py \
                run_name=${RUN_NAME} \
                use_wandb=true \
                dynamics.depth=${DEPTH} \
                dynamics.d_model=${D_MODEL} \
                dynamics.n_heads=${N_HEADS} \
                dynamics.n_kv_heads=${N_HEADS} \
                scaling_flops_budget=${FLOPS} \
                hydra.run.dir=${ISOFLOP_DIR}/${RUN_NAME}"
        else
            echo "Unknown model type: ${MODEL_TYPE}. Use 'tokenizer' or 'dynamics'." | tee -a "$LOG_FILE"
            exit 1
        fi

        if [ "$DRY_RUN" == "--dry-run" ]; then
            echo "[DRY RUN] Would execute:" | tee -a "$LOG_FILE"
            echo "$CMD" | tee -a "$LOG_FILE"
            echo "${FLOPS},${DEPTH},${D_MODEL},${N_HEADS},dry_run,0" >> "$RESULTS_FILE"
        else
            echo "Executing command..." | tee -a "$LOG_FILE"
            START_TIME=$(date +%s)

            # Run with error handling
            if eval "$CMD" 2>&1 | tee -a "${ISOFLOP_DIR}/${RUN_NAME}.log"; then
                END_TIME=$(date +%s)
                DURATION=$((END_TIME - START_TIME))
                echo "Completed F${FLOPS}_d${DEPTH} in ${DURATION}s" | tee -a "$LOG_FILE"
                echo "${FLOPS},${DEPTH},${D_MODEL},${N_HEADS},success,${DURATION}" >> "$RESULTS_FILE"
            else
                END_TIME=$(date +%s)
                DURATION=$((END_TIME - START_TIME))
                echo "FAILED: F${FLOPS}_d${DEPTH} after ${DURATION}s" | tee -a "$LOG_FILE"
                echo "${FLOPS},${DEPTH},${D_MODEL},${N_HEADS},failed,${DURATION}" >> "$RESULTS_FILE"
            fi
        fi
    done
done

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "All experiments completed at $(date)" | tee -a "$LOG_FILE"
echo "Total runs: ${TOTAL_RUNS}" | tee -a "$LOG_FILE"
echo "Results saved to: ${RESULTS_FILE}" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Next step: Run analysis to find optimal tokens_per_param ratio:" | tee -a "$LOG_FILE"
echo "  python scripts/analyze_isoflop.py --entity YOUR_ENTITY --project YOUR_PROJECT" | tee -a "$LOG_FILE"

# Print summary
echo ""
echo "Results summary:"
cat "$RESULTS_FILE"
