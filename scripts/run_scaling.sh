#!/bin/bash
# Scaling Laws Experiment Runner
# Usage: ./scripts/run_scaling.sh [tokenizer|dynamics] [--dry-run]
#        CUDA_DEVICES=0,1 ./scripts/run_scaling.sh tokenizer
#
# Trains a family of models at varying depths to study scaling laws.
# Based on nanochat methodology: https://github.com/karpathy/nanochat/discussions/420

set -e

MODEL_TYPE=${1:-tokenizer}
DRY_RUN=${2:-}

# GPU selection (set CUDA_DEVICES env var to override, e.g., CUDA_DEVICES=0,1)
export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES}}

# Scaling configurations
DEPTHS=(1 2 3 4 5 6)
D_MODEL_MULT=64
TOKENS_PER_PARAM=4063.7

# Output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCALING_DIR="logs/scaling_${MODEL_TYPE}_${TIMESTAMP}"
mkdir -p "$SCALING_DIR"

# Log file
LOG_FILE="${SCALING_DIR}/experiment.log"
echo "========================================" | tee "$LOG_FILE"
echo "Scaling Laws Experiment" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "Model type: ${MODEL_TYPE}" | tee -a "$LOG_FILE"
echo "Depths: ${DEPTHS[*]}" | tee -a "$LOG_FILE"
echo "d_model multiplier: ${D_MODEL_MULT}" | tee -a "$LOG_FILE"
echo "Tokens per param: ${TOKENS_PER_PARAM}" | tee -a "$LOG_FILE"
echo "Output dir: ${SCALING_DIR}" | tee -a "$LOG_FILE"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Results CSV
RESULTS_FILE="${SCALING_DIR}/results.csv"
echo "depth,d_model,n_heads,status,duration_sec" > "$RESULTS_FILE"

# Run experiments
for DEPTH in "${DEPTHS[@]}"; do
    D_MODEL=$((DEPTH * D_MODEL_MULT))
    N_HEADS=$((D_MODEL / 64))

    RUN_NAME="scaling_${MODEL_TYPE}_d${DEPTH}"

    echo "" | tee -a "$LOG_FILE"
    echo "==================================================" | tee -a "$LOG_FILE"
    echo "Running: depth=${DEPTH}, d_model=${D_MODEL}, n_heads=${N_HEADS}" | tee -a "$LOG_FILE"
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
            scaling_tokens_per_param=${TOKENS_PER_PARAM} \
            hydra.run.dir=${SCALING_DIR}/${RUN_NAME}"
    elif [ "$MODEL_TYPE" == "dynamics" ]; then
        CMD="uv run scripts/train_dynamics.py \
            run_name=${RUN_NAME} \
            use_wandb=true \
            dynamics.depth=${DEPTH} \
            dynamics.d_model=${D_MODEL} \
            dynamics.n_heads=${N_HEADS} \
            dynamics.n_kv_heads=${N_HEADS} \
            scaling_tokens_per_param=${TOKENS_PER_PARAM} \
            hydra.run.dir=${SCALING_DIR}/${RUN_NAME}"
    else
        echo "Unknown model type: ${MODEL_TYPE}. Use 'tokenizer' or 'dynamics'." | tee -a "$LOG_FILE"
        exit 1
    fi

    if [ "$DRY_RUN" == "--dry-run" ]; then
        echo "[DRY RUN] Would execute:" | tee -a "$LOG_FILE"
        echo "$CMD" | tee -a "$LOG_FILE"
        echo "${DEPTH},${D_MODEL},${N_HEADS},dry_run,0" >> "$RESULTS_FILE"
    else
        echo "Executing command..." | tee -a "$LOG_FILE"
        START_TIME=$(date +%s)

        # Run with error handling
        if eval "$CMD" 2>&1 | tee -a "${SCALING_DIR}/${RUN_NAME}.log"; then
            END_TIME=$(date +%s)
            DURATION=$((END_TIME - START_TIME))
            echo "Completed d${DEPTH} in ${DURATION}s" | tee -a "$LOG_FILE"
            echo "${DEPTH},${D_MODEL},${N_HEADS},success,${DURATION}" >> "$RESULTS_FILE"
        else
            END_TIME=$(date +%s)
            DURATION=$((END_TIME - START_TIME))
            echo "FAILED: d${DEPTH} after ${DURATION}s" | tee -a "$LOG_FILE"
            echo "${DEPTH},${D_MODEL},${N_HEADS},failed,${DURATION}" >> "$RESULTS_FILE"
        fi
    fi
done

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "All experiments completed at $(date)" | tee -a "$LOG_FILE"
echo "Results saved to: ${RESULTS_FILE}" | tee -a "$LOG_FILE"
echo "Full logs in: ${SCALING_DIR}" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Print summary
echo ""
echo "Results summary:"
cat "$RESULTS_FILE"
