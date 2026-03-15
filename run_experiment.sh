#!/bin/bash
# ===========================================================================
# EnhPPFL Experiment Runner
# ===========================================================================
# Runs the full federated training pipeline and optionally verifies the
# defence mechanism via gradient inversion attack simulation.
#
# Usage:
#   # CIFAR-10 / ResNet-18 (200 rounds)
#   ./run_thesis_experiment.sh --model resnet18 --dataset cifar10 --rounds 200
#
#   # NSL-KDD / MLP (200 rounds)
#   ./run_thesis_experiment.sh --model mlp --dataset nslkdd --rounds 200
# ===========================================================================

set -e

# ============================================================================
# DEFAULT CONFIGURATION
# ============================================================================

MODEL_TYPE="resnet18"
DATASET="cifar10"
NUM_CLIENTS=10
MIN_CLIENTS=5
TOTAL_ROUNDS=200
PRIVACY_EPSILON=1.0
PRIVACY_DELTA=1e-5
BASE_NOISE=2.0
BASE_CLIPPING=1.0
# λ=0.28 yields ~73% cumulative bandwidth reduction (paper Table 5 default).
LAMBDA_SPARSE=0.28
# NSL-KDD input dimension after one-hot encoding of the three categorical columns.
INPUT_DIM=122
LEARNING_RATE=0.01
LOCAL_EPOCHS=1
BATCH_SIZE=32
DEVICE="cpu"
DATA_DIR="./data"
LOG_DIR="./logs"
RUN_DEFENCE_VERIFICATION="yes"
NUM_ATTACK_SAMPLES=50

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)                  MODEL_TYPE="$2";                  shift 2 ;;
        --dataset)                DATASET="$2";                     shift 2 ;;
        --clients)                NUM_CLIENTS="$2";                 shift 2 ;;
        --min-clients)            MIN_CLIENTS="$2";                 shift 2 ;;
        --rounds)                 TOTAL_ROUNDS="$2";                shift 2 ;;
        --epsilon)                PRIVACY_EPSILON="$2";             shift 2 ;;
        --noise)                  BASE_NOISE="$2";                  shift 2 ;;
        --device)                 DEVICE="$2";                      shift 2 ;;
        --skip-defence-verification) RUN_DEFENCE_VERIFICATION="no"; shift   ;;
        *)  echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ============================================================================
# SETUP
# ============================================================================

EXPERIMENT_NAME="enhppfl_${MODEL_TYPE}_${DATASET}_$(date +%Y%m%d_%H%M%S)"
EXPERIMENT_DIR="${LOG_DIR}/${EXPERIMENT_NAME}"
mkdir -p "${EXPERIMENT_DIR}"

echo "=========================================================================="
echo "EnhPPFL Federated Learning Experiment"
echo "=========================================================================="
echo ""
echo "Configuration:"
echo "  Model:          ${MODEL_TYPE}"
echo "  Dataset:        ${DATASET}"
echo "  Clients:        ${NUM_CLIENTS}  (min ${MIN_CLIENTS} per round)"
echo "  Rounds:         ${TOTAL_ROUNDS}"
echo "  Privacy:        ε=${PRIVACY_EPSILON}, δ=${PRIVACY_DELTA}"
echo "  σ_base:         ${BASE_NOISE}"
echo "  λ (sparsity):   ${LAMBDA_SPARSE}"
echo "  Device:         ${DEVICE}"
echo "  Log directory:  ${EXPERIMENT_DIR}"
echo ""

# ============================================================================
# PHASE 1: FEDERATED TRAINING
# ============================================================================

echo "=========================================================================="
echo "Phase 1: Federated Training"
echo "=========================================================================="
echo ""

echo "[$(date '+%H:%M:%S')] Starting server..."
python server.py \
    --server-address "0.0.0.0:8080" \
    --model-type "${MODEL_TYPE}" \
    --total-rounds "${TOTAL_ROUNDS}" \
    --min-clients "${MIN_CLIENTS}" \
    --min-available-clients "${MIN_CLIENTS}" \
    --fraction-fit "$(python -c "print(${MIN_CLIENTS}/${NUM_CLIENTS})")" \
    --fraction-evaluate 0.1 \
    --privacy-epsilon "${PRIVACY_EPSILON}" \
    --privacy-delta "${PRIVACY_DELTA}" \
    --learning-rate "${LEARNING_RATE}" \
    --input-dim "${INPUT_DIM}" \
    --device "${DEVICE}" \
    > "${EXPERIMENT_DIR}/server.log" 2>&1 &

SERVER_PID=$!
echo "  Server PID: ${SERVER_PID}"
echo "${SERVER_PID}" > "${EXPERIMENT_DIR}/server.pid"
sleep 5

echo "[$(date '+%H:%M:%S')] Starting ${NUM_CLIENTS} clients..."
for ((i=0; i<NUM_CLIENTS; i++)); do
    python client.py \
        --client-id "${i}" \
        --total-clients "${NUM_CLIENTS}" \
        --server-address "localhost:8080" \
        --model-type "${MODEL_TYPE}" \
        --dataset "${DATASET}" \
        --device "${DEVICE}" \
        --privacy-epsilon "${PRIVACY_EPSILON}" \
        --privacy-delta "${PRIVACY_DELTA}" \
        --base-noise "${BASE_NOISE}" \
        --base-clipping "${BASE_CLIPPING}" \
        --lambda-sparse "${LAMBDA_SPARSE}" \
        --learning-rate "${LEARNING_RATE}" \
        --local-epochs "${LOCAL_EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --data-dir "${DATA_DIR}" \
        > "${EXPERIMENT_DIR}/client_${i}.log" 2>&1 &
    echo "${!}" >> "${EXPERIMENT_DIR}/client_pids.txt"
    sleep 1
done

echo ""
echo "All processes started."
echo "  Server log:   tail -f ${EXPERIMENT_DIR}/server.log"
echo "  Client 0 log: tail -f ${EXPERIMENT_DIR}/client_0.log"
echo ""

wait "${SERVER_PID}"
echo ""
echo "[$(date '+%H:%M:%S')] Training complete."

if [ -f "${EXPERIMENT_DIR}/client_pids.txt" ]; then
    while read PID; do
        kill "${PID}" 2>/dev/null || true
    done < "${EXPERIMENT_DIR}/client_pids.txt"
fi

# ============================================================================
# PHASE 2: EXTRACT METRICS
# ============================================================================

echo ""
echo "=========================================================================="
echo "Phase 2: Metrics"
echo "=========================================================================="
echo ""

FINAL_EPSILON=$(grep "Final ε" "${EXPERIMENT_DIR}/server.log" | tail -1 | grep -oP 'ε = \K[0-9.]+' || echo "N/A")

if [ "${MODEL_TYPE}" = "mlp" ]; then
    FINAL_F1=$(grep "F1 (weighted)" "${EXPERIMENT_DIR}/server.log" | tail -1 | grep -oP 'F1 \(weighted\): \K[0-9.]+' || echo "N/A")
    FINAL_UTILITY="${FINAL_F1}"
    UTILITY_LABEL="F1 (weighted)"
    UTILITY_TARGET="0.919"
else
    FINAL_ACC=$(grep "Accuracy:" "${EXPERIMENT_DIR}/server.log" | tail -1 | grep -oP 'Accuracy:\s+\K[0-9.]+' || echo "N/A")
    FINAL_UTILITY="${FINAL_ACC}"
    UTILITY_LABEL="Accuracy"
    UTILITY_TARGET="0.886"
fi

COMM_REDUCTION=$(grep "cumul" "${EXPERIMENT_DIR}/server.log" | tail -1 | grep -oP 'reduction \K[\d.]+' || echo "N/A")

echo "[1] ${UTILITY_LABEL}: ${FINAL_UTILITY}  (target ${UTILITY_TARGET})"
echo "[2] Privacy ε:        ${FINAL_EPSILON}   (target ≤ 1.0)"
echo "[3] Comm. reduction:  ${COMM_REDUCTION}%  (target ≥ 73%)"
echo ""

# ============================================================================
# PHASE 3: DEFENCE VERIFICATION
# ============================================================================

if [ "${RUN_DEFENCE_VERIFICATION}" = "yes" ] && [ "${MODEL_TYPE}" = "resnet18" ]; then
    echo "=========================================================================="
    echo "Phase 3: Defence Verification (Gradient Inversion)"
    echo "=========================================================================="
    echo ""
    echo "Testing ${NUM_ATTACK_SAMPLES} samples (may take 10–30 minutes)..."
    echo ""

    python verify_defense.py \
        --model-type "${MODEL_TYPE}" \
        --dataset "${DATASET}" \
        --num-samples "${NUM_ATTACK_SAMPLES}" \
        --noise-multiplier "${BASE_NOISE}" \
        --clipping-threshold "${BASE_CLIPPING}" \
        --device "${DEVICE}" \
        --data-dir "${DATA_DIR}" \
        --save-plot \
        > "${EXPERIMENT_DIR}/defence_verification.log" 2>&1

    ATTACK_SUCCESS=$(grep "With defence" "${EXPERIMENT_DIR}/defence_verification.log" | \
                     grep -oP 'success rate=\K[\d.]+' || echo "N/A")

    echo "[4] Attack success (with defence): ${ATTACK_SUCCESS}%  (target ≤ 8%)"

    if [ -f "defence_verification.png" ]; then
        mv defence_verification.png "${EXPERIMENT_DIR}/"
    fi
elif [ "${RUN_DEFENCE_VERIFICATION}" = "yes" ] && [ "${MODEL_TYPE}" = "mlp" ]; then
    echo "[4] Defence verification: skipped (image-based attack not applicable to MLP/NSL-KDD)"
fi

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "=========================================================================="
echo "Experiment complete — ${EXPERIMENT_DIR}"
echo "=========================================================================="

{
    echo "EnhPPFL Experiment Summary"
    echo "=========================="
    echo ""
    echo "Name:    ${EXPERIMENT_NAME}"
    echo "Date:    $(date)"
    echo ""
    echo "Configuration"
    echo "  Model:   ${MODEL_TYPE}"
    echo "  Dataset: ${DATASET}"
    echo "  Clients: ${NUM_CLIENTS}"
    echo "  Rounds:  ${TOTAL_ROUNDS}"
    echo "  σ_base:  ${BASE_NOISE},  λ=${LAMBDA_SPARSE},  ε=${PRIVACY_EPSILON}"
    echo ""
    echo "Results"
    echo "  ${UTILITY_LABEL}: ${FINAL_UTILITY}"
    echo "  Privacy ε:       ${FINAL_EPSILON}"
    echo "  Comm. reduction: ${COMM_REDUCTION}%"
} > "${EXPERIMENT_DIR}/summary.txt"

echo "Summary saved to: ${EXPERIMENT_DIR}/summary.txt"
echo ""
