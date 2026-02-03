#!/bin/bash
# ===========================================================================
# EnhPPFL Thesis Defense Experiment Runner
# ===========================================================================
# This script runs the complete pipeline to verify ALL thesis claims:
# 1. Utility: 91.9% F1 Score on NSL-KDD
# 2. Communication: 73% bandwidth reduction
# 3. Defense: ≤8% attack success rate
#
# Usage:
#   # CIFAR-10 with ResNet-18
#   ./run_thesis_experiment.sh --model resnet18 --dataset cifar10 --rounds 200
#
#   # NSL-KDD with MLP (for 91.9% F1 claim)
#   ./run_thesis_experiment.sh --model mlp --dataset nslkdd --rounds 100
# ===========================================================================

set -e  # Exit on error

# ============================================================================
# PARSE ARGUMENTS
# ============================================================================

MODEL_TYPE="resnet18"
DATASET="cifar10"
NUM_CLIENTS=10
MIN_CLIENTS=5
TOTAL_ROUNDS=100
PRIVACY_EPSILON=1.0
PRIVACY_DELTA=1e-5
BASE_NOISE=2.0
BASE_CLIPPING=1.0
LAMBDA_SPARSE=0.5
LEARNING_RATE=0.01
LOCAL_EPOCHS=1
BATCH_SIZE=32
DEVICE="cpu"
DATA_DIR="./data"
LOG_DIR="./logs"
RUN_DEFENSE_VERIFICATION="yes"
NUM_ATTACK_SAMPLES=50

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --clients)
            NUM_CLIENTS="$2"
            shift 2
            ;;
        --min-clients)
            MIN_CLIENTS="$2"
            shift 2
            ;;
        --rounds)
            TOTAL_ROUNDS="$2"
            shift 2
            ;;
        --epsilon)
            PRIVACY_EPSILON="$2"
            shift 2
            ;;
        --noise)
            BASE_NOISE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --skip-defense-verification)
            RUN_DEFENSE_VERIFICATION="no"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================================
# SETUP
# ============================================================================

EXPERIMENT_NAME="thesis_${MODEL_TYPE}_${DATASET}_$(date +%Y%m%d_%H%M%S)"
EXPERIMENT_DIR="${LOG_DIR}/${EXPERIMENT_NAME}"
mkdir -p "${EXPERIMENT_DIR}"

echo "=========================================================================="
echo "EnhPPFL THESIS DEFENSE EXPERIMENT"
echo "=========================================================================="
echo ""
echo "Configuration:"
echo "  Model: ${MODEL_TYPE}"
echo "  Dataset: ${DATASET}"
echo "  Clients: ${NUM_CLIENTS} (min ${MIN_CLIENTS} per round)"
echo "  Rounds: ${TOTAL_ROUNDS}"
echo "  Privacy: ε=${PRIVACY_EPSILON}, δ=${PRIVACY_DELTA}"
echo "  Device: ${DEVICE}"
echo "  Log Directory: ${EXPERIMENT_DIR}"
echo ""
echo "Thesis Claims to Verify:"
echo "  [1] Utility: 91.9% F1 (NSL-KDD) or 85%+ Acc (CIFAR-10)"
echo "  [2] Communication Reduction: ≥73%"
echo "  [3] Attack Success Rate: ≤8%"
echo ""

# ============================================================================
# PHASE 1: FEDERATED TRAINING
# ============================================================================

echo "=========================================================================="
echo "PHASE 1: FEDERATED TRAINING"
echo "=========================================================================="
echo ""

# Start server in background
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting server..."
python server.py \
    --server-address "0.0.0.0:9090" \

    --model-type "${MODEL_TYPE}" \
    --total-rounds "${TOTAL_ROUNDS}" \
    --min-clients "${MIN_CLIENTS}" \
    --min-available-clients "${MIN_CLIENTS}" \
    --fraction-fit $(python -c "print(${MIN_CLIENTS}/${NUM_CLIENTS})") \
    --fraction-evaluate 0.1 \
    --privacy-epsilon "${PRIVACY_EPSILON}" \
    --privacy-delta "${PRIVACY_DELTA}" \
    --learning-rate "${LEARNING_RATE}" \
    --device "${DEVICE}" \
    > "${EXPERIMENT_DIR}/server.log" 2>&1 &

SERVER_PID=$!
echo "  Server PID: ${SERVER_PID}"
echo ${SERVER_PID} > "${EXPERIMENT_DIR}/server.pid"
sleep 5

# Start clients in background
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${NUM_CLIENTS} clients..."
for ((i=0; i<${NUM_CLIENTS}; i++)); do
    python client.py \
        --client-id ${i} \
        --total-clients ${NUM_CLIENTS} \
        --server-address "localhost:9090" \
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
    
    CLIENT_PID=$!
    echo ${CLIENT_PID} >> "${EXPERIMENT_DIR}/client_pids.txt"
    sleep 1  # Stagger starts
done

echo ""
echo "All processes started! Monitoring progress..."
echo "  Server log: tail -f ${EXPERIMENT_DIR}/server.log"
echo "  Client 0 log: tail -f ${EXPERIMENT_DIR}/client_0.log"
echo ""

# Wait for training to complete
wait ${SERVER_PID}

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training complete!"

# Kill any remaining client processes
if [ -f "${EXPERIMENT_DIR}/client_pids.txt" ]; then
    while read PID; do
        kill ${PID} 2>/dev/null || true
    done < "${EXPERIMENT_DIR}/client_pids.txt"
fi

# ============================================================================
# PHASE 2: EXTRACT METRICS FROM LOGS
# ============================================================================

echo ""
echo "=========================================================================="
echo "PHASE 2: EXTRACTING THESIS METRICS"
echo "=========================================================================="
echo ""

# Extract final privacy budget
FINAL_EPSILON=$(grep "Final Privacy Budget" "${EXPERIMENT_DIR}/server.log" | tail -1 | grep -oP 'ε = \K[0-9.]+' || echo "N/A")

# Extract final utility
if [ "${MODEL_TYPE}" = "mlp" ]; then
    FINAL_F1=$(grep "F1 Score:" "${EXPERIMENT_DIR}/server.log" | tail -1 | grep -oP 'F1 Score: \K[0-9.]+' || echo "N/A")
    FINAL_UTILITY="${FINAL_F1}"
    UTILITY_TYPE="F1 Score"
    UTILITY_TARGET="0.919"
else
    FINAL_ACC=$(grep "Accuracy:" "${EXPERIMENT_DIR}/server.log" | tail -1 | grep -oP 'Accuracy: \K[0-9.]+' || echo "N/A")
    FINAL_UTILITY="${FINAL_ACC}"
    UTILITY_TYPE="Accuracy"
    UTILITY_TARGET="0.85"
fi

# Extract communication reduction
COMM_REDUCTION=$(grep "CUMULATIVE REDUCTION:" "${EXPERIMENT_DIR}/server.log" | tail -1 | grep -oP '\d+\.\d+(?=%)' || echo "N/A")

echo "[1] UTILITY (${UTILITY_TYPE}):"
echo "    Achieved: ${FINAL_UTILITY}"
echo "    Target: ${UTILITY_TARGET}"
if [ "${FINAL_UTILITY}" != "N/A" ]; then
    if (( $(echo "${FINAL_UTILITY} >= ${UTILITY_TARGET}" | bc -l) )); then
        echo "    Status: ✓ PASS"
    else
        echo "    Status: ✗ FAIL"
    fi
else
    echo "    Status: ? UNKNOWN (check logs)"
fi
echo ""

echo "[2] PRIVACY BUDGET:"
echo "    Final ε: ${FINAL_EPSILON}"
echo "    Target: ≤1.0"
if [ "${FINAL_EPSILON}" != "N/A" ]; then
    if (( $(echo "${FINAL_EPSILON} <= 1.0" | bc -l) )); then
        echo "    Status: ✓ PASS"
    else
        echo "    Status: ✗ FAIL"
    fi
else
    echo "    Status: ? UNKNOWN (check logs)"
fi
echo ""

echo "[3] COMMUNICATION REDUCTION:"
echo "    Achieved: ${COMM_REDUCTION}%"
echo "    Target: ≥73%"
if [ "${COMM_REDUCTION}" != "N/A" ]; then
    if (( $(echo "${COMM_REDUCTION} >= 73.0" | bc -l) )); then
        echo "    Status: ✓ PASS"
    else
        echo "    Status: ✗ FAIL"
    fi
else
    echo "    Status: ? UNKNOWN (check logs)"
fi
echo ""

# ============================================================================
# PHASE 3: DEFENSE VERIFICATION
# ============================================================================

if [ "${RUN_DEFENSE_VERIFICATION}" = "yes" ] && [ "${MODEL_TYPE}" = "resnet18" ]; then
    echo ""
    echo "=========================================================================="
    echo "PHASE 3: DEFENSE VERIFICATION (GRADIENT INVERSION ATTACK)"
    echo "=========================================================================="
    echo ""
    
    echo "Running gradient inversion attack simulation..."
    echo "This will test ${NUM_ATTACK_SAMPLES} samples (may take 10-30 minutes)"
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
        > "${EXPERIMENT_DIR}/defense_verification.log" 2>&1
    
    # Extract attack success rate
    ATTACK_SUCCESS=$(grep "Attack Success Rate:" "${EXPERIMENT_DIR}/defense_verification.log" | tail -1 | grep -oP '\d+\.\d+(?=%)' || echo "N/A")
    
    echo ""
    echo "[4] DEFENSE EFFECTIVENESS (Attack Success Rate):"
    echo "    Achieved: ${ATTACK_SUCCESS}%"
    echo "    Target: ≤8%"
    if [ "${ATTACK_SUCCESS}" != "N/A" ]; then
        if (( $(echo "${ATTACK_SUCCESS} <= 8.0" | bc -l) )); then
            echo "    Status: ✓ PASS"
        else
            echo "    Status: ✗ FAIL"
        fi
    else
        echo "    Status: ? UNKNOWN (check logs)"
    fi
    
    # Move plot if it exists
    if [ -f "defense_verification.png" ]; then
        mv defense_verification.png "${EXPERIMENT_DIR}/"
        echo ""
        echo "    Visualization saved: ${EXPERIMENT_DIR}/defense_verification.png"
    fi
elif [ "${RUN_DEFENSE_VERIFICATION}" = "yes" ] && [ "${MODEL_TYPE}" = "mlp" ]; then
    echo ""
    echo "[4] DEFENSE VERIFICATION:"
    echo "    Skipped for MLP model (requires image-based attacks)"
    echo "    Use --model resnet18 --dataset cifar10 for defense verification"
fi

# ============================================================================
# FINAL SUMMARY
# ============================================================================

echo ""
echo "=========================================================================="
echo "THESIS DEFENSE EXPERIMENT COMPLETE"
echo "=========================================================================="
echo ""
echo "Results Directory: ${EXPERIMENT_DIR}"
echo ""
echo "Files Generated:"
echo "  - server.log: Server training logs"
echo "  - client_*.log: Client training logs (${NUM_CLIENTS} files)"
echo "  - defense_verification.log: Attack simulation results (if run)"
echo "  - defense_verification.png: Visualization (if run)"
echo "  - enhppfl_${MODEL_TYPE}_final.pt: Trained model"
echo ""
echo "Summary of Thesis Claims:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Claim 1: Utility
echo "[1] MODEL UTILITY:"
if [ "${MODEL_TYPE}" = "mlp" ]; then
    echo "    Claim: 91.9% F1 Score on NSL-KDD"
    echo "    Achieved: $([ "${FINAL_UTILITY}" != "N/A" ] && echo "${FINAL_UTILITY}" || echo "Check logs")"
else
    echo "    Claim: 85%+ Accuracy on CIFAR-10"
    echo "    Achieved: $([ "${FINAL_UTILITY}" != "N/A" ] && echo "${FINAL_UTILITY}" || echo "Check logs")"
fi
echo ""

# Claim 2: Communication
echo "[2] COMMUNICATION EFFICIENCY:"
echo "    Claim: 73% Bandwidth Reduction"
echo "    Achieved: $([ "${COMM_REDUCTION}" != "N/A" ] && echo "${COMM_REDUCTION}%" || echo "Check logs")"
echo ""

# Claim 3: Defense
if [ "${RUN_DEFENSE_VERIFICATION}" = "yes" ] && [ "${MODEL_TYPE}" = "resnet18" ]; then
    echo "[3] DEFENSE ROBUSTNESS:"
    echo "    Claim: ≤8% Attack Success Rate"
    echo "    Achieved: $([ "${ATTACK_SUCCESS}" != "N/A" ] && echo "${ATTACK_SUCCESS}%" || echo "Check logs")"
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "To review detailed results:"
echo "  Server: less ${EXPERIMENT_DIR}/server.log"
echo "  Client 0: less ${EXPERIMENT_DIR}/client_0.log"
if [ "${RUN_DEFENSE_VERIFICATION}" = "yes" ]; then
    echo "  Defense: less ${EXPERIMENT_DIR}/defense_verification.log"
fi
echo ""
echo "=========================================================================="

# Save summary to file
{
    echo "EnhPPFL Thesis Defense Experiment Summary"
    echo "=========================================="
    echo ""
    echo "Experiment: ${EXPERIMENT_NAME}"
    echo "Date: $(date)"
    echo ""
    echo "Configuration:"
    echo "  Model: ${MODEL_TYPE}"
    echo "  Dataset: ${DATASET}"
    echo "  Clients: ${NUM_CLIENTS}"
    echo "  Rounds: ${TOTAL_ROUNDS}"
    echo "  Privacy: ε=${PRIVACY_EPSILON}"
    echo ""
    echo "Results:"
    echo "  Utility (${UTILITY_TYPE}): ${FINAL_UTILITY}"
    echo "  Privacy Budget (ε): ${FINAL_EPSILON}"
    echo "  Communication Reduction: ${COMM_REDUCTION}%"
    if [ "${RUN_DEFENSE_VERIFICATION}" = "yes" ] && [ "${MODEL_TYPE}" = "resnet18" ]; then
        echo "  Attack Success Rate: ${ATTACK_SUCCESS}%"
    fi
} > "${EXPERIMENT_DIR}/summary.txt"

echo "Summary saved to: ${EXPERIMENT_DIR}/summary.txt"
echo ""
