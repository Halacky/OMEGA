#!/bin/bash
# Run H10 Enhanced DG Pipeline — ablation variants across 4 GPUs.
# Phase 1: A B C D (baseline + cosine/LS + SWAD + SupCon)
# Phase 2: E F G H (BandMask + CORAL + SAM + best combo)
# Usage: bash scripts/run_h10_parallel_4gpu.sh [--full] [--phase 1|2|both]

set -e

FULL_FLAG="${1:---full}"
PHASE="${3:-both}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="experiments_output/h10_enhanced_dg_${TIMESTAMP}"
mkdir -p "$OUT_DIR"

echo "Starting H10 Enhanced DG Pipeline at $(date)"
echo "Output: $OUT_DIR"
echo "Mode: $FULL_FLAG, Phase: $PHASE"

run_phase1() {
    echo "=== Phase 1: A B C D ==="
    CUDA_VISIBLE_DEVICES=0 python3 experiments/exp_h10_enhanced_dg_pipeline_loso.py \
        $FULL_FLAG --variants A --output_dir "${OUT_DIR}/variant_A" \
        > "${OUT_DIR}/gpu0_A.log" 2>&1 &
    local P0=$!

    CUDA_VISIBLE_DEVICES=1 python3 experiments/exp_h10_enhanced_dg_pipeline_loso.py \
        $FULL_FLAG --variants B --output_dir "${OUT_DIR}/variant_B" \
        > "${OUT_DIR}/gpu1_B.log" 2>&1 &
    local P1=$!

    CUDA_VISIBLE_DEVICES=2 python3 experiments/exp_h10_enhanced_dg_pipeline_loso.py \
        $FULL_FLAG --variants C --output_dir "${OUT_DIR}/variant_C" \
        > "${OUT_DIR}/gpu2_C.log" 2>&1 &
    local P2=$!

    CUDA_VISIBLE_DEVICES=3 python3 experiments/exp_h10_enhanced_dg_pipeline_loso.py \
        $FULL_FLAG --variants D --output_dir "${OUT_DIR}/variant_D" \
        > "${OUT_DIR}/gpu3_D.log" 2>&1 &
    local P3=$!

    echo "Phase 1 launched: A(PID=$P0) B(PID=$P1) C(PID=$P2) D(PID=$P3)"
    wait $P0 $P1 $P2 $P3
    echo "Phase 1 completed at $(date)"
}

run_phase2() {
    echo "=== Phase 2: E F G H ==="
    CUDA_VISIBLE_DEVICES=0 python3 experiments/exp_h10_enhanced_dg_pipeline_loso.py \
        $FULL_FLAG --variants E --output_dir "${OUT_DIR}/variant_E" \
        > "${OUT_DIR}/gpu0_E.log" 2>&1 &
    local P0=$!

    CUDA_VISIBLE_DEVICES=1 python3 experiments/exp_h10_enhanced_dg_pipeline_loso.py \
        $FULL_FLAG --variants F --output_dir "${OUT_DIR}/variant_F" \
        > "${OUT_DIR}/gpu1_F.log" 2>&1 &
    local P1=$!

    CUDA_VISIBLE_DEVICES=2 python3 experiments/exp_h10_enhanced_dg_pipeline_loso.py \
        $FULL_FLAG --variants G --output_dir "${OUT_DIR}/variant_G" \
        > "${OUT_DIR}/gpu2_G.log" 2>&1 &
    local P2=$!

    CUDA_VISIBLE_DEVICES=3 python3 experiments/exp_h10_enhanced_dg_pipeline_loso.py \
        $FULL_FLAG --variants H --output_dir "${OUT_DIR}/variant_H" \
        > "${OUT_DIR}/gpu3_H.log" 2>&1 &
    local P3=$!

    echo "Phase 2 launched: E(PID=$P0) F(PID=$P1) G(PID=$P2) H(PID=$P3)"
    wait $P0 $P1 $P2 $P3
    echo "Phase 2 completed at $(date)"
}

case "$PHASE" in
    1) run_phase1 ;;
    2) run_phase2 ;;
    both|*) run_phase1; run_phase2 ;;
esac

echo ""
echo "All variants completed at $(date)"
echo ""

# Print summary
for v in A B C D E F G H; do
    LOG=$(ls "${OUT_DIR}"/gpu*_${v}.log 2>/dev/null | head -1)
    if [ -n "$LOG" ]; then
        echo "=== Variant $v ==="
        grep -E '>>>' "$LOG" 2>/dev/null || echo "No results found"
    fi
done
