#!/bin/bash
# Run SCG-Net 4 variants in parallel across 4 GPUs.
# Usage: bash scripts/run_scg_parallel_4gpu.sh

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="experiments_output/scg_net_4gpu_${TIMESTAMP}"
mkdir -p "$OUT_DIR"

echo "Starting SCG-Net parallel run at $(date)"
echo "Output: $OUT_DIR"

# GPU 0: Variant A (Sinc + SCG, adversary ON)
CUDA_VISIBLE_DEVICES=0 python3 experiments/exp_scg_net_loso.py \
    --full --variants A \
    --output_dir "${OUT_DIR}/variant_A" \
    > "${OUT_DIR}/gpu0_A.log" 2>&1 &
PID_A=$!

# GPU 1: Variant B (UVMD + SCG, adversary ON)
CUDA_VISIBLE_DEVICES=1 python3 experiments/exp_scg_net_loso.py \
    --full --variants B \
    --output_dir "${OUT_DIR}/variant_B" \
    > "${OUT_DIR}/gpu1_B.log" 2>&1 &
PID_B=$!

# GPU 2: Variant C (Sinc + SCG, no adversary)
CUDA_VISIBLE_DEVICES=2 python3 experiments/exp_scg_net_loso.py \
    --full --variants C \
    --output_dir "${OUT_DIR}/variant_C" \
    > "${OUT_DIR}/gpu2_C.log" 2>&1 &
PID_C=$!

# GPU 3: Variant D (Sinc + full IN, γ=1 fixed)
CUDA_VISIBLE_DEVICES=3 python3 experiments/exp_scg_net_loso.py \
    --full --variants D \
    --output_dir "${OUT_DIR}/variant_D" \
    > "${OUT_DIR}/gpu3_D.log" 2>&1 &
PID_D=$!

echo "GPU 0: Variant A started (PID=$PID_A)"
echo "GPU 1: Variant B started (PID=$PID_B)"
echo "GPU 2: Variant C started (PID=$PID_C)"
echo "GPU 3: Variant D started (PID=$PID_D)"
echo ""
echo "All 4 variants launched. PIDs: A=$PID_A, B=$PID_B, C=$PID_C, D=$PID_D"
echo "Monitor with: tail -f ${OUT_DIR}/gpu*.log"

wait $PID_A $PID_B $PID_C $PID_D
echo ""
echo "All variants completed at $(date)"

# Print summary
for v in A B C D; do
    echo "=== Variant $v ==="
    grep '>>>' "${OUT_DIR}/gpu*_${v}.log" 2>/dev/null || echo "No results found"
done
