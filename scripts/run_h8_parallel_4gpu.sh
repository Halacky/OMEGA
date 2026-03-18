#!/bin/bash
# Run H8 experiment in parallel across 4 GPUs on 4x RTX 4090
# Each variant gets its own GPU:
#   GPU 0: Variant G (Sinc + Adaptive MixStyle)
#   GPU 1: Variant I (Sinc + Reversed gradient)
#   GPU 2: Variant H (UVMD + Adaptive MixStyle)
#   GPU 3: Variant J (UVMD + Reversed gradient)

set -e
cd /home/kirill/omega_experiments

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_BASE="experiments_output/h8_4gpu_${TIMESTAMP}"
mkdir -p "$OUT_BASE"

echo "Starting H8 parallel run at $(date)"
echo "Output: $OUT_BASE"

# GPU 0: Variant G (Sinc, fast ~5 min/fold)
CUDA_VISIBLE_DEVICES=0 python3 experiments/h8_adaptive_mixstyle_loso.py \
    --full --variants G \
    --output_dir "${OUT_BASE}/variant_G" \
    > "${OUT_BASE}/gpu0_G.log" 2>&1 &
PID_G=$!
echo "GPU 0: Variant G started (PID=$PID_G)"

# GPU 1: Variant I (Sinc reversed, fast ~5 min/fold)
CUDA_VISIBLE_DEVICES=1 python3 experiments/h8_adaptive_mixstyle_loso.py \
    --full --variants I \
    --output_dir "${OUT_BASE}/variant_I" \
    > "${OUT_BASE}/gpu1_I.log" 2>&1 &
PID_I=$!
echo "GPU 1: Variant I started (PID=$PID_I)"

# GPU 2: Variant H (UVMD, slower ~10 min/fold)
CUDA_VISIBLE_DEVICES=2 python3 experiments/h8_adaptive_mixstyle_loso.py \
    --full --variants H \
    --output_dir "${OUT_BASE}/variant_H" \
    > "${OUT_BASE}/gpu2_H.log" 2>&1 &
PID_H=$!
echo "GPU 2: Variant H started (PID=$PID_H)"

# GPU 3: Variant J (UVMD reversed, slower ~10 min/fold)
CUDA_VISIBLE_DEVICES=3 python3 experiments/h8_adaptive_mixstyle_loso.py \
    --full --variants J \
    --output_dir "${OUT_BASE}/variant_J" \
    > "${OUT_BASE}/gpu3_J.log" 2>&1 &
PID_J=$!
echo "GPU 3: Variant J started (PID=$PID_J)"

echo ""
echo "All 4 variants launched. PIDs: G=$PID_G, I=$PID_I, H=$PID_H, J=$PID_J"
echo "Monitor with: tail -f ${OUT_BASE}/gpu*.log"
echo ""

# Wait for all
wait $PID_G
echo "$(date): Variant G finished (exit $?)"
wait $PID_I
echo "$(date): Variant I finished (exit $?)"
wait $PID_H
echo "$(date): Variant H finished (exit $?)"
wait $PID_J
echo "$(date): Variant J finished (exit $?)"

echo ""
echo "All variants complete at $(date)"
echo "Results in: $OUT_BASE"

# Print summary from logs
echo ""
echo "=== SUMMARY ==="
for v in G I H J; do
    grep ">>>" "${OUT_BASE}/gpu*_${v}.log" 2>/dev/null || echo "Variant $v: no summary found"
done
