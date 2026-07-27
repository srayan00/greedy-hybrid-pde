#!/bin/bash
# Local/CPU smoke test of the FNO + ApproxGreedy lite-router pipeline.
set -eo pipefail
source ~/.bashrc || true
conda activate greedy

ROOT=/home/srayan/greedy-hybrid-pde/playground/lite_criteria
CKP=$ROOT/checkpoints_smoke
RES=$ROOT/results_smoke
mkdir -p "$CKP" "$RES"
cd "$ROOT"

N=31
DEV=${DEV:-cpu}

echo "=== smoke: train FNO ==="
python -u train_fno.py \
  --N "$N" --epochs 5 --n_train 256 --n_val 64 \
  --trunc_mode 8 --hidden_size 32 --num_layers 2 --batch_size 64 \
  --ckp_dir "$CKP" --tag fno_smoke --device "$DEV"

FNO=$CKP/fno_smoke_N${N}_best.pth

echo "=== smoke: train router ==="
python -u train_and_bench.py train_router \
  --N "$N" --fno_ckp "$FNO" --out "$CKP/lite_router_smoke.pth" \
  --n_inst 32 --max_iters 80 --epochs 3 --batches_per_epoch 2 --device "$DEV"

echo "=== smoke: bench ==="
python -u train_and_bench.py bench \
  --N "$N" --fno_ckp "$FNO" --router_ckp "$CKP/lite_router_smoke.pth" \
  --n_test 8 --hints_tau 25 --out "$RES/smoke.json" --device "$DEV"

echo "SMOKE OK"
