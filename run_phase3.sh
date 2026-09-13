#!/bin/zsh
# Reviewer-response experiments: strong baselines, MG pairing, scaling to 256^2/512^2, overheads.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-/Users/yash/miniconda3/envs/ansatz/bin/python}
until [ -f logs/correctors_large.done ]; do sleep 20; done
# --- 128^2: strong baselines + multigrid pairing
for EQ in Poisson ConvDiff; do
  $PY bench_baselines.py --equation $EQ --N 128 --n_test 64 > logs/baselines_${EQ}_128.log 2>&1
  $PY bench.py --equation $EQ --N 128 --solvers mg --train_only > logs/routers_${EQ}_128_mg.log 2>&1
  $PY bench.py --equation $EQ --N 128 --solvers mg --n_test 64 > logs/bench_${EQ}_128_mg.log 2>&1
done
# --- 256^2: all pairings + mg, 32 instances
for EQ in Poisson ConvDiff; do
  $PY bench.py --equation $EQ --N 256 --solvers jacobi,jacobi_0.67,gs,ssor,sor_1.5,mg --measure_only > logs/costs_${EQ}_256.log 2>&1
  $PY bench.py --equation $EQ --N 256 --solvers jacobi,jacobi_0.67,gs,ssor,sor_1.5,mg --train_only --router_inst 64 --router_max_epochs 1500 > logs/routers_${EQ}_256.log 2>&1
  $PY bench.py --equation $EQ --N 256 --solvers jacobi,jacobi_0.67,gs,ssor,sor_1.5,mg --n_test 32 --max_ops 40000 > logs/bench_${EQ}_256.log 2>&1
  $PY bench_baselines.py --equation $EQ --N 256 --n_test 32 > logs/baselines_${EQ}_256.log 2>&1
done
# --- 512^2: main pairings + mg, 16 instances, classical runs capped
for EQ in Poisson ConvDiff; do
  $PY bench.py --equation $EQ --N 512 --solvers jacobi,gs,sor_1.5,mg --measure_only > logs/costs_${EQ}_512.log 2>&1
  $PY bench.py --equation $EQ --N 512 --solvers jacobi,gs,sor_1.5,mg --train_only --router_inst 32 --router_max_epochs 400 --router_err_stop 1e-8 > logs/routers_${EQ}_512.log 2>&1
  $PY bench.py --equation $EQ --N 512 --solvers jacobi,gs,sor_1.5,mg --n_test 16 --max_ops 8000 > logs/bench_${EQ}_512.log 2>&1
  $PY bench_baselines.py --equation $EQ --N 512 --n_test 16 --max_iter 3000 > logs/baselines_${EQ}_512.log 2>&1
done
$PY bench_overheads.py > logs/overheads.log 2>&1
echo done > logs/phase3.done
