#!/bin/zsh
# Fill the pairings that were skipped for compute: damped Jacobi and SymGS at 512^2 (Poisson, ConvDiff),
# SymGS at 256^2 (AnisoDiff); then regenerate the usage traces of those grids (all pairings of the grid).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-/Users/yash/miniconda3/envs/ansatz/bin/python}
for EQ in Poisson ConvDiff; do
  S=jacobi_0.67,ssor
  $PY bench.py --equation $EQ --N 512 --solvers $S --measure_only > logs/costs_${EQ}_512_fill.log 2>&1
  $PY bench.py --equation $EQ --N 512 --solvers $S --train_only --retrain_router --router_inst 32 --router_max_epochs 400 --router_err_stop 1e-8 > logs/routers_${EQ}_512_fill.log 2>&1
  $PY bench.py --equation $EQ --N 512 --solvers $S --n_test 16 --max_ops 8000 > logs/bench_${EQ}_512_fill.log 2>&1
done
$PY bench.py --equation AnisoDiff --N 256 --solvers ssor --measure_only > logs/costs_AnisoDiff_256_ssor.log 2>&1
$PY bench.py --equation AnisoDiff --N 256 --solvers ssor --train_only --retrain_router --router_inst 64 --router_max_epochs 1500 > logs/routers_AnisoDiff_256_ssor.log 2>&1
$PY bench.py --equation AnisoDiff --N 256 --solvers ssor --n_test 16 --max_ops 40000 > logs/bench_AnisoDiff_256_ssor.log 2>&1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 MKL_NUM_THREADS=2
$PY make_usage_data.py --equation Poisson  --N 512 --solvers jacobi,jacobi_0.67,gs,ssor,sor_1.5,mg --n_test 16 > logs/usage_Poisson_512.log 2>&1
$PY make_usage_data.py --equation ConvDiff --N 512 --solvers jacobi,jacobi_0.67,gs,ssor,sor_1.5,mg --n_test 16 > logs/usage_ConvDiff_512.log 2>&1
$PY make_usage_data.py --equation AnisoDiff --N 256 --solvers jacobi,jacobi_0.67,gs,ssor,sor_1.5 --n_test 16 > logs/usage_AnisoDiff_256.log 2>&1
echo done > logs/fill.done
