#!/bin/zsh
# Retrain the 512^2 ConvDiff/Jacobi router with a larger imitation budget and re-run that cell (after everything else).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-/Users/yash/miniconda3/envs/ansatz/bin/python}
until [ -f logs/phase6.done ]; do sleep 30; done
$PY bench.py --equation ConvDiff --N 512 --solvers jacobi --train_only --retrain_router --router_inst 64 --router_max_epochs 800 --router_err_stop 1e-8 > logs/routers_ConvDiff_512_jacobi_v2.log 2>&1
$PY bench.py --equation ConvDiff --N 512 --solvers jacobi --n_test 16 --max_ops 8000 > logs/bench_ConvDiff_512_jacobi_v2.log 2>&1
echo done > logs/phase7.done
