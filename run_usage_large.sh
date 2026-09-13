#!/bin/zsh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 MKL_NUM_THREADS=2
PY=/Users/yash/miniconda3/envs/ansatz/bin/python
$PY make_usage_data.py --equation Poisson  --N 256 --solvers jacobi,jacobi_0.67,gs,ssor,sor_1.5,mg --n_test 32 > logs/usage_Poisson_256.log 2>&1
$PY make_usage_data.py --equation ConvDiff --N 256 --solvers jacobi,jacobi_0.67,gs,ssor,sor_1.5,mg --n_test 32 > logs/usage_ConvDiff_256.log 2>&1
$PY make_usage_data.py --equation AnisoDiff --N 256 --solvers jacobi,gs,sor_1.5 --n_test 16 > logs/usage_AnisoDiff_256.log 2>&1
$PY make_usage_data.py --equation Poisson  --N 512 --solvers jacobi,gs,sor_1.5,mg --n_test 16 > logs/usage_Poisson_512.log 2>&1
$PY make_usage_data.py --equation ConvDiff --N 512 --solvers jacobi,gs,sor_1.5,mg --n_test 16 > logs/usage_ConvDiff_512.log 2>&1
echo done > logs/usage_large.done
