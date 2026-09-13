#!/bin/zsh
# Full pairwise study for one equation: costs -> routers -> benchmark.
# Usage: ./run_pairwise.sh Poisson [N] [NTEST]
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-/Users/yash/miniconda3/envs/ansatz/bin/python}
EQ=$1
N=${2:-128}
NTEST=${3:-64}
SOLVERS=jacobi,jacobi_0.67,gs,ssor,sor_1.5
$PY bench.py --equation $EQ --N $N --solvers $SOLVERS --measure_only > logs/costs_${EQ}_${N}.log 2>&1
$PY bench.py --equation $EQ --N $N --solvers $SOLVERS --train_only > logs/routers_${EQ}_${N}.log 2>&1
$PY bench.py --equation $EQ --N $N --solvers $SOLVERS --n_test $NTEST > logs/bench_${EQ}_${N}.log 2>&1
