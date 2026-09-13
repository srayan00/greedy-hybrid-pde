#!/bin/zsh
# Ensemble study for one equation: routers over NO + W for the paper's four ensembles.
# Usage: ./run_ensembles.sh Poisson
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-/Users/yash/miniconda3/envs/ansatz/bin/python}
EQ=$1
N=${2:-128}
NTEST=${3:-64}
for W in jacobi,gs jacobi,gs,ssor jacobi,gs,ssor,jacobi_0.67 jacobi,gs,ssor,jacobi_0.67,sor_1.5; do
  TAG=${W//,/+}
  $PY bench.py --equation $EQ --N $N --solvers $W --ensemble --train_only > logs/routers_ens_${EQ}_${N}_${TAG}.log 2>&1
  $PY bench.py --equation $EQ --N $N --solvers $W --ensemble --n_test $NTEST --policies greedy,oracle,router > logs/bench_ens_${EQ}_${N}_${TAG}.log 2>&1
done
