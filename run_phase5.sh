#!/bin/zsh
# Ensemble routers retrained with more DAgger data (4 rounds, 256 instances), after phase 4.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-/Users/yash/miniconda3/envs/ansatz/bin/python}
until [ -f logs/phase4.done ]; do sleep 30; done
for EQ in Poisson ConvDiff; do
  for W in jacobi,gs jacobi,gs,ssor jacobi,gs,ssor,jacobi_0.67 jacobi,gs,ssor,jacobi_0.67,sor_1.5; do
    TAG=${W//,/+}
    $PY bench.py --equation $EQ --N 128 --solvers $W --ensemble --train_only --retrain_router --dagger_rounds 4 --router_inst 256 > logs/routers_ens2_${EQ}_128_${TAG}.log 2>&1
    $PY bench.py --equation $EQ --N 128 --solvers $W --ensemble --n_test 64 --policies greedy,oracle,router > logs/bench_ens2_${EQ}_128_${TAG}.log 2>&1
  done
done
echo done > logs/phase5.done
