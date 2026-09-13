#!/bin/zsh
# Re-time all nested-ensemble cells on an idle machine (routers already trained; no retraining).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-/Users/yash/miniconda3/envs/ansatz/bin/python}
until [ -f logs/nest.done ]; do sleep 30; done
sleep 60
for EQ in Poisson ConvDiff AnisoDiff; do
  for W in jacobi,jacobi_0.67 jacobi,jacobi_0.67,gs; do
    TAG=${W//,/+}
    $PY bench.py --equation $EQ --N 128 --solvers $W --ensemble --n_test 64 --max_ops 4000 --policies greedy,oracle,router --with_pairwise > logs/retime_${EQ}_128_${TAG}.log 2>&1
  done
done
for EQ in Poisson ConvDiff AnisoDiff; do
  NT=32; [ "$EQ" = "AnisoDiff" ] && NT=16
  for W in jacobi,jacobi_0.67 jacobi,jacobi_0.67,gs; do
    TAG=${W//,/+}
    $PY bench.py --equation $EQ --N 256 --solvers $W --ensemble --n_test $NT --max_ops 3000 --policies greedy,oracle,router --with_pairwise > logs/retime_${EQ}_256_${TAG}.log 2>&1
  done
done
echo done > logs/retime.done
