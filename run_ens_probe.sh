#!/bin/zsh
# probe: can the router capture the ensemble gain at 256^2 / 1e-8?  {jacobi, gs} and {jacobi, gs, ssor} on Poisson
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
PY=/Users/yash/miniconda3/envs/ansatz/bin/python
for W in jacobi,gs jacobi,gs,ssor; do
  TAG=${W//,/+}
  $PY bench.py --equation Poisson --N 256 --solvers $W --ensemble --retrain_router --router_inst 64 --router_max_epochs 600 --router_err_stop 1e-9 --dagger_rounds 3 --n_test 32 --max_ops 2000 --policies greedy,oracle,router > logs/bench_ens_Poisson_256_${TAG}.log 2>&1
done
echo done > logs/ens_probe.done
