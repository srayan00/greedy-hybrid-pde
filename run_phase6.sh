#!/bin/zsh
# Ensemble routers with larger capacity / more imitation data (hyperparameters only), after the master chain.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-/Users/yash/miniconda3/envs/ansatz/bin/python}
until [ -f logs/aniso.done ]; do sleep 30; done
mkdir -p results_ens_big
for EQ in Poisson ConvDiff; do
  for W in jacobi,gs jacobi,gs,ssor jacobi,gs,ssor,jacobi_0.67 jacobi,gs,ssor,jacobi_0.67,sor_1.5; do
    TAG=${W//,/+}
    $PY bench.py --equation $EQ --N 128 --solvers $W --ensemble --train_only --retrain_router --router_tag _big --dagger_rounds 6 --router_inst 256 --router_hidden 128 --router_epochs 300 > logs/routers_ensbig_${EQ}_128_${TAG}.log 2>&1
    $PY bench.py --equation $EQ --N 128 --solvers $W --ensemble --n_test 64 --policies greedy,oracle,router --router_tag _big --out_dir results_ens_big > logs/bench_ensbig_${EQ}_128_${TAG}.log 2>&1
  done
done
echo done > logs/phase6.done
