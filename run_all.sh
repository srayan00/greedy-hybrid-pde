#!/bin/zsh
# Master chain (unified cost rule: unit = corrector call; m_j = round(u/c_j); per-unit-cost exponent).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-/Users/yash/miniconda3/envs/ansatz/bin/python}
S128=jacobi,jacobi_0.67,gs,ssor,sor_1.5,mg
S256=jacobi,jacobi_0.67,gs,ssor,sor_1.5,mg
S512=jacobi,gs,sor_1.5,mg
# ---- 128^2 pairwise (routers retrained under the unified rule)
for EQ in Poisson ConvDiff; do
  $PY bench.py --equation $EQ --N 128 --solvers $S128 --train_only --retrain_router > logs/routers_${EQ}_128.log 2>&1
  $PY bench.py --equation $EQ --N 128 --solvers $S128 --n_test 64 > logs/bench_${EQ}_128.log 2>&1
done
# ---- 128^2 ensembles (4 DAgger rounds, 256 instances)
for EQ in Poisson ConvDiff; do
  for W in jacobi,gs jacobi,gs,ssor jacobi,gs,ssor,jacobi_0.67 jacobi,gs,ssor,jacobi_0.67,sor_1.5; do
    TAG=${W//,/+}
    $PY bench.py --equation $EQ --N 128 --solvers $W --ensemble --train_only --retrain_router --dagger_rounds 4 --router_inst 256 > logs/routers_ens_${EQ}_128_${TAG}.log 2>&1
    $PY bench.py --equation $EQ --N 128 --solvers $W --ensemble --n_test 64 --policies greedy,oracle,router > logs/bench_ens_${EQ}_128_${TAG}.log 2>&1
  done
done
echo done > logs/stage128.done
# ---- usage traces (untimed)
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
for EQ in Poisson ConvDiff; do
  $PY make_usage_data.py --equation $EQ --N 128 > logs/usage_${EQ}_128.log 2>&1
done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
# ---- 256^2
for EQ in Poisson ConvDiff; do
  $PY bench.py --equation $EQ --N 256 --solvers $S256 --measure_only > logs/costs_${EQ}_256.log 2>&1
  $PY bench.py --equation $EQ --N 256 --solvers $S256 --train_only --retrain_router --router_inst 64 --router_max_epochs 1500 > logs/routers_${EQ}_256.log 2>&1
  $PY bench.py --equation $EQ --N 256 --solvers $S256 --n_test 32 --max_ops 40000 > logs/bench_${EQ}_256.log 2>&1
  $PY bench_baselines.py --equation $EQ --N 256 --n_test 32 > logs/baselines_${EQ}_256.log 2>&1
done
echo done > logs/stage256.done
# ---- 512^2
for EQ in Poisson ConvDiff; do
  $PY bench.py --equation $EQ --N 512 --solvers $S512 --measure_only > logs/costs_${EQ}_512.log 2>&1
  $PY bench.py --equation $EQ --N 512 --solvers $S512 --train_only --retrain_router --router_inst 32 --router_max_epochs 400 --router_err_stop 1e-8 > logs/routers_${EQ}_512.log 2>&1
  $PY bench.py --equation $EQ --N 512 --solvers $S512 --n_test 16 --max_ops 8000 > logs/bench_${EQ}_512.log 2>&1
  $PY bench_baselines.py --equation $EQ --N 512 --n_test 16 --max_iter 3000 > logs/baselines_${EQ}_512.log 2>&1
done
echo done > logs/stage512.done
# ---- overheads, then seed trials at 128^2
$PY bench_overheads.py > logs/overheads.log 2>&1
for EQ in Poisson ConvDiff; do
  $PY bench_seeds.py --equation $EQ --N 128 > logs/seeds_${EQ}_128.log 2>&1
done
echo done > logs/all.done
