#!/bin/zsh
# Anisotropic diffusion (-eps u_xx - u_yy = f, eps = 0.01): same pipeline; the corrector's sensor grid keeps
# full resolution along x (the direction the point smoothers cannot damp) and 1/4 along y.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-/Users/yash/miniconda3/envs/ansatz/bin/python}
until [ -f logs/all.done ]; do sleep 30; done
EQ=AnisoDiff
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8
$PY corrector.py --equation $EQ --N 128 --coarsen_x 1 --coarsen_y 4 --n_train 64000 --n_val 2000 > logs/deeponet_${EQ}_128.log 2>&1
$PY corrector.py --equation $EQ --N 256 --coarsen_x 1 --coarsen_y 4 --n_train 48000 --n_val 1000 > logs/deeponet_${EQ}_256.log 2>&1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
S128=jacobi,jacobi_0.67,gs,ssor,sor_1.5
$PY bench.py --equation $EQ --N 128 --solvers $S128 --measure_only > logs/costs_${EQ}_128.log 2>&1
$PY bench.py --equation $EQ --N 128 --solvers $S128 --train_only --retrain_router > logs/routers_${EQ}_128.log 2>&1
$PY bench.py --equation $EQ --N 128 --solvers $S128 --n_test 64 > logs/bench_${EQ}_128.log 2>&1
for W in jacobi,gs jacobi,gs,ssor jacobi,gs,ssor,jacobi_0.67 jacobi,gs,ssor,jacobi_0.67,sor_1.5; do
  TAG=${W//,/+}
  $PY bench.py --equation $EQ --N 128 --solvers $W --ensemble --train_only --retrain_router --dagger_rounds 4 --router_inst 256 > logs/routers_ens_${EQ}_128_${TAG}.log 2>&1
  $PY bench.py --equation $EQ --N 128 --solvers $W --ensemble --n_test 64 --policies greedy,oracle,router > logs/bench_ens_${EQ}_128_${TAG}.log 2>&1
done
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
$PY make_usage_data.py --equation $EQ --N 128 > logs/usage_${EQ}_128.log 2>&1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
S256=jacobi,gs,sor_1.5
$PY bench.py --equation $EQ --N 256 --solvers $S256 --measure_only > logs/costs_${EQ}_256.log 2>&1
$PY bench.py --equation $EQ --N 256 --solvers $S256 --train_only --retrain_router --router_inst 64 --router_max_epochs 1500 > logs/routers_${EQ}_256.log 2>&1
$PY bench.py --equation $EQ --N 256 --solvers $S256 --n_test 16 --max_ops 40000 > logs/bench_${EQ}_256.log 2>&1
$PY bench_seeds.py --equation $EQ --N 128 > logs/seeds_${EQ}_128.log 2>&1
echo done > logs/aniso.done
