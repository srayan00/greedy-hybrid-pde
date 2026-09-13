#!/bin/zsh
# Nested ensembles {J} < {J, dJ} < {J, dJ, GS} on all equations at 128^2 and 256^2, with the
# members' pairwise routers evaluated in the same session (--with_pairwise). Costs of the
# ensemble are assembled from the pairwise cache (identical macro-action sizes).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
PY=/Users/yash/miniconda3/envs/ansatz/bin/python
COMMON="--policies greedy,oracle,router --with_pairwise --router_err_stop 1e-9 --dagger_rounds 3 --retrain_router"
for EQ in Poisson ConvDiff AnisoDiff; do
  for W in jacobi,jacobi_0.67 jacobi,jacobi_0.67,gs; do
    TAG=${W//,/+}
    $PY bench.py --equation $EQ --N 128 --solvers $W --ensemble --n_test 64 --max_ops 4000 --router_inst 128 --router_max_epochs 600 ${=COMMON} > logs/bench_nest_${EQ}_128_${TAG}.log 2>&1
  done
done
# damped-Jacobi pairing on anisotropic diffusion at 256^2 (needed as a single-solver baseline)
$PY bench.py --equation AnisoDiff --N 256 --solvers jacobi_0.67 --measure_only > logs/costs_AnisoDiff_256_jacobi_0.67.log 2>&1
$PY bench.py --equation AnisoDiff --N 256 --solvers jacobi_0.67 --train_only --retrain_router --router_inst 64 --router_max_epochs 1500 > logs/routers_AnisoDiff_256_jacobi_0.67.log 2>&1
$PY bench.py --equation AnisoDiff --N 256 --solvers jacobi_0.67 --n_test 16 --max_ops 40000 > logs/bench_AnisoDiff_256_jacobi_0.67.log 2>&1
for EQ in Poisson ConvDiff AnisoDiff; do
  NT=32; [ "$EQ" = "AnisoDiff" ] && NT=16
  for W in jacobi,jacobi_0.67 jacobi,jacobi_0.67,gs; do
    TAG=${W//,/+}
    $PY bench.py --equation $EQ --N 256 --solvers $W --ensemble --n_test $NT --max_ops 3000 --router_inst 64 --router_max_epochs 600 ${=COMMON} > logs/bench_nest_${EQ}_256_${TAG}.log 2>&1
  done
done
echo done > logs/nest.done
