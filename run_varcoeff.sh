#!/bin/zsh
# Variable-coefficient diffusion (no Fourier inverse): same confirmatory protocol as run_final.sh.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-/Users/yash/miniconda3/envs/ansatz/bin/python}
POL=classical,hints2,hints5,hints10,hints25,hints50,phints5,phints10,phints25,phints50,oneshot,greedy,oracle,router
SEED="--seed 73 --timed_reps 3"
until [ -f logs/final.done ]; do sleep 60; done
EQ=VarCoeff
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
[ -f checkpoints/deeponet_${EQ}_256_best.pth ] || $PY corrector.py --equation $EQ --N 256 --coarsen 4 --n_train 32000 --n_val 1000 > logs/deeponet_${EQ}_256.log 2>&1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
S=jacobi,jacobi_0.67,gs,ssor,sor_1.5,mg
$PY bench.py --equation $EQ --N 128 --solvers $S --measure_only --remeasure_costs > logs/final_costs_${EQ}_128.log 2>&1
$PY bench.py --equation $EQ --N 128 --solvers $S --train_only --retrain_router > logs/final_routers_${EQ}_128.log 2>&1
$PY bench.py --equation $EQ --N 128 --solvers $S --n_test 64 --max_ops 60000 --policies $POL ${=SEED} > logs/final_bench_${EQ}_128.log 2>&1
$PY bench_baselines.py --equation $EQ --N 128 --n_test 64 --seed 73 > logs/final_baselines_${EQ}_128.log 2>&1
for W in jacobi,jacobi_0.67 jacobi,jacobi_0.67,gs; do
  $PY bench.py --equation $EQ --N 128 --solvers $W --ensemble --retrain_router --router_inst 256 --router_max_epochs 600 --router_err_stop 1e-9 --dagger_rounds 6 --router_hidden 128 --router_epochs 300 --n_test 64 --max_ops 4000 --policies greedy,oracle,router --with_pairwise ${=SEED} > logs/final_nest_${EQ}_128_${W//,/+}.log 2>&1
done
$PY bench.py --equation $EQ --N 256 --solvers $S --measure_only --remeasure_costs > logs/final_costs_${EQ}_256.log 2>&1
$PY bench.py --equation $EQ --N 256 --solvers $S --train_only --retrain_router --router_inst 64 --router_max_epochs 1500 > logs/final_routers_${EQ}_256.log 2>&1
$PY bench.py --equation $EQ --N 256 --solvers $S --n_test 32 --max_ops 40000 --policies $POL ${=SEED} > logs/final_bench_${EQ}_256.log 2>&1
$PY bench_baselines.py --equation $EQ --N 256 --n_test 32 --seed 73 > logs/final_baselines_${EQ}_256.log 2>&1
for W in jacobi,jacobi_0.67 jacobi,jacobi_0.67,gs; do
  $PY bench.py --equation $EQ --N 256 --solvers $W --ensemble --retrain_router --router_inst 128 --router_max_epochs 600 --router_err_stop 1e-9 --dagger_rounds 6 --router_hidden 128 --router_epochs 300 --n_test 32 --max_ops 3000 --policies greedy,oracle,router --with_pairwise ${=SEED} > logs/final_nest_${EQ}_256_${W//,/+}.log 2>&1
done
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
$PY make_usage_data.py --equation $EQ --N 128 --solvers $S --n_test 64 --seed 73 > logs/final_usage_${EQ}_128.log 2>&1
$PY make_usage_data.py --equation $EQ --N 256 --solvers $S --n_test 32 --seed 73 > logs/final_usage_${EQ}_256.log 2>&1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
$PY bench_seeds.py --equation $EQ --N 128 --seed 73 > logs/final_seeds_${EQ}_128.log 2>&1
for N in 128 256; do
  $PY check_assumptions.py --equation $EQ --N $N --seed 73 > logs/final_assumptions_${EQ}_${N}.log 2>&1
  $PY check_assumptions.py --equation $EQ --N $N --seed 73 --paths_only --path_tol $($PY -c "print(1.0/$N**2)") > logs/final_paths_h2_${EQ}_${N}.log 2>&1
  $PY screen_ensembles.py --equation $EQ --N $N --members $S --n_inst 16 --seed 73 --max_ops $([ $N = 128 ] && echo 2000 || echo 1000) > logs/final_screen_${EQ}_${N}.log 2>&1
done
$PY check_theorem.py --equation $EQ --N 128 --n_inst 8 --seed 73 > logs/final_theorem_${EQ}_128.log 2>&1
$PY discretization_error.py > logs/final_discretization.log 2>&1
echo done > logs/varcoeff.done
