#!/bin/zsh
# Confirmatory re-run of the whole wall-clock study with the compiled stencil kernels (libstencil.so),
# a fresh test seed (73; every configuration choice was made on seed 72 with the numpy kernels, archived in
# results_dev/), frozen recipes, the complete fixed-schedule baseline set and three timed replays.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-/Users/yash/miniconda3/envs/ansatz/bin/python}
POL=classical,hints2,hints5,hints10,hints25,hints50,phints5,phints10,phints25,phints50,oneshot,greedy,oracle,router
SEED="--seed 73 --timed_reps 3"
stage() { echo "$(date '+%F %T') $1" >> logs/final_progress.log; }
# ---------------------------------------------------------------- 128^2
for EQ in Poisson ConvDiff AnisoDiff; do
  S=jacobi,jacobi_0.67,gs,ssor,sor_1.5; [ "$EQ" != "AnisoDiff" ] && S=$S,mg
  [ -f checkpoints/costs_${EQ}_128.json ] || $PY bench.py --equation $EQ --N 128 --solvers $S --measure_only --remeasure_costs > logs/final_costs_${EQ}_128.log 2>&1
  [ -f checkpoints/router_${EQ}_128_sor_1.5.pth ] && [ checkpoints/router_${EQ}_128_sor_1.5.pth -nt libstencil.so ] || $PY bench.py --equation $EQ --N 128 --solvers $S --train_only --retrain_router > logs/final_routers_${EQ}_128.log 2>&1
  $PY bench.py --equation $EQ --N 128 --solvers $S --n_test 64 --max_ops 60000 --policies $POL ${=SEED} > logs/final_bench_${EQ}_128.log 2>&1
  [ "$EQ" != "AnisoDiff" ] && $PY bench_baselines.py --equation $EQ --N 128 --n_test 64 --seed 73 > logs/final_baselines_${EQ}_128.log 2>&1
  stage "128 $EQ pairwise done"
done
echo done > logs/final_128.done
# ---------------------------------------------------------------- ensembles at 128^2 (nested + the four original sets)
for EQ in Poisson ConvDiff AnisoDiff; do
  for W in jacobi,jacobi_0.67 jacobi,jacobi_0.67,gs; do
    $PY bench.py --equation $EQ --N 128 --solvers $W --ensemble --retrain_router --router_inst 256 --router_max_epochs 600 --router_err_stop 1e-9 --dagger_rounds 6 --router_hidden 128 --router_epochs 300 --n_test 64 --max_ops 4000 --policies greedy,oracle,router --with_pairwise ${=SEED} > logs/final_nest_${EQ}_128_${W//,/+}.log 2>&1
  done
  for W in jacobi,gs jacobi,gs,ssor jacobi,gs,ssor,jacobi_0.67 jacobi,gs,ssor,jacobi_0.67,sor_1.5; do
    $PY bench.py --equation $EQ --N 128 --solvers $W --ensemble --retrain_router --dagger_rounds 4 --router_inst 256 --n_test 64 --policies greedy,oracle,router --with_pairwise ${=SEED} > logs/final_ens_${EQ}_128_${W//,/+}.log 2>&1
  done
  stage "128 $EQ ensembles done"
done
echo done > logs/final_ens128.done
# ---------------------------------------------------------------- 256^2
for EQ in Poisson ConvDiff AnisoDiff; do
  S=jacobi,jacobi_0.67,gs,ssor,sor_1.5; NT=32; [ "$EQ" != "AnisoDiff" ] && S=$S,mg; [ "$EQ" = "AnisoDiff" ] && NT=16
  $PY bench.py --equation $EQ --N 256 --solvers $S --measure_only --remeasure_costs > logs/final_costs_${EQ}_256.log 2>&1
  $PY bench.py --equation $EQ --N 256 --solvers $S --train_only --retrain_router --router_inst 64 --router_max_epochs 1500 > logs/final_routers_${EQ}_256.log 2>&1
  $PY bench.py --equation $EQ --N 256 --solvers $S --n_test $NT --max_ops 40000 --policies $POL ${=SEED} > logs/final_bench_${EQ}_256.log 2>&1
  [ "$EQ" != "AnisoDiff" ] && $PY bench_baselines.py --equation $EQ --N 256 --n_test 32 --seed 73 > logs/final_baselines_${EQ}_256.log 2>&1
  for W in jacobi,jacobi_0.67 jacobi,jacobi_0.67,gs; do
    $PY bench.py --equation $EQ --N 256 --solvers $W --ensemble --retrain_router --router_inst 128 --router_max_epochs 600 --router_err_stop 1e-9 --dagger_rounds 6 --router_hidden 128 --router_epochs 300 --n_test $NT --max_ops 3000 --policies greedy,oracle,router --with_pairwise ${=SEED} > logs/final_nest_${EQ}_256_${W//,/+}.log 2>&1
  done
  stage "256 $EQ done"
done
echo done > logs/final_256.done
# ---------------------------------------------------------------- 512^2 (isotropic equations)
for EQ in Poisson ConvDiff; do
  S=jacobi,jacobi_0.67,gs,ssor,sor_1.5,mg
  $PY bench.py --equation $EQ --N 512 --solvers $S --measure_only --remeasure_costs > logs/final_costs_${EQ}_512.log 2>&1
  $PY bench.py --equation $EQ --N 512 --solvers $S --train_only --retrain_router --router_inst 32 --router_max_epochs 400 --router_err_stop 1e-8 > logs/final_routers_${EQ}_512.log 2>&1
  if [ "$EQ" = "ConvDiff" ]; then   # frozen per-cell exception of the development phase
    $PY bench.py --equation $EQ --N 512 --solvers jacobi --train_only --retrain_router --router_inst 64 --router_max_epochs 800 --router_err_stop 1e-8 > logs/final_routers_${EQ}_512_jacobi.log 2>&1
  fi
  $PY bench.py --equation $EQ --N 512 --solvers $S --n_test 16 --max_ops 8000 --policies $POL ${=SEED} > logs/final_bench_${EQ}_512.log 2>&1
  $PY bench_baselines.py --equation $EQ --N 512 --n_test 16 --max_iter 3000 --seed 73 > logs/final_baselines_${EQ}_512.log 2>&1
  stage "512 $EQ done"
done
echo done > logs/final_512.done
# ---------------------------------------------------------------- auxiliary: usage traces, seeds, overheads, assumptions, screening, theorem check
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 MKL_NUM_THREADS=2
$PY make_usage_data.py --equation Poisson  --N 128 --solvers jacobi,jacobi_0.67,gs,ssor,sor_1.5,mg --n_test 64 --seed 73 > logs/final_usage_Poisson_128.log 2>&1
$PY make_usage_data.py --equation ConvDiff --N 128 --solvers jacobi,jacobi_0.67,gs,ssor,sor_1.5,mg --n_test 64 --seed 73 > logs/final_usage_ConvDiff_128.log 2>&1
$PY make_usage_data.py --equation AnisoDiff --N 128 --solvers jacobi,jacobi_0.67,gs,ssor,sor_1.5 --n_test 64 --seed 73 > logs/final_usage_AnisoDiff_128.log 2>&1
$PY make_usage_data.py --equation Poisson  --N 256 --solvers jacobi,jacobi_0.67,gs,ssor,sor_1.5,mg --n_test 32 --seed 73 > logs/final_usage_Poisson_256.log 2>&1
$PY make_usage_data.py --equation ConvDiff --N 256 --solvers jacobi,jacobi_0.67,gs,ssor,sor_1.5,mg --n_test 32 --seed 73 > logs/final_usage_ConvDiff_256.log 2>&1
$PY make_usage_data.py --equation AnisoDiff --N 256 --solvers jacobi,jacobi_0.67,gs,ssor,sor_1.5 --n_test 16 --seed 73 > logs/final_usage_AnisoDiff_256.log 2>&1
$PY make_usage_data.py --equation Poisson  --N 512 --solvers jacobi,jacobi_0.67,gs,ssor,sor_1.5,mg --n_test 16 --seed 73 > logs/final_usage_Poisson_512.log 2>&1
$PY make_usage_data.py --equation ConvDiff --N 512 --solvers jacobi,jacobi_0.67,gs,ssor,sor_1.5,mg --n_test 16 --seed 73 > logs/final_usage_ConvDiff_512.log 2>&1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
for EQ in Poisson ConvDiff AnisoDiff; do
  $PY bench_seeds.py --equation $EQ --N 128 --seed 73 > logs/final_seeds_${EQ}_128.log 2>&1
  NT=32; [ "$EQ" = "AnisoDiff" ] && NT=16
  $PY bench_seeds.py --equation $EQ --N 256 --solvers jacobi --n_test $NT --router_inst 64 --router_max_epochs 1500 --max_ops 40000 --seed 73 > logs/final_seeds_${EQ}_256.log 2>&1
done
$PY bench_overheads.py > logs/final_overheads.log 2>&1
for N in 128 256 512; do for EQ in Poisson ConvDiff AnisoDiff; do
  [ -f checkpoints/costs_${EQ}_${N}.json ] || continue
  $PY check_assumptions.py --equation $EQ --N $N --seed 73 > logs/final_assumptions_${EQ}_${N}.log 2>&1
  $PY check_assumptions.py --equation $EQ --N $N --seed 73 --paths_only --path_tol $($PY -c "print(1.0/$N**2)") > logs/final_paths_h2_${EQ}_${N}.log 2>&1
  M=jacobi,jacobi_0.67,gs,ssor,sor_1.5; [ "$EQ" != "AnisoDiff" ] && M=$M,mg
  $PY screen_ensembles.py --equation $EQ --N $N --members $M --n_inst 16 --seed 73 --max_ops $([ $N = 128 ] && echo 2000 || echo 1000) > logs/final_screen_${EQ}_${N}.log 2>&1
done; done
for EQ in Poisson ConvDiff AnisoDiff; do $PY check_theorem.py --equation $EQ --N 128 --n_inst 8 --seed 73 > logs/final_theorem_${EQ}_128.log 2>&1; done
$PY discretization_error.py > logs/final_discretization.log 2>&1
echo done > logs/final.done
stage "ALL DONE"
