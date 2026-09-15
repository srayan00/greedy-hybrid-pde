#!/bin/zsh
# Confirmatory re-run of the wall-clock study (revision 2, after the third review):
#   compiled stencil kernels; corrected (Hermitian) GRF sampler; live per-operation costs (update +
#   residual + norm, median of blocks); Alg. 1 exponents (1) for every action not dearer than the unit;
#   HINTS tau = 15 (the 2-D period of the HINTS paper) in the schedule family; residual-decay controls;
#   classical baselines timed inside the same replay loop as the policies; drift guard per instance;
#   fresh test seed 73 (all configuration choices were made on seed 72, archived in results_dev/);
#   three timed replays in random order with warm-up. Fail-fast: any failing command stops the chain.
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-/Users/yash/miniconda3/envs/ansatz/bin/python}
POL=classical,hints2,hints5,hints10,hints15,hints25,hints50,phints5,phints10,phints15,phints25,phints50,oneshot,decay0.9,decay0.95,decay0.98,greedy,oracle,router
SEED="--seed 73 --timed_reps 3"
stage() { echo "$(date '+%F %T') $1" >> logs/final_progress.log; }
run() { "$@" || { echo "$(date '+%F %T') FAILED (exit $?): $*" >> logs/final_progress.log; echo FAILED > logs/final.failed; exit 1; }; }
solvers() { cat config/solvers_$1; }
rm -f logs/final.failed
# ---------------------------------------------------------------- 128^2 (pairwise + same-session baselines)
done_cells() { for s in $(echo $2 | tr , ' '); do [ -f results/${1}_${3}_${s}.json ] || return 1; done; return 0; }
for EQ in Poisson ConvDiff AnisoDiff; do
  S=$(solvers $EQ)
  done_cells $EQ $S 128 && continue
  [ -f checkpoints/costs_${EQ}_128.json ] || run $PY bench.py --equation $EQ --N 128 --solvers $S --measure_only --remeasure_costs > logs/final_costs_${EQ}_128.log 2>&1
  [ -f checkpoints/router_${EQ}_128_sor_1.5.pth ] && [ checkpoints/router_${EQ}_128_sor_1.5.pth -nt libstencil.so ] || run $PY bench.py --equation $EQ --N 128 --solvers $S --train_only --retrain_router > logs/final_routers_${EQ}_128.log 2>&1
  # pairings whose result file exists are skipped (resumable after an interruption)
  TODO=$(for s in $(echo $S | tr , ' '); do [ -f results/${EQ}_128_${s}.json ] || printf "%s," $s; done); TODO=${TODO%,}
  run $PY bench.py --equation $EQ --N 128 --solvers $TODO --n_test 64 --max_ops 60000 --policies $POL ${=SEED} > logs/final_bench_${EQ}_128.log 2>&1
  stage "128 $EQ pairwise done"
done
echo done > logs/final_128.done
# ---------------------------------------------------------------- ensembles at 128^2 (nested + the four original sets)
for EQ in Poisson ConvDiff AnisoDiff; do
  for W in jacobi,jacobi_0.67 jacobi,jacobi_0.67,gs; do
    [ -f results/${EQ}_128_ens_${W//,/+}.json ] && continue
    run $PY bench.py --equation $EQ --N 128 --solvers $W --ensemble --retrain_router --router_inst 256 --router_max_epochs 600 --router_err_stop 1e-9 --dagger_rounds 6 --router_hidden 128 --router_epochs 300 --n_test 64 --max_ops 4000 --policies greedy,oracle,router --with_pairwise ${=SEED} > logs/final_nest_${EQ}_128_${W//,/+}.log 2>&1
  done
  for W in jacobi,gs jacobi,gs,ssor jacobi,gs,ssor,jacobi_0.67 jacobi,gs,ssor,jacobi_0.67,sor_1.5; do
    [ -f results/${EQ}_128_ens_${W//,/+}.json ] && continue
    run $PY bench.py --equation $EQ --N 128 --solvers $W --ensemble --retrain_router --dagger_rounds 4 --router_inst 256 --n_test 64 --policies greedy,oracle,router --with_pairwise ${=SEED} > logs/final_ens_${EQ}_128_${W//,/+}.log 2>&1
  done
  stage "128 $EQ ensembles done"
done
echo done > logs/final_ens128.done
# ---------------------------------------------------------------- 256^2
for EQ in Poisson ConvDiff AnisoDiff; do
  S=$(solvers $EQ); NT=32; [ "$EQ" = "AnisoDiff" ] && NT=16
  if ! done_cells $EQ $S 256; then
    [ -f checkpoints/costs_${EQ}_256.json ] || run $PY bench.py --equation $EQ --N 256 --solvers $S --measure_only --remeasure_costs > logs/final_costs_${EQ}_256.log 2>&1
    [ -f checkpoints/router_${EQ}_256_sor_1.5.pth ] && [ checkpoints/router_${EQ}_256_sor_1.5.pth -nt libstencil.so ] || run $PY bench.py --equation $EQ --N 256 --solvers $S --train_only --retrain_router --router_inst 64 --router_max_epochs 1500 > logs/final_routers_${EQ}_256.log 2>&1
    TODO=$(for s in $(echo $S | tr , ' '); do [ -f results/${EQ}_256_${s}.json ] || printf "%s," $s; done); TODO=${TODO%,}
    run $PY bench.py --equation $EQ --N 256 --solvers $TODO --n_test $NT --max_ops 40000 --policies $POL ${=SEED} > logs/final_bench_${EQ}_256.log 2>&1
  fi
  for W in jacobi,jacobi_0.67 jacobi,jacobi_0.67,gs; do
    [ -f results/${EQ}_256_ens_${W//,/+}.json ] && continue
    run $PY bench.py --equation $EQ --N 256 --solvers $W --ensemble --retrain_router --router_inst 128 --router_max_epochs 600 --router_err_stop 1e-9 --dagger_rounds 6 --router_hidden 128 --router_epochs 300 --n_test $NT --max_ops 3000 --policies greedy,oracle,router --with_pairwise ${=SEED} > logs/final_nest_${EQ}_256_${W//,/+}.log 2>&1
  done
  stage "256 $EQ done"
done
echo done > logs/final_256.done
# ---------------------------------------------------------------- 512^2 (isotropic equations)
for EQ in Poisson ConvDiff; do
  S=$(solvers $EQ)
  done_cells $EQ $S 512 && continue
  [ -f checkpoints/costs_${EQ}_512.json ] || run $PY bench.py --equation $EQ --N 512 --solvers $S --measure_only --remeasure_costs > logs/final_costs_${EQ}_512.log 2>&1
  [ -f checkpoints/router_${EQ}_512_sor_1.5.pth ] && [ checkpoints/router_${EQ}_512_sor_1.5.pth -nt libstencil.so ] || run $PY bench.py --equation $EQ --N 512 --solvers $S --train_only --retrain_router --router_inst 32 --router_max_epochs 400 --router_err_stop 1e-8 > logs/final_routers_${EQ}_512.log 2>&1
  if [ "$EQ" = "ConvDiff" ]; then   # frozen per-cell exception of the development phase
    run $PY bench.py --equation $EQ --N 512 --solvers jacobi --train_only --retrain_router --router_inst 64 --router_max_epochs 800 --router_err_stop 1e-8 > logs/final_routers_${EQ}_512_jacobi.log 2>&1
  fi
  TODO=$(for s in $(echo $S | tr , ' '); do [ -f results/${EQ}_512_${s}.json ] || printf "%s," $s; done); TODO=${TODO%,}
  run $PY bench.py --equation $EQ --N 512 --solvers $TODO --n_test 16 --max_ops 8000 --max_iter_baseline 3000 --baselines fft,mg,$([ "$EQ" = "Poisson" ] && echo cg,pcg_ssor,pcg_mg || echo bicgstab,bicgstab_mg) --policies $POL ${=SEED} > logs/final_bench_${EQ}_512.log 2>&1
  stage "512 $EQ done"
done
echo done > logs/final_512.done
# ---------------------------------------------------------------- auxiliary: usage traces, seeds, overheads, assumptions, screening, theorem check, discretisation
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 MKL_NUM_THREADS=2
for EQ in Poisson ConvDiff AnisoDiff; do
  S=$(solvers $EQ)
  run $PY make_usage_data.py --equation $EQ --N 128 --solvers $S --n_test 64 --seed 73 > logs/final_usage_${EQ}_128.log 2>&1
  NT=32; [ "$EQ" = "AnisoDiff" ] && NT=16
  run $PY make_usage_data.py --equation $EQ --N 256 --solvers $S --n_test $NT --seed 73 > logs/final_usage_${EQ}_256.log 2>&1
  [ "$EQ" = "AnisoDiff" ] || run $PY make_usage_data.py --equation $EQ --N 512 --solvers $S --n_test 16 --seed 73 > logs/final_usage_${EQ}_512.log 2>&1
done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
for EQ in Poisson ConvDiff AnisoDiff; do
  S=$(solvers $EQ)
  run $PY bench_seeds.py --equation $EQ --N 128 --solvers $S --seed 73 > logs/final_seeds_${EQ}_128.log 2>&1
  NT=32; [ "$EQ" = "AnisoDiff" ] && NT=16
  run $PY bench_seeds.py --equation $EQ --N 256 --solvers jacobi --n_test $NT --router_inst 64 --router_max_epochs 1500 --max_ops 40000 --seed 73 > logs/final_seeds_${EQ}_256.log 2>&1
done
run $PY bench_overheads.py > logs/final_overheads.log 2>&1
for N in 128 256 512; do for EQ in Poisson ConvDiff AnisoDiff; do
  [ -f checkpoints/costs_${EQ}_${N}.json ] || continue
  M=$(solvers $EQ)
  # the assumption checker needs the transposes of the triangular sweeps, which the numpy/scipy
  # fallback provides (splu); the operators are identical and the macro sizes come from the cost cache
  STENCIL_NUMPY=1 run $PY check_assumptions.py --equation $EQ --N $N --seed 73 > logs/final_assumptions_${EQ}_${N}.log 2>&1
  STENCIL_NUMPY=1 run $PY check_assumptions.py --equation $EQ --N $N --seed 73 --paths_only --path_tol $($PY -c "print(1.0/$N**2)") > logs/final_paths_h2_${EQ}_${N}.log 2>&1
  run $PY screen_ensembles.py --equation $EQ --N $N --members $M --n_inst 16 --seed 73 --max_ops $([ $N = 128 ] && echo 2000 || echo 1000) > logs/final_screen_${EQ}_${N}.log 2>&1
done; done
for EQ in Poisson ConvDiff AnisoDiff; do run $PY check_theorem.py --equation $EQ --N 128 --n_inst 8 --seed 73 > logs/final_theorem_${EQ}_128.log 2>&1; done
run $PY discretization_error.py 73 > logs/final_discretization.log 2>&1
echo done > logs/final.done
stage "ALL DONE (main chain)"
