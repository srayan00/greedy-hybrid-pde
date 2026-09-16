#!/bin/zsh
# Confirmatory re-run of the wall-clock study (revision 2, after the third review):
#   compiled stencil kernels; corrected (Hermitian) GRF sampler; live per-operation costs (update +
#   residual + norm, median of blocks); Alg. 1 exponents (1) for every action not dearer than the unit;
#   HINTS tau = 15 (the 2-D period of the HINTS paper) in the schedule family; residual-decay controls;
#   classical baselines timed inside the same replay loop as the policies; drift guard per instance;
#   fresh test seed 73 (all configuration choices were made on seed 72, archived in results_dev/);
#   three timed replays in random order with warm-up. Fail-fast: any failing command stops the chain.
# Phases: ONLY_N=<N> restricts every stage to that grid, ONLY_EQ=<equation> to that equation (the grid-spanning overhead and discretisation studies
#   then do not run) and SKIP_AUX=1 skips the auxiliary stage; run_all.sh uses both to complete every 128^2
#   result before the larger grids. Every step whose output already exists is skipped (the result files of the
#   benchmark cells, the closing "saved" line of the log for the auxiliary scripts), so the script resumes after
#   an interruption and a later call without filters runs only what is missing.
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-/Users/yash/miniconda3/envs/ansatz/bin/python}
POL=classical,hints2,hints5,hints10,hints15,hints25,hints50,phints5,phints10,phints15,phints25,phints50,oneshot,decay0.9,decay0.95,decay0.98,greedy,oracle,router
SEED="--seed 73 --timed_reps 3"
stage() { echo "$(date '+%F %T') $1" >> logs/final_progress.log; }
run() { "$@" || { rc=$?; echo "$(date '+%F %T') FAILED (exit $rc): $*" >> logs/final_progress.log; echo FAILED > logs/final.failed; exit 1; }; }
solvers() { cat config/solvers_$1; }
rm -f logs/final.failed
want() { [ -z "${ONLY_N:-}" ] || [ "$ONLY_N" = "$1" ]; }    # grid filter
wanteq() { [ -z "${ONLY_EQ:-}" ] || [ "$ONLY_EQ" = "$1" ]; }  # equation filter (ONLY_EQ=<equation>)
fin() { tail -n 3 "$1" 2>/dev/null | grep -q "^saved"; }       # an auxiliary step is complete once its log ends with the saved-output line
# ---------------------------------------------------------------- 128^2 (pairwise + same-session baselines)
done_cells() { for s in $(echo $2 | tr , ' '); do [ -f results/${1}_${3}_${s}.json ] || return 1; done; return 0; }
if want 128; then
for EQ in Poisson ConvDiff AnisoDiff; do
  wanteq $EQ || continue
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
  wanteq $EQ || continue
  new=0
  for W in jacobi,jacobi_0.67 jacobi,jacobi_0.67,gs; do
    [ -f results/${EQ}_128_ens_${W//,/+}.json ] && continue
    run $PY bench.py --equation $EQ --N 128 --solvers $W --ensemble --retrain_router --router_inst 256 --router_max_epochs 600 --router_err_stop 1e-9 --dagger_rounds 6 --router_hidden 128 --router_epochs 300 --n_test 64 --max_ops 4000 --policies greedy,oracle,router --with_pairwise ${=SEED} > logs/final_nest_${EQ}_128_${W//,/+}.log 2>&1
    new=1
  done
  for W in jacobi,gs jacobi,gs,ssor jacobi,gs,ssor,jacobi_0.67 jacobi,gs,ssor,jacobi_0.67,sor_1.5; do
    [ -f results/${EQ}_128_ens_${W//,/+}.json ] && continue
    run $PY bench.py --equation $EQ --N 128 --solvers $W --ensemble --retrain_router --dagger_rounds 4 --router_inst 256 --n_test 64 --policies greedy,oracle,router --with_pairwise ${=SEED} > logs/final_ens_${EQ}_128_${W//,/+}.log 2>&1
    new=1
  done
  if [ $new = 1 ]; then stage "128 $EQ ensembles done"; fi
done
echo done > logs/final_ens128.done
fi
# ---------------------------------------------------------------- 256^2
if want 256; then
for EQ in Poisson ConvDiff AnisoDiff; do
  wanteq $EQ || continue
  S=$(solvers $EQ); NT=32; [ "$EQ" = "AnisoDiff" ] && NT=16; new=0
  if ! done_cells $EQ $S 256; then
    [ -f checkpoints/costs_${EQ}_256.json ] || run $PY bench.py --equation $EQ --N 256 --solvers $S --measure_only --remeasure_costs > logs/final_costs_${EQ}_256.log 2>&1
    [ -f checkpoints/router_${EQ}_256_sor_1.5.pth ] && [ checkpoints/router_${EQ}_256_sor_1.5.pth -nt libstencil.so ] || run $PY bench.py --equation $EQ --N 256 --solvers $S --train_only --retrain_router --router_inst 64 --router_max_epochs 1500 > logs/final_routers_${EQ}_256.log 2>&1
    TODO=$(for s in $(echo $S | tr , ' '); do [ -f results/${EQ}_256_${s}.json ] || printf "%s," $s; done); TODO=${TODO%,}
    run $PY bench.py --equation $EQ --N 256 --solvers $TODO --n_test $NT --max_ops 40000 --policies $POL ${=SEED} > logs/final_bench_${EQ}_256.log 2>&1
    new=1
  fi
  for W in jacobi,jacobi_0.67 jacobi,jacobi_0.67,gs; do
    [ -f results/${EQ}_256_ens_${W//,/+}.json ] && continue
    run $PY bench.py --equation $EQ --N 256 --solvers $W --ensemble --retrain_router --router_inst 128 --router_max_epochs 600 --router_err_stop 1e-9 --dagger_rounds 6 --router_hidden 128 --router_epochs 300 --n_test $NT --max_ops 3000 --policies greedy,oracle,router --with_pairwise ${=SEED} > logs/final_nest_${EQ}_256_${W//,/+}.log 2>&1
    new=1
  done
  if [ $new = 1 ]; then stage "256 $EQ done"; fi
done
echo done > logs/final_256.done
fi
# ---------------------------------------------------------------- 512^2 (isotropic equations)
if want 512; then
for EQ in Poisson ConvDiff; do
  wanteq $EQ || continue
  S=$(solvers $EQ)
  done_cells $EQ $S 512 && continue
  [ -f checkpoints/costs_${EQ}_512.json ] || run $PY bench.py --equation $EQ --N 512 --solvers $S --measure_only --remeasure_costs > logs/final_costs_${EQ}_512.log 2>&1
  # 512^2 routers: the training rollouts must cover the test horizon (max_ops 8000 sweeps), i.e. 8000 / m decisions
  # (m = 6 Jacobi sweeps, 2 GS/SOR, 1 SymGS/multigrid per corrector call); a first training with 400-decision rollouts
  # produced routers that kept calling the corrector beyond the horizon they had seen (archived in results_superseded/)
  RJ=$(for s in jacobi jacobi_0.67; do { [ -f checkpoints/router_${EQ}_512_${s}.pth ] && [ checkpoints/router_${EQ}_512_${s}.pth -nt libstencil.so ]; } || printf "%s," $s; done); RJ=${RJ%,}
  [ -z "$RJ" ] || run $PY bench.py --equation $EQ --N 512 --solvers $RJ --train_only --retrain_router --router_inst 32 --router_max_epochs 1400 --router_err_stop 1e-8 > logs/final_routers_${EQ}_512_jacobi.log 2>&1
  RO=$(for s in $(echo $S | tr , ' '); do case $s in jacobi|jacobi_0.67) continue;; esac; { [ -f checkpoints/router_${EQ}_512_${s}.pth ] && [ checkpoints/router_${EQ}_512_${s}.pth -nt libstencil.so ]; } || printf "%s," $s; done); RO=${RO%,}
  [ -z "$RO" ] || run $PY bench.py --equation $EQ --N 512 --solvers $RO --train_only --retrain_router --router_inst 32 --router_max_epochs 4000 --router_err_stop 1e-8 > logs/final_routers_${EQ}_512.log 2>&1
  TODO=$(for s in $(echo $S | tr , ' '); do [ -f results/${EQ}_512_${s}.json ] || printf "%s," $s; done); TODO=${TODO%,}
  run $PY bench.py --equation $EQ --N 512 --solvers $TODO --n_test 16 --max_ops 8000 --max_iter_baseline 3000 --baselines fft,mg,$([ "$EQ" = "Poisson" ] && echo cg,pcg_ssor,pcg_mg || echo bicgstab,bicgstab_mg) --policies $POL ${=SEED} > logs/final_bench_${EQ}_512.log 2>&1
  stage "512 $EQ done"
done
echo done > logs/final_512.done
fi
# ---------------------------------------------------------------- auxiliary: usage traces, seeds, overheads, assumptions, screening, theorem check, discretisation
if [ -z "${SKIP_AUX:-}" ]; then
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 MKL_NUM_THREADS=2
for EQ in Poisson ConvDiff AnisoDiff; do
  wanteq $EQ || continue
  S=$(solvers $EQ)
  if want 128 && ! fin logs/final_usage_${EQ}_128.log; then run $PY make_usage_data.py --equation $EQ --N 128 --solvers $S --n_test 64 --seed 73 > logs/final_usage_${EQ}_128.log 2>&1; fi
  NT=32; [ "$EQ" = "AnisoDiff" ] && NT=16
  if want 256 && ! fin logs/final_usage_${EQ}_256.log; then run $PY make_usage_data.py --equation $EQ --N 256 --solvers $S --n_test $NT --seed 73 > logs/final_usage_${EQ}_256.log 2>&1; fi
  if [ "$EQ" != "AnisoDiff" ] && want 512 && ! fin logs/final_usage_${EQ}_512.log; then run $PY make_usage_data.py --equation $EQ --N 512 --solvers $S --n_test 16 --seed 73 > logs/final_usage_${EQ}_512.log 2>&1; fi
done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
for EQ in Poisson ConvDiff AnisoDiff; do
  wanteq $EQ || continue
  S=$(solvers $EQ)
  if want 128 && ! fin logs/final_seeds_${EQ}_128.log; then run $PY bench_seeds.py --equation $EQ --N 128 --solvers $S --seed 73 > logs/final_seeds_${EQ}_128.log 2>&1; fi
  NT=32; [ "$EQ" = "AnisoDiff" ] && NT=16
  if want 256 && ! fin logs/final_seeds_${EQ}_256.log; then run $PY bench_seeds.py --equation $EQ --N 256 --solvers jacobi --n_test $NT --router_inst 64 --router_max_epochs 1500 --max_ops 40000 --seed 73 > logs/final_seeds_${EQ}_256.log 2>&1; fi
done
if [ -z "${ONLY_N:-}" ] && ! fin logs/final_overheads.log; then run $PY bench_overheads.py > logs/final_overheads.log 2>&1; fi
for N in 128 256 512; do want $N || continue; for EQ in Poisson ConvDiff AnisoDiff; do
  wanteq $EQ || continue
  [ -f checkpoints/costs_${EQ}_${N}.json ] || continue
  M=$(solvers $EQ)
  # the assumption checker needs the transposes of the triangular sweeps, which the numpy/scipy
  # fallback provides (splu); the operators are identical and the macro sizes come from the cost cache
  if ! fin logs/final_assumptions_${EQ}_${N}.log; then
    STENCIL_NUMPY=1 run $PY check_assumptions.py --equation $EQ --N $N --seed 73 > logs/final_assumptions_${EQ}_${N}.log 2>&1
    rm -f logs/final_paths_h2_${EQ}_${N}.log     # a fresh assumptions file needs the path check again
  fi
  fin logs/final_paths_h2_${EQ}_${N}.log || STENCIL_NUMPY=1 run $PY check_assumptions.py --equation $EQ --N $N --seed 73 --paths_only --path_tol $($PY -c "print(1.0/$N**2)") > logs/final_paths_h2_${EQ}_${N}.log 2>&1
  fin logs/final_screen_${EQ}_${N}.log || run $PY screen_ensembles.py --equation $EQ --N $N --members $M --n_inst 16 --seed 73 --max_ops $([ $N = 128 ] && echo 2000 || echo 1000) > logs/final_screen_${EQ}_${N}.log 2>&1
done; done
for EQ in Poisson ConvDiff AnisoDiff; do if want 128 && ! fin logs/final_theorem_${EQ}_128.log; then run $PY check_theorem.py --equation $EQ --N 128 --n_inst 8 --seed 73 > logs/final_theorem_${EQ}_128.log 2>&1; fi; done
if [ -z "${ONLY_N:-}" ]; then run $PY discretization_error.py 73 > logs/final_discretization.log 2>&1; fi
fi   # SKIP_AUX
if [ -z "${ONLY_N:-}" ] && [ -z "${SKIP_AUX:-}" ]; then
  echo done > logs/final.done
  stage "ALL DONE (main chain)"
else
  stage "run_final.sh pass done (ONLY_N=${ONLY_N:-all}, SKIP_AUX=${SKIP_AUX:-0})"
fi
exit 0
