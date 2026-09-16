#!/bin/zsh
# Variable-coefficient diffusion (no Fourier diagonalisation): same confirmatory protocol as run_final.sh.
# Accepts ONLY_N=<N> and SKIP_AUX=1 like run_final.sh and skips every step whose output already exists.
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-/Users/yash/miniconda3/envs/ansatz/bin/python}
POL=classical,hints2,hints5,hints10,hints15,hints25,hints50,phints5,phints10,phints15,phints25,phints50,oneshot,decay0.9,decay0.95,decay0.98,greedy,oracle,router
SEED="--seed 73 --timed_reps 3"
stage() { echo "$(date '+%F %T') $1" >> logs/final_progress.log; }
run() { "$@" || { rc=$?; echo "$(date '+%F %T') FAILED (exit $rc): $*" >> logs/final_progress.log; echo FAILED > logs/varcoeff.failed; exit 1; }; }
EQ=VarCoeff
S=$(cat config/solvers_$EQ)
want() { [ -z "${ONLY_N:-}" ] || [ "$ONLY_N" = "$1" ]; }    # grid filter
fin() { tail -n 3 "$1" 2>/dev/null | grep -q "^saved"; }       # an auxiliary step is complete once its log ends with the saved-output line
done_cells() { for s in $(echo $2 | tr , ' '); do [ -f results/${1}_${3}_${s}.json ] || return 1; done; return 0; }
rm -f logs/varcoeff.failed
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
for N in 128 256; do
  want $N || continue
  [ -f checkpoints/deeponet_${EQ}_${N}_best.pth ] || run $PY corrector.py --equation $EQ --N $N --coarsen $((N / 64)) --n_train 32000 --n_val 1000 > logs/deeponet_${EQ}_${N}.log 2>&1
done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for N in 128 256; do
  want $N || continue
  NT=64; RI=""; [ $N = 256 ] && NT=32 && RI="--router_inst 64 --router_max_epochs 1500"
  MO=60000; [ $N = 256 ] && MO=40000
  new=0
  if ! done_cells $EQ $S $N; then
  [ -f checkpoints/costs_${EQ}_${N}.json ] || run $PY bench.py --equation $EQ --N $N --solvers $S --measure_only --remeasure_costs > logs/final_costs_${EQ}_${N}.log 2>&1
  # routers are trained unless the router of the last pairing (trained last) exists and is newer than the compiled kernels
  [ -f checkpoints/router_${EQ}_${N}_${S##*,}.pth ] && [ checkpoints/router_${EQ}_${N}_${S##*,}.pth -nt libstencil.so ] || run $PY bench.py --equation $EQ --N $N --solvers $S --train_only --retrain_router ${=RI} > logs/final_routers_${EQ}_${N}.log 2>&1
  TODO=$(for s in $(echo $S | tr , ' '); do [ -f results/${EQ}_${N}_${s}.json ] || printf "%s," $s; done); TODO=${TODO%,}
  run $PY bench.py --equation $EQ --N $N --solvers $TODO --n_test $NT --max_ops $MO --policies $POL ${=SEED} > logs/final_bench_${EQ}_${N}.log 2>&1
  new=1
  fi
  for W in jacobi,jacobi_0.67 jacobi,jacobi_0.67,gs; do
    [ -f results/${EQ}_${N}_ens_${W//,/+}.json ] && continue
    RI2="--router_inst 256"; [ $N = 256 ] && RI2="--router_inst 128"
    run $PY bench.py --equation $EQ --N $N --solvers $W --ensemble --retrain_router ${=RI2} --router_max_epochs 600 --router_err_stop 1e-9 --dagger_rounds 6 --router_hidden 128 --router_epochs 300 --n_test $NT --max_ops $(( N == 128 ? 4000 : 3000 )) --policies greedy,oracle,router --with_pairwise ${=SEED} > logs/final_nest_${EQ}_${N}_${W//,/+}.log 2>&1
    new=1
  done
  if [ $new = 1 ]; then stage "$N $EQ done"; fi
done
if [ -z "${SKIP_AUX:-}" ]; then
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
if want 128 && ! fin logs/final_usage_${EQ}_128.log; then run $PY make_usage_data.py --equation $EQ --N 128 --solvers $S --n_test 64 --seed 73 > logs/final_usage_${EQ}_128.log 2>&1; fi
if want 256 && ! fin logs/final_usage_${EQ}_256.log; then run $PY make_usage_data.py --equation $EQ --N 256 --solvers $S --n_test 32 --seed 73 > logs/final_usage_${EQ}_256.log 2>&1; fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
if want 128 && ! fin logs/final_seeds_${EQ}_128.log; then run $PY bench_seeds.py --equation $EQ --N 128 --solvers $S --seed 73 > logs/final_seeds_${EQ}_128.log 2>&1; fi
for N in 128 256; do
  want $N || continue
  if ! fin logs/final_assumptions_${EQ}_${N}.log; then
    STENCIL_NUMPY=1 run $PY check_assumptions.py --equation $EQ --N $N --seed 73 > logs/final_assumptions_${EQ}_${N}.log 2>&1
    rm -f logs/final_paths_h2_${EQ}_${N}.log     # a fresh assumptions file needs the path check again
  fi
  fin logs/final_paths_h2_${EQ}_${N}.log || STENCIL_NUMPY=1 run $PY check_assumptions.py --equation $EQ --N $N --seed 73 --paths_only --path_tol $($PY -c "print(1.0/$N**2)") > logs/final_paths_h2_${EQ}_${N}.log 2>&1
  fin logs/final_screen_${EQ}_${N}.log || run $PY screen_ensembles.py --equation $EQ --N $N --members $S --n_inst 16 --seed 73 --max_ops $([ $N = 128 ] && echo 2000 || echo 1000) > logs/final_screen_${EQ}_${N}.log 2>&1
done
if want 128 && ! fin logs/final_theorem_${EQ}_128.log; then run $PY check_theorem.py --equation $EQ --N 128 --n_inst 8 --seed 73 > logs/final_theorem_${EQ}_128.log 2>&1; fi
if [ -z "${ONLY_N:-}" ]; then run $PY discretization_error.py 73 > logs/final_discretization.log 2>&1; fi
fi   # SKIP_AUX
if [ -z "${ONLY_N:-}" ] && [ -z "${SKIP_AUX:-}" ]; then
  echo done > logs/varcoeff.done
  stage "ALL DONE (variable-coefficient chain)"
else
  stage "run_varcoeff.sh pass done (ONLY_N=${ONLY_N:-all}, SKIP_AUX=${SKIP_AUX:-0})"
fi
exit 0
