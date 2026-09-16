#!/bin/zsh
# Decision-granularity ablation at 128^2: unit of cost = 1/4 of a corrector call (finer macro-actions; the
# corrector is then an action dearer than the unit and is scored per unit of cost). Pairwise cells and the
# nested two-member ensemble, routers retrained with the same recipe, same confirmatory instances.
# Every step whose output already exists is skipped, so the script resumes after an interruption.
# GRAN_ONLY="Poisson:gs,ssor ConvDiff:jacobi_0.67" restricts a pass to those pairings (no ensemble step); run_all.sh uses it
# to run first the pairings in which the router lost to the best fixed schedule at 1e-8 with the unit of one corrector call.
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-/Users/yash/miniconda3/envs/ansatz/bin/python}
POL=classical,hints15,hints25,phints5,phints10,phints15,oneshot,decay0.9,decay0.95,decay0.98,greedy,oracle,router
stage() { echo "$(date '+%F %T') $1" >> logs/final_progress.log; }
run() { "$@" || { rc=$?; echo "$(date '+%F %T') FAILED (exit $rc): $*" >> logs/final_progress.log; echo FAILED > logs/granularity.failed; exit 1; }; }
for EQ in Poisson ConvDiff AnisoDiff VarCoeff; do
  S=$(cat config/solvers_$EQ); new=0
  if [ -n "${GRAN_ONLY:-}" ]; then
    S=""; for item in ${=GRAN_ONLY}; do [ "${item%%:*}" = "$EQ" ] && S=${item#*:}; done
    [ -n "$S" ] || continue
  fi
  # pairings whose result file exists are skipped; routers are trained unless the router of the last pairing
  # (trained last) exists and is newer than the compiled kernels
  TODO=$(for s in $(echo $S | tr , ' '); do [ -f results/${EQ}_128_${s}_u4.json ] || printf "%s," $s; done); TODO=${TODO%,}
  if [ -n "$TODO" ]; then
    # routers of the pairings to run, trained unless they exist and are newer than the compiled kernels
    RT=$(for s in $(echo $TODO | tr , ' '); do { [ -f checkpoints/router_${EQ}_128_${s}_u4.pth ] && [ checkpoints/router_${EQ}_128_${s}_u4.pth -nt libstencil.so ]; } || printf "%s," $s; done); RT=${RT%,}
    [ -z "$RT" ] || run $PY bench.py --equation $EQ --N 128 --solvers $RT --train_only --retrain_router --unit_frac 0.25 --router_tag _u4 > logs/final_routers_${EQ}_128_u4${GRAN_ONLY:+_priority}.log 2>&1
    run $PY bench.py --equation $EQ --N 128 --solvers $TODO --n_test 64 --max_ops 60000 --unit_frac 0.25 --router_tag _u4 --tag _u4 --baselines none --policies $POL --seed 73 --timed_reps 3 > logs/final_bench_${EQ}_128_u4${GRAN_ONLY:+_priority}.log 2>&1
    new=1
  fi
  if [ -z "${GRAN_ONLY:-}" ] && [ ! -f results/${EQ}_128_ens_jacobi+jacobi_0.67_u4.json ]; then
    run $PY bench.py --equation $EQ --N 128 --solvers jacobi,jacobi_0.67 --ensemble --retrain_router --unit_frac 0.25 --router_tag _u4 --tag _u4 --router_inst 256 --router_max_epochs 600 --router_err_stop 1e-9 --dagger_rounds 6 --router_hidden 128 --router_epochs 300 --n_test 64 --max_ops 8000 --policies greedy,oracle,router --with_pairwise --seed 73 --timed_reps 3 > logs/final_nest_${EQ}_128_u4.log 2>&1
    new=1
  fi
  if [ $new = 1 ]; then stage "granularity $EQ ${GRAN_ONLY:+(priority pairings) }done"; fi
done
# (the re-timing pass of the 128^2 cells written by an earlier driver revision is run_retime128.sh)
if [ -z "${GRAN_ONLY:-}" ]; then echo done > logs/granularity.done; stage "ALL DONE (granularity chain)"; fi
exit 0
