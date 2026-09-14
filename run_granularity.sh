#!/bin/zsh
# Decision-granularity ablation at 128^2: unit of cost = 1/4 of a corrector call (finer macro-actions; the
# corrector is then an action dearer than the unit and is scored per unit of cost). Pairwise cells and the
# nested two-member ensemble, routers retrained with the same recipe, same confirmatory instances.
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-/Users/yash/miniconda3/envs/ansatz/bin/python}
POL=classical,hints15,hints25,phints5,phints10,phints15,oneshot,decay0.9,decay0.95,decay0.98,greedy,oracle,router
stage() { echo "$(date '+%F %T') $1" >> logs/final_progress.log; }
run() { "$@" || { echo "$(date '+%F %T') FAILED (exit $?): $*" >> logs/final_progress.log; echo FAILED > logs/granularity.failed; exit 1; }; }
for EQ in Poisson ConvDiff AnisoDiff VarCoeff; do
  S=$(cat config/solvers_$EQ)
  run $PY bench.py --equation $EQ --N 128 --solvers $S --train_only --retrain_router --unit_frac 0.25 --router_tag _u4 > logs/final_routers_${EQ}_128_u4.log 2>&1
  run $PY bench.py --equation $EQ --N 128 --solvers $S --n_test 64 --max_ops 60000 --unit_frac 0.25 --router_tag _u4 --tag _u4 --baselines none --policies $POL --seed 73 --timed_reps 3 > logs/final_bench_${EQ}_128_u4.log 2>&1
  run $PY bench.py --equation $EQ --N 128 --solvers jacobi,jacobi_0.67 --ensemble --retrain_router --unit_frac 0.25 --router_tag _u4 --tag _u4 --router_inst 256 --router_max_epochs 600 --router_err_stop 1e-9 --dagger_rounds 6 --router_hidden 128 --router_epochs 300 --n_test 64 --max_ops 8000 --policies greedy,oracle,router --with_pairwise --seed 73 --timed_reps 3 > logs/final_nest_${EQ}_128_u4.log 2>&1
  stage "granularity $EQ done"
done
# re-timing pass for the 128^2 Poisson and ConvDiff pairwise cells, which were produced before the
# in-run re-timing of drift-flagged instances existed (same routers and cost caches; only the flagged
# instances are timed again, with the drift guard waiting up to 15 min)
POLF=classical,hints2,hints5,hints10,hints15,hints25,hints50,phints5,phints10,phints15,phints25,phints50,oneshot,decay0.9,decay0.95,decay0.98,greedy,oracle,router
for EQ in Poisson ConvDiff; do for S in $(cat config/solvers_$EQ | tr , ' '); do
  run $PY bench.py --equation $EQ --N 128 --solvers $S --n_test 64 --max_ops 60000 --policies $POLF --seed 73 --timed_reps 3 --retime_only --retime_all > logs/final_retime_${EQ}_128_$S.log 2>&1
done; done
stage "re-timing pass done"
echo done > logs/granularity.done
stage "ALL DONE (granularity chain)"
