#!/bin/zsh
# Decision-granularity ablation at 128^2: unit of cost = 1/4 of a corrector call (finer macro-actions;
# the corrector is then an action dearer than the unit and is scored per unit of cost).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-/Users/yash/miniconda3/envs/ansatz/bin/python}
until [ -f logs/varcoeff.done ]; do sleep 60; done
for EQ in Poisson ConvDiff AnisoDiff VarCoeff; do
  S=jacobi,jacobi_0.67,gs,ssor,sor_1.5; [ "$EQ" != "AnisoDiff" ] && S=$S,mg
  $PY bench.py --equation $EQ --N 128 --solvers $S --train_only --retrain_router --unit_frac 0.25 --router_tag _u4 > logs/final_routers_${EQ}_128_u4.log 2>&1
  $PY bench.py --equation $EQ --N 128 --solvers $S --n_test 64 --max_ops 60000 --unit_frac 0.25 --router_tag _u4 --tag _u4 --policies classical,hints25,phints5,phints10,oneshot,greedy,oracle,router --seed 73 --timed_reps 3 > logs/final_bench_${EQ}_128_u4.log 2>&1
done
echo done > logs/granularity.done
