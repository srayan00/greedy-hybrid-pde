#!/bin/zsh
# Selection of the best fixed schedule on the development instances (seed 72) for every equation and grid whose
# corrector and cost cache exist (select_schedule.py): untimed, deterministic, resumable (a log ending in "saved").
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-/Users/yash/miniconda3/envs/ansatz/bin/python}
stage() { echo "$(date '+%F %T') $1" >> logs/final_progress.log; }
run() { "$@" || { rc=$?; echo "$(date '+%F %T') FAILED (exit $rc): $*" >> logs/final_progress.log; echo FAILED > logs/devsched.failed; exit 1; }; }
fin() { tail -n 3 "$1" 2>/dev/null | grep -q "^saved"; }
new=0
for N in 128 256 512; do for EQ in Poisson ConvDiff AnisoDiff VarCoeff; do
  [ -f checkpoints/costs_${EQ}_${N}.json ] && [ -f checkpoints/deeponet_${EQ}_${N}_best.pth ] || continue
  fin logs/final_devsched_${EQ}_${N}.log && continue
  NT=64; [ $N = 256 ] && NT=32; [ $N = 512 ] && NT=16; [ "$EQ" = "AnisoDiff" ] && [ $N = 256 ] && NT=16
  MO=60000; [ $N = 256 ] && MO=40000; [ $N = 512 ] && MO=8000
  run $PY select_schedule.py --equation $EQ --N $N --solvers $(cat config/solvers_$EQ) --n_test $NT --seed 72 --max_ops $MO > logs/final_devsched_${EQ}_${N}.log 2>&1
  new=1
done; done
if [ $new = 1 ]; then stage "development-set schedule selection done (grids with a cost cache)"; fi
exit 0
