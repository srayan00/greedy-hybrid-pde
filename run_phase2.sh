#!/bin/zsh
# usage data (untimed, both PDEs) -> ensemble study (timed, both PDEs)
PY=${PY:-/Users/yash/miniconda3/envs/ansatz/bin/python}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
for EQ in Poisson ConvDiff; do
  $PY make_usage_data.py --equation $EQ --N 128 > logs/usage_${EQ}_128.log 2>&1
done
for EQ in Poisson ConvDiff; do
  ./run_ensembles.sh $EQ 128 64
done
echo PHASE2_DONE > logs/phase2.done
