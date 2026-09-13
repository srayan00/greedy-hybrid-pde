#!/bin/zsh
# Backfill work-unit times into the Poisson/ConvDiff seed trials (untimed; after everything else).
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
PY=${PY:-/Users/yash/miniconda3/envs/ansatz/bin/python}
until [ -f logs/phase7.done ]; do sleep 30; done
for EQ in Poisson ConvDiff; do $PY seeds_wu.py --equation $EQ --N 128 > logs/seeds_wu_${EQ}.log 2>&1; done
echo done > logs/phase8.done
