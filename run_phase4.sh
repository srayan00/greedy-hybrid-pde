#!/bin/zsh
# Multi-trial (router training seed) robustness at 128^2, after phase 3.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-/Users/yash/miniconda3/envs/ansatz/bin/python}
until [ -f logs/phase3.done ]; do sleep 30; done
for EQ in Poisson ConvDiff; do
  $PY bench_seeds.py --equation $EQ --N 128 > logs/seeds_${EQ}_128.log 2>&1
done
echo done > logs/phase4.done
