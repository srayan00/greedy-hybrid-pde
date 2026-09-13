#!/bin/zsh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-/Users/yash/miniconda3/envs/ansatz/bin/python}
for N in 128 256 512; do
  for EQ in Poisson ConvDiff AnisoDiff; do
    [ -f checkpoints/costs_${EQ}_${N}.json ] || continue
    $PY check_assumptions.py --equation $EQ --N $N > logs/assumptions_${EQ}_${N}.log 2>&1
  done
done
echo done > logs/assumptions.done
