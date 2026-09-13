#!/bin/zsh
export OMP_NUM_THREADS=8
PY=${PY:-/Users/yash/miniconda3/envs/ansatz/bin/python}
for EQ in Poisson ConvDiff; do
  $PY corrector.py --equation $EQ --N 128 --coarsen 2 --n_train 64000 --n_val 2000 > logs/deeponet_${EQ}_128.log 2>&1
done
