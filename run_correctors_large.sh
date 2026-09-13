#!/bin/zsh
export OMP_NUM_THREADS=8
PY=${PY:-/Users/yash/miniconda3/envs/ansatz/bin/python}
for N in 256 512; do
  C=$((N/64))
  for EQ in Poisson ConvDiff; do
    $PY corrector.py --equation $EQ --N $N --coarsen $C --n_train 32000 --n_val 1000 > logs/deeponet_${EQ}_${N}.log 2>&1
  done
done
echo done > logs/correctors_large.done
