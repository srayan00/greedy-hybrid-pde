#!/bin/zsh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
PY=/Users/yash/miniconda3/envs/ansatz/bin/python
until [ -f logs/paths.done ]; do sleep 20; done
for N in 128 256 512; do for EQ in Poisson ConvDiff AnisoDiff; do
  [ -f results/assumptions_${EQ}_${N}.json ] || continue
  H2=$($PY -c "print(1.0/$N**2)")
  $PY check_assumptions.py --equation $EQ --N $N --paths_only --path_tol $H2 > logs/paths_h2_${EQ}_${N}.log 2>&1
done; done
echo done > logs/paths_h2.done
