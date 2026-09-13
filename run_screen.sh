#!/bin/zsh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-/Users/yash/miniconda3/envs/ansatz/bin/python}
for N in 128 256 512; do
  for EQ in Poisson ConvDiff AnisoDiff; do
    [ -f checkpoints/deeponet_${EQ}_${N}_best.pth ] || continue
    if [ "$EQ" = "AnisoDiff" ]; then M=jacobi,jacobi_0.67,gs,ssor,sor_1.5; else M=jacobi,jacobi_0.67,gs,ssor,sor_1.5,mg; fi
    $PY screen_ensembles.py --equation $EQ --N $N --members $M --n_inst 16 --max_ops $([ $N = 128 ] && echo 2000 || echo 1000) > logs/screen_${EQ}_${N}.log 2>&1
  done
done
echo done > logs/screen.done
