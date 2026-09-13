#!/bin/zsh
# Confirmatory runs requested by the review: (1) one-shot schedule baseline on every pairwise cell;
# (2) seed trials at 256^2; (3) held-out evaluation (fresh test seed 73, three timed replays, frozen
# configurations) of all 128^2/256^2 pairwise cells, the retrained 512^2 cell and the nested ensembles.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-/Users/yash/miniconda3/envs/ansatz/bin/python}
until [ -f logs/fill.done ]; do sleep 60; done
sleep 30
# (1) one-shot schedule
$PY bench.py --equation Poisson  --N 128 --solvers jacobi,jacobi_0.67,gs,ssor,sor_1.5,mg --policies oneshot --tag _oneshot --n_test 64 --max_ops 60000 > logs/oneshot_Poisson_128.log 2>&1
$PY bench.py --equation ConvDiff --N 128 --solvers jacobi,jacobi_0.67,gs,ssor,sor_1.5,mg --policies oneshot --tag _oneshot --n_test 64 --max_ops 60000 > logs/oneshot_ConvDiff_128.log 2>&1
$PY bench.py --equation AnisoDiff --N 128 --solvers jacobi,jacobi_0.67,gs,ssor,sor_1.5 --policies oneshot --tag _oneshot --n_test 64 --max_ops 60000 > logs/oneshot_AnisoDiff_128.log 2>&1
$PY bench.py --equation Poisson  --N 256 --solvers jacobi,jacobi_0.67,gs,ssor,sor_1.5,mg --policies oneshot --tag _oneshot --n_test 32 --max_ops 40000 > logs/oneshot_Poisson_256.log 2>&1
$PY bench.py --equation ConvDiff --N 256 --solvers jacobi,jacobi_0.67,gs,ssor,sor_1.5,mg --policies oneshot --tag _oneshot --n_test 32 --max_ops 40000 > logs/oneshot_ConvDiff_256.log 2>&1
$PY bench.py --equation AnisoDiff --N 256 --solvers jacobi,jacobi_0.67,gs,ssor,sor_1.5 --policies oneshot --tag _oneshot --n_test 16 --max_ops 40000 > logs/oneshot_AnisoDiff_256.log 2>&1
$PY bench.py --equation Poisson  --N 512 --solvers jacobi,jacobi_0.67,gs,ssor,sor_1.5,mg --policies oneshot --tag _oneshot --n_test 16 --max_ops 8000 > logs/oneshot_Poisson_512.log 2>&1
$PY bench.py --equation ConvDiff --N 512 --solvers jacobi,jacobi_0.67,gs,ssor,sor_1.5,mg --policies oneshot --tag _oneshot --n_test 16 --max_ops 8000 > logs/oneshot_ConvDiff_512.log 2>&1
echo oneshot > logs/review_stage1.done
# (2) seed trials at 256^2 (Jacobi pairing of every equation)
$PY bench_seeds.py --equation Poisson  --N 256 --solvers jacobi --n_test 32 --router_inst 64 --router_max_epochs 1500 --max_ops 40000 > logs/seeds_Poisson_256.log 2>&1
$PY bench_seeds.py --equation ConvDiff --N 256 --solvers jacobi --n_test 32 --router_inst 64 --router_max_epochs 1500 --max_ops 40000 > logs/seeds_ConvDiff_256.log 2>&1
$PY bench_seeds.py --equation AnisoDiff --N 256 --solvers jacobi --n_test 16 --router_inst 64 --router_max_epochs 1500 --max_ops 40000 > logs/seeds_AnisoDiff_256.log 2>&1
echo seeds > logs/review_stage2.done
# (3) held-out evaluation, frozen configurations, fresh seed, 3 timed replays
H="--seed 73 --tag _seed73 --timed_reps 3"
$PY bench.py --equation Poisson  --N 128 --solvers jacobi,jacobi_0.67,gs,ssor,sor_1.5,mg --n_test 64 --max_ops 60000 ${=H} > logs/heldout_Poisson_128.log 2>&1
$PY bench.py --equation ConvDiff --N 128 --solvers jacobi,jacobi_0.67,gs,ssor,sor_1.5,mg --n_test 64 --max_ops 60000 ${=H} > logs/heldout_ConvDiff_128.log 2>&1
$PY bench.py --equation AnisoDiff --N 128 --solvers jacobi,jacobi_0.67,gs,ssor,sor_1.5 --n_test 64 --max_ops 60000 ${=H} > logs/heldout_AnisoDiff_128.log 2>&1
for EQ in Poisson ConvDiff AnisoDiff; do
  NT=64
  for W in jacobi,jacobi_0.67 jacobi,jacobi_0.67,gs; do
    $PY bench.py --equation $EQ --N 128 --solvers $W --ensemble --n_test $NT --max_ops 4000 --policies greedy,oracle,router --with_pairwise ${=H} > logs/heldout_nest_${EQ}_128_${W//,/+}.log 2>&1
  done
done
$PY bench.py --equation ConvDiff --N 512 --solvers jacobi --n_test 16 --max_ops 8000 ${=H} > logs/heldout_ConvDiff_512.log 2>&1
$PY bench.py --equation Poisson  --N 256 --solvers jacobi,jacobi_0.67,gs,ssor,sor_1.5,mg --n_test 32 --max_ops 40000 ${=H} > logs/heldout_Poisson_256.log 2>&1
$PY bench.py --equation ConvDiff --N 256 --solvers jacobi,jacobi_0.67,gs,ssor,sor_1.5,mg --n_test 32 --max_ops 40000 ${=H} > logs/heldout_ConvDiff_256.log 2>&1
$PY bench.py --equation AnisoDiff --N 256 --solvers jacobi,jacobi_0.67,gs,ssor,sor_1.5 --n_test 16 --max_ops 40000 ${=H} > logs/heldout_AnisoDiff_256.log 2>&1
for EQ in Poisson ConvDiff AnisoDiff; do
  NT=32; [ "$EQ" = "AnisoDiff" ] && NT=16
  for W in jacobi,jacobi_0.67 jacobi,jacobi_0.67,gs; do
    $PY bench.py --equation $EQ --N 256 --solvers $W --ensemble --n_test $NT --max_ops 3000 --policies greedy,oracle,router --with_pairwise ${=H} > logs/heldout_nest_${EQ}_256_${W//,/+}.log 2>&1
  done
done
echo done > logs/review.done
