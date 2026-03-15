#!/bin/bash
#SBATCH --job-name=verify_poi_multi
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/verify_poi_multi_%j.out
#SBATCH --error=logs/verify_poi_multi_%j.err

mkdir -p logs

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate greedy

cd /mnt/sharefs/user35/greedy-hybrid-pde

echo "=== Multi-solver oracle (no router) ==="
python verify_pipeline.py \
    --model_name hier_v2 \
    --n_test 128 \
    --dim 2 \
    --equation Poisson \
    --grf_mode hierarchical \
    --solver_type jacobi_0.67,jacobi,gs \
    --max_iters 300

echo ""
echo "=== Multi-solver with LSTM Router ==="
python verify_pipeline.py \
    --model_name hier_v2 \
    --n_test 128 \
    --dim 2 \
    --equation Poisson \
    --grf_mode hierarchical \
    --solver_type jacobi_0.67,jacobi,gs \
    --router_model_name router_poi_multi \
    --numerical_solvers jacobi_0.67,jacobi,gs \
    --max_iters 300
