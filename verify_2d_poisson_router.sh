#!/bin/bash
#SBATCH --job-name=verify_2d_router
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/verify_2d_poisson_router_%j.out
#SBATCH --error=logs/verify_2d_poisson_router_%j.err

mkdir -p logs

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate greedy

cd /mnt/sharefs/user35/greedy-hybrid-pde

echo "Running 2D Poisson verification with LSTM Router comparison"
python verify_pipeline.py \
    --model_name hier_v2 \
    --n_test 128 \
    --dim 2 \
    --equation Poisson \
    --grf_mode hierarchical \
    --router_model_name hier_v1 \
    --numerical_solvers jacobi \
    --max_iters 300
