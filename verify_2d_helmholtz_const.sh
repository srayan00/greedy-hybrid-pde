#!/bin/bash
#SBATCH --job-name=verify_helm_const
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/verify_helm_const_%j.out
#SBATCH --error=logs/verify_helm_const_%j.err

mkdir -p logs

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate greedy

cd /mnt/sharefs/user35/greedy-hybrid-pde

echo "Running 2D Helmholtz (const k2=10) verification"
python verify_pipeline.py \
    --model_name hier_v5 \
    --n_test 128 \
    --dim 2 \
    --equation Helmholtz \
    --grf_mode hierarchical \
    --numerical_solvers jacobi \
    --max_iters 300 \
    --k2_mode const
