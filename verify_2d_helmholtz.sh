#!/bin/bash
#SBATCH --job-name=verify_2d_helm
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/verify_2d_helmholtz_%j.out
#SBATCH --error=logs/verify_2d_helmholtz_%j.err

mkdir -p logs

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate greedy

cd /mnt/sharefs/user35/greedy-hybrid-pde

echo "Running 2D Helmholtz verification (oracle only, no router yet)"
python verify_pipeline.py \
    --model_name hier_v3 \
    --n_test 128 \
    --dim 2 \
    --equation Helmholtz \
    --grf_mode hierarchical \
    --numerical_solvers jacobi \
    --max_iters 300
