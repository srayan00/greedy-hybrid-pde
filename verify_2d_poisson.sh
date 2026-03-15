#!/bin/bash
#SBATCH --job-name=verify_2d_poisson
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/verify_2d_poisson_%j.out
#SBATCH --error=logs/verify_2d_poisson_%j.err

mkdir -p logs

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate greedy

cd /mnt/sharefs/user35/greedy-hybrid-pde

python verify_pipeline.py \
    --model_name hier_v2 \
    --n_test 64 \
    --max_iters 300 \
    --tau 24 \
    --dim 2 \
    --equation Poisson \
    --grf_mode hierarchical
