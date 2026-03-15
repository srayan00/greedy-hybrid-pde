#!/bin/bash
#SBATCH --job-name=verify_pipeline
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/verify_pipeline_%j.out
#SBATCH --error=logs/verify_pipeline_%j.err

mkdir -p logs

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate greedy

cd /mnt/sharefs/user35/greedy-hybrid-pde

echo "=== Run: 1D Poisson, hierarchical GRF, ID model ==="
python verify_pipeline.py \
    --model_name hier_v1 \
    --n_test 128 \
    --max_iters 300 \
    --tau 24 \
    --dim 1 \
    --grf_mode hierarchical
