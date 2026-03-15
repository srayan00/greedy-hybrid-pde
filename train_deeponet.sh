#!/bin/bash
#SBATCH --job-name=train_deeponet_1d_poisson
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/train_deeponet_%j.out
#SBATCH --error=logs/train_deeponet_%j.err

mkdir -p logs

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate greedy

cd /mnt/sharefs/user35/greedy-hybrid-pde

python train_ml_solver.py \
    --model deeponet \
    --model_name test_v1 \
    --equation Poisson \
    --dim 1 \
    --boundary Periodic \
    --in_channels 1
