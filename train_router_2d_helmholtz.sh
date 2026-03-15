#!/bin/bash
#SBATCH --job-name=train_router_helm
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_router_2d_helmholtz_%j.out
#SBATCH --error=logs/train_router_2d_helmholtz_%j.err

mkdir -p logs

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate greedy

cd /mnt/sharefs/user35/greedy-hybrid-pde

python train_router.py \
    --ml_model deeponet \
    --ml_model_name hier_v3 \
    --model lstm \
    --model_name hier_v1 \
    --numerical_solvers jacobi \
    --equation Helmholtz \
    --dim 2 \
    --boundary Periodic \
    --in_channels 1 \
    --grf_mode hierarchical \
    --data_name hier_
