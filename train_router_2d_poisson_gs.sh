#!/bin/bash
#SBATCH --job-name=train_router_poi_gs
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_router_poi_gs_%j.out
#SBATCH --error=logs/train_router_poi_gs_%j.err

mkdir -p logs

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate greedy

cd /mnt/sharefs/user35/greedy-hybrid-pde

python train_router.py \
    --ml_model deeponet \
    --ml_model_name hier_v2 \
    --model lstm \
    --model_name router_poi_gs \
    --numerical_solvers gs \
    --equation Poisson \
    --dim 2 \
    --boundary Periodic \
    --in_channels 1 \
    --grf_mode hierarchical \
    --data_name hier_
