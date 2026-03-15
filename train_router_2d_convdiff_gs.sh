#!/bin/bash
#SBATCH --job-name=train_router_cd_gs
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_router_cd_gs_%j.out
#SBATCH --error=logs/train_router_cd_gs_%j.err

mkdir -p logs

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate greedy

cd /mnt/sharefs/user35/greedy-hybrid-pde

python train_router.py \
    --ml_model deeponet \
    --ml_model_name hier_cd2 \
    --model_name router_cd_gs \
    --model lstm \
    --equation ConvDiff \
    --dim 2 \
    --boundary Periodic \
    --in_channels 1 \
    --grf_mode hierarchical \
    --data_name hier_cd_ \
    --numerical_solvers gs \
    --b_vel 20.0
