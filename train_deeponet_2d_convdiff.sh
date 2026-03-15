#!/bin/bash
#SBATCH --job-name=train_don_cd
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_don_cd_%j.out
#SBATCH --error=logs/train_don_cd_%j.err

mkdir -p logs

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate greedy

cd /mnt/sharefs/user35/greedy-hybrid-pde

python train_ml_solver.py \
    --model deeponet \
    --model_name hier_cd \
    --equation ConvDiff \
    --dim 2 \
    --boundary Periodic \
    --in_channels 1 \
    --grf_mode hierarchical \
    --data_name hier_cd_ \
    --loss_alpha 0.0 \
    --b_vel 20.0
