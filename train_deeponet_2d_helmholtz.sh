#!/bin/bash
#SBATCH --job-name=train_don_2d_helmholtz
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_deeponet_2d_helmholtz_%j.out
#SBATCH --error=logs/train_deeponet_2d_helmholtz_%j.err

mkdir -p logs

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate greedy

cd /mnt/sharefs/user35/greedy-hybrid-pde

python train_ml_solver.py \
    --model deeponet \
    --model_name hier_v3 \
    --equation Helmholtz \
    --dim 2 \
    --boundary Periodic \
    --in_channels 1 \
    --grf_mode hierarchical \
    --data_name hier_ \
    --loss_alpha 0.0 \
    --args_file args/deeponet_helmholtz_args.json
