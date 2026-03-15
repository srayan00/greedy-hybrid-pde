#!/bin/bash
#SBATCH --job-name=fno_unroll
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/fno_unrolled_poi_%j.out
#SBATCH --error=logs/fno_unrolled_poi_%j.err

mkdir -p logs

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate greedy

cd /mnt/sharefs/user35/greedy-hybrid-pde

export PYTHONUNBUFFERED=1

python -u train_fno_unrolled.py \
    --fno_model_name hier_fno \
    --save_name hier_fno_unrolled \
    --equation Poisson \
    --solver_specs "sor_1.0,sor_1.3,sor_1.6" \
    --data_name hier_ \
    --n_train 2048 \
    --n_val 256 \
    --batch_size 32 \
    --epochs 100 \
    --lr 5e-5 \
    --T_unroll 30 \
    --T_unroll_max 200 \
    --T_grow_every 15 \
    --T_grow_by 20 \
    --loss_mode fno_error \
    --N 31
