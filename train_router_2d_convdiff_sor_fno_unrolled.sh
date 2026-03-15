#!/bin/bash
#SBATCH --job-name=rtr_cd_unr
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/router_cd_sor_fno_unrolled_%j.out
#SBATCH --error=logs/router_cd_sor_fno_unrolled_%j.err

mkdir -p logs

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate greedy

cd /mnt/sharefs/user35/greedy-hybrid-pde

export PYTHONUNBUFFERED=1

python -u train_router.py \
    --ml_model fno \
    --ml_model_name hier_cd_fno_unrolled_scratch \
    --model_name hier_cd_sor_fno_unrolled \
    --numerical_solvers "sor_1.0,sor_1.3,sor_1.6" \
    --equation ConvDiff \
    --dim 2 \
    --boundary Periodic \
    --in_channels 1 \
    --grf_mode hierarchical \
    --data_name hier_cd_ \
    --b_vel 20.0
