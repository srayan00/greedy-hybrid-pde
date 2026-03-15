#!/bin/bash
#SBATCH --job-name=rtr_poi_jnt
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/router_poi_sor_fno_joint_%j.out
#SBATCH --error=logs/router_poi_sor_fno_joint_%j.err

mkdir -p logs

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate greedy

cd /mnt/sharefs/user35/greedy-hybrid-pde

export PYTHONUNBUFFERED=1

python -u train_router.py \
    --ml_model fno \
    --ml_model_name hier_fno_unrolled \
    --model_name hier_sor_fno_joint \
    --numerical_solvers "sor_1.0,sor_1.3,sor_1.6" \
    --equation Poisson \
    --dim 2 \
    --boundary Periodic \
    --in_channels 1 \
    --grf_mode hierarchical \
    --data_name hier_ \
    --joint
