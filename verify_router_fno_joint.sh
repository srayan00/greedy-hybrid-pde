#!/bin/bash
#SBATCH --job-name=verify_jnt
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/verify_router_fno_joint_%j.out
#SBATCH --error=logs/verify_router_fno_joint_%j.err

mkdir -p logs

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate greedy

cd /mnt/sharefs/user35/greedy-hybrid-pde

export PYTHONUNBUFFERED=1

python -u verify_pipeline.py \
    --model_name hier_fno_unrolled \
    --ml_model fno \
    --n_test 128 \
    --dim 2 \
    --equation Poisson \
    --grf_mode hierarchical \
    --router_model_name hier_sor_fno_joint \
    --numerical_solvers "sor_1.0,sor_1.3,sor_1.6" \
    --solver_type "sor_1.0,sor_1.3,sor_1.6" \
    --max_iters 300 \
    --save_dir ./plots_joint
