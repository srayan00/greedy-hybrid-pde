#!/bin/bash
#SBATCH --job-name=verify_poi_multi_orc
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/verify_poi_multi_oracle_%j.out
#SBATCH --error=logs/verify_poi_multi_oracle_%j.err

mkdir -p logs

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate greedy

cd /mnt/sharefs/user35/greedy-hybrid-pde

python verify_pipeline.py \
    --model_name hier_v2 \
    --n_test 128 \
    --dim 2 \
    --equation Poisson \
    --grf_mode hierarchical \
    --solver_type jacobi_0.67,jacobi,gs \
    --max_iters 300
