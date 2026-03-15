#!/bin/bash
#SBATCH --job-name=verify_poi_sor
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/verify_poi_sor_%j.out
#SBATCH --error=logs/verify_poi_sor_%j.err

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
    --solver_type sor_1.0,sor_1.3,sor_1.6 \
    --max_iters 300
