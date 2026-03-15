#!/bin/bash
#SBATCH --job-name=verify_cr
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/verify_cr_%j.out
#SBATCH --error=logs/verify_cr_%j.err

mkdir -p logs

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate greedy

cd /mnt/sharefs/user35/greedy-hybrid-pde

echo "Running 2D ConvReact verification (with LSTM router)"
python verify_pipeline.py \
    --model_name hier_cr \
    --n_test 128 \
    --dim 2 \
    --equation ConvDiff \
    --grf_mode hierarchical \
    --numerical_solvers jacobi \
    --max_iters 300 \
    --b_vel 20.0 \
    --reaction_c 5.0 \
    --router_model_name router_cr \
    --save_dir ./plots_cr
