#!/bin/bash
#SBATCH --job-name=analyze_multi
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/analyze_multi_%j.out
#SBATCH --error=logs/analyze_multi_%j.err

mkdir -p logs

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate greedy

cd /mnt/sharefs/user35/greedy-hybrid-pde

python analyze_multi_solver.py
