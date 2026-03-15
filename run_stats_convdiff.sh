#!/bin/bash
#SBATCH --job-name=stats_cd
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/stats_cd_%j.out
#SBATCH --error=logs/stats_cd_%j.err

mkdir -p logs

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate greedy

cd /mnt/sharefs/user35/greedy-hybrid-pde

python compute_stats_convdiff.py
