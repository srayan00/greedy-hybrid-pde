#!/bin/bash
#SBATCH --job-name=compare_fno
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/compare_fno_%j.out
#SBATCH --error=logs/compare_fno_%j.err

mkdir -p logs

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate greedy

cd /mnt/sharefs/user35/greedy-hybrid-pde

export PYTHONUNBUFFERED=1

python -u compare_fno_unrolled.py \
    --equation Poisson \
    --solver_specs "sor_1.0,sor_1.3,sor_1.6" \
    --pretrained_name hier_fno \
    --unrolled_name hier_fno_unrolled \
    --n_test 128 \
    --max_iters 300
