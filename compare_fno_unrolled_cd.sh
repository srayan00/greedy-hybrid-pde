#!/bin/bash
#SBATCH --job-name=compare_cd
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/compare_fno_cd_%j.out
#SBATCH --error=logs/compare_fno_cd_%j.err

mkdir -p logs

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate greedy

cd /mnt/sharefs/user35/greedy-hybrid-pde

export PYTHONUNBUFFERED=1

python -u compare_fno_unrolled.py \
    --equation ConvDiff \
    --solver_specs "sor_1.0,sor_1.3,sor_1.6" \
    --pretrained_name hier_cd_fno \
    --unrolled_name hier_cd_fno_unrolled \
    --n_test 128 \
    --max_iters 300
