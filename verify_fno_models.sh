#!/bin/bash
#SBATCH --job-name=verify_fno
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/verify_fno_%j.out
#SBATCH --error=logs/verify_fno_%j.err

mkdir -p logs

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate greedy

cd /mnt/sharefs/user35/greedy-hybrid-pde

echo "=========================================="
echo "1. FNO + Poisson + SOR portfolio"
echo "=========================================="
python verify_pipeline.py \
    --model_name hier_fno \
    --ml_model fno \
    --n_test 128 --dim 2 --equation Poisson \
    --grf_mode hierarchical \
    --solver_type sor_1.0,sor_1.3,sor_1.6 \
    --max_iters 300

echo ""
echo "=========================================="
echo "2. FNO + ConvDiff + SOR portfolio"
echo "=========================================="
python verify_pipeline.py \
    --model_name hier_cd_fno \
    --ml_model fno \
    --n_test 128 --dim 2 --equation ConvDiff \
    --grf_mode hierarchical \
    --solver_type sor_1.0,sor_1.3,sor_1.6 \
    --b_vel 20.0 \
    --max_iters 300
