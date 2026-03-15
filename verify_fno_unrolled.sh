#!/bin/bash
#SBATCH --job-name=verify_unroll
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/verify_fno_unrolled_%j.out
#SBATCH --error=logs/verify_fno_unrolled_%j.err

mkdir -p logs

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate greedy

cd /mnt/sharefs/user35/greedy-hybrid-pde

export PYTHONUNBUFFERED=1

echo "============================================"
echo " 1) Oracle comparison with UNROLLED FNO"
echo "============================================"
python -u verify_pipeline.py \
    --model_name hier_fno_unrolled \
    --ml_model fno \
    --n_test 128 --dim 2 --equation Poisson \
    --grf_mode hierarchical \
    --solver_type sor_1.0,sor_1.3,sor_1.6 \
    --max_iters 300

echo ""
echo "============================================"
echo " 2) Oracle comparison with PRE-TRAINED FNO"
echo "============================================"
python -u verify_pipeline.py \
    --model_name hier_fno \
    --ml_model fno \
    --n_test 128 --dim 2 --equation Poisson \
    --grf_mode hierarchical \
    --solver_type sor_1.0,sor_1.3,sor_1.6 \
    --max_iters 300
