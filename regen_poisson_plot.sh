#!/bin/bash
#SBATCH --job-name=regen_plot
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=logs/regen_poisson_plot_%j.out
#SBATCH --error=logs/regen_poisson_plot_%j.err

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate greedy
cd /mnt/sharefs/user35/greedy-hybrid-pde
export PYTHONUNBUFFERED=1

python -u compare_fno_unrolled.py \
    --equation Poisson \
    --solver_specs "sor_1.0,sor_1.3,sor_1.6" \
    --unrolled_name hier_fno_unrolled \
    --n_test 128 \
    --max_iters 300 \
    --no_pretrained

cp plots/2d_poisson_hierarchical_multi_sor_1.0_sor_1.3_sor_1.6_fno/fno_comparison.png \
   cfd-hack-slides/images/poisson_fno_convergence.png
echo "Copied to slides images"
