# Reproducing Results: Greedy PDE Router

This document describes how to reproduce the experimental results for the
**Greedy Hybrid PDE Solver** — a learned router that blends a neural operator
(DeepONet) with a classical Jacobi solver to accelerate PDE convergence.

We present results on three 2D periodic PDEs (grid N=31, 300 Jacobi iterations):

| PDE | Equation |
|-----|----------|
| **Poisson** | −∇·(a∇u) = f |
| **Convection-Diffusion (ConvDiff)** | −∇·(a∇u) + b·∇u = f, b=(20,20) |
| **Convection-Reaction-Diffusion (ConvReact)** | −∇·(a∇u) + b·∇u + cu = f, b=(20,20), c=5 |

All use hierarchical Gaussian Random Fields for forcing and a=1 (single input channel).

---

## Prerequisites

```bash
conda env create -f environment.yml
conda activate greedy
```

All SLURM scripts assume a single GPU and the conda environment `greedy`.
Before running, edit the `cd /mnt/sharefs/...` line in each `.sh` file to
point to your local clone of this repository.

---

## Overview of the Pipeline

For each PDE, the pipeline has four stages:

1. **Train DeepONet** — Learn a neural surrogate that maps residuals to corrections.
2. **Train LSTM Router** — Learn a routing policy that decides when to apply the DeepONet vs. Jacobi.
3. **Verify & Plot** — Run Jacobi, HINTS, Oracle Greedy, and LSTM Router for 300 iterations; produce convergence plots and routing pattern visualizations.
4. **Compute Statistics** — Report per-sample mean/std and paired t-tests.

---

## 1. 2D Poisson

### 1a. Train DeepONet

```bash
sbatch train_deeponet_2d_poisson.sh
```

This trains a DeepONet (`args/deeponet_args.json`) on 10,000 hierarchical GRF
samples with MSE loss (alpha=0). Checkpoint: `checkpoints/deeponet_hier_v2_Poisson_Periodic_2d_1c_best.pth`.

### 1b. Train LSTM Router

```bash
sbatch train_router_2d_poisson.sh
```

Trains a 3-layer LSTM router (`args/lstm_args.json`) for 300 epochs using the
DeepONet from step 1a. The router learns when applying the DeepONet yields a
larger error reduction than Jacobi. Checkpoint: `checkpoints/lstmrouter_hier_v1_Poisson_Periodic_2d_1c_jacobi_best.pth`.

### 1c. Verify & Plot

```bash
sbatch verify_2d_poisson_router.sh
```

Runs 128 test samples through 300 iterations of each strategy and saves plots to
`plots/2d_poisson_hierarchical/`:
- `convergence_comparison.png` — Mean L2 error vs iteration for all strategies
- `deeponet_predictions.png` — DeepONet vs true solution visualizations
- `greedy_routing_pattern.png` — Oracle greedy routing decisions
- `router_routing_pattern.png` — LSTM router routing decisions

### 1d. Compute Statistics Table

```bash
sbatch run_stats_poisson.sh
```

> **Note:** You need to create `run_stats_poisson.sh` wrapping `compute_stats.py`
> (it runs on GPU). Alternatively, run interactively on a GPU node:
> ```bash
> srun --gres=gpu:1 --time=00:30:00 python compute_stats.py
> ```

Output: plaintext table with mean (std) and paired t-test p-values (LSTM < Jacobi, LSTM < HINTS).

---

## 2. 2D Convection-Diffusion (ConvDiff)

### 2a. Train DeepONet

```bash
sbatch train_deeponet_2d_convdiff_v2.sh
```

Uses the larger DeepONet architecture (`args/deeponet_helmholtz_args.json`:
branch_dim=128, hidden_branch=256, 3 layers). Checkpoint:
`checkpoints/deeponet_hier_cd2_ConvDiff_Periodic_2d_1c_best.pth`.

### 2b. Train LSTM Router

```bash
sbatch train_router_2d_convdiff.sh
```

Checkpoint: `checkpoints/lstmrouter_router_cd_ConvDiff_Periodic_2d_1c_jacobi_best.pth`.

### 2c. Verify & Plot

```bash
sbatch verify_2d_convdiff.sh
```

Plots saved to `plots/2d_convdiff_hierarchical/`.

### 2d. Compute Statistics Table

```bash
sbatch run_stats_convdiff.sh
```

This runs `compute_stats_convdiff.py` on GPU. Output: plaintext table with
mean (std) and paired t-test p-values.

---

## 3. 2D Convection-Reaction-Diffusion (ConvReact)

This is the same ConvDiff equation with an added reaction term (c=5), which
removes the null space and ensures the greedy advantage persists at large
iteration counts.

### 3a. Train DeepONet

```bash
sbatch train_deeponet_2d_convreact.sh
```

Uses the larger DeepONet architecture with `--reaction_c 5.0`. Checkpoint:
`checkpoints/deeponet_hier_cr_ConvDiff_Periodic_2d_1c_best.pth`.

### 3b. Train LSTM Router

```bash
sbatch train_router_2d_convreact.sh
```

Checkpoint: `checkpoints/lstmrouter_router_cr_ConvDiff_Periodic_2d_1c_jacobi_best.pth`.

### 3c. Verify & Plot

```bash
sbatch verify_2d_convreact.sh
```

Plots saved to `plots_cr/2d_convdiff_hierarchical/` (separate directory
to avoid overwriting ConvDiff plots, since both use `--equation ConvDiff`).

---

## Configuration Files

| File | Description |
|------|-------------|
| `args/deeponet_args.json` | Small DeepONet (branch_dim=60, 2 layers). Used for Poisson. |
| `args/deeponet_helmholtz_args.json` | Large DeepONet (branch_dim=128, 3 layers). Used for ConvDiff, ConvReact. |
| `args/lstm_args.json` | LSTM router config (hidden_dim=256, 3 layers, 300 epochs). |
| `args/grf_args.json` | GRF parameters (alpha=1, beta=9, gamma=2). |

## Key Command-Line Arguments

| Argument | Description |
|----------|-------------|
| `--equation` | PDE type: `Poisson`, `Helmholtz`, `ConvDiff` |
| `--dim` | Spatial dimension (1 or 2) |
| `--boundary` | Boundary condition (`Periodic` or `Dirichlet`) |
| `--grf_mode` | GRF sampling: `fixed` (single set of params) or `hierarchical` (sampled params) |
| `--b_vel` | Advection velocity for ConvDiff (default 20.0; b_vec = (b_vel, b_vel)) |
| `--reaction_c` | Reaction coefficient for ConvDiff (default 0.0; set to 5.0 for ConvReact) |
| `--in_channels` | Number of input channels (1 for single-channel PDEs) |
| `--loss_alpha` | MSE loss weighting (0.0 = plain MSE) |
| `--router_model_name` | If provided, also runs LSTM router comparison in verify_pipeline.py |

## File Descriptions

| File | Role |
|------|------|
| `pde.py` | PDE discretizations: Poisson, Helmholtz, ConvDiff (1D & 2D) |
| `data_generation.py` | Gaussian Random Field generators (fixed & hierarchical) |
| `ml_solver.py` | DeepONet and FNO model definitions |
| `numerical_solver.py` | Weighted Jacobi, Gauss-Seidel, Multigrid solvers |
| `hybrid_solver.py` | HybridSolver, routers (Constant, HINTS, LSTMGreedy) |
| `trainer.py` | Training loop, loss functions, scheduled sampling, BPTT |
| `train_ml_solver.py` | Script to train DeepONet/FNO surrogates |
| `train_router.py` | Script to train LSTM routing policy |
| `verify_pipeline.py` | End-to-end evaluation: data gen, iteration loop, plotting |
| `compute_stats.py` | Per-sample statistics and t-tests for 2D Poisson |
| `compute_stats_convdiff.py` | Per-sample statistics and t-tests for 2D ConvDiff |

## Expected Results

### 2D Poisson (300 iterations, 128 test samples)

| Strategy | Final L2 Error | AUC |
|----------|---------------|-----|
| Jacobi Only | ~3.5e-4 | ~4.3e-1 |
| HINTS | ~2.9e-4 | ~1.5e-1 |
| **LSTM Router** | **~1.8e-4** | **~1.1e-1** |
| True Greedy | ~6.9e-5 | ~5.9e-2 |

### 2D ConvDiff (300 iterations, 128 test samples)

| Strategy | Final L2 Error | AUC |
|----------|---------------|-----|
| Jacobi Only | ~1.5e-4 | ~3.4e-1 |
| HINTS | ~1.8e-4 | ~1.9e-1 |
| **LSTM Router** | **~4.2e-5** | **~1.4e-1** |
| True Greedy | ~1.3e-5 | ~7.7e-2 |

### 2D ConvReact (300 iterations, 128 test samples)

| Strategy | Final L2 Error | AUC |
|----------|---------------|-----|
| Jacobi Only | ~1.0e-4 | ~3.0e-1 |
| HINTS | ~1.8e-4 | ~1.5e-1 |
| **LSTM Router** | **~1.2e-4** | **~9.4e-2** |
| True Greedy | ~7.7e-5 | ~5.3e-2 |

> Exact values vary slightly with random seed but relative ordering is consistent.
> All paired t-tests (LSTM < Jacobi, LSTM < HINTS) yield p < 1e-4.
