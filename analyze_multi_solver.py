"""Analyze why the oracle switches between GS and Jacobi(w=0.67) at late iterations for ConvDiff."""
import torch
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, ".")
from pde import ConvectionDiffusion2D, PoissonEquation2D
from numerical_solver import WeightedJacobiSolver, GaussSeidelSolver
from verify_pipeline import generate_test_data, load_deeponet, _build_deeponet_input

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("args/deeponet_args.json") as f:
    N = json.load(f)["N"]

equation = "ConvDiff"
b_vel = 20.0
n_test = 128

inputs, u_sol, f_raw, x, pde, k2_raw = generate_test_data(
    n_test, N, 2, device, equation=equation, grf_mode="hierarchical",
    b_vel=b_vel, reaction_c=0.0)

f_raw_flat = f_raw.reshape(n_test, N * N) if f_raw.dim() > 2 else f_raw

deeponet = load_deeponet("./checkpoints", "hier_cd2", N, 2, equation, device)

# Build PDE and solvers
y = torch.linspace(0, 1, N + 1, device=device)[:-1]
a_func = lambda xi, yi: 1.0
pde2 = ConvectionDiffusion2D(a_func=a_func, f_func=f_raw_flat,
                              b_vec=(b_vel, b_vel),
                              boundary="Periodic", x=x, y=y, A=None, solve=False, device=device,
                              reaction=0.0)
A = pde2.A
b = pde2.b

jacobi_067 = WeightedJacobiSolver(equation=pde2, device=device, weight=0.67)
jacobi_1 = WeightedJacobiSolver(equation=pde2, device=device, weight=1.0)
gs = GaussSeidelSolver(equation=pde2, device=device)

solvers = [jacobi_067, jacobi_1, gs]
solver_names = ["Jacobi(w=0.67)", "Jacobi(w=1)", "GS"]

# Run greedy and track per-solver error reductions
max_iters = 300
u_prev = torch.zeros_like(u_sol)

# Track: at each step, what's the error after each solver candidate?
# Also track the spectral content of the error
best_solver_history = np.zeros((max_iters, n_test), dtype=int)
error_reductions = np.zeros((max_iters, n_test, 4))  # 3 classical + deeponet
error_norms = np.zeros((max_iters, n_test, 4))

# Also compute the spectral energy of the error at key points
spectral_snapshots = {}

for t in range(max_iters):
    if t % 50 == 0:
        print(f"Iteration {t}/{max_iters}")

    err_prev = torch.linalg.norm(u_prev - u_sol, dim=-1)  # (n_test,)

    candidates = []
    for i, solver in enumerate(solvers):
        u_cand = solver.iteration(u_prev)
        u_cand = u_cand - torch.mean(u_cand, dim=-1, keepdim=True)
        candidates.append(u_cand)
        err_cand = torch.linalg.norm(u_cand - u_sol, dim=-1)
        error_norms[t, :, i] = err_cand.cpu().numpy()
        error_reductions[t, :, i] = (err_prev - err_cand).cpu().numpy()

    # DeepONet
    residual = b - torch.bmm(A, u_prev.unsqueeze(-1)).squeeze(-1)
    inp, rnorm = _build_deeponet_input(residual, None, equation)
    with torch.no_grad():
        correction = deeponet(inp).reshape(u_prev.shape)
    if rnorm is not None:
        correction = correction * rnorm
    u_don = u_prev + correction
    u_don = u_don - torch.mean(u_don, dim=-1, keepdim=True)
    candidates.append(u_don)
    err_don = torch.linalg.norm(u_don - u_sol, dim=-1)
    error_norms[t, :, 3] = err_don.cpu().numpy()
    error_reductions[t, :, 3] = (err_prev - err_don).cpu().numpy()

    # Best solver per sample
    all_errors = torch.stack([torch.linalg.norm(c - u_sol, dim=-1) for c in candidates], dim=0)
    best = torch.argmin(all_errors, dim=0)
    best_solver_history[t] = best.cpu().numpy()

    u_new = torch.stack(candidates, dim=0)[best, torch.arange(n_test)]
    u_prev = u_new

    # Spectral snapshots at key iterations
    if t + 1 in [1, 10, 50, 100, 200, 250, 275, 300]:
        error_field = (u_prev - u_sol).reshape(n_test, N, N)
        fft_error = torch.fft.fft2(error_field)
        power = (fft_error.abs() ** 2).mean(dim=0)  # average over samples
        spectral_snapshots[t + 1] = power.cpu().numpy()

print("\n=== Solver selection patterns ===")
for t_range, label in [((0, 50), "iters 1-50"), ((50, 150), "iters 51-150"),
                        ((150, 250), "iters 151-250"), ((250, 300), "iters 251-300")]:
    chunk = best_solver_history[t_range[0]:t_range[1]]
    print(f"\n{label}:")
    for k, name in enumerate(solver_names + ["DeepONet"]):
        frac = (chunk == k).mean()
        print(f"  {name:20s}: {frac:.4f}")

# Analyze which samples switch to Jacobi(w=0.67) late
late_chunk = best_solver_history[250:]  # last 50 iterations
uses_j067_late = (late_chunk == 0).sum(axis=0)  # per-sample count of Jacobi(w=0.67)
switchers = np.where(uses_j067_late > 5)[0]
non_switchers = np.where(uses_j067_late == 0)[0][:len(switchers)]

print(f"\n=== Samples that switch to Jacobi(w=0.67) late (>{5} uses in last 50 iters) ===")
print(f"  Count: {len(switchers)} samples")
print(f"  Indices: {switchers[:20]}")

# Compare error reduction per solver at iteration 290 for switchers vs non-switchers
t_late = 289
print(f"\n=== Error reduction at iteration {t_late+1} ===")
print(f"  Switchers (samples using Jacobi(0.67) late):")
for k, name in enumerate(solver_names + ["DeepONet"]):
    reds = error_reductions[t_late, switchers, k]
    print(f"    {name:20s}: mean reduction = {reds.mean():.2e}, std = {reds.std():.2e}")
print(f"  Non-switchers (always GS):")
for k, name in enumerate(solver_names + ["DeepONet"]):
    reds = error_reductions[t_late, non_switchers, k]
    print(f"    {name:20s}: mean reduction = {reds.mean():.2e}, std = {reds.std():.2e}")

# Compare the A matrix spectral properties for switchers vs non-switchers
# Since ConvDiff has constant coefficients, A is shared -> must be the solution/error structure
print(f"\n=== Solution norms ===")
print(f"  Switchers:     u_sol norm mean = {torch.linalg.norm(u_sol[switchers], dim=-1).mean():.4e}")
print(f"  Non-switchers: u_sol norm mean = {torch.linalg.norm(u_sol[non_switchers], dim=-1).mean():.4e}")

# Error at iteration 250 for both groups
print(f"\n=== Error at iteration 250 ===")
err_250 = error_norms[249]  # after iter 250
for k, name in enumerate(solver_names + ["DeepONet"]):
    e_sw = err_250[switchers, k]
    e_nsw = err_250[non_switchers, k]
    print(f"  {name:20s}: switchers={e_sw.mean():.2e}, non-switchers={e_nsw.mean():.2e}")

# Plot 1: Error reduction margin (GS - Jacobi0.67) over time for switchers vs non-switchers
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
margin = error_reductions[:, :, 2] - error_reductions[:, :, 0]  # GS reduction - J067 reduction
# positive means GS is better
if len(switchers) > 0:
    ax.plot(np.arange(1, max_iters+1), margin[:, switchers].mean(axis=1),
            label=f"Switchers (n={len(switchers)})", color="blue", alpha=0.8)
if len(non_switchers) > 0:
    ax.plot(np.arange(1, max_iters+1), margin[:, non_switchers].mean(axis=1),
            label=f"Non-switchers (n={len(non_switchers)})", color="green", alpha=0.8)
ax.axhline(0, color='black', linestyle='--', alpha=0.3)
ax.set_xlabel("Iteration")
ax.set_ylabel("GS advantage (error reduction GS - J067)")
ax.set_title("Per-step advantage of GS over Jacobi(w=0.67)")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Spectral power at different iterations
ax = axes[0, 1]
freqs = np.arange(N)
for step, power in sorted(spectral_snapshots.items()):
    # 1D radial average of 2D power spectrum
    kx = np.fft.fftfreq(N, d=1.0/N)
    ky = np.fft.fftfreq(N, d=1.0/N)
    KX, KY = np.meshgrid(kx, ky)
    K_rad = np.sqrt(KX**2 + KY**2)
    max_k = int(N / 2)
    radial_power = np.zeros(max_k)
    for ki in range(max_k):
        mask = (K_rad >= ki) & (K_rad < ki + 1)
        if mask.any():
            radial_power[ki] = power[mask].mean()
    ax.semilogy(np.arange(max_k), radial_power + 1e-30, label=f"iter {step}", alpha=0.8)
ax.set_xlabel("Wavenumber |k|")
ax.set_ylabel("Mean error power")
ax.set_title("Radial error power spectrum evolution")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Plot 3: Per-solver error reduction over time (all samples)
ax = axes[1, 0]
for k, (name, color) in enumerate(zip(solver_names + ["DeepONet"],
                                       ["blue", "orange", "green", "red"])):
    mean_red = error_reductions[:, :, k].mean(axis=1)
    ax.plot(np.arange(1, max_iters+1), mean_red, label=name, color=color, alpha=0.8)
ax.set_xlabel("Iteration")
ax.set_ylabel("Mean error reduction")
ax.set_title("Per-solver error reduction over time")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Zoomed error reduction for last 100 iters
ax = axes[1, 1]
t_start = 200
for k, (name, color) in enumerate(zip(solver_names + ["DeepONet"],
                                       ["blue", "orange", "green", "red"])):
    mean_red = error_reductions[t_start:, :, k].mean(axis=1)
    ax.plot(np.arange(t_start+1, max_iters+1), mean_red, label=name, color=color, alpha=0.8)
ax.set_xlabel("Iteration")
ax.set_ylabel("Mean error reduction")
ax.set_title("Per-solver error reduction (zoomed, iters 201-300)")
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
save_dir = "plots/multi_solver_analysis"
os.makedirs(save_dir, exist_ok=True)
fig.savefig(os.path.join(save_dir, "convdiff_solver_switching_analysis.png"), dpi=150)
plt.close(fig)
print(f"\nSaved analysis plot to {save_dir}/convdiff_solver_switching_analysis.png")

# Compute the spectral radius of each solver's iteration matrix
print("\n=== Spectral analysis of solver iteration matrices ===")
A_shared = A[0]  # shared for constant coefficients
D_inv = torch.diag(1.0 / torch.diag(A_shared))
I_mat = torch.eye(A_shared.shape[0], device=device)

# Jacobi iteration matrix: I - omega * D^{-1} * A
M_j067 = I_mat - 0.67 * D_inv @ A_shared
M_j1 = I_mat - 1.0 * D_inv @ A_shared

# GS iteration matrix: I - (D + L)^{-1} * A  where L is strictly lower triangular
DL = torch.tril(A_shared)
DL_inv = torch.linalg.inv(DL)
M_gs = I_mat - DL_inv @ A_shared

eigs_j067 = torch.linalg.eigvals(M_j067).abs()
eigs_j1 = torch.linalg.eigvals(M_j1).abs()
eigs_gs = torch.linalg.eigvals(M_gs).abs()

print(f"  Jacobi(w=0.67) spectral radius: {eigs_j067.max():.6f}")
print(f"  Jacobi(w=1.0)  spectral radius: {eigs_j1.max():.6f}")
print(f"  Gauss-Seidel    spectral radius: {eigs_gs.max():.6f}")

# Look at the eigenvalue magnitudes sorted
eigs_j067_sorted = eigs_j067.sort(descending=True).values[:20]
eigs_j1_sorted = eigs_j1.sort(descending=True).values[:20]
eigs_gs_sorted = eigs_gs.sort(descending=True).values[:20]

print(f"\n  Top 10 eigenvalue magnitudes:")
print(f"  {'k':>3s}  {'J(0.67)':>10s}  {'J(1.0)':>10s}  {'GS':>10s}")
for i in range(10):
    print(f"  {i:3d}  {eigs_j067_sorted[i].item():10.6f}  {eigs_j1_sorted[i].item():10.6f}  {eigs_gs_sorted[i].item():10.6f}")

# Find modes where Jacobi(0.67) has smaller eigenvalue than GS
j067_better = (eigs_j067 < eigs_gs)
n_j067_better = j067_better.sum().item()
n_total = len(eigs_j067)
print(f"\n  Modes where |lambda_J067| < |lambda_GS|: {n_j067_better}/{n_total} ({100*n_j067_better/n_total:.1f}%)")

# Plot eigenvalue comparison
fig, ax = plt.subplots(figsize=(10, 6))
idx = np.arange(len(eigs_j067))
ax.scatter(eigs_gs.cpu().numpy(), eigs_j067.cpu().numpy(), s=3, alpha=0.5)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='|lambda_J067| = |lambda_GS|')
ax.set_xlabel("|lambda_GS|")
ax.set_ylabel("|lambda_J067|")
ax.set_title("Eigenvalue magnitudes: GS vs Jacobi(w=0.67)\nPoints below diagonal: J067 is better")
ax.legend()
ax.grid(True, alpha=0.3)
fig.savefig(os.path.join(save_dir, "eigenvalue_comparison.png"), dpi=150)
plt.close(fig)
print(f"Saved eigenvalue comparison to {save_dir}/eigenvalue_comparison.png")
