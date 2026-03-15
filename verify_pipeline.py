"""
Verification script for the greedy hybrid PDE solver pipeline.
Focuses on 1D Poisson with Periodic BCs, K=2 (Jacobi + DeepONet).

Steps:
  1. Generate test data (GRF forcing -> solve Poisson)
  2. Load trained DeepONet and evaluate its predictions
  3. Visualize: true solution vs DeepONet prediction for a few samples
  4. Run iterative solvers and compare convergence:
     - Jacobi only
     - DeepONet only (apply correction each step)
     - HINTS (Jacobi + DeepONet, fixed schedule)
     - True greedy (oracle: pick best solver at each step using true error)
  5. Plot convergence histories
"""

import os
import json
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_generation import GaussianRandomField, GaussianRandomFieldHierarchical
from pde import PoissonEquation1D, PoissonEquation2D, HelmholtzEquation1D, HelmholtzEquation2D, ConvectionDiffusion2D, ReactionDiffusion2D
from ml_solver import DeepONet
from numerical_solver import WeightedJacobiSolver
from hybrid_solver import HybridSolver, ConstantRouter, HINTSRouter, LSTMGreedyRouter
import models


def _make_grf(N, dim, grf_mode, device, seed):
    with open("args/grf_args.json") as f:
        grf_args = json.load(f)
    if grf_mode == "fixed":
        return GaussianRandomField(
            num_samples=N, dim=dim,
            alpha=grf_args["alpha"], beta=grf_args["beta"], gamma=grf_args["gamma"],
            device=device, seed=seed,
        )
    elif grf_mode == "hierarchical":
        return GaussianRandomFieldHierarchical(
            num_samples=N, dim=dim,
            alpha_min=0.01, alpha_max=100.0,
            beta_min=0.1, beta_max=1000.0,
            gamma_list=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
            device=device, seed=seed,
        )
    else:
        raise ValueError(f"Unknown grf_mode: {grf_mode}")


def generate_test_data(n_test, N, dim, device, equation="Poisson",
                       grf_mode="fixed", seed=9999, k2_mode="exp", b_vel=20.0,
                       reaction_c=0.0):
    """Generate test forcing functions + reference solutions for Periodic PDEs."""
    needs_mean_zero = equation == "Poisson" or (equation == "ConvDiff" and reaction_c == 0.0)
    grf = _make_grf(N, dim, grf_mode, device, seed)

    if dim == 1:
        pushfwd = lambda x: x - torch.mean(x, dim=-1, keepdim=True)
    else:
        pushfwd = lambda x: x - torch.mean(x, dim=(-2, -1), keepdim=True)

    if needs_mean_zero:
        f_raw = grf.generate(n_test, pushfoward=pushfwd)
    else:
        f_raw = grf.generate(n_test, pushfoward=None)

    # For Helmholtz, normalize f to unit L2 norm per sample (matches DeepONet training)
    if equation == "Helmholtz":
        f_flat_tmp = f_raw.reshape(n_test, -1)
        f_norms = torch.linalg.norm(f_flat_tmp, dim=-1, keepdim=True).clamp(min=1e-15)
        if dim == 1:
            f_raw = f_raw / f_norms
        else:
            f_raw = f_raw / f_norms.unsqueeze(-1)

    # For Helmholtz, also generate k2 field
    k2_raw = None
    if equation == "Helmholtz":
        if k2_mode == "const":
            k2_shape = (n_test, N, N) if dim == 2 else (n_test, N)
            k2_raw = 10.0 * torch.ones(k2_shape, device=device)
        elif k2_mode == "mild":
            grf_k2 = _make_grf(N, dim, grf_mode, device, seed + 1)
            k2_raw = grf_k2.generate(n_test, pushfoward=lambda x: 10.0 + 5.0 * torch.tanh(x))
        else:
            grf_k2 = _make_grf(N, dim, grf_mode, device, seed + 1)
            k2_raw = grf_k2.generate(n_test)

    x = torch.linspace(0, 1, N + 1, device=device)[:-1]
    y = torch.linspace(0, 1, N + 1, device=device)[:-1] if dim == 2 else None

    if dim == 1:
        a_func = lambda xi: 1.0
        if equation == "Poisson":
            pde = PoissonEquation1D(a_func=a_func, f_func=f_raw,
                                    boundary="Periodic", x=x, device=device)
        else:
            pde = HelmholtzEquation1D(a_func=a_func, f_func=f_raw, k2=k2_raw,
                                      boundary="Periodic", x=x, device=device)
        u_sol = pde.u.clone().detach().float() if isinstance(pde.u, torch.Tensor) else torch.tensor(pde.u, dtype=torch.float32, device=device)
        if needs_mean_zero:
            u_sol = u_sol - torch.mean(u_sol, dim=-1, keepdim=True)
        if equation == "Helmholtz":
            inputs = torch.cat([k2_raw[:, None, :], f_raw[:, None, :]], dim=1)
        else:
            inputs = f_raw[:, None, :]
        return inputs, u_sol, f_raw, x, pde, k2_raw
    else:
        f_flat = f_raw.reshape(n_test, N * N)
        a_func = lambda xi, yi: 1.0
        if equation == "Poisson":
            pde = PoissonEquation2D(a_func=a_func, f_func=f_flat,
                                    boundary="Periodic", x=x, y=y, device=device)
        elif equation == "ConvDiff":
            pde = ConvectionDiffusion2D(a_func=a_func, f_func=f_flat,
                                        b_vec=(b_vel, b_vel),
                                        boundary="Periodic", x=x, y=y, device=device,
                                        reaction=reaction_c)
        elif equation == "Reaction":
            pde = ReactionDiffusion2D(a_func=a_func, f_func=f_flat,
                                      reaction=reaction_c,
                                      boundary="Periodic", x=x, y=y, device=device)
        else:
            k2_flat = k2_raw.reshape(n_test, N * N)
            pde = HelmholtzEquation2D(a_func=a_func, f_func=f_flat, k2=k2_flat,
                                      boundary="Periodic", x=x, y=y, device=device)
        u_sol = pde.u.clone().detach().float() if isinstance(pde.u, torch.Tensor) else torch.tensor(pde.u, dtype=torch.float32, device=device)
        u_sol_flat = u_sol.reshape(n_test, N * N)
        if needs_mean_zero:
            u_sol_flat = u_sol_flat - torch.mean(u_sol_flat, dim=-1, keepdim=True)
        if equation == "Helmholtz":
            k2_flat = k2_raw.reshape(n_test, N * N)
            inputs = torch.cat([k2_flat[:, None, :], f_flat[:, None, :]], dim=1)
        else:
            inputs = f_flat[:, None, :]
        return inputs, u_sol_flat, f_flat, x, pde, k2_raw


def load_deeponet(ckp_dir, model_name, N, dim, equation, device):
    """Load the best DeepONet checkpoint."""
    args_path = os.path.join(ckp_dir, f"deeponet_{model_name}_{equation}_Periodic_{dim}d_1c_args.json")
    if os.path.exists(args_path):
        with open(args_path) as f:
            args = json.load(f)
    else:
        with open("args/deeponet_args.json") as f:
            args = json.load(f)

    in_channels = 2 if equation == "Helmholtz" else 1

    model = DeepONet(
        N=N, dim=dim, in_channels=in_channels, device=device, boundary="Periodic",
        branch_dim=args["branch_dim"],
        hidden_branch=args["hidden_branch"],
        num_branch_layers=args["num_branch_layers"],
        hidden_trunk=args["hidden_trunk"],
        num_trunk_layers=args["num_trunk_layers"],
    ).to(device)

    ckp_path = os.path.join(ckp_dir, f"deeponet_{model_name}_{equation}_Periodic_{dim}d_{1}c_best.pth")
    if not os.path.exists(ckp_path):
        ckp_path = os.path.join(ckp_dir, f"deeponet_{model_name}_{equation}_Periodic_{dim}d_{1}c_full.pth")
    if not os.path.exists(ckp_path):
        raise FileNotFoundError(f"No checkpoint found at {ckp_path}")

    ckp = torch.load(ckp_path, map_location=device)
    model.load_state_dict(ckp["model"])
    model.eval()
    print(f"Loaded DeepONet from {ckp_path} (epoch {ckp['epoch']})")
    return model


def load_lstm_router(ckp_dir, router_model_name, N, dim, equation, numerical_solvers_str, device):
    """Load a trained LSTM router checkpoint."""
    with open("args/lstm_args.json") as f:
        lstm_args = json.load(f)

    in_channels = 1
    new_in_channels = in_channels + 1 if equation == "Helmholtz" else in_channels
    num_solvers = len(numerical_solvers_str.split(",")) + 1  # numerical + ML

    if dim == 1:
        input_dim = N * (new_in_channels + 1)
    else:
        input_dim = N * N * (new_in_channels + 1)

    router = LSTMGreedyRouter(
        None, input_dim,
        lstm_args["hidden_dim"], lstm_args["num_layers"],
        num_solvers, lstm_args["dropout"],
    ).to(device)

    ckp_path = os.path.join(
        ckp_dir,
        f"lstmrouter_{router_model_name}_{equation}_Periodic_{dim}d_{in_channels}c_{numerical_solvers_str}_best.pth"
    )
    if not os.path.exists(ckp_path):
        ckp_path = ckp_path.replace("_best.pth", "_full.pth")
    if not os.path.exists(ckp_path):
        raise FileNotFoundError(f"No router checkpoint found at {ckp_path}")

    ckp = torch.load(ckp_path, map_location=device, weights_only=False)
    router.load_state_dict(ckp["model"])
    router.eval()
    print(f"Loaded LSTM router from {ckp_path} (epoch {ckp.get('epoch', '?')})")
    return router


def evaluate_deeponet(model, inputs, u_sol, device, equation="Poisson", reaction_c=0.0):
    """Compute per-sample L2 errors of the DeepONet predictions."""
    needs_mean_zero = equation == "Poisson" or (equation == "ConvDiff" and reaction_c == 0.0)
    with torch.no_grad():
        preds = model(inputs.to(device))
    preds = preds.reshape(preds.shape[0], -1)
    if needs_mean_zero:
        preds = preds - torch.mean(preds, dim=-1, keepdim=True)
    errors = torch.linalg.norm(preds - u_sol, dim=-1)
    rel_errors = errors / torch.linalg.norm(u_sol, dim=-1).clamp(min=1e-12)
    return preds, errors, rel_errors


def visualize_deeponet(x, u_sol, preds, N, dim, equation, n_show=4, save_dir="plots"):
    """Plot true solution vs DeepONet prediction for a few samples."""
    os.makedirs(save_dir, exist_ok=True)

    if dim == 1:
        xnp = x.cpu().numpy()
        fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 3.5), sharey=True)
        for i in range(n_show):
            ax = axes[i]
            ax.plot(xnp, u_sol[i].cpu().numpy(), label="True", linewidth=2)
            ax.plot(xnp, preds[i].cpu().numpy(), "--", label="DeepONet", linewidth=2)
            err = torch.linalg.norm(preds[i] - u_sol[i]).item()
            ax.set_title(f"Sample {i}  |  L2 err = {err:.4e}")
            ax.set_xlabel("x")
            if i == 0:
                ax.set_ylabel("u(x)")
                ax.legend(fontsize=8)
        fig.suptitle(f"DeepONet vs true ({dim}D {equation}, Periodic)")
    else:
        fig, axes = plt.subplots(3, n_show, figsize=(4 * n_show, 10))
        for i in range(n_show):
            u_true_2d = u_sol[i].cpu().reshape(N, N).numpy()
            u_pred_2d = preds[i].cpu().reshape(N, N).numpy()
            err_2d = u_pred_2d - u_true_2d
            vmin, vmax = u_true_2d.min(), u_true_2d.max()

            im0 = axes[0][i].imshow(u_true_2d, cmap="RdBu_r", vmin=vmin, vmax=vmax)
            axes[0][i].set_title(f"True (sample {i})")
            plt.colorbar(im0, ax=axes[0][i], fraction=0.046)

            im1 = axes[1][i].imshow(u_pred_2d, cmap="RdBu_r", vmin=vmin, vmax=vmax)
            axes[1][i].set_title("DeepONet")
            plt.colorbar(im1, ax=axes[1][i], fraction=0.046)

            im2 = axes[2][i].imshow(err_2d, cmap="RdBu_r")
            l2 = torch.linalg.norm(preds[i] - u_sol[i]).item()
            axes[2][i].set_title(f"Error  |  L2={l2:.4e}")
            plt.colorbar(im2, ax=axes[2][i], fraction=0.046)
        fig.suptitle(f"DeepONet vs true ({dim}D {equation}, Periodic)")

    fig.tight_layout()
    path = os.path.join(save_dir, "deeponet_predictions.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved DeepONet visualization to {path}")


def _build_deeponet_input(residual, k2_flat, equation):
    """Build DeepONet input: residual only for Poisson, [k2, residual] for Helmholtz.
    For Helmholtz, normalizes residual to unit norm and returns the norm for rescaling."""
    if equation == "Helmholtz" and k2_flat is not None:
        res_norms = torch.linalg.norm(residual, dim=-1, keepdim=True).clamp(min=1e-15)
        residual_normed = residual / res_norms
        inp = torch.cat([k2_flat[:, None, :], residual_normed[:, None, :]], dim=1)
        return inp, res_norms
    return residual[:, None, :], None


def run_iterative_comparison(deeponet, f_raw, u_sol, N, dim, device,
                             equation="Poisson", k2_flat=None,
                             lstm_router=None, b_vel=20.0,
                             reaction_c=0.0,
                             max_iters=300, tau=24, snapshot_steps=None):
    """
    Compare convergence of Jacobi only, HINTS, True Greedy, and optionally LSTM Router.
    Supports 1D/2D Poisson, Helmholtz, ConvDiff, and Reaction.
    """
    if snapshot_steps is None:
        snapshot_steps = []

    subtract_mean = equation == "Poisson" or (equation == "ConvDiff" and reaction_c == 0.0)
    n_test = f_raw.shape[0]
    x = torch.linspace(0, 1, N + 1, device=device)[:-1]

    if dim == 1:
        a_func = lambda xi: 1.0
        if equation == "Poisson":
            pde = PoissonEquation1D(a_func=a_func, f_func=f_raw,
                                    boundary="Periodic", x=x, A=None, solve=False, device=device)
        else:
            pde = HelmholtzEquation1D(a_func=a_func, f_func=f_raw, k2=k2_flat,
                                      boundary="Periodic", x=x, A=None, solve=False, device=device)
    else:
        y = torch.linspace(0, 1, N + 1, device=device)[:-1]
        a_func = lambda xi, yi: 1.0
        if equation == "Poisson":
            pde = PoissonEquation2D(a_func=a_func, f_func=f_raw,
                                    boundary="Periodic", x=x, y=y, A=None, solve=False, device=device)
        elif equation == "ConvDiff":
            pde = ConvectionDiffusion2D(a_func=a_func, f_func=f_raw,
                                        b_vec=(b_vel, b_vel),
                                        boundary="Periodic", x=x, y=y, A=None, solve=False, device=device,
                                        reaction=reaction_c)
        elif equation == "Reaction":
            pde = ReactionDiffusion2D(a_func=a_func, f_func=f_raw,
                                      reaction=reaction_c,
                                      boundary="Periodic", x=x, y=y, A=None, solve=False, device=device)
        else:
            pde = HelmholtzEquation2D(a_func=a_func, f_func=f_raw, k2=k2_flat,
                                      boundary="Periodic", x=x, y=y, A=None, solve=False, device=device)
    A = pde.A
    b = pde.b

    u_prev_jacobi = torch.zeros_like(u_sol)
    u_prev_hints = torch.zeros_like(u_sol)
    u_prev_greedy = torch.zeros_like(u_sol)

    errors_jacobi, errors_hints, errors_greedy = [], [], []
    errors_jacobi_ps, errors_hints_ps, errors_greedy_ps = [], [], []
    greedy_choices, greedy_choices_per_sample = [], []
    snapshots = {}

    # LSTM Router track
    run_router = lstm_router is not None
    u_prev_router = torch.zeros_like(u_sol) if run_router else None
    errors_router = [] if run_router else None
    errors_router_ps = [] if run_router else None
    router_choices = [] if run_router else None
    router_choices_per_sample = [] if run_router else None
    hidden_state = None

    jacobi_solver = WeightedJacobiSolver(equation=pde, device=device, weight=1.0)

    for t in range(max_iters):
        if t % 50 == 0:
            print(f"  Iteration {t}/{max_iters}")

        # --- Jacobi only ---
        u_new_jacobi = jacobi_solver.iteration(u_prev_jacobi)
        if subtract_mean:
            u_new_jacobi = u_new_jacobi - torch.mean(u_new_jacobi, dim=-1, keepdim=True)
        err_jacobi = torch.linalg.norm(u_new_jacobi - u_sol, dim=-1)
        errors_jacobi.append(err_jacobi.mean().item())
        errors_jacobi_ps.append(err_jacobi.cpu().numpy())
        u_prev_jacobi = u_new_jacobi

        # --- HINTS ---
        if (t + 1) % tau == 0:
            residual_hints = b - torch.bmm(A, u_prev_hints.unsqueeze(-1)).squeeze(-1)
            inp_hints, rnorm_hints = _build_deeponet_input(residual_hints, k2_flat, equation)
            with torch.no_grad():
                correction = deeponet(inp_hints).reshape(u_prev_hints.shape)
            if rnorm_hints is not None:
                correction = correction * rnorm_hints
            u_new_hints = u_prev_hints + correction
        else:
            u_new_hints = jacobi_solver.iteration(u_prev_hints)
        if subtract_mean:
            u_new_hints = u_new_hints - torch.mean(u_new_hints, dim=-1, keepdim=True)
        err_hints = torch.linalg.norm(u_new_hints - u_sol, dim=-1)
        errors_hints.append(err_hints.mean().item())
        errors_hints_ps.append(err_hints.cpu().numpy())
        u_prev_hints = u_new_hints

        # --- True greedy (oracle) ---
        u_jacobi_candidate = jacobi_solver.iteration(u_prev_greedy)
        if subtract_mean:
            u_jacobi_candidate = u_jacobi_candidate - torch.mean(u_jacobi_candidate, dim=-1, keepdim=True)

        residual_greedy = b - torch.bmm(A, u_prev_greedy.unsqueeze(-1)).squeeze(-1)
        inp_greedy, rnorm_greedy = _build_deeponet_input(residual_greedy, k2_flat, equation)
        with torch.no_grad():
            correction_greedy = deeponet(inp_greedy).reshape(u_prev_greedy.shape)
        if rnorm_greedy is not None:
            correction_greedy = correction_greedy * rnorm_greedy
        u_deeponet_candidate = u_prev_greedy + correction_greedy
        if subtract_mean:
            u_deeponet_candidate = u_deeponet_candidate - torch.mean(u_deeponet_candidate, dim=-1, keepdim=True)

        err_jac = torch.linalg.norm(u_jacobi_candidate - u_sol, dim=-1)
        err_don = torch.linalg.norm(u_deeponet_candidate - u_sol, dim=-1)
        use_deeponet = (err_don < err_jac)

        u_new_greedy = torch.where(use_deeponet.unsqueeze(-1), u_deeponet_candidate, u_jacobi_candidate)
        err_greedy = torch.linalg.norm(u_new_greedy - u_sol, dim=-1)
        errors_greedy.append(err_greedy.mean().item())
        errors_greedy_ps.append(err_greedy.cpu().numpy())
        greedy_choices.append(use_deeponet.float().mean().item())
        greedy_choices_per_sample.append(use_deeponet.cpu().numpy())
        u_prev_greedy = u_new_greedy

        # --- LSTM Router ---
        if run_router:
            residual_router = b - torch.bmm(A, u_prev_router.unsqueeze(-1)).squeeze(-1)
            inp_router, rnorm_router = _build_deeponet_input(residual_router, k2_flat, equation)
            recurrent_input = torch.cat((inp_router, u_prev_router.unsqueeze(1)), dim=1)
            bs = recurrent_input.shape[0]
            with torch.no_grad():
                chosen_solver, _, hidden_state = lstm_router.predict(
                    recurrent_input.reshape(bs, -1), hidden_state, with_scores=True)

            use_don_router = (chosen_solver == 1)

            u_jac_r = jacobi_solver.iteration(u_prev_router)
            with torch.no_grad():
                correction_r = deeponet(inp_router).reshape(u_prev_router.shape)
            if rnorm_router is not None:
                correction_r = correction_r * rnorm_router
            u_don_r = u_prev_router + correction_r

            if subtract_mean:
                u_jac_r = u_jac_r - torch.mean(u_jac_r, dim=-1, keepdim=True)
                u_don_r = u_don_r - torch.mean(u_don_r, dim=-1, keepdim=True)

            u_new_router = torch.where(use_don_router.unsqueeze(-1), u_don_r, u_jac_r)
            err_router = torch.linalg.norm(u_new_router - u_sol, dim=-1)
            errors_router.append(err_router.mean().item())
            errors_router_ps.append(err_router.cpu().numpy())
            router_choices.append(use_don_router.float().mean().item())
            router_choices_per_sample.append(use_don_router.cpu().numpy())
            u_prev_router = u_new_router

        step_num = t + 1
        if step_num in snapshot_steps:
            snapshots[step_num] = {
                "jacobi": u_prev_jacobi.cpu().clone(),
                "hints": u_prev_hints.cpu().clone(),
                "greedy": u_prev_greedy.cpu().clone(),
            }
            if run_router:
                snapshots[step_num]["router"] = u_prev_router.cpu().clone()

    return (
        np.array(errors_jacobi),
        np.array(errors_hints),
        np.array(errors_greedy),
        np.array(greedy_choices),
        np.array(greedy_choices_per_sample),
        snapshots,
        np.array(errors_router) if run_router else None,
        np.array(router_choices) if run_router else None,
        np.array(router_choices_per_sample) if run_router else None,
        np.array(errors_jacobi_ps),
        np.array(errors_hints_ps),
        np.array(errors_greedy_ps),
        np.array(errors_router_ps) if run_router else None,
    )


def visualize_greedy_routing(greedy_choices_per_sample, save_dir="plots",
                             label="Oracle Greedy", filename="greedy_routing_pattern.png"):
    """
    Visualize per-sample routing decisions.
    greedy_choices_per_sample: (max_iters, n_test) bool array, True = DeepONet chosen
    """
    os.makedirs(save_dir, exist_ok=True)
    choices = greedy_choices_per_sample  # (T, n_test)
    T, n_test = choices.shape

    deeponet_counts = choices.sum(axis=0)
    sort_idx = np.argsort(-deeponet_counts)
    choices_sorted = choices[:, sort_idx]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10),
                             gridspec_kw={"height_ratios": [2, 1], "width_ratios": [3, 1]})

    ax_heat = axes[0][0]
    ax_heat.imshow(choices_sorted.T, aspect="auto", cmap="Greens",
                   interpolation="nearest", extent=[1, T, n_test - 0.5, -0.5])
    ax_heat.set_xlabel("Iteration")
    ax_heat.set_ylabel("Sample (sorted by DeepONet usage)")
    ax_heat.set_title(f"{label} routing: green = DeepONet, white = Jacobi")

    # --- 2. Bar chart: DeepONet calls per sample ---
    ax_bar = axes[0][1]
    ax_bar.barh(np.arange(n_test), deeponet_counts[sort_idx], color="tab:green", alpha=0.7)
    ax_bar.set_xlabel("# DeepONet calls")
    ax_bar.set_ylabel("Sample")
    ax_bar.set_ylim(n_test - 0.5, -0.5)
    ax_bar.set_title("Total DeepONet calls")

    # --- 3. Aggregate: fraction using DeepONet at each step ---
    ax_frac = axes[1][0]
    frac_per_step = choices.mean(axis=1)
    ax_frac.bar(np.arange(1, T + 1), frac_per_step, color="tab:green", alpha=0.7, width=1.0)
    ax_frac.set_xlabel("Iteration")
    ax_frac.set_ylabel("Fraction of samples\nchoosing DeepONet")
    ax_frac.set_title("DeepONet selection rate per iteration")
    ax_frac.set_xlim(0.5, T + 0.5)
    ax_frac.set_ylim(0, 1.05)

    # --- 4. Histogram: total DeepONet calls distribution ---
    ax_hist = axes[1][1]
    ax_hist.hist(deeponet_counts, bins=np.arange(-0.5, deeponet_counts.max() + 1.5, 1),
                 color="tab:green", alpha=0.7, edgecolor="white")
    ax_hist.set_xlabel("# DeepONet calls")
    ax_hist.set_ylabel("# Samples")
    ax_hist.set_title("Distribution of DeepONet usage")

    fig.tight_layout()
    path = os.path.join(save_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved routing pattern to {path}")

    print(f"\n  Per-sample DeepONet usage ({label}):")
    print(f"    Min calls:    {int(deeponet_counts.min())}")
    print(f"    Max calls:    {int(deeponet_counts.max())}")
    print(f"    Mean calls:   {deeponet_counts.mean():.1f}")
    print(f"    Median calls: {np.median(deeponet_counts):.0f}")
    print(f"    Std calls:    {deeponet_counts.std():.1f}")
    unique, counts = np.unique(deeponet_counts.astype(int), return_counts=True)
    print(f"    Distribution: {dict(zip(unique, counts))}")

    steps_with_deeponet = np.where(choices.any(axis=1))[0] + 1
    print(f"    Iterations where DeepONet was chosen by >= 1 sample: {steps_with_deeponet.tolist()}")


def visualize_solution_snapshots(x, u_sol, snapshots, snapshot_steps,
                                 sample_indices=None, save_dir="plots"):
    """
    For each sample, plot true solution vs iterates from each method at each snapshot step.
    Rows = snapshot steps, Columns = samples.
    """
    os.makedirs(save_dir, exist_ok=True)
    xnp = x.cpu().numpy()

    if sample_indices is None:
        sample_indices = [0, 1, 2, 3]

    n_steps = len(snapshot_steps)
    n_samples = len(sample_indices)

    fig, axes = plt.subplots(n_steps, n_samples, figsize=(4.5 * n_samples, 3.2 * n_steps),
                             squeeze=False)

    for row, step in enumerate(snapshot_steps):
        snap = snapshots[step]
        for col, si in enumerate(sample_indices):
            ax = axes[row][col]
            u_true = u_sol[si].cpu().numpy()
            ax.plot(xnp, u_true, "k-", label="True", linewidth=2, alpha=0.8)
            ax.plot(xnp, snap["jacobi"][si].numpy(), "--", label="Jacobi",
                    linewidth=1.5, color="tab:blue")
            ax.plot(xnp, snap["hints"][si].numpy(), "--", label="HINTS",
                    linewidth=1.5, color="tab:orange")
            ax.plot(xnp, snap["greedy"][si].numpy(), "--", label="Greedy",
                    linewidth=1.5, color="tab:green")

            if row == 0:
                ax.set_title(f"Sample {si}", fontsize=11)
            if col == 0:
                ax.set_ylabel(f"Step {step}", fontsize=11)
            if row == n_steps - 1:
                ax.set_xlabel("x")
            if row == 0 and col == n_samples - 1:
                ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Solution iterates: True vs Jacobi vs HINTS vs True Greedy",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    path = os.path.join(save_dir, "solution_snapshots.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved solution snapshots to {path}")

    # Also plot the error fields (iterate - true) for each method
    fig2, axes2 = plt.subplots(n_steps, n_samples, figsize=(4.5 * n_samples, 3.2 * n_steps),
                               squeeze=False)
    for row, step in enumerate(snapshot_steps):
        snap = snapshots[step]
        for col, si in enumerate(sample_indices):
            ax = axes2[row][col]
            u_true = u_sol[si].cpu().numpy()
            ax.plot(xnp, snap["jacobi"][si].numpy() - u_true, label="Jacobi",
                    linewidth=1.5, color="tab:blue")
            ax.plot(xnp, snap["hints"][si].numpy() - u_true, label="HINTS",
                    linewidth=1.5, color="tab:orange")
            ax.plot(xnp, snap["greedy"][si].numpy() - u_true, label="Greedy",
                    linewidth=1.5, color="tab:green")
            ax.axhline(0, color="k", linewidth=0.5, alpha=0.5)

            if row == 0:
                ax.set_title(f"Sample {si}", fontsize=11)
            if col == 0:
                ax.set_ylabel(f"Step {step}\nerror", fontsize=11)
            if row == n_steps - 1:
                ax.set_xlabel("x")
            if row == 0 and col == n_samples - 1:
                ax.legend(fontsize=7, loc="upper right")

    fig2.suptitle("Error fields (estimate − true) at each iteration",
                  fontsize=13, y=1.01)
    fig2.tight_layout()
    path2 = os.path.join(save_dir, "error_snapshots.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved error snapshots to {path2}")


def plot_convergence(errors_jacobi, errors_hints, errors_greedy, greedy_choices,
                     max_iters, title_suffix="", save_dir="plots",
                     errors_router=None, router_choices=None):
    """Plot convergence histories and greedy routing fractions."""
    os.makedirs(save_dir, exist_ok=True)
    iters = np.arange(1, max_iters + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.semilogy(iters, errors_jacobi, label="Jacobi Only", linewidth=2)
    ax1.semilogy(iters, errors_hints, label="HINTS (Jacobi + DeepONet)", linewidth=2)
    ax1.semilogy(iters, errors_greedy, label="True Greedy (oracle)", linewidth=2, color="green")
    if errors_router is not None:
        ax1.semilogy(iters, errors_router, label="LSTM Router (learned)", linewidth=2,
                     color="red", linestyle="--")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Mean L2 Error")
    ax1.set_title(f"Convergence (Periodic){title_suffix}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(iters, greedy_choices, color="green", alpha=0.7, label="Oracle Greedy")
    if router_choices is not None:
        ax2.plot(iters, router_choices, color="red", alpha=0.7, linestyle="--",
                 label="LSTM Router")
        ax2.legend()
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Fraction choosing DeepONet")
    ax2.set_title("DeepONet selection rate")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(save_dir, "convergence_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved convergence plot to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="test_v1")
    parser.add_argument("--ckp_dir", type=str, default="./checkpoints")
    parser.add_argument("--n_test", type=int, default=64)
    parser.add_argument("--max_iters", type=int, default=300)
    parser.add_argument("--tau", type=int, default=24, help="HINTS period")
    parser.add_argument("--save_dir", type=str, default="./plots")
    parser.add_argument("--dim", type=int, default=1, choices=[1, 2])
    parser.add_argument("--equation", type=str, default="Poisson",
                        choices=["Poisson", "Helmholtz", "ConvDiff", "Reaction"])
    parser.add_argument("--b_vel", type=float, default=20.0,
                        help="Advection velocity for ConvDiff (b_vec=(b_vel,b_vel))")
    parser.add_argument("--reaction_c", type=float, default=0.0,
                        help="Reaction coefficient (used by Reaction and optionally ConvDiff equations)")
    parser.add_argument("--grf_mode", type=str, default="fixed",
                        choices=["fixed", "hierarchical"],
                        help="GRF mode: 'fixed' (single PSD) or 'hierarchical' (varied PSDs)")
    parser.add_argument("--k2_mode", type=str, default="exp", choices=["exp", "mild", "const"],
                        help="Helmholtz k2 pushforward: 'exp', 'mild', or 'const'")
    parser.add_argument("--router_model_name", type=str, default=None,
                        help="LSTM router checkpoint name (if provided, also runs router comparison)")
    parser.add_argument("--numerical_solvers", type=str, default="jacobi",
                        help="Numerical solvers string for router checkpoint path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Config: dim={args.dim}, equation={args.equation}, grf_mode={args.grf_mode}")

    save_dir = os.path.join(args.save_dir, f"{args.dim}d_{args.equation.lower()}_{args.grf_mode}")
    os.makedirs(save_dir, exist_ok=True)

    with open("args/deeponet_args.json") as f:
        N = json.load(f)["N"]

    # 1. Generate test data
    print("\n=== Step 1: Generate test data ===")
    inputs, u_sol, f_raw, x, pde, k2_raw = generate_test_data(
        args.n_test, N, args.dim, device,
        equation=args.equation, grf_mode=args.grf_mode, k2_mode=args.k2_mode,
        b_vel=args.b_vel, reaction_c=args.reaction_c)
    print(f"  Generated {args.n_test} test samples, grid size N={N}")
    print(f"  u_sol shape: {u_sol.shape}, f shape: {f_raw.shape}")
    if k2_raw is not None:
        print(f"  k2 shape: {k2_raw.shape}")

    # For 2D iterative comparison, f_raw must be flat (n_test, N*N)
    if args.dim == 2:
        f_raw_flat = f_raw.reshape(args.n_test, N * N) if f_raw.dim() > 2 else f_raw
        k2_flat = k2_raw.reshape(args.n_test, N * N) if k2_raw is not None and k2_raw.dim() > 2 else k2_raw
    else:
        f_raw_flat = f_raw
        k2_flat = k2_raw

    # 2. Load DeepONet
    print("\n=== Step 2: Load trained DeepONet ===")
    deeponet = load_deeponet(args.ckp_dir, args.model_name, N, args.dim, args.equation, device)

    # 3. Evaluate DeepONet
    print("\n=== Step 3: Evaluate DeepONet ===")
    preds, errors, rel_errors = evaluate_deeponet(deeponet, inputs, u_sol, device, equation=args.equation, reaction_c=args.reaction_c)
    print(f"  Mean L2 error:     {errors.mean().item():.6e} ± {errors.std().item():.6e}")
    print(f"  Mean relative L2:  {rel_errors.mean().item():.6e} ± {rel_errors.std().item():.6e}")
    print(f"  Max L2 error:      {errors.max().item():.6e}")

    # 4. Visualize DeepONet predictions
    print("\n=== Step 4: Visualize DeepONet predictions ===")
    visualize_deeponet(x, u_sol, preds, N, args.dim, args.equation, n_show=4, save_dir=save_dir)

    # 5. Optionally load LSTM router
    lstm_router = None
    if args.router_model_name:
        print("\n=== Step 5a: Load LSTM Router ===")
        lstm_router = load_lstm_router(
            args.ckp_dir, args.router_model_name, N, args.dim,
            args.equation, args.numerical_solvers, device)

    # 6. Run iterative comparison
    print("\n=== Step 5: Iterative solver comparison ===")
    snapshot_steps = [1, 5, 10, 25, 50, 75, 100, 150, 200, 300]
    snapshot_steps = [s for s in snapshot_steps if s <= args.max_iters]
    strategies = "Jacobi, HINTS, True Greedy"
    if lstm_router:
        strategies += ", LSTM Router"
    print(f"  Running {args.max_iters} iterations of {strategies}...")
    print(f"  Capturing snapshots at steps: {snapshot_steps}")
    result = run_iterative_comparison(
        deeponet, f_raw_flat, u_sol, N, args.dim, device,
        equation=args.equation, k2_flat=k2_flat,
        lstm_router=lstm_router, b_vel=args.b_vel,
        reaction_c=args.reaction_c,
        max_iters=args.max_iters, tau=args.tau,
        snapshot_steps=snapshot_steps,
    )
    (errors_jacobi, errors_hints, errors_greedy, greedy_choices, greedy_choices_per_sample,
     snapshots, errors_router, router_choices, router_choices_per_sample,
     errors_jacobi_ps, errors_hints_ps, errors_greedy_ps, errors_router_ps) = result

    print(f"\n  Final errors after {args.max_iters} iterations:")
    print(f"    Jacobi only:    {errors_jacobi[-1]:.6e}")
    print(f"    HINTS:          {errors_hints[-1]:.6e}")
    print(f"    True Greedy:    {errors_greedy[-1]:.6e}")
    if errors_router is not None:
        print(f"    LSTM Router:    {errors_router[-1]:.6e}")
    print(f"\n  AUC (sum of errors, lower = better):")
    print(f"    Jacobi only:    {errors_jacobi.sum():.6e}")
    print(f"    HINTS:          {errors_hints.sum():.6e}")
    print(f"    True Greedy:    {errors_greedy.sum():.6e}")
    if errors_router is not None:
        print(f"    LSTM Router:    {errors_router.sum():.6e}")
    print(f"\n  Greedy DeepONet usage: {np.mean(greedy_choices):.2%} of steps (avg across samples)")
    if router_choices is not None:
        print(f"  Router DeepONet usage: {np.mean(router_choices):.2%} of steps (avg across samples)")

    # 6. Visualize routing patterns
    print("\n=== Step 6: Routing patterns ===")
    visualize_greedy_routing(greedy_choices_per_sample, save_dir=save_dir,
                             label="Oracle Greedy", filename="greedy_routing_pattern.png")
    if router_choices_per_sample is not None:
        visualize_greedy_routing(router_choices_per_sample, save_dir=save_dir,
                                 label="LSTM Router", filename="router_routing_pattern.png")

    # 7. Visualize solution snapshots (1D only for now)
    if args.dim == 1:
        print("\n=== Step 7: Visualize solution snapshots ===")
        visualize_solution_snapshots(x, u_sol, snapshots, snapshot_steps,
                                     sample_indices=[0, 1, 2, 3], save_dir=save_dir)

    # 8. Plot convergence
    print("\n=== Step 8: Plot convergence ===")
    title_suffix = f": {args.dim}D {args.equation}"
    plot_convergence(errors_jacobi, errors_hints, errors_greedy, greedy_choices,
                     args.max_iters, title_suffix=title_suffix, save_dir=save_dir,
                     errors_router=errors_router, router_choices=router_choices)

    print("\nDone! Check the plots/ directory for visualizations.")


if __name__ == "__main__":
    main()
