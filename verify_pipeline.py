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
from pde import PoissonEquation1D, PoissonEquation2D, HelmholtzEquation1D, HelmholtzEquation2D, ConvectionDiffusion2D
from ml_solver import DeepONet, FNOforPDE
from numerical_solver import WeightedJacobiSolver, GaussSeidelSolver, SORSolver, MultigridSolver
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


def load_fno(ckp_dir, model_name, N, dim, equation, device, args_file=None):
    """Load a trained FNO checkpoint."""
    args_path = os.path.join(ckp_dir, f"fno_{model_name}_{equation}_Periodic_{dim}d_1c_args.json")
    if os.path.exists(args_path):
        with open(args_path) as f:
            args = json.load(f)
    else:
        default = args_file or "args/fno_n31_args.json"
        with open(default) as f:
            args = json.load(f)

    in_channels = 2 if equation == "Helmholtz" else 1

    model = FNOforPDE(
        trunc_mode=args["trunc_mode"], dim=dim, in_channels=in_channels,
        hidden_size=args["hidden_size"], num_layers=args["num_layers"],
    ).to(device)

    ckp_path = os.path.join(ckp_dir, f"fno_{model_name}_{equation}_Periodic_{dim}d_1c_best.pth")
    if not os.path.exists(ckp_path):
        ckp_path = os.path.join(ckp_dir, f"fno_{model_name}_{equation}_Periodic_{dim}d_1c_full.pth")
    if not os.path.exists(ckp_path):
        raise FileNotFoundError(f"No FNO checkpoint found at {ckp_path}")

    ckp = torch.load(ckp_path, map_location=device, weights_only=False)
    state = ckp["model"]
    state = {k: v for k, v in state.items() if not k.startswith("_")}
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded FNO from {ckp_path} (epoch {ckp['epoch']})")
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


def _make_classical_solver(solver_spec, pde, device):
    """Instantiate one classical solver from a spec string like 'jacobi', 'jacobi_0.67', 'gs', 'sor_1.5', 'mg_3'."""
    parts = solver_spec.split("_")
    name = parts[0]
    if name == "jacobi":
        weight = float(parts[1]) if len(parts) > 1 else 1.0
        return WeightedJacobiSolver(equation=pde, device=device, weight=weight)
    elif name == "gs":
        return GaussSeidelSolver(equation=pde, device=device)
    elif name == "sor":
        omega = float(parts[1]) if len(parts) > 1 else 1.5
        return SORSolver(equation=pde, omega=omega, device=device)
    elif name == "mg":
        levels = int(parts[1]) if len(parts) > 1 else 3
        return MultigridSolver(equation=pde, levels=levels, device=device)
    else:
        raise ValueError(f"Unknown solver spec: {solver_spec}")


def _solver_display_name(spec):
    """Human-readable name for a solver spec."""
    parts = spec.split("_")
    name = parts[0]
    if name == "jacobi":
        w = parts[1] if len(parts) > 1 else "1"
        return f"Jacobi(w={w})"
    elif name == "gs":
        return "Gauss-Seidel"
    elif name == "sor":
        w = parts[1] if len(parts) > 1 else "1.5"
        return f"SOR(w={w})"
    elif name == "mg":
        return "Multigrid"
    return spec


def _build_pde(f_raw, N, dim, device, equation, k2_flat, b_vel, reaction_c):
    """Build PDE object for iterative comparison (no solve)."""
    x = torch.linspace(0, 1, N + 1, device=device)[:-1]
    if dim == 1:
        a_func = lambda xi: 1.0
        if equation == "Poisson":
            return PoissonEquation1D(a_func=a_func, f_func=f_raw,
                                     boundary="Periodic", x=x, A=None, solve=False, device=device)
        else:
            return HelmholtzEquation1D(a_func=a_func, f_func=f_raw, k2=k2_flat,
                                       boundary="Periodic", x=x, A=None, solve=False, device=device)
    else:
        y = torch.linspace(0, 1, N + 1, device=device)[:-1]
        a_func = lambda xi, yi: 1.0
        if equation == "Poisson":
            return PoissonEquation2D(a_func=a_func, f_func=f_raw,
                                     boundary="Periodic", x=x, y=y, A=None, solve=False, device=device)
        elif equation == "ConvDiff":
            return ConvectionDiffusion2D(a_func=a_func, f_func=f_raw,
                                         b_vec=(b_vel, b_vel),
                                         boundary="Periodic", x=x, y=y, A=None, solve=False, device=device,
                                         reaction=reaction_c)
        else:
            return HelmholtzEquation2D(a_func=a_func, f_func=f_raw, k2=k2_flat,
                                       boundary="Periodic", x=x, y=y, A=None, solve=False, device=device)


def run_iterative_comparison(deeponet, f_raw, u_sol, N, dim, device,
                             equation="Poisson", k2_flat=None,
                             lstm_router=None, b_vel=20.0,
                             reaction_c=0.0,
                             max_iters=300, tau=24, snapshot_steps=None,
                             solver_type="jacobi"):
    """
    Compare convergence of classical-only, HINTS, True Greedy, and optionally LSTM Router.

    solver_type can be a single spec ('jacobi', 'gs', 'mg_3') or a comma-separated
    list for multi-solver routing ('jacobi_0.67,jacobi,gs').  When multiple solvers
    are specified, the oracle and router choose among all K classical solvers + DeepONet.
    The classical-only and HINTS baselines always use the first solver in the list.
    """
    if snapshot_steps is None:
        snapshot_steps = []

    subtract_mean = equation == "Poisson" or (equation == "ConvDiff" and reaction_c == 0.0)
    n_test = f_raw.shape[0]

    pde = _build_pde(f_raw, N, dim, device, equation, k2_flat, b_vel, reaction_c)
    A = pde.A
    b = pde.b

    # Build solver list
    solver_spec_list = [s.strip() for s in solver_type.split(",")]
    solver_names = [_solver_display_name(s) for s in solver_spec_list] + ["DeepONet"]
    classical_solvers = [_make_classical_solver(s, pde, device) for s in solver_spec_list]
    K = len(classical_solvers) + 1   # total solvers including DeepONet
    ml_idx = len(classical_solvers)   # DeepONet index

    # Per-solver baselines: each classical solver run independently
    n_classical = len(classical_solvers)
    u_prev_baselines = [torch.zeros_like(u_sol) for _ in range(n_classical)]
    errors_baselines = [[] for _ in range(n_classical)]
    errors_baselines_ps = [[] for _ in range(n_classical)]

    u_prev_hints = torch.zeros_like(u_sol)
    u_prev_greedy = torch.zeros_like(u_sol)

    errors_hints, errors_greedy = [], []
    errors_hints_ps, errors_greedy_ps = [], []
    greedy_choices, greedy_choices_per_sample = [], []
    snapshots = {}

    run_router = lstm_router is not None
    u_prev_router = torch.zeros_like(u_sol) if run_router else None
    errors_router = [] if run_router else None
    errors_router_ps = [] if run_router else None
    router_choices = [] if run_router else None
    router_choices_per_sample = [] if run_router else None
    hidden_state = None

    primary_solver = classical_solvers[0]

    for t in range(max_iters):
        if t % 50 == 0:
            print(f"  Iteration {t}/{max_iters}")

        # --- Classical-only baselines (one per solver) ---
        for si, solver in enumerate(classical_solvers):
            u_new_bl = solver.iteration(u_prev_baselines[si])
            if subtract_mean:
                u_new_bl = u_new_bl - torch.mean(u_new_bl, dim=-1, keepdim=True)
            err_bl = torch.linalg.norm(u_new_bl - u_sol, dim=-1)
            errors_baselines[si].append(err_bl.mean().item())
            errors_baselines_ps[si].append(err_bl.cpu().numpy())
            u_prev_baselines[si] = u_new_bl

        # --- HINTS (first solver + periodic DeepONet) ---
        if (t + 1) % tau == 0:
            residual_hints = b - torch.bmm(A, u_prev_hints.unsqueeze(-1)).squeeze(-1)
            inp_hints, rnorm_hints = _build_deeponet_input(residual_hints, k2_flat, equation)
            with torch.no_grad():
                correction = deeponet(inp_hints).reshape(u_prev_hints.shape)
            if rnorm_hints is not None:
                correction = correction * rnorm_hints
            u_new_hints = u_prev_hints + correction
        else:
            u_new_hints = primary_solver.iteration(u_prev_hints)
        if subtract_mean:
            u_new_hints = u_new_hints - torch.mean(u_new_hints, dim=-1, keepdim=True)
        err_hints = torch.linalg.norm(u_new_hints - u_sol, dim=-1)
        errors_hints.append(err_hints.mean().item())
        errors_hints_ps.append(err_hints.cpu().numpy())
        u_prev_hints = u_new_hints

        # --- True greedy oracle (all K solvers) ---
        candidates = []
        candidate_errors = []
        for solver in classical_solvers:
            u_cand = solver.iteration(u_prev_greedy)
            if subtract_mean:
                u_cand = u_cand - torch.mean(u_cand, dim=-1, keepdim=True)
            candidates.append(u_cand)
            candidate_errors.append(torch.linalg.norm(u_cand - u_sol, dim=-1))

        residual_greedy = b - torch.bmm(A, u_prev_greedy.unsqueeze(-1)).squeeze(-1)
        inp_greedy, rnorm_greedy = _build_deeponet_input(residual_greedy, k2_flat, equation)
        with torch.no_grad():
            correction_greedy = deeponet(inp_greedy).reshape(u_prev_greedy.shape)
        if rnorm_greedy is not None:
            correction_greedy = correction_greedy * rnorm_greedy
        u_don_cand = u_prev_greedy + correction_greedy
        if subtract_mean:
            u_don_cand = u_don_cand - torch.mean(u_don_cand, dim=-1, keepdim=True)
        candidates.append(u_don_cand)
        candidate_errors.append(torch.linalg.norm(u_don_cand - u_sol, dim=-1))

        all_candidates = torch.stack(candidates, dim=0)       # (K, B, N^2)
        all_errors = torch.stack(candidate_errors, dim=0)     # (K, B)
        best_solver = torch.argmin(all_errors, dim=0)         # (B,)
        u_new_greedy = all_candidates[best_solver, torch.arange(n_test)]

        err_greedy = torch.linalg.norm(u_new_greedy - u_sol, dim=-1)
        errors_greedy.append(err_greedy.mean().item())
        errors_greedy_ps.append(err_greedy.cpu().numpy())
        greedy_choices.append((best_solver == ml_idx).float().mean().item())
        greedy_choices_per_sample.append(best_solver.cpu().numpy())
        u_prev_greedy = u_new_greedy

        # --- LSTM Router (all K solvers) ---
        if run_router:
            residual_router = b - torch.bmm(A, u_prev_router.unsqueeze(-1)).squeeze(-1)
            inp_router, rnorm_router = _build_deeponet_input(residual_router, k2_flat, equation)
            recurrent_input = torch.cat((inp_router, u_prev_router.unsqueeze(1)), dim=1)
            bs = recurrent_input.shape[0]
            with torch.no_grad():
                chosen_solver, _, hidden_state = lstm_router.predict(
                    recurrent_input.reshape(bs, -1), hidden_state, with_scores=True)

            u_new_router = torch.zeros_like(u_prev_router)
            for i, solver in enumerate(classical_solvers):
                mask_i = (chosen_solver == i)
                if mask_i.any():
                    u_cand_r = solver.iteration(u_prev_router)
                    if subtract_mean:
                        u_cand_r = u_cand_r - torch.mean(u_cand_r, dim=-1, keepdim=True)
                    u_new_router[mask_i] = u_cand_r[mask_i]

            mask_ml = (chosen_solver == ml_idx)
            if mask_ml.any():
                with torch.no_grad():
                    correction_r = deeponet(inp_router).reshape(u_prev_router.shape)
                if rnorm_router is not None:
                    correction_r = correction_r * rnorm_router
                u_don_r = u_prev_router + correction_r
                if subtract_mean:
                    u_don_r = u_don_r - torch.mean(u_don_r, dim=-1, keepdim=True)
                u_new_router[mask_ml] = u_don_r[mask_ml]

            err_router = torch.linalg.norm(u_new_router - u_sol, dim=-1)
            errors_router.append(err_router.mean().item())
            errors_router_ps.append(err_router.cpu().numpy())
            router_choices.append((chosen_solver == ml_idx).float().mean().item())
            router_choices_per_sample.append(chosen_solver.cpu().numpy())
            u_prev_router = u_new_router

        step_num = t + 1
        if step_num in snapshot_steps:
            snapshots[step_num] = {
                "baseline0": u_prev_baselines[0].cpu().clone(),
                "hints": u_prev_hints.cpu().clone(),
                "greedy": u_prev_greedy.cpu().clone(),
            }
            if run_router:
                snapshots[step_num]["router"] = u_prev_router.cpu().clone()

    return {
        "errors_baselines": [np.array(e) for e in errors_baselines],
        "errors_baselines_ps": [np.array(e) for e in errors_baselines_ps],
        "errors_hints": np.array(errors_hints),
        "errors_greedy": np.array(errors_greedy),
        "greedy_choices": np.array(greedy_choices),
        "greedy_choices_per_sample": np.array(greedy_choices_per_sample),
        "snapshots": snapshots,
        "errors_router": np.array(errors_router) if run_router else None,
        "router_choices": np.array(router_choices) if run_router else None,
        "router_choices_per_sample": np.array(router_choices_per_sample) if run_router else None,
        "errors_hints_ps": np.array(errors_hints_ps),
        "errors_greedy_ps": np.array(errors_greedy_ps),
        "errors_router_ps": np.array(errors_router_ps) if run_router else None,
        "solver_names": solver_names,
        "baseline_names": [_solver_display_name(s) for s in solver_spec_list],
    }


def visualize_greedy_routing(greedy_choices_per_sample, save_dir="plots",
                             label="Oracle Greedy", filename="greedy_routing_pattern.png",
                             solver_names=None):
    """
    Visualize per-sample routing decisions.
    greedy_choices_per_sample: (max_iters, n_test) int array of solver indices.
    solver_names: list of solver names corresponding to indices 0..K-1.
    """
    os.makedirs(save_dir, exist_ok=True)
    choices = greedy_choices_per_sample.astype(int)  # (T, n_test)
    T, n_test = choices.shape
    K = int(choices.max()) + 1
    ml_idx = K - 1

    if solver_names is None:
        solver_names = [f"Solver {i}" for i in range(K - 1)] + ["DeepONet"]

    deeponet_counts = (choices == ml_idx).sum(axis=0)
    sort_idx = np.argsort(-deeponet_counts)
    choices_sorted = choices[:, sort_idx]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10),
                             gridspec_kw={"height_ratios": [2, 1], "width_ratios": [3, 1]})

    # --- 1. Heatmap: K-way routing ---
    from matplotlib.colors import ListedColormap, BoundaryNorm
    base_colors = plt.cm.tab10.colors
    cmap_colors = [base_colors[i % 10] for i in range(K)]
    cmap = ListedColormap(cmap_colors[:K])
    bounds = np.arange(-0.5, K + 0.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    ax_heat = axes[0][0]
    im = ax_heat.imshow(choices_sorted.T, aspect="auto", cmap=cmap, norm=norm,
                        interpolation="nearest", extent=[1, T, n_test - 0.5, -0.5])
    ax_heat.set_xlabel("Iteration")
    ax_heat.set_ylabel("Sample (sorted by DeepONet usage)")
    ax_heat.set_title(f"{label} routing")
    cbar = fig.colorbar(im, ax=ax_heat, ticks=np.arange(K), fraction=0.03)
    cbar.ax.set_yticklabels(solver_names[:K], fontsize=8)

    # --- 2. Bar chart: DeepONet calls per sample ---
    ax_bar = axes[0][1]
    ax_bar.barh(np.arange(n_test), deeponet_counts[sort_idx], color=cmap_colors[ml_idx], alpha=0.7)
    ax_bar.set_xlabel("# DeepONet calls")
    ax_bar.set_ylabel("Sample")
    ax_bar.set_ylim(n_test - 0.5, -0.5)
    ax_bar.set_title("Total DeepONet calls")

    # --- 3. Per-solver fraction over time ---
    ax_frac = axes[1][0]
    iters = np.arange(1, T + 1)
    for k in range(K):
        frac_k = (choices == k).mean(axis=1)
        ax_frac.plot(iters, frac_k, color=cmap_colors[k], alpha=0.8,
                     label=solver_names[k], linewidth=1.5)
    ax_frac.set_xlabel("Iteration")
    ax_frac.set_ylabel("Fraction of samples")
    ax_frac.set_title("Solver selection rate per iteration")
    ax_frac.set_xlim(0.5, T + 0.5)
    ax_frac.set_ylim(-0.05, 1.05)
    ax_frac.legend(fontsize=7, loc="upper right")

    # --- 4. Histogram: total DeepONet calls distribution ---
    ax_hist = axes[1][1]
    max_calls = max(int(deeponet_counts.max()), 1)
    ax_hist.hist(deeponet_counts, bins=np.arange(-0.5, max_calls + 1.5, 1),
                 color=cmap_colors[ml_idx], alpha=0.7, edgecolor="white")
    ax_hist.set_xlabel("# DeepONet calls")
    ax_hist.set_ylabel("# Samples")
    ax_hist.set_title("Distribution of DeepONet usage")

    fig.tight_layout()
    path = os.path.join(save_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved routing pattern to {path}")

    print(f"\n  Per-sample solver usage ({label}):")
    for k in range(K):
        k_counts = (choices == k).sum(axis=0)
        print(f"    {solver_names[k]:20s}: mean={k_counts.mean():.1f}, "
              f"min={int(k_counts.min())}, max={int(k_counts.max())}")
    print(f"    DeepONet fraction: {(choices == ml_idx).mean():.4f}")


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


def plot_convergence(errors_baselines, errors_hints, errors_greedy, greedy_choices,
                     max_iters, title_suffix="", save_dir="plots",
                     errors_router=None, router_choices=None,
                     solver_label="Jacobi",
                     greedy_choices_per_sample=None, router_choices_per_sample=None,
                     solver_names=None, baseline_names=None,
                     max_plot_iters=None, no_hints=False):
    """Plot convergence histories and routing fractions.

    errors_baselines: list of arrays, one per classical solver baseline.
    baseline_names: display name for each baseline curve.
    """
    os.makedirs(save_dir, exist_ok=True)
    T = max_plot_iters if max_plot_iters is not None else max_iters
    iters = np.arange(1, T + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    baseline_colors = plt.cm.Set2.colors
    if baseline_names is None:
        baseline_names = [solver_label]
    for bi, (errs, bname) in enumerate(zip(errors_baselines, baseline_names)):
        ax1.semilogy(iters, errs[:T], label=f"{bname} Only",
                     linewidth=1.5, color=baseline_colors[bi % len(baseline_colors)],
                     alpha=0.8)
    if not no_hints:
        ax1.semilogy(iters, errors_hints[:T], label=f"HINTS ({baseline_names[0]} + ML)", linewidth=1.5,
                     color="orange", linestyle="--")
    ax1.semilogy(iters, errors_greedy[:T], label="True Greedy (oracle)", linewidth=2.5, color="green")
    if errors_router is not None:
        ax1.semilogy(iters, errors_router[:T], label="LSTM Router (learned)", linewidth=2,
                     color="red", linestyle="--")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Mean L2 Error")
    ax1.set_title(f"Convergence (Periodic){title_suffix}")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    K = len(solver_names) if solver_names else 2
    if K > 2 and greedy_choices_per_sample is not None:
        base_colors = plt.cm.tab10.colors
        for k in range(K):
            frac_k = (greedy_choices_per_sample == k).mean(axis=1)[:T]
            ax2.plot(iters, frac_k, color=base_colors[k % 10], alpha=0.7,
                     label=f"Oracle: {solver_names[k]}", linewidth=1.5)
        if router_choices_per_sample is not None:
            for k in range(K):
                frac_k = (router_choices_per_sample == k).mean(axis=1)[:T]
                ax2.plot(iters, frac_k, color=base_colors[k % 10], alpha=0.7,
                         linestyle="--", label=f"Router: {solver_names[k]}", linewidth=1.0)
        ax2.legend(fontsize=6, loc="upper right")
        ax2.set_ylabel("Fraction of samples")
        ax2.set_title("Per-solver selection rate")
    else:
        ax2.plot(iters, greedy_choices[:T], color="green", alpha=0.7, label="Oracle Greedy")
        if router_choices is not None:
            ax2.plot(iters, router_choices[:T], color="red", alpha=0.7, linestyle="--",
                     label="LSTM Router")
            ax2.legend()
        ax2.set_ylabel("Fraction choosing DeepONet")
        ax2.set_title("DeepONet selection rate")
    ax2.set_xlabel("Iteration")
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
                        choices=["Poisson", "Helmholtz", "ConvDiff"])
    parser.add_argument("--b_vel", type=float, default=20.0,
                        help="Advection velocity for ConvDiff (b_vec=(b_vel,b_vel))")
    parser.add_argument("--reaction_c", type=float, default=0.0,
                        help="Reaction coefficient for ConvDiff (c in -div(a grad u) + b.grad u + c*u = f)")
    parser.add_argument("--grf_mode", type=str, default="fixed",
                        choices=["fixed", "hierarchical"],
                        help="GRF mode: 'fixed' (single PSD) or 'hierarchical' (varied PSDs)")
    parser.add_argument("--k2_mode", type=str, default="exp", choices=["exp", "mild", "const"],
                        help="Helmholtz k2 pushforward: 'exp', 'mild', or 'const'")
    parser.add_argument("--router_model_name", type=str, default=None,
                        help="LSTM router checkpoint name (if provided, also runs router comparison)")
    parser.add_argument("--numerical_solvers", type=str, default="jacobi",
                        help="Numerical solvers string for router checkpoint path")
    parser.add_argument("--solver_type", type=str, default="jacobi",
                        help="Classical solver: 'jacobi', 'gs', or 'mg' / 'mg_3'")
    parser.add_argument("--ml_model", type=str, default="deeponet",
                        choices=["deeponet", "fno"],
                        help="ML solver type: 'deeponet' or 'fno'")
    parser.add_argument("--max_plot_iters", type=int, default=None,
                        help="Truncate convergence plot at this iteration (default: max_iters)")
    parser.add_argument("--no_hints", action="store_true",
                        help="Skip HINTS baseline in plots")
    args = parser.parse_args()
    if args.max_plot_iters is None:
        args.max_plot_iters = args.max_iters

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Config: dim={args.dim}, equation={args.equation}, grf_mode={args.grf_mode}")

    if "," in args.solver_type:
        solver_suffix = "_multi_" + args.solver_type.replace(",", "_")
    elif args.solver_type != "jacobi":
        solver_suffix = f"_{args.solver_type}"
    else:
        solver_suffix = ""
    ml_suffix = f"_{args.ml_model}" if args.ml_model != "deeponet" else ""
    save_dir = os.path.join(args.save_dir, f"{args.dim}d_{args.equation.lower()}_{args.grf_mode}{solver_suffix}{ml_suffix}")
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

    # 2. Load ML model
    print(f"\n=== Step 2: Load trained {args.ml_model.upper()} ===")
    if args.ml_model == "fno":
        deeponet = load_fno(args.ckp_dir, args.model_name, N, args.dim, args.equation, device)
    else:
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
    specs = [s.strip() for s in args.solver_type.split(",")]
    solver_label = _solver_display_name(specs[0])
    n_solvers = len(specs) + 1
    strategies = f"{solver_label} only, HINTS, True Greedy ({n_solvers} solvers)"
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
        solver_type=args.solver_type,
    )
    errors_baselines = result["errors_baselines"]
    errors_hints = result["errors_hints"]
    errors_greedy = result["errors_greedy"]
    greedy_choices = result["greedy_choices"]
    greedy_choices_per_sample = result["greedy_choices_per_sample"]
    snapshots = result["snapshots"]
    errors_router = result["errors_router"]
    router_choices = result["router_choices"]
    router_choices_per_sample = result["router_choices_per_sample"]
    solver_names = result["solver_names"]
    baseline_names = result["baseline_names"]

    print(f"\n  Final errors after {args.max_iters} iterations:")
    for bi, bname in enumerate(baseline_names):
        print(f"    {bname} only: {errors_baselines[bi][-1]:.6e}")
    print(f"    HINTS:          {errors_hints[-1]:.6e}")
    print(f"    True Greedy:    {errors_greedy[-1]:.6e}")
    if errors_router is not None:
        print(f"    LSTM Router:    {errors_router[-1]:.6e}")
    print(f"\n  AUC (sum of errors, lower = better):")
    for bi, bname in enumerate(baseline_names):
        print(f"    {bname} only: {errors_baselines[bi].sum():.6e}")
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
                             label="Oracle Greedy", filename="greedy_routing_pattern.png",
                             solver_names=solver_names)
    if router_choices_per_sample is not None:
        visualize_greedy_routing(router_choices_per_sample, save_dir=save_dir,
                                 label="LSTM Router", filename="router_routing_pattern.png",
                                 solver_names=solver_names)

    # 7. Visualize solution snapshots (1D only for now)
    if args.dim == 1:
        print("\n=== Step 7: Visualize solution snapshots ===")
        visualize_solution_snapshots(x, u_sol, snapshots, snapshot_steps,
                                     sample_indices=[0, 1, 2, 3], save_dir=save_dir)

    # 8. Plot convergence
    print("\n=== Step 8: Plot convergence ===")
    title_suffix = f": {args.dim}D {args.equation}"
    plot_convergence(errors_baselines, errors_hints, errors_greedy, greedy_choices,
                     args.max_iters, title_suffix=title_suffix, save_dir=save_dir,
                     errors_router=errors_router, router_choices=router_choices,
                     solver_label=solver_label,
                     greedy_choices_per_sample=greedy_choices_per_sample,
                     router_choices_per_sample=router_choices_per_sample,
                     solver_names=solver_names, baseline_names=baseline_names,
                     max_plot_iters=args.max_plot_iters, no_hints=args.no_hints)

    print("\nDone! Check the plots/ directory for visualizations.")


if __name__ == "__main__":
    main()
