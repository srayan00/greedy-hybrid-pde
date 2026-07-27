"""
Error vs cumulative FLOPs / wall-clock for hybrid PDE routers.

Reuses the same evaluation setup as results.py, then converts solver-decision
trajectories into cost-aware curves and time-to-threshold tables.

Example (paired Jacobi setting, mirrors batch/test.sbat paths):
  python -u error_vs_cost.py \\
    --ckp_dir /nfs/turbo/lsa-tewaria/sahana/greedy/poisson/checkpoints \\
    --data_dir /nfs/turbo/lsa-tewaria/sahana/greedy/poisson/data \\
    --results_dir /nfs/turbo/lsa-tewaria/sahana/greedy/poisson/results \\
    --ml_model_name msealphaepsilon_loss_new_data4 \\
    --model_name side_info_deeponet \\
    --data_name new_new_new_ \\
    --model lstm_side_info \\
    --equation Poisson --boundary Periodic --dim 2 --in_channels 1 \\
    --numerical_solvers jacobi --n_test 128 --hints_tau 24
"""

import argparse
import json
import os
import time
from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import results as results_mod
from data_generation import PDEDataset2
from hybrid_solver import (
    ConstantRouter,
    HINTSRouter,
    HybridSolver,
    LSTMGreedyRouter,
    LSTMGreedyRouter_SideInfo,
)
from ml_solver import DeepONet
from numerical_solver import (
    GaussSeidelSolver,
    MultigridSolver,
    RichardsonSolver,
    SuccessiveOverRelaxationSolver,
    SymmetricSuccessiveOverRelaxationSolver,
    WeightedJacobiSolver,
)
from pde import ConvectionDiffusion2D, PoissonEquation1D, PoissonEquation2D
from results import test_model, true_greedy_model
from trainer import ApproxGreedyRouterLoss


# ---------------------------------------------------------------------------
# FLOP cost model (mul+add counted as 2 FLOPs for dense Linear/matvec ops)
# ---------------------------------------------------------------------------

def linear_flops(in_features: int, out_features: int, bias: bool = True) -> float:
    return 2.0 * in_features * out_features + (out_features if bias else 0.0)


def mlp_flops(sizes: Sequence[int]) -> float:
    return float(sum(linear_flops(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)))


def lstm_cell_flops(input_size: int, hidden_size: int) -> float:
    # i2h: input -> 4h, h2h: h -> 4h, plus gate/state elementwise ~ O(h) (ignored)
    return linear_flops(input_size, 4 * hidden_size) + linear_flops(hidden_size, 4 * hidden_size)


def lstm_stack_flops(input_size: int, hidden_size: int, num_layers: int, num_outputs: int) -> float:
    flops = lstm_cell_flops(input_size, hidden_size)
    for _ in range(num_layers - 1):
        flops += lstm_cell_flops(hidden_size, hidden_size)
    flops += linear_flops(hidden_size, num_outputs)
    return float(flops)


def sparse_jacobi_flops(n_dof: int) -> float:
    """Ideal 2D 5-point Jacobi: Au (~10N) + diag update (~3N)."""
    return 13.0 * n_dof


def sparse_gs_flops(n_dof: int) -> float:
    """Gauss–Seidel / SOR roughly comparable to one sparse sweep."""
    return 13.0 * n_dof


def sparse_ssor_flops(n_dof: int) -> float:
    return 2.0 * sparse_gs_flops(n_dof)


def sparse_mg_flops(n_dof: int, pre: int = 3, post: int = 3, coarse: int = 3) -> float:
    """2-level MG as in numerical_solver.py: pre + coarse + post Jacobi sweeps (+ restrict/prolong ~ O(N))."""
    return (pre + post + coarse) * sparse_jacobi_flops(n_dof) + 4.0 * n_dof


def deeponet_flops(
    n_dof: int,
    branch_dim: int,
    hidden_branch: int,
    num_branch_layers: int,
    hidden_trunk: int,
    num_trunk_layers: int,
    dim: int = 2,
    cached_trunk: bool = True,
) -> float:
    branch_sizes = [n_dof] + [hidden_branch] * num_branch_layers + [branch_dim]
    trunk_sizes = [dim] + [hidden_trunk] * num_trunk_layers + [branch_dim]
    flops = mlp_flops(branch_sizes) + 2.0 * branch_dim * n_dof
    if not cached_trunk:
        flops += mlp_flops(trunk_sizes) * n_dof
    return float(flops)


def build_cost_table(
    n: int,
    dim: int,
    in_channels: int,
    numerical_solver_names: Sequence[str],
    ml_args: dict,
    router_args: dict,
    router_type: str,
    cached_trunk: bool = True,
) -> Dict[str, float]:
    """
    Per-step FLOPs for each suite index.
    Suite order matches HybridSolver: numerical solvers then DeepONet (last index).
    """
    n_dof = n if dim == 1 else n * n
    num_solvers = len(numerical_solver_names) + 1

    costs = {}
    for name in numerical_solver_names:
        base = name.split("_")[0]
        if base == "jacobi" or base == "rich":
            costs[name] = sparse_jacobi_flops(n_dof)
        elif base in ("gs", "sor"):
            costs[name] = sparse_gs_flops(n_dof)
        elif base == "ssor":
            costs[name] = sparse_ssor_flops(n_dof)
        elif base == "mg":
            costs[name] = sparse_mg_flops(n_dof)
        else:
            raise ValueError(f"Unknown numerical solver for cost model: {name}")

    costs["deeponet"] = deeponet_flops(
        n_dof=n_dof,
        branch_dim=ml_args["branch_dim"],
        hidden_branch=ml_args["hidden_branch"],
        num_branch_layers=ml_args["num_branch_layers"],
        hidden_trunk=ml_args["hidden_trunk"],
        num_trunk_layers=ml_args["num_trunk_layers"],
        dim=dim,
        cached_trunk=cached_trunk,
    )

    # Router input: fields flattened + optional side info
    field_channels = in_channels + 1  # inputs (+a/f) concatenated with u_prev
    decoder_dim = n_dof * field_channels
    if router_type == "lstm_side_info":
        decoder_dim += 3

    costs["router"] = lstm_stack_flops(
        input_size=decoder_dim,
        hidden_size=router_args["hidden_dim"],
        num_layers=router_args["num_layers"],
        num_outputs=num_solvers,
    )
    costs["jacobi_ref"] = sparse_jacobi_flops(n_dof)
    return costs


def suite_index_costs(
    numerical_solver_names: Sequence[str],
    cost_table: Dict[str, float],
) -> np.ndarray:
    """Map suite index -> solver FLOPs (excludes router)."""
    per_idx = [cost_table[name] for name in numerical_solver_names] + [cost_table["deeponet"]]
    return np.asarray(per_idx, dtype=np.float64)


def cumulative_cost_from_decisions(
    errors: np.ndarray,
    decisions: Optional[np.ndarray],
    per_index_flops: np.ndarray,
    router_flops: float,
    charge_router: bool,
    method: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    errors: (T+1, B) including t=0
    decisions: (T, B) suite indices for steps 1..T, or None for constant numerical (index 0)
    Returns mean_error (T+1,), mean_cumcost (T+1,)
    """
    t_plus, batch = errors.shape
    t_steps = t_plus - 1
    if decisions is None:
        decisions = np.zeros((t_steps, batch), dtype=np.int64)
    elif decisions.shape[0] == t_plus:
        # some dumps may include a dummy first row
        decisions = decisions[1:]

    step_cost = per_index_flops[decisions]  # (T, B)
    if charge_router:
        step_cost = step_cost + router_flops

    cum = np.cumsum(step_cost, axis=0)  # (T, B)
    cum_full = np.concatenate([np.zeros((1, batch), dtype=np.float64), cum], axis=0)
    return errors.mean(axis=1), cum_full.mean(axis=1)


def cost_to_threshold(errors: np.ndarray, cumcost: np.ndarray, thresholds: Sequence[float]) -> Dict[str, float]:
    """
    errors/cumcost: (T+1, B). For each sample, first time mean? -> per-sample first hitting time, then average.
    """
    out = {}
    t_plus, batch = errors.shape
    for eps in thresholds:
        hits = []
        for b in range(batch):
            idx = np.where(errors[:, b] <= eps)[0]
            if len(idx) == 0:
                hits.append(np.nan)
            else:
                hits.append(cumcost[idx[0], b])
        out[f"cost_to_{eps:g}"] = float(np.nanmean(hits))
        out[f"frac_hit_{eps:g}"] = float(np.mean(~np.isnan(hits)))
    return out


def per_sample_cumulative(
    decisions: Optional[np.ndarray],
    t_plus: int,
    batch: int,
    per_index_flops: np.ndarray,
    router_flops: float,
    charge_router: bool,
) -> np.ndarray:
    t_steps = t_plus - 1
    if decisions is None:
        decisions = np.zeros((t_steps, batch), dtype=np.int64)
    elif decisions.shape[0] == t_plus:
        decisions = decisions[1:]
    step_cost = per_index_flops[decisions]
    if charge_router:
        step_cost = step_cost + router_flops
    cum = np.cumsum(step_cost, axis=0)
    return np.concatenate([np.zeros((1, batch)), cum], axis=0)


# ---------------------------------------------------------------------------
# Model / data loading (aligned with results.py)
# ---------------------------------------------------------------------------

def build_numerical_solvers(names: Sequence[str], pde, device):
    solvers = []
    for solver in names:
        split = solver.split("_")
        if split[0] == "jacobi":
            weight = float(split[1]) if len(split) > 1 else 1.0
            solvers.append(WeightedJacobiSolver(pde, device, weight))
        elif split[0] == "gs":
            solvers.append(GaussSeidelSolver(pde, device))
        elif split[0] == "mg":
            levels = int(split[1]) if len(split) > 1 else 2
            solvers.append(MultigridSolver(pde, levels, device))
        elif split[0] == "sor":
            omega = float(split[1]) if len(split) > 1 else 1.0
            solvers.append(SuccessiveOverRelaxationSolver(pde, device, omega))
        elif split[0] == "ssor":
            omega = float(split[1]) if len(split) > 1 else 1.0
            solvers.append(SymmetricSuccessiveOverRelaxationSolver(pde, device, omega))
        elif split[0] == "rich":
            omega = float(split[1]) if len(split) > 1 else 1.0
            solvers.append(RichardsonSolver(pde, device, omega))
        else:
            raise ValueError(f"Invalid numerical solver: {solver}")
    return solvers


def load_ml_model(ml_model_type: str, ml_arguments: dict, dim: int, boundary: str, in_channels: int, device):
    if ml_model_type == "deeponet":
        return DeepONet(
            N=ml_arguments["N"],
            dim=dim,
            device=device,
            in_channels=in_channels,
            boundary=boundary,
            branch_dim=ml_arguments["branch_dim"],
            hidden_branch=ml_arguments["hidden_branch"],
            num_branch_layers=ml_arguments["num_branch_layers"],
            hidden_trunk=ml_arguments["hidden_trunk"],
            num_trunk_layers=ml_arguments["num_trunk_layers"],
        ).to(device)
    raise ValueError(f"Unsupported ml_model_type for cost script: {ml_model_type}")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed_cuda_ms(fn, device: torch.device, warmup: int, iters: int) -> float:
    """Median wall time (ms) of fn() over `iters` timed runs after `warmup`."""
    for _ in range(warmup):
        fn()
    _sync(device)
    times = []
    for _ in range(iters):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(times))


def benchmark_component_ms(
    numerical_solver,
    ml_model,
    router,
    pde,
    device: torch.device,
    dim: int,
    n: int,
    in_channels: int,
    batch_size: int = 1,
    warmup: int = 20,
    iters: int = 100,
) -> Dict[str, float]:
    """
    Measure ms/step for Jacobi-like numerical update, DeepONet, and LSTM router.
    Uses batch_size samples (default 1 = per-sample latency).
    """
    n_dof = n if dim == 1 else n * n
    u = torch.randn(batch_size, n_dof, device=device)
    # PDE batch fields
    if dim == 1:
        f = torch.randn(batch_size, n, device=device)
        inputs = f.unsqueeze(1) if in_channels == 1 else torch.randn(batch_size, in_channels, n, device=device)
    else:
        f = torch.randn(batch_size, n, n, device=device)
        inputs = f.unsqueeze(1) if in_channels == 1 else torch.randn(batch_size, in_channels, n, n, device=device)

    # Attach batch equation like HybridSolver does
    numerical_solver.equation = pde
    # Ensure A/b exist for this PDE object; use random residual-compatible state via iteration API
    # For WeightedJacobi etc., equation.b and equation.A must be set. Mirror hybrid path:
    # build a temporary equation batch by calling the same interface HybridSolver uses is heavy;
    # instead time matmul-equivalent through solver.iteration after setting b from residual of zeros.
    if not hasattr(pde, "A") or pde.A is None:
        raise RuntimeError("PDE must have assembled A for wall-clock benchmarks")

    # Expand PDE to batch if needed
    A = pde.A
    if A.dim() == 2:
        # single operator shared across batch
        b = torch.randn(batch_size, n_dof, device=device)
        # monkey-patch a thin wrapper object
        class _Eq:
            pass
        eq = _Eq()
        eq.A = A
        eq.b = b
        eq.is_batch = True
        eq.in_channels = in_channels
        eq.compute_residual = lambda u, mask=None: b - (A @ u.T).T if False else b - torch.matmul(u, A.T)
        # Prefer torch.matmul(A, u.unsqueeze(-1)) like solvers
        def _resid(u_in, mask=None):
            return b - torch.matmul(A, u_in.unsqueeze(-1)).squeeze(-1)
        eq.compute_residual = _resid
        numerical_solver.equation = eq
    else:
        numerical_solver.equation = pde

    def run_num():
        with torch.no_grad():
            numerical_solver.iteration(u)

    # Warm DeepONet trunk once
    with torch.no_grad():
        _ = ml_model(inputs)

    def run_ml():
        with torch.no_grad():
            ml_model(inputs)

    # Router input matches HybridSolver: concat fields + u, flatten; side-info adds 3
    flat_fields = inputs.reshape(batch_size, -1)
    flat_u = u
    router_in = torch.cat([flat_fields, flat_u], dim=-1)
    if getattr(router, "type", "") == "LSTMGreedySideInfo":
        side = torch.zeros(batch_size, 3, device=device)
        router_in = torch.cat([router_in, side], dim=-1)
    hidden = None

    def run_router():
        nonlocal hidden
        with torch.no_grad():
            _, _, hidden = router.predict(router_in, hidden, with_scores=True)

    ms_num = _timed_cuda_ms(run_num, device, warmup, iters)
    ms_ml = _timed_cuda_ms(run_ml, device, warmup, iters)
    ms_router = _timed_cuda_ms(run_router, device, warmup, iters)
    return {
        "numerical_ms": ms_num,
        "deeponet_ms": ms_ml,
        "router_ms": ms_router,
        "batch_size": float(batch_size),
    }


def time_method_end_to_end_s(model, test_loader, in_channels, dim, loss, centered, device) -> float:
    """Wall-clock seconds for one full test_model pass (all iters, all batches)."""
    _sync(device)
    t0 = time.perf_counter()
    test_model(model, test_loader, in_channels, dim, loss=loss, centered=centered, loss_t=False)
    _sync(device)
    return time.perf_counter() - t0


def cumulative_wallclock_from_decisions(
    errors: np.ndarray,
    decisions: Optional[np.ndarray],
    numerical_ms: float,
    deeponet_ms: float,
    router_ms: float,
    charge_router: bool,
    ml_index: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build per-sample cumulative wall time (seconds) from measured ms/op.
    Returns mean_err, mean_cum_sec, cum_sec (T+1, B).
    """
    t_plus, batch = errors.shape
    t_steps = t_plus - 1
    if decisions is None:
        decisions = np.zeros((t_steps, batch), dtype=np.int64)
    elif decisions.shape[0] == t_plus:
        decisions = decisions[1:]

    step_ms = np.where(decisions == ml_index, deeponet_ms, numerical_ms).astype(np.float64)
    if charge_router:
        step_ms = step_ms + router_ms
    cum_ms = np.cumsum(step_ms, axis=0)
    cum_sec = np.concatenate([np.zeros((1, batch)), cum_ms * 1e-3], axis=0)
    return errors.mean(axis=1), cum_sec.mean(axis=1), cum_sec


def main():
    parser = argparse.ArgumentParser(description="Error vs FLOPs / wall-clock plots")
    parser.add_argument("--ml_model", type=str, default="deeponet")
    parser.add_argument("--n_test", type=int, default=128)
    parser.add_argument("--model", type=str, default="lstm_side_info")
    parser.add_argument("--extra", type=int, default=200)
    parser.add_argument("--ckp_dir", type=str, required=True)
    parser.add_argument("--ml_model_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--data_name", type=str, default="")
    parser.add_argument("--grf_mode", type=str, default="hierarchical")
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--boundary", type=str, default="Periodic")
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--numerical_solvers", type=str, default="jacobi")
    parser.add_argument("--equation", type=str, default="Poisson")
    parser.add_argument("--reaction_c", type=float, default=0.0)
    parser.add_argument("--b_vel", type=float, default=20.0)
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--results_tag", type=str, default="cost")
    parser.add_argument("--hints_tau", type=int, default=24)
    parser.add_argument("--thresholds", type=str, default="1e-2,1e-3,1e-4")
    parser.add_argument("--no_cached_trunk", action="store_true")
    parser.add_argument(
        "--measure_wallclock",
        action="store_true",
        help="Microbenchmark Jacobi/DeepONet/router ms and plot error vs measured wall-clock",
    )
    parser.add_argument("--bench_warmup", type=int, default=20)
    parser.add_argument("--bench_iters", type=int, default=100)
    parser.add_argument("--bench_batch_size", type=int, default=1)
    parser.add_argument(
        "--jacobi_ms",
        type=float,
        default=None,
        help="Optional override ms/Jacobi; ignored if --measure_wallclock is set",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=None,
        help="If set, also plot one sample trajectory; default uses mean curves",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    numerical_solvers = args.numerical_solvers.split(",")
    if len(numerical_solvers) != 1:
        raise ValueError("This script currently supports a single numerical solver + DeepONet (K=2).")

    num_solvers = len(numerical_solvers) + 1
    ml_ckp_path = (
        f"{args.ckp_dir}/{args.ml_model}_{args.ml_model_name}_{args.grf_mode}_"
        f"{args.equation}_{args.boundary}_{args.dim}d_{args.in_channels}c_best.pth"
    )
    ml_args_path = (
        f"{args.ckp_dir}/{args.ml_model}_{args.ml_model_name}_{args.grf_mode}_"
        f"{args.equation}_{args.boundary}_{args.dim}d_{args.in_channels}c_args.json"
    )
    router_ckp = (
        f"{args.ckp_dir}/{args.model}router_{args.model_name}_{args.grf_mode}_"
        f"{args.equation}_{args.boundary}_{args.dim}d_{args.in_channels}c_{args.numerical_solvers}_best.pth"
    )
    router_args_path = (
        f"{args.ckp_dir}/{args.model}router_{args.model_name}_{args.grf_mode}_"
        f"{args.equation}_{args.boundary}_{args.dim}d_{args.in_channels}c_{args.numerical_solvers}args.json"
    )

    with open(ml_args_path, "r") as f:
        ml_arguments = json.load(f)
    if os.path.exists(router_args_path):
        with open(router_args_path, "r") as f:
            arguments = json.load(f)
    else:
        with open(f"args/{args.model}_args.json", "r") as f:
            arguments = json.load(f)

    n = ml_arguments["N"]
    dim = args.dim
    new_in_channels = args.in_channels + 1 if args.equation == "Helmholtz" else args.in_channels

    cost_table = build_cost_table(
        n=n,
        dim=dim,
        in_channels=new_in_channels,
        numerical_solver_names=numerical_solvers,
        ml_args=ml_arguments,
        router_args=arguments,
        router_type=args.model,
        cached_trunk=not args.no_cached_trunk,
    )
    per_index = suite_index_costs(numerical_solvers, cost_table)
    print("=== Per-step FLOP costs ===")
    for k, v in cost_table.items():
        print(f"  {k:12s}: {v:.3e}  ({v / cost_table['jacobi_ref']:.1f}x Jacobi)")
    print(f"  suite index costs: {per_index}")

    # Data (same naming convention as results.py)
    test_path = (
        f"{args.data_dir}/{args.data_name}router_test_data_{args.grf_mode}_{args.equation}_"
        f"{args.boundary}_{dim}d_{args.in_channels}c_{args.n_test}s.pt"
    )
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing test data: {test_path}")
    test_data = torch.load(test_path, map_location=device, weights_only=False)
    test_dataset = PDEDataset2(test_data)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=arguments.get("batch_size", 32), shuffle=False
    )

    # Grid / PDE
    if dim == 1:
        x = torch.linspace(0, 1, n + 1)[:-1] if args.boundary == "Periodic" else torch.linspace(0, 1, n)
        y = None
    else:
        x = torch.linspace(0, 1, n + 1)[:-1] if args.boundary == "Periodic" else torch.linspace(0, 1, n)
        y = x.clone()

    if args.equation == "Poisson":
        if dim == 1:
            pde = PoissonEquation1D(lambda x: 1, lambda x: 1, args.boundary, x, device=device, solve=False)
        else:
            pde = PoissonEquation2D(lambda x, y: 1, lambda x, y: 1, args.boundary, x, y, device=device, solve=False)
    elif args.equation == "ConvDiff":
        pde = ConvectionDiffusion2D(
            lambda x, y: 1,
            lambda x, y: 1,
            (args.b_vel, args.b_vel),
            args.boundary,
            x,
            y,
            device=device,
            reaction=args.reaction_c,
            solve=False,
        )
    else:
        raise ValueError("Only Poisson/ConvDiff supported in this script currently")

    ml_model = load_ml_model(args.ml_model, ml_arguments, dim, args.boundary, new_in_channels, device)
    ml_ckp = torch.load(ml_ckp_path, map_location=device, weights_only=False)
    ml_model.load_state_dict(ml_ckp["model"])
    ml_model.eval()

    list_of_solvers = build_numerical_solvers(numerical_solvers, pde, device)
    decoder_dim = (n * n if dim == 2 else n) * (new_in_channels + 1)
    if args.model == "lstm":
        router = LSTMGreedyRouter(
            None,
            decoder_dim,
            arguments["hidden_dim"],
            arguments["num_layers"],
            num_solvers,
            arguments.get("dropout", 0.0),
        ).to(device)
    else:
        router = LSTMGreedyRouter_SideInfo(
            None,
            decoder_dim,
            arguments["hidden_dim"],
            arguments["num_layers"],
            num_solvers,
            arguments.get("dropout", 0.0),
        ).to(device)
    router_ckp_obj = torch.load(router_ckp, map_location=device, weights_only=False)
    router.load_state_dict(router_ckp_obj["model"])
    router.eval()

    # true_greedy_model references module-level `device` inside results.py
    results_mod.device = device

    wall_ms = None
    if args.measure_wallclock:
        print("=== Measuring wall-clock (CUDA-synced microbenchmarks) ===")
        # Ensure A is on the right device
        if hasattr(pde, "A") and pde.A is not None:
            pde.A = pde.A.to(device)
        if hasattr(pde, "b") and pde.b is not None:
            pde.b = pde.b.to(device)
        wall_ms = benchmark_component_ms(
            numerical_solver=list_of_solvers[0],
            ml_model=ml_model,
            router=router,
            pde=pde,
            device=device,
            dim=dim,
            n=n,
            in_channels=new_in_channels,
            batch_size=args.bench_batch_size,
            warmup=args.bench_warmup,
            iters=args.bench_iters,
        )
        print(
            f"  numerical: {wall_ms['numerical_ms']:.4f} ms/step "
            f"(batch={int(wall_ms['batch_size'])})"
        )
        print(f"  deeponet:  {wall_ms['deeponet_ms']:.4f} ms/step")
        print(f"  router:    {wall_ms['router_ms']:.4f} ms/step")
        print(
            f"  relative: deeponet={wall_ms['deeponet_ms']/wall_ms['numerical_ms']:.1f}x num, "
            f"router={wall_ms['router_ms']/wall_ms['numerical_ms']:.1f}x num"
        )

    suite = list_of_solvers + [ml_model]
    needs_mean_zero = (args.equation == "Poisson") or (args.equation == "ConvDiff" and args.reaction_c == 0.0)
    centered = needs_mean_zero and args.boundary == "Periodic" and args.in_channels == 1
    loss = ApproxGreedyRouterLoss(centered=centered)

    model_greedy = HybridSolver(
        N=n, dim=dim, in_channels=args.in_channels, boundary=args.boundary, equation=pde,
        suite_solver=suite, router=router, tol=1e-7, max_iters=arguments["max_iters"], threshold=0.1,
    ).to(device)
    model_const = HybridSolver(
        N=n, dim=dim, in_channels=args.in_channels, boundary=args.boundary, equation=pde,
        suite_solver=suite, router=ConstantRouter(num_solvers, 0, device=device),
        tol=1e-7, max_iters=arguments["max_iters"], threshold=0.1,
    ).to(device)
    model_hints = HybridSolver(
        N=n, dim=dim, in_channels=args.in_channels, boundary=args.boundary, equation=pde,
        suite_solver=suite, router=HINTSRouter(num_solvers, args.hints_tau, device=device),
        tol=1e-7, max_iters=arguments["max_iters"], threshold=0.1,
    ).to(device)

    e2e = {}
    print("Evaluating Learned Greedy...")
    if args.measure_wallclock:
        _sync(device)
        t0 = time.perf_counter()
    err_g, _, _, _, _, _, dec_g, _, _ = test_model(
        model_greedy, test_loader, args.in_channels, dim, loss=loss, centered=centered, loss_t=False
    )
    if args.measure_wallclock:
        _sync(device)
        e2e["Learned Greedy"] = time.perf_counter() - t0
        print(f"  end-to-end wall time: {e2e['Learned Greedy']:.2f} s")

    print("Evaluating numerical-only...")
    if args.measure_wallclock:
        _sync(device)
        t0 = time.perf_counter()
    err_c, _, _, _, _, _, dec_c, _, _ = test_model(
        model_const, test_loader, args.in_channels, dim, loss=loss, centered=centered, loss_t=False
    )
    if args.measure_wallclock:
        _sync(device)
        e2e[f"{numerical_solvers[0]} only"] = time.perf_counter() - t0
        print(f"  end-to-end wall time: {e2e[f'{numerical_solvers[0]} only']:.2f} s")

    print("Evaluating HINTS...")
    if args.measure_wallclock:
        _sync(device)
        t0 = time.perf_counter()
    err_h, _, _, _, _, _, dec_h, _, _ = test_model(
        model_hints, test_loader, args.in_channels, dim, loss=loss, centered=centered, loss_t=False
    )
    if args.measure_wallclock:
        _sync(device)
        e2e[f"HINTS-{numerical_solvers[0]}"] = time.perf_counter() - t0
        print(f"  end-to-end wall time: {e2e[f'HINTS-{numerical_solvers[0]}']:.2f} s")

    print("Evaluating True Greedy...")
    if args.measure_wallclock:
        _sync(device)
        t0 = time.perf_counter()
    err_t, dec_t = true_greedy_model(
        model_hints, test_loader, args.in_channels, dim, loss, centered=centered,
        loss_t=False, max_iters=arguments["max_iters"],
    )
    if args.measure_wallclock:
        _sync(device)
        e2e["True-Greedy"] = time.perf_counter() - t0
        print(f"  end-to-end wall time: {e2e['True-Greedy']:.2f} s")

    methods = {
        f"{numerical_solvers[0]} only": (err_c, None, False),
        f"HINTS-{numerical_solvers[0]}": (err_h, dec_h, False),
        f"Learned Greedy-{numerical_solvers[0]}": (err_g, dec_g, True),
        f"True-Greedy-{numerical_solvers[0]}": (err_t, dec_t, False),
    }
    # Also report learned greedy solver-only wall cost (no LSTM) for fair hybrid comparison
    if args.measure_wallclock:
        methods[f"Learned Greedy-{numerical_solvers[0]} (no router charge)"] = (err_g, dec_g, False)

    thresholds = [float(x) for x in args.thresholds.split(",")]
    os.makedirs(args.results_dir, exist_ok=True)
    tag = (
        f"{args.results_tag}_{args.ml_model}_{args.equation}_{args.boundary}_"
        f"{dim}d_{numerical_solvers[0]}"
    )

    save_kwargs = dict(
        errors_constant=err_c,
        errors_hints=err_h,
        errors_greedy=err_g,
        errors_true_greedy=err_t,
        decisions_constant=dec_c,
        decisions_hints=dec_h,
        decisions_greedy=dec_g,
        decisions_true_greedy=dec_t,
        per_index_flops=per_index,
        router_flops=cost_table["router"],
        jacobi_ref_flops=cost_table["jacobi_ref"],
        cost_table_json=np.array(json.dumps(cost_table)),
    )
    if wall_ms is not None:
        save_kwargs["wall_ms_json"] = np.array(json.dumps(wall_ms))
        save_kwargs["e2e_seconds_json"] = np.array(json.dumps(e2e))
    np.savez_compressed(f"{args.results_dir}/{tag}_trajectories.npz", **save_kwargs)

    # Plots + table
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    rows = []

    for name, (errors, decisions, charge_router) in methods.items():
        router_cost = cost_table["router"] if charge_router else 0.0
        mean_err, mean_cost = cumulative_cost_from_decisions(
            errors, decisions, per_index, router_cost, charge_router=charge_router, method=name
        )
        cum_flops = per_sample_cumulative(
            decisions, errors.shape[0], errors.shape[1], per_index, router_cost, charge_router
        )
        # Left: FLOPs (skip duplicate no-router-charge curve to keep FLOP panel clean)
        if "no router charge" not in name:
            axes[0].semilogy(mean_cost, mean_err, label=name)

        row = {"method": name, "charge_router": charge_router}
        hit_flops = cost_to_threshold(errors, cum_flops, thresholds)
        for k, v in hit_flops.items():
            row[f"flops_{k}"] = v
        row["jacobi_eq_to_0.01"] = hit_flops["cost_to_0.01"] / cost_table["jacobi_ref"]

        if wall_ms is not None:
            mean_err_w, mean_sec, cum_sec = cumulative_wallclock_from_decisions(
                errors,
                decisions,
                numerical_ms=wall_ms["numerical_ms"],
                deeponet_ms=wall_ms["deeponet_ms"],
                router_ms=wall_ms["router_ms"],
                charge_router=charge_router,
                ml_index=len(numerical_solvers),  # DeepONet is last suite index
            )
            axes[1].semilogy(mean_sec, mean_err_w, label=name)
            hit_sec = cost_to_threshold(errors, cum_sec, thresholds)
            for eps in thresholds:
                row[f"sec_to_{eps:g}"] = hit_sec[f"cost_to_{eps:g}"]
                row[f"frac_hit_{eps:g}"] = hit_sec[f"frac_hit_{eps:g}"]
        else:
            axes[1].semilogy(mean_cost / cost_table["jacobi_ref"], mean_err, label=name)
            for eps in thresholds:
                row[f"jacobi_eq_to_{eps:g}"] = hit_flops[f"cost_to_{eps:g}"] / cost_table["jacobi_ref"]
                row[f"frac_hit_{eps:g}"] = hit_flops[f"frac_hit_{eps:g}"]

        rows.append(row)

    if args.measure_wallclock:
        for row in rows:
            m = row["method"]
            if m.startswith("Learned Greedy") and "no router" not in m:
                row["e2e_seconds_full_eval"] = e2e.get("Learned Greedy")
            elif m.startswith("HINTS"):
                row["e2e_seconds_full_eval"] = e2e.get(f"HINTS-{numerical_solvers[0]}")
            elif m.endswith("only"):
                row["e2e_seconds_full_eval"] = e2e.get(f"{numerical_solvers[0]} only")
            elif m.startswith("True-Greedy"):
                row["e2e_seconds_full_eval"] = e2e.get("True-Greedy")

    axes[0].set_xlabel("Cumulative FLOPs")
    axes[0].set_ylabel(r"Mean $\|e^{(t)}\|_2$")
    axes[0].set_title(f"{args.equation} / {numerical_solvers[0]}")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, which="both", alpha=0.3)

    if wall_ms is not None:
        axes[1].set_xlabel("Cumulative wall-clock (s) from measured ms/op")
        axes[1].set_title("Measured wall-clock convergence")
    else:
        axes[1].set_xlabel("Cumulative cost in Jacobi-equivalents")
        axes[1].set_title("Cost-normalized convergence")
    axes[1].set_ylabel(r"Mean $\|e^{(t)}\|_2$")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig_path = f"{args.results_dir}/{tag}_error_vs_cost.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"Saved figure: {fig_path}")

    df = pd.DataFrame(rows)
    csv_path = f"{args.results_dir}/{tag}_time_to_threshold.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved table: {csv_path}")
    print(df.to_string(index=False))

    if wall_ms is not None:
        ms_path = f"{args.results_dir}/{tag}_wallclock_ms.json"
        with open(ms_path, "w") as f:
            json.dump({"per_op_ms": wall_ms, "e2e_seconds": e2e}, f, indent=2)
        print(f"Saved wall-clock JSON: {ms_path}")

    readme = f"{args.results_dir}/{tag}_cost_model.txt"
    with open(readme, "w") as f:
        f.write("Cost model assumptions\n")
        f.write("- Jacobi/GS/SOR: sparse 5-point stencil FLOPs (~13 N)\n")
        f.write("- SymGS: 2x GS\n")
        f.write("- MG: (pre+post+coarse)=9 Jacobi sweeps + O(N) transfer\n")
        f.write(f"- DeepONet trunk cached: {not args.no_cached_trunk}\n")
        f.write("- Learned Greedy charges LSTM router every iteration; HINTS/const/true-greedy do not\n")
        f.write("- Wall-clock uses CUDA-synced microbenchmarks when --measure_wallclock is set\n")
        f.write(json.dumps(cost_table, indent=2))
        if wall_ms is not None:
            f.write("\n\nMeasured ms/op:\n")
            f.write(json.dumps(wall_ms, indent=2))
            f.write("\n\nEnd-to-end seconds:\n")
            f.write(json.dumps(e2e, indent=2))
    print(f"Saved cost model notes: {readme}")


if __name__ == "__main__":
    main()