"""
Side-by-side comparison of pre-trained vs unrolled FNO in the oracle greedy loop.
Produces a single convergence plot with both greedy curves + all solver baselines.
"""
import argparse, json, os, torch, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml_solver import FNOforPDE
from numerical_solver import SORSolver, GaussSeidelSolver, WeightedJacobiSolver
from pde import PoissonEquation2D, ConvectionDiffusion2D
from data_generation import GaussianRandomFieldHierarchical
from verify_pipeline import (load_fno, _make_classical_solver, _solver_display_name,
                              _build_pde, _build_deeponet_input)

def run_oracle(deeponet, classical_solvers, pde, u_sol, max_iters, subtract_mean):
    """Run oracle greedy and return per-iteration mean errors + FNO usage."""
    A, b = pde.A, pde.b
    n_test = u_sol.shape[0]
    ml_idx = len(classical_solvers)

    u_prev = torch.zeros_like(u_sol)
    errors, fno_usage = [], []

    for t in range(max_iters):
        candidates, candidate_errors = [], []
        for solver in classical_solvers:
            solver.equation = pde
            u_cand = solver.iteration(u_prev)
            if subtract_mean:
                u_cand = u_cand - torch.mean(u_cand, dim=-1, keepdim=True)
            candidates.append(u_cand)
            candidate_errors.append(torch.linalg.norm(u_cand - u_sol, dim=-1))

        residual = b - torch.bmm(A, u_prev.unsqueeze(-1)).squeeze(-1)
        inp = residual[:, None, :]
        with torch.no_grad():
            correction = deeponet(inp).reshape(u_prev.shape)
        u_fno = u_prev + correction
        if subtract_mean:
            u_fno = u_fno - torch.mean(u_fno, dim=-1, keepdim=True)
        candidates.append(u_fno)
        candidate_errors.append(torch.linalg.norm(u_fno - u_sol, dim=-1))

        all_cands = torch.stack(candidates, dim=0)
        all_errs = torch.stack(candidate_errors, dim=0)
        best = torch.argmin(all_errs, dim=0)
        u_prev = all_cands[best, torch.arange(n_test)]

        err = torch.linalg.norm(u_prev - u_sol, dim=-1)
        errors.append(err.mean().item())
        fno_usage.append((best == ml_idx).float().mean().item())

    return np.array(errors), np.array(fno_usage)


def run_single_solver(solver, pde, u_sol, max_iters, subtract_mean):
    """Run a single classical solver baseline."""
    solver.equation = pde
    u_prev = torch.zeros_like(u_sol)
    errors = []
    for t in range(max_iters):
        u_prev = solver.iteration(u_prev)
        if subtract_mean:
            u_prev = u_prev - torch.mean(u_prev, dim=-1, keepdim=True)
        err = torch.linalg.norm(u_prev - u_sol, dim=-1)
        errors.append(err.mean().item())
    return np.array(errors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--equation", default="Poisson", choices=["Poisson", "ConvDiff"])
    parser.add_argument("--solver_specs", default="sor_1.0,sor_1.3,sor_1.6")
    parser.add_argument("--pretrained_name", default="hier_fno")
    parser.add_argument("--unrolled_name", default="hier_fno_unrolled")
    parser.add_argument("--ckp_dir", default="./checkpoints")
    parser.add_argument("--n_test", type=int, default=128)
    parser.add_argument("--max_iters", type=int, default=300)
    parser.add_argument("--save_dir", default="./plots")
    parser.add_argument("--b_vel", type=float, default=20.0)
    parser.add_argument("--grf_mode", default="hierarchical")
    parser.add_argument("--no_pretrained", action="store_true",
                        help="Skip pre-trained FNO; only show baselines + unrolled")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open("args/deeponet_args.json") as f:
        N = json.load(f)["N"]

    # Generate test data
    print("Generating test data...")
    grf = GaussianRandomFieldHierarchical(
        num_samples=N, dim=2,
        alpha_min=0.01, alpha_max=100.0, beta_min=0.1, beta_max=1000.0,
        gamma_list=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
        device=device, seed=42)

    subtract_mean = args.equation in ("Poisson", "ConvDiff")
    pushforward = (lambda f: f - torch.mean(f, dim=(-2, -1), keepdim=True)) if subtract_mean else None
    f = grf.generate(args.n_test, pushfoward=pushforward)
    f_flat = f.reshape(args.n_test, -1)

    x = torch.linspace(0, 1, N + 1, device=device)[:-1]
    y = torch.linspace(0, 1, N + 1, device=device)[:-1]
    a_func = lambda xi, yi: 1.0

    if args.equation == "Poisson":
        pde = PoissonEquation2D(a_func=a_func, f_func=f_flat, boundary="Periodic",
                                 x=x, y=y, device=device)
    else:
        pde = ConvectionDiffusion2D(a_func=a_func, f_func=f_flat,
                                     b_vec=(args.b_vel, args.b_vel),
                                     boundary="Periodic", x=x, y=y, device=device)
    u_sol = torch.tensor(pde.u, dtype=torch.float32, device=device).reshape(args.n_test, -1)
    if subtract_mean:
        u_sol = u_sol - torch.mean(u_sol, dim=-1, keepdim=True)

    # Build classical solvers
    specs = [s.strip() for s in args.solver_specs.split(",")]
    classical_solvers = [_make_classical_solver(s, pde, device) for s in specs]
    baseline_names = [_solver_display_name(s) for s in specs]

    # Load FNOs
    fno_pre = None
    if not args.no_pretrained:
        print("Loading pre-trained FNO...")
        fno_pre = load_fno(args.ckp_dir, args.pretrained_name, N, 2, args.equation, device)
    print("Loading unrolled FNO...")
    fno_unr = load_fno(args.ckp_dir, args.unrolled_name, N, 2, args.equation, device)

    # Run baselines
    print(f"\nRunning {len(specs)} solver baselines ({args.max_iters} iters)...")
    baseline_errors = {}
    for si, (solver, bname) in enumerate(zip(classical_solvers, baseline_names)):
        solver_copy = _make_classical_solver(specs[si], pde, device)
        errs = run_single_solver(solver_copy, pde, u_sol, args.max_iters, subtract_mean)
        baseline_errors[bname] = errs
        final = errs[-1]
        auc = errs.sum()
        status = "DIVERGES" if final > 1e3 else f"{final:.2e}"
        print(f"  {bname}: final={status}, AUC={auc:.3e}")

    # Run oracle greedy with pre-trained FNO
    errs_pre, usage_pre = None, None
    if fno_pre is not None:
        print("\nRunning oracle greedy with PRE-TRAINED FNO...")
        errs_pre, usage_pre = run_oracle(fno_pre, classical_solvers, pde, u_sol,
                                          args.max_iters, subtract_mean)
        print(f"  Final={errs_pre[-1]:.2e}, AUC={errs_pre.sum():.3e}, FNO usage={usage_pre.mean():.2%}")

    # Run oracle greedy with unrolled FNO
    print("Running oracle greedy with UNROLLED FNO...")
    errs_unr, usage_unr = run_oracle(fno_unr, classical_solvers, pde, u_sol,
                                      args.max_iters, subtract_mean)
    print(f"  Final={errs_unr[-1]:.2e}, AUC={errs_unr.sum():.3e}, FNO usage={usage_unr.mean():.2%}")

    # Plot
    print("\nGenerating comparison plot...")
    iters = np.arange(1, args.max_iters + 1)
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5.5))

    baseline_colors = ["#66c2a5", "#fc8d62", "#8da0cb"]
    for bi, bname in enumerate(baseline_names):
        errs = baseline_errors[bname]
        clipped = np.clip(errs, None, errs[0] * 100)
        diverges = errs[-1] > 1e3
        ax1.semilogy(iters, clipped, label=f"{bname} Only" + (" (diverges)" if diverges else ""),
                     linewidth=1.3, color=baseline_colors[bi % len(baseline_colors)],
                     alpha=0.7)

    if errs_pre is not None:
        ax1.semilogy(iters, errs_pre, label="Greedy + Pre-trained FNO",
                     linewidth=2, color="#984ea3")
    ax1.semilogy(iters, errs_unr, label="Greedy + Unrolled FNO",
                 linewidth=2.5, color="#e41a1c")
    ax1.set_xlabel("Iteration", fontsize=11)
    ax1.set_ylabel("Mean L2 Error", fontsize=11)
    ax1.set_title(f"Convergence: 2D {args.equation}", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    solver_tag = args.solver_specs.replace(",", "_")
    save_dir = os.path.join(args.save_dir,
                            f"2d_{args.equation.lower()}_{args.grf_mode}_multi_{solver_tag}_fno")
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "fno_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison plot to {path}")

    # Summary table
    print(f"\n{'='*65}")
    print(f"{'Strategy':<30} {'Final L2':>12} {'AUC':>12}")
    print(f"{'-'*65}")
    for bname in baseline_names:
        errs = baseline_errors[bname]
        final = errs[-1]
        tag = "DIVERGES" if final > 1e3 else f"{final:.2e}"
        print(f"{bname + ' Only':<30} {tag:>12} {errs.sum():>12.3e}")
    if errs_pre is not None:
        print(f"{'Greedy + Pre-trained FNO':<30} {errs_pre[-1]:>12.2e} {errs_pre.sum():>12.3e}")
    print(f"{'Greedy + Unrolled FNO':<30} {errs_unr[-1]:>12.2e} {errs_unr.sum():>12.3e}")
    print(f"{'='*65}")
    if errs_pre is not None:
        improvement = errs_pre.sum() / errs_unr.sum()
        print(f"Unrolled vs pre-trained AUC improvement: {improvement:.1f}×")
    best_classical_auc = min(baseline_errors[b].sum() for b in baseline_names)
    print(f"Unrolled vs best classical AUC improvement: {best_classical_auc / errs_unr.sum():.1f}×")


if __name__ == "__main__":
    main()
