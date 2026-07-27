"""Error/residual convergence curves + AUC for lite GRU routers.

Mirrors the main-repo protocol in results.py / experiments.py:
  - fixed iteration budget
  - absolute L2 error of demeaned (u - u*)
  - AUC = trapezoid(error_trajectory) per sample
  - mean curve plots (log-y) for classical / HINTS / GRU / oracle
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from deeponet_corrector import DeepONetCorrector
from fast_pde import GRF2D, demean, l2, make_solver
from pde_factory import make_pde
from train_and_bench import load_router


def rollout_abs(pde, solver, corrector, f, u_truth, policy, router, *,
                max_iters, hints_tau):
    """Return abs L2 error & residual trajectories of shape (max_iters,).

    Matches main-repo test_model: error = ||u - u*||_2 (demeaned for singular ops).
    """
    u = np.zeros_like(f)
    fn = float(l2(f)[0])
    u_star = demean(u_truth)
    if router is not None:
        router.reset()
    prev_dec = 0
    errs = np.empty(max_iters)
    ress = np.empty(max_iters)
    for it in range(max_iters):
        r = pde.residual(u, f)
        errs[it] = float(l2(demean(u) - u_star)[0])
        ress[it] = float(l2(r)[0]) / fn
        if policy == "classical":
            d = 0
        elif policy == "hints":
            d = 1 if (it + 1) % hints_tau == 0 else 0
        elif policy == "router":
            d = router.decide(ress[it], it, prev_dec, r=r, pde=pde)
        elif policy == "oracle":
            u_c = solver.step(u, f, r)
            u_n = u + corrector.correct(r)
            e_c = float(l2(demean(u_c) - u_star)[0])
            e_n = float(l2(demean(u_n) - u_star)[0])
            d = 1 if e_n < e_c else 0
            u = u_n if d else u_c
            prev_dec = d
            continue
        else:
            raise ValueError(policy)
        if d:
            u = u + corrector.correct(r)
        else:
            u = solver.step(u, f, r)
        prev_dec = d
    return errs, ress


def fmt_msd(mean, std, scale=1.0):
    return f"{mean * scale:.3f} ({std * scale:.3f})"


def run_one(args):
    torch.set_num_threads(8)
    pde = make_pde(args.equation, args.N, b_vel=args.b_vel,
                   eps_x=args.eps_x, eps_y=args.eps_y)
    solver = make_solver(pde, args.solver)
    corrector = DeepONetCorrector(args.don_ckp, device="cpu", threads=8)
    router = load_router(args.router_ckp)

    grf = GRF2D(args.N, rng=np.random.default_rng(args.seed))
    f = grf.sample(args.n_test)
    u_truth = demean(pde.solve_direct(f))

    policies = ["classical", "hints", "router", "oracle"]
    labels = {
        "classical": f"{args.solver} only",
        "hints": f"HINTS-{args.hints_tau}",
        "router": "GRU (ApproxGreedy)",
        "oracle": "True-Greedy",
    }
    colors = {
        "classical": "tab:orange",
        "hints": "tab:red",
        "router": "tab:blue",
        "oracle": "k",
    }

    curves_e, curves_r = {}, {}
    rows = []
    for pol in policies:
        E = np.zeros((args.max_iters, args.n_test))
        R = np.zeros((args.max_iters, args.n_test))
        for i in range(args.n_test):
            print(f"[{pol}] sample {i+1}/{args.n_test}", flush=True)
            e, r = rollout_abs(
                pde, solver, corrector, f[i:i + 1], u_truth[i:i + 1],
                pol, router if pol == "router" else None,
                max_iters=args.max_iters, hints_tau=args.hints_tau,
            )
            E[:, i] = e
            R[:, i] = r
        curves_e[pol] = E
        curves_r[pol] = R
        auc_e = np.trapezoid(E, axis=0)
        auc_r = np.trapezoid(R, axis=0)
        row = {
            "Methods": labels[pol],
            "Mean_FinalError": float(np.mean(E[-1])),
            "Std_FinalError": float(np.std(E[-1])),
            "Mean_AUC_Error": float(np.mean(auc_e)),
            "Std_AUC_Error": float(np.std(auc_e)),
            "FinalError": fmt_msd(np.mean(E[-1]), np.std(E[-1]), 1e3),
            "AUC_Error": fmt_msd(np.mean(auc_e), np.std(auc_e), 1e3),
            "Mean_FinalResidual": float(np.mean(R[-1])),
            "Std_FinalResidual": float(np.std(R[-1])),
            "Mean_AUC_Residual": float(np.mean(auc_r)),
            "Std_AUC_Residual": float(np.std(auc_r)),
            "FinalResidual": fmt_msd(np.mean(R[-1]), np.std(R[-1]), 1e3),
            "AUC_Residual": fmt_msd(np.mean(auc_r), np.std(auc_r), 1e3),
        }
        rows.append(row)
        print(
            f"[{pol}] final_err×1e3={row['FinalError']}  "
            f"AUC_err×1e3={row['AUC_Error']}  "
            f"final_res×1e3={row['FinalResidual']}  "
            f"AUC_res×1e3={row['AUC_Residual']}",
            flush=True,
        )

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    csv_path = f"{args.out_prefix}_auc.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("saved", csv_path)

    # compact JSON with mean curves for later replotting
    json_path = f"{args.out_prefix}_curves.json"
    payload = {
        "args": vars(args),
        "labels": labels,
        "auc_table": rows,
        "mean_error": {p: curves_e[p].mean(axis=1).tolist() for p in policies},
        "mean_residual": {p: curves_r[p].mean(axis=1).tolist() for p in policies},
    }
    with open(json_path, "w") as fh:
        json.dump(payload, fh)
    print("saved", json_path)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xlab = "V-cycles" if args.solver == "mg" else "Iteration"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for pol in policies:
        axes[0].plot(curves_e[pol].mean(axis=1), label=labels[pol],
                     color=colors[pol], lw=2)
        axes[1].plot(curves_r[pol].mean(axis=1), label=labels[pol],
                     color=colors[pol], lw=2)
    for ax, ylab, title in zip(
        axes,
        ["Error (L2)", "Relative residual"],
        ["Error Comparison of Different Routing Strategies",
         "Residual Comparison of Different Routing Strategies"],
    ):
        ax.set_yscale("log")
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.suptitle(f"{args.equation} N={args.N} {args.solver}  "
                 f"(GRU lite, unnormalized ApproxGreedy)", fontsize=11)
    fig.tight_layout()
    png = f"{args.out_prefix}_error_residual.png"
    fig.savefig(png, dpi=140)
    print("saved", png)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--equation", required=True)
    p.add_argument("--N", type=int, default=63)
    p.add_argument("--solver", required=True)
    p.add_argument("--eps_x", type=float, default=0.01)
    p.add_argument("--eps_y", type=float, default=1.0)
    p.add_argument("--b_vel", type=float, default=20.0)
    p.add_argument("--don_ckp", required=True)
    p.add_argument("--router_ckp", required=True)
    p.add_argument("--n_test", type=int, default=32)
    p.add_argument("--max_iters", type=int, required=True)
    p.add_argument("--hints_tau", type=int, default=25)
    p.add_argument("--seed", type=int, default=72)
    p.add_argument("--out_prefix", required=True)
    args = p.parse_args()
    run_one(args)


if __name__ == "__main__":
    main()
