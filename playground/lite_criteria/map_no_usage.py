"""Collect per-iteration NO decisions across samples for aniso policies.

Writes a compact JSON used for plotting: for each policy, a (n_samples, T)
binary matrix truncated to T_max, plus mean NO rate vs iteration.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from deeponet_corrector import DeepONetCorrector
from fast_pde import GRF2D, demean, make_solver
from pde_factory import make_pde
from train_and_bench import LiteRouter, run_rollout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=31)
    p.add_argument("--eps_x", type=float, default=0.01)
    p.add_argument("--eps_y", type=float, default=1.0)
    p.add_argument("--solver", type=str, default="jacobi_0.67")
    p.add_argument("--n_test", type=int, default=24)
    p.add_argument("--t_max", type=int, default=2500, help="plot horizon in iterations")
    p.add_argument("--max_iters", type=int, default=10000)
    p.add_argument("--time_cap", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=72)
    p.add_argument("--hints_tau", type=int, default=25)
    p.add_argument("--don_ckp", type=str, required=True)
    p.add_argument("--router_ckp", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    torch.set_num_threads(1)
    pde = make_pde("AnisoDiff", args.N, eps_x=args.eps_x, eps_y=args.eps_y)
    solver = make_solver(pde, args.solver)
    corrector = DeepONetCorrector(args.don_ckp, device=args.device, threads=1)
    router = LiteRouter.load(args.router_ckp)

    grf = GRF2D(args.N, rng=np.random.default_rng(args.seed))
    f_test = grf.sample(args.n_test)
    u_truth = demean(pde.solve_direct(f_test))
    _ = corrector.correct(f_test[:1])
    _ = solver.step(np.zeros_like(f_test[:1]), f_test[:1])

    policies = ["hints25", "router", "oracle"]
    out = {
        "args": vars(args),
        "policies": {},
    }
    T = args.t_max

    for policy in policies:
        mat = np.zeros((args.n_test, T), dtype=np.int8)
        lengths = []
        n_no = []
        for i in range(args.n_test):
            print(f"[{policy}] sample {i+1}/{args.n_test}", flush=True)
            tr = run_rollout(
                pde,
                solver,
                f_test[i:i + 1],
                u_truth[i:i + 1],
                policy if policy != "hints25" else f"hints{args.hints_tau}",
                corrector=corrector,
                router=router if policy == "router" else None,
                max_iters=args.max_iters,
                time_cap=args.time_cap,
            )
            d = tr["decision"]
            d = d[d >= 0]  # drop terminal -1
            lengths.append(int(len(d)))
            n_no.append(int((d == 1).sum()))
            L = min(T, len(d))
            mat[i, :L] = d[:L].astype(np.int8)
            # if shorter than T, leave zeros (Jacobi) — or mark inactive?
            # Better: use -1 for padded so mean only over active
            if L < T:
                mat[i, L:] = -1

        active = mat >= 0
        no = (mat == 1).astype(np.float64)
        mean_no = np.zeros(T)
        for t in range(T):
            a = active[:, t]
            mean_no[t] = float(no[a, t].mean()) if a.any() else float("nan")

        # downsample heatmap for canvas (every k iters, keep first 80 cols)
        # Store full mean_no but downsample matrix
        stride = max(1, T // 80)
        mat_ds = mat[:, ::stride]
        # still truncate display width
        if mat_ds.shape[1] > 100:
            mat_ds = mat_ds[:, :100]

        out["policies"][policy] = {
            "lengths": lengths,
            "n_no": n_no,
            "mean_no_rate": [None if (isinstance(x, float) and np.isnan(x)) else float(x) for x in mean_no.tolist()],
            "heatmap": mat_ds.tolist(),  # -1 pad, 0 jacobi, 1 NO
            "heatmap_stride": stride,
            "t_max": T,
        }
        print(
            f"[{policy}] median len={np.median(lengths):.0f} "
            f"median NO={np.median(n_no):.0f} "
            f"mean NO rate@t=0..200={np.nanmean(mean_no[:200]):.3f}",
            flush=True,
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh)
    print("saved", args.out)


if __name__ == "__main__":
    main()
