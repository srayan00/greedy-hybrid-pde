"""Iteration-wise NO-usage diagnostic: oracle vs learned routers.

For each policy we roll out per sample and record, at every iteration, the
policy's decision AND the oracle label at that same state (one-step lookahead
with both experts). This separates two failure modes:
  - imitation gap: router disagrees with the oracle label on its own states
  - timing gap:    router fires NO at different phases of the solve

Outputs a JSON with per-sample decision/label sequences plus a summary, and a
PNG with NO-rate curves, agreement curves, and decision rasters.
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
from fast_pde import GRF2D, demean, l2, make_solver
from pde_factory import make_pde
from train_and_bench import load_router


def rollout(pde, solver, corrector, f, u_truth, chooser, *, max_iters, err_tol):
    """f, u_truth: (1, N, N). chooser: 'oracle' | 'classical' | callable."""
    u = np.zeros_like(f)
    fn = float(l2(f)[0])
    un = float(l2(demean(u_truth))[0])
    prev_dec = 0
    dec, lab, rerr = [], [], []
    for it in range(max_iters):
        r = pde.residual(u, f)
        rel_res = float(l2(r)[0]) / fn
        rel_err = float(l2(demean(u - u_truth))[0]) / un
        if rel_err <= err_tol or rel_res <= 1e-13:
            break
        u_c = solver.step(u, f, r)
        u_n = u + corrector.correct(r)
        e_c = float(l2(demean(u_c - u_truth))[0])
        e_n = float(l2(demean(u_n - u_truth))[0])
        label = int(e_n < e_c)
        if chooser == "oracle":
            d = label
        elif chooser == "classical":
            d = 0
        else:
            d = chooser(rel_res, it, prev_dec, r)
        u = u_n if d else u_c
        dec.append(d)
        lab.append(label)
        rerr.append(rel_err)
        prev_dec = d
    return {"dec": dec, "lab": lab, "rel_err": rerr, "iters": len(dec)}


def summarize(runs):
    iters = [r["iters"] for r in runs]
    no_calls = [int(np.sum(r["dec"])) for r in runs]
    # timing of NO calls as fraction of trajectory length
    pos = []
    for r in runs:
        d = np.asarray(r["dec"])
        w = np.where(d == 1)[0]
        if len(w) and r["iters"] > 1:
            pos.extend((w / (r["iters"] - 1)).tolist())
    dec = np.concatenate([np.asarray(r["dec"]) for r in runs])
    lab = np.concatenate([np.asarray(r["lab"]) for r in runs])
    agree = float((dec == lab).mean()) if len(dec) else float("nan")
    # precision/recall of the NO action vs oracle label, on-policy
    tp = float(((dec == 1) & (lab == 1)).sum())
    prec = tp / max(float((dec == 1).sum()), 1.0)
    rec = tp / max(float((lab == 1).sum()), 1.0)
    return {
        "median_iters": float(np.median(iters)),
        "median_no_calls": float(np.median(no_calls)),
        "label_no_frac": float(lab.mean()) if len(lab) else float("nan"),
        "agree": agree,
        "no_precision": prec,
        "no_recall": rec,
        "no_pos_quartiles": ([float(q) for q in np.quantile(pos, [0.25, 0.5, 0.75])]
                             if pos else None),
    }


def binned_rate(runs, key, n_bins=40):
    """Mean of `key` vs normalized solve progress, averaged over samples."""
    out = np.full((len(runs), n_bins), np.nan)
    for i, r in enumerate(runs):
        v = np.asarray(r[key], dtype=float)
        if len(v) < 2:
            continue
        edges = np.linspace(0, len(v), n_bins + 1).astype(int)
        for b in range(n_bins):
            s, e = edges[b], max(edges[b + 1], edges[b] + 1)
            out[i, b] = v[s:e].mean()
    return np.nanmean(out, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--equation", type=str, required=True)
    p.add_argument("--N", type=int, default=63)
    p.add_argument("--solver", type=str, required=True)
    p.add_argument("--eps_x", type=float, default=0.01)
    p.add_argument("--eps_y", type=float, default=1.0)
    p.add_argument("--b_vel", type=float, default=20.0)
    p.add_argument("--don_ckp", type=str, required=True)
    p.add_argument("--routers", type=str, nargs="+", default=[],
                   help="label=path pairs, e.g. gru=ckp/x.pth mlp=ckp/y.pth")
    p.add_argument("--n_test", type=int, default=12)
    p.add_argument("--err_tol", type=float, default=1e-6)
    p.add_argument("--max_iters", type=int, default=8000)
    p.add_argument("--seed", type=int, default=72)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--plot", type=str, default=None)
    args = p.parse_args()

    torch.set_num_threads(8)
    pde = make_pde(args.equation, args.N, b_vel=args.b_vel,
                   eps_x=args.eps_x, eps_y=args.eps_y)
    solver = make_solver(pde, args.solver)
    corrector = DeepONetCorrector(args.don_ckp, device="cpu", threads=8)

    grf = GRF2D(args.N, rng=np.random.default_rng(args.seed))
    f_test = grf.sample(args.n_test)
    u_truth = demean(pde.solve_direct(f_test))

    policies = {"oracle": "oracle"}
    for spec in args.routers:
        label, path = spec.split("=", 1)
        policies[label] = load_router(path)

    results = {"args": vars(args), "policies": {}}
    for name, pol in policies.items():
        runs = []
        for i in range(args.n_test):
            if pol == "oracle":
                chooser = "oracle"
            else:
                pol.reset()
                chooser = (lambda rel, it, pd, r, _p=pol:
                           _p.decide(rel, it, pd, r=r, pde=pde))
            runs.append(rollout(pde, solver, corrector,
                                f_test[i:i + 1], u_truth[i:i + 1],
                                chooser, max_iters=args.max_iters,
                                err_tol=args.err_tol))
        s = summarize(runs)
        results["policies"][name] = {
            "summary": s,
            "runs": [{"dec": r["dec"], "lab": r["lab"], "iters": r["iters"]}
                     for r in runs],
        }
        print(f"[{name}] iters(med)={s['median_iters']:.0f} "
              f"NO(med)={s['median_no_calls']:.0f} "
              f"label_no_frac={s['label_no_frac']:.3f} "
              f"agree={s['agree']:.3f} prec={s['no_precision']:.3f} "
              f"rec={s['no_recall']:.3f} pos_q={s['no_pos_quartiles']}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh)
    print("saved", args.out)

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = list(results["policies"].keys())
        colors = {"oracle": "k", "gru": "tab:red", "mlp_norm": "tab:blue",
                  "mlp_raw": "tab:cyan"}
        fig, axes = plt.subplots(3, 1, figsize=(9, 10))

        ax = axes[0]
        x = np.linspace(0, 1, 40)
        for name in names:
            runs = results["policies"][name]["runs"]
            ax.plot(x, binned_rate(runs, "dec"), label=f"{name} (chosen)",
                    color=colors.get(name, None), lw=2)
        ax.plot(x, binned_rate(results["policies"]["oracle"]["runs"], "lab"),
                "k--", lw=1, label="oracle label")
        ax.set_ylabel("NO-call rate")
        ax.set_xlabel("solve progress (fraction of iterations to tol)")
        ax.set_title(f"{args.equation} N={args.N} {args.solver}: when does each "
                     f"policy call the NO?")
        ax.legend(fontsize=8)

        ax = axes[1]
        for name in names:
            if name == "oracle":
                continue
            runs = results["policies"][name]["runs"]
            agree = [{"agree": (np.asarray(r["dec"]) == np.asarray(r["lab"])).astype(float).tolist(),
                      "iters": r["iters"]} for r in runs]
            ax.plot(x, binned_rate(agree, "agree"), label=name,
                    color=colors.get(name, None), lw=2)
        ax.axhline(1.0, color="k", ls=":", lw=0.5)
        ax.set_ylabel("on-policy agreement with oracle label")
        ax.set_xlabel("solve progress")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)

        ax = axes[2]
        yt, yl = [], []
        row = 0
        for name in names:
            runs = results["policies"][name]["runs"][:6]
            for j, r in enumerate(runs):
                d = np.asarray(r["dec"])
                w = np.where(d == 1)[0]
                if r["iters"] > 1:
                    ax.scatter(w / (r["iters"] - 1), np.full(len(w), row),
                               s=4, color=colors.get(name, "gray"), marker="|")
                row += 1
            yt.append(row - len(runs) / 2 - 0.5)
            yl.append(name)
            row += 1
        ax.set_yticks(yt)
        ax.set_yticklabels(yl)
        ax.set_xlabel("solve progress")
        ax.set_title("NO calls per sample (first 6 samples each)")
        ax.set_xlim(0, 1)

        fig.tight_layout()
        fig.savefig(args.plot, dpi=140)
        print("saved", args.plot)


if __name__ == "__main__":
    main()
