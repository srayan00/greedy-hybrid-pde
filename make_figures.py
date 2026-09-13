"""Figures for the cost-aware wall-clock study.

  paper/neurips_images/ca_deeponet_predictions.png   corrector one-shot predictions
  paper/neurips_images/ca_router_usage.png           corrector-call frequency vs iteration (128^2, all equations)
  paper/neurips_images/ca_usage_<eq>.png             corrector-call frequency vs iteration, rows = grids
  paper/neurips_images/ca_router_decisions_<eq>.png  decision rasters across test instances
  paper/neurips_images/ca_convergence.png            error vs wall-clock time, representative instances
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fast_pde import FastStencilPDE, GRF2D, demean, l2
from corrector import DeepONetCorrector

SOLVER_NAMES = {"jacobi": "Jacobi", "jacobi_0.67": "Jacobi (0.67)", "gs": "GS",
                "ssor": "SymGS", "sor_1.5": "SOR (1.5)"}
SOLVER_ORDER = ["jacobi", "jacobi_0.67", "gs", "ssor", "sor_1.5"]
EQS = ["Poisson", "ConvDiff", "AnisoDiff"]
EQ_NAMES = {"Poisson": "Poisson", "ConvDiff": "ConvDiff", "AnisoDiff": "AnisoDiff"}
OUT = "paper/neurips_images"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                     "legend.fontsize": 7.5, "figure.dpi": 150})


RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")


def load():
    R = {}
    for path in sorted(glob.glob(f"{RESULTS_DIR}/*.json")):
        d = json.load(open(path))
        if "args" not in d or "ensemble" not in d["args"]:  # usage/seeds/overheads files, not benchmark output
            continue
        a = d["args"]
        for gkey, g in d["groups"].items():
            R[(a["equation"], a["N"], gkey, bool(a["ensemble"]))] = (d, g)
    return R


def fig_predictions(N=128, seed=72, idx=(0, 1)):
    eqs = [eq for eq in EQS if os.path.exists(f"checkpoints/deeponet_{eq}_{N}_best.pth")]
    fig, axes = plt.subplots(2 * len(eqs), 4, figsize=(9.2, 2.2 * 2 * len(eqs)))
    for ei, eq in enumerate(eqs):
        pde = FastStencilPDE(N, equation=eq)
        corr = DeepONetCorrector(f"checkpoints/deeponet_{eq}_{N}_best.pth")
        f = GRF2D(N, rng=np.random.default_rng(seed)).sample(max(idx) + 1)
        u = pde.solve_direct(f)
        du = corr.correct(f)
        for r, i in enumerate(idx):
            ax = axes[2 * ei + r]
            rel = float(l2(demean(du[i] - u[i])) / l2(u[i]))
            ims = [f[i], u[i], du[i], u[i] - du[i]]
            titles = [f"{eq}: forcing $f$", "solution $u_h$",
                      f"DeepONet $C_{{\\mathrm{{NO}}}}(f)$ (rel. err. {rel:.1e})", "error $u_h - C_{\\mathrm{NO}}(f)$"]
            for a, im, t in zip(ax, ims, titles):
                vmax = np.abs(im).max()
                m = a.imshow(im.T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
                a.set_title(t, pad=4)
                a.set_xticks([])
                a.set_yticks([])
                cb = plt.colorbar(m, ax=a, fraction=0.046, pad=0.02, ticks=[-vmax, 0, vmax])
                cb.ax.set_yticklabels([f"{-vmax:.1e}", "0", f"{vmax:.1e}"], fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/ca_deeponet_predictions.png", bbox_inches="tight")
    plt.close(fig)


def load_usage():
    """Untimed decision traces of all test instances (make_usage_data.py);
    falls back to the 4 stored benchmark curves when absent."""
    U = {}
    for path in sorted(glob.glob(f"{RESULTS_DIR}/usage_*.json")):
        d = json.load(open(path))
        a = d["args"]
        for spec, g in d["groups"].items():
            U[(a["equation"], a["N"], spec)] = g
    return U


def op_sequences(R, U, eq, spec, pol):
    k = [k for k in R if k[0] == eq and k[2] == spec and not k[3]]
    if (eq, k[0][1], spec) in U and pol in U[(eq, k[0][1], spec)]["policies"]:
        g = U[(eq, k[0][1], spec)]
        return [np.asarray(s_) for s_ in g["policies"][pol]], len(g["ops"]) - 1
    g = R[k[0]][1]
    return [np.asarray(cv["op"]) for cv in g["curves"].get(pol, [])], len(g["ops"]) - 1


def usage_curve(seqs, T, K_no):
    """Fraction of test instances that execute a corrector iteration at each
    iteration index (runs that already converged count as 'no call')."""
    cnt = np.zeros(T)
    for op in seqs:
        m = min(T, len(op))
        cnt[:m] += (op[:m] == K_no)
    return cnt / max(len(seqs), 1)


def fig_usage(R, T=60):
    U = load_usage()
    specs = [s for s in SOLVER_ORDER if any(k[2] == s and not k[3] for k in R)]
    fig, axes = plt.subplots(len(EQS), len(specs), figsize=(2.2 * len(specs), 1.9 * len(EQS)), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for ei, eq in enumerate(EQS):
        for si, spec in enumerate(specs):
            ax = axes[ei, si]
            k = [k for k in R if k[0] == eq and k[2] == spec and not k[3]]
            if not k:
                ax.axis("off")
                continue
            for pol, lab, st in [("router", "learned router", "-"), ("oracle", "cost-aware oracle", "--"),
                                 ("hints25", "HINTS ($\\tau{=}25$)", ":")]:
                seqs, K_no = op_sequences(R, U, eq, spec, pol)
                if seqs:
                    ax.plot(np.arange(1, T + 1), usage_curve(seqs, T, K_no), st, lw=1.3, label=lab)
            ax.set_title(f"{eq}, {SOLVER_NAMES[spec]}")
            if ei == len(EQS) - 1:
                ax.set_xlabel("iteration")
            if si == 0:
                ax.set_ylabel("fraction of runs\ncalling the corrector")
    axes[0, 0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{OUT}/ca_router_usage.png", bbox_inches="tight")
    plt.close(fig)


def fig_usage_grids(R, T=60):
    """Per equation: corrector-call frequency vs iteration, rows = grids, cols = pairings
    (router / cost-aware oracle / HINTS-25), from the untimed traces of all test instances."""
    U = load_usage()
    order = SOLVER_ORDER + ["mg"]
    names = dict(SOLVER_NAMES, mg="Multigrid")
    for eq in EQS:
        Ns = sorted(set(k[1] for k in R if k[0] == eq and not k[3]))
        specs = [s for s in order if any(k[0] == eq and k[2] == s and not k[3] for k in R)]
        if not Ns or not specs:
            continue
        fig, axes = plt.subplots(len(Ns), len(specs), figsize=(2.2 * len(specs), 1.9 * len(Ns)), sharex=True, sharey=True, squeeze=False)
        for ni, N in enumerate(Ns):
            for si, spec in enumerate(specs):
                ax = axes[ni, si]
                k = [k for k in R if k[0] == eq and k[1] == N and k[2] == spec and not k[3]]
                if not k:
                    ax.axis("off")
                    continue
                g = R[k[0]][1]
                for pol, lab, st in [("router", "learned router", "-"), ("oracle", "cost-aware oracle", "--"),
                                     ("hints25", "HINTS ($\\tau{=}25$)", ":")]:
                    if (eq, N, spec) in U and pol in U[(eq, N, spec)]["policies"]:
                        seqs = [np.asarray(s_) for s_ in U[(eq, N, spec)]["policies"][pol]]
                    else:
                        seqs = [np.asarray(cv["op"]) for cv in g["curves"].get(pol, [])]
                    K_no = len(g["ops"]) - 1
                    if seqs:
                        ax.plot(np.arange(1, T + 1), usage_curve(seqs, T, K_no), st, lw=1.3, label=lab)
                ax.set_title(f"${N}^2$, {names[spec]}")
                if ni == len(Ns) - 1:
                    ax.set_xlabel("iteration")
                if si == 0:
                    ax.set_ylabel("fraction of runs\ncalling the corrector")
        axes[0, 0].legend(loc="upper right")
        fig.suptitle(f"{EQ_NAMES[eq]}: corrector usage of the learned router, the cost-aware oracle and HINTS", y=1.01)
        fig.tight_layout()
        fig.savefig(f"{OUT}/ca_usage_{eq.lower()}.png", bbox_inches="tight")
        plt.close(fig)


def fig_decisions(R, T=60):
    U = load_usage()
    specs = [s for s in SOLVER_ORDER if any(k[2] == s and not k[3] for k in R)]
    for eq in EQS:
        fig, axes = plt.subplots(2, len(specs), figsize=(2.2 * len(specs), 3.6), sharex=True, sharey=True)
        axes = np.atleast_2d(axes)
        for si, spec in enumerate(specs):
            k = [k for k in R if k[0] == eq and k[2] == spec and not k[3]]
            if not k:
                continue
            for pi, pol in enumerate(["router", "oracle"]):
                ax = axes[pi, si]
                seqs, K_no = op_sequences(R, U, eq, spec, pol)
                M = np.full((len(seqs), T), np.nan)
                for i, op in enumerate(seqs):
                    op = op[:T]
                    M[i, :len(op)] = (op == K_no)
                ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
                if pi == 0:
                    ax.set_title(f"{SOLVER_NAMES[spec]}")
                if si == 0:
                    ax.set_ylabel("learned router\ntest instance" if pol == "router" else "cost-aware oracle\ntest instance")
                if pi == 1:
                    ax.set_xlabel("iteration")
        fig.suptitle(f"{eq}: corrector calls (dark) over the first {T} iterations")
        fig.tight_layout()
        fig.savefig(f"{OUT}/ca_router_decisions_{eq.lower()}.png", bbox_inches="tight")
        plt.close(fig)


def fig_convergence(R, inst=0):
    specs = [s for s in SOLVER_ORDER if any(k[2] == s and not k[3] for k in R)]
    fig, axes = plt.subplots(len(EQS), len(specs), figsize=(2.3 * len(specs), 2.0 * len(EQS)), sharey=True)
    axes = np.atleast_2d(axes)
    for ei, eq in enumerate(EQS):
        for si, spec in enumerate(specs):
            ax = axes[ei, si]
            k = [k for k in R if k[0] == eq and k[2] == spec and not k[3]]
            if not k:
                ax.axis("off")
                continue
            d, g = R[k[0]]
            h2 = d["h2"]
            for pol, lab, st in [("classical", "solver only", "-"), ("hints25", "HINTS ($\\tau{=}25$)", ":"),
                                 ("hints5", "HINTS ($\\tau{=}5$)", "-."), ("router", "learned router", "-"),
                                 ("oracle", "cost-aware oracle", "--")]:
                if pol not in g["curves"]:
                    continue
                cv = g["curves"][pol][inst]
                t = np.asarray(cv["t_live"]) * 1e3
                e = np.asarray(cv["rel_err"])
                n = min(len(t), len(e))
                ax.semilogy(t[:n], e[:n], st, lw=1.2, label=lab)
            ax.axhline(h2, color="gray", lw=0.6, ls="--")
            ax.set_xscale("log")
            ax.set_title(f"{eq}, {SOLVER_NAMES[spec]}")
            if ei == len(EQS) - 1:
                ax.set_xlabel("wall-clock time [ms]")
            if si == 0:
                ax.set_ylabel("relative error")
            ax.set_ylim(1e-10, 2)
    axes[0, 0].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/ca_convergence.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import sys
    R = load()
    which = sys.argv[1:] or ["pred", "usage", "dec", "conv"]
    if "pred" in which:
        fig_predictions()
    if "usage" in which and R:
        fig_usage(R)
        fig_usage_grids(R)
    if "dec" in which and R:
        fig_decisions(R)
    if "conv" in which and R:
        fig_convergence(R)
    print("figures written to", OUT)
