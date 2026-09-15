"""Figures for the cost-aware wall-clock study.

  paper/neurips_images/ca_deeponet_predictions.png   corrector one-shot predictions
  paper/neurips_images/ca_router_usage.png           corrector-call frequency vs iteration (128^2, all equations)
  paper/neurips_images/ca_usage_<eq>.png             corrector-call frequency vs iteration, rows = grids
  paper/neurips_images/ca_router_decisions_<eq>.png  learned-router operations on every test instance (pairings + nested ensemble)
  paper/neurips_images/ca_router_decisions_overview.png  the same across equations, with HINTS on the same instances
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
EQS = ["Poisson", "ConvDiff", "AnisoDiff", "VarCoeff"]
EQ_NAMES = {"Poisson": "Poisson", "ConvDiff": "ConvDiff", "AnisoDiff": "AnisoDiff", "VarCoeff": "VarCoeff"}
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
                                 ("hints15", "HINTS ($\\tau{=}15$)", ":")]:
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
                                     ("hints15", "HINTS ($\\tau{=}15$)", ":")]:
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


# ---------------------------------------------------------------- decision timelines across test instances
# Colours: corrector calls are ink dots everywhere; sweeps of the single paired solver are muted gray (the panel title
# names the solver); only the ensemble panels colour solvers, with categorical slots 1-3 of the reference palette
# (validated all-pairs on a white surface).
DEC_INK, DEC_INK2, DEC_MUTED, DEC_AXIS = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
DEC_CAT = {"jacobi": "#2a78d6", "jacobi_0.67": "#eb6834", "gs": "#1baf7a"}
DEC_SPECS = ["jacobi", "jacobi_0.67", "gs", "ssor", "sor_1.5", "mg", "linegs"]
DEC_NAMES = dict(SOLVER_NAMES, mg="Multigrid", linegs="Line GS")
NESTED3 = "jacobi+jacobi_0.67+gs"


def load_cell(eq, N, spec, ens=False):
    """One benchmark result file (None if absent or being rewritten)."""
    path = f"{RESULTS_DIR}/{eq}_{N}_{'ens_' if ens else ''}{spec}.json"
    try:
        d = json.load(open(path))
    except (OSError, ValueError):
        return None
    return d, next(iter(d["groups"].values()))


def decision_timelines(g, pol, tol=1e-8):
    """Per test instance, from the operation sequence stored for its timed replay, cut where the relative error first
    falls below `tol`: the solver runs (operation, start, end) and the corrector-call positions, in units of one
    corrector call. None if the cell predates stored sequences."""
    ops, costs = g["ops"], g["costs"]
    k_no = ops.index("no")
    unit = [costs[o] / costs["no"] for o in ops]
    T = []
    for r in g["policies"].get(pol, []):
        if not r.get("op_rle"):
            return None
        key = min(r["tol"], key=lambda k: abs(float(k) - tol))
        it = r["tol"][key]["iters"]
        limit = it if it is not None else sum(n for _, n in r["op_rle"])
        runs, calls, x, done = [], [], 0.0, 0
        for op, n in r["op_rle"]:
            n = min(n, limit - done)
            if n <= 0:
                break
            if op == k_no:
                calls.extend(x + j * unit[op] for j in range(n))
            else:
                runs.append((op, x, x + n * unit[op]))
            x += n * unit[op]
            done += n
        T.append({"runs": runs, "calls": calls, "total": x, "reached": it is not None})
    return T or None


def draw_timelines(ax, T, order, ops, ens, x_max):
    """One row per test instance: solver runs as thin lines and corrector calls as dots at the middle of their cost
    interval; x = cumulative cost in corrector calls, linear below one call and logarithmic above."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    dense = max(len(T[i]["calls"]) for i in order) > 12          # a fixed schedule: many calls per instance
    for row, i in enumerate(order):
        for op, s, e in T[i]["runs"]:
            ax.hlines(row, s, e, color=DEC_CAT.get(ops[op], DEC_INK2) if ens else DEC_MUTED, lw=1.0)
        c = np.asarray(T[i]["calls"]) + 0.5
        ax.scatter(c, np.full(len(c), row), s=1.0 if dense else 3.2, color=DEC_INK, alpha=0.45 if dense else 1.0,
                   linewidths=0, zorder=3, rasterized=dense)
    ax.set_xscale("symlog", linthresh=1.0, linscale=0.35)
    ax.set_xlim(0, x_max)
    ticks = [0, 1] + [v for v in (3, 10, 30, 100, 300, 1000, 3000) if v <= x_max]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_yticks([0, len(order) - 1])
    ax.set_yticklabels(["1", str(len(order))])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(DEC_AXIS)
    ax.tick_params(colors=DEC_INK2, labelsize=7)


def decision_handles(ens_ops, paired=True):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=DEC_INK, marker="o", ls="", ms=3.5, label="corrector call")]
    if paired:
        h.append(Line2D([], [], color=DEC_MUTED, lw=1.5, label="sweeps of the paired solver"))
    h += [Line2D([], [], color=DEC_CAT[o], lw=1.5, label=f"{DEC_NAMES[o]} sweeps (ensemble)") for o in ens_ops]
    return h


def pending(ax, title):
    ax.set_title(title, color=DEC_INK)
    ax.text(0.5, 0.5, "sequences pending", ha="center", va="center", transform=ax.transAxes, color=DEC_INK2, fontsize=7.5)
    ax.set_xticks([])
    ax.set_yticks([])


def fig_decisions(eqs=EQS, N=128):
    """Per equation: the learned router's operations on every test instance (rows, sorted by cost to 1e-8) for every
    pairing and the nested ensemble {Jacobi, Jacobi (0.67), GS}, from the stored sequences of the timed replays."""
    for eq in eqs:
        cells = [(s, False) for s in DEC_SPECS if os.path.exists(f"{RESULTS_DIR}/{eq}_{N}_{s}.json")]
        if os.path.exists(f"{RESULTS_DIR}/{eq}_{N}_ens_{NESTED3}.json"):
            cells.append((NESTED3, True))
        if not cells:
            continue
        ncol = 4
        nrow = int(np.ceil((len(cells) + 1) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(2.0 * ncol, 2.0 * nrow), squeeze=False)
        for ci, (spec, ens) in enumerate(cells):
            ax = axes.flat[ci]
            title = "ensemble {J, J(0.67), GS}" if ens else DEC_NAMES[spec]
            cell = load_cell(eq, N, spec, ens)
            T = decision_timelines(cell[1], "router") if cell else None
            if T is None:
                pending(ax, title)
                continue
            order = np.argsort([t["total"] for t in T], kind="stable")
            draw_timelines(ax, T, order, cell[1]["ops"], ens, 1.25 * max(t["total"] for t in T) + 0.5)
            ax.set_title(title, color=DEC_INK)
            if ci % ncol == 0:
                ax.set_ylabel("test instance", color=DEC_INK2)
            if ci + ncol >= len(cells) + 1:
                ax.set_xlabel("cost (corrector calls)", color=DEC_INK2)
            del cell
        lg = axes.flat[len(cells)]
        lg.axis("off")
        lg.legend(handles=decision_handles(["jacobi", "jacobi_0.67", "gs"] if any(e for _, e in cells) else []),
                  loc="center", frameon=False, fontsize=7.5)
        for ax in axes.flat[len(cells) + 1:]:
            ax.axis("off")
        fig.suptitle(f"{EQ_NAMES[eq]}, ${N}^2$: learned-router operations on each test instance until relative error $10^{{-8}}$",
                     color=DEC_INK)
        fig.tight_layout()
        fig.savefig(f"{OUT}/ca_router_decisions_{eq.lower()}.png", bbox_inches="tight", dpi=200)
        plt.close(fig)


def fig_decisions_overview(eqs=EQS, N=128):
    """One row per equation: the learned router in the Jacobi pairing, HINTS (tau = 15) in the same pairing on the same
    test instances in the same order, and the learned router in the nested ensemble {Jacobi, Jacobi (0.67), GS}."""
    fig, axes = plt.subplots(len(eqs), 3, figsize=(7.2, 1.85 * len(eqs)), squeeze=False)
    titles = ["learned router\nJacobi pairing", "HINTS ($\\tau{=}15$)\nJacobi pairing", "learned router\nensemble {J, J(0.67), GS}"]
    for ei, eq in enumerate(eqs):
        pw, en = load_cell(eq, N, "jacobi"), load_cell(eq, N, NESTED3, True)
        T_r = decision_timelines(pw[1], "router") if pw else None
        T_h = decision_timelines(pw[1], "hints15") if pw else None
        panels = [(T_r, pw, False), (T_h, pw, False), (decision_timelines(en[1], "router") if en else None, en, True)]
        order_pw = np.argsort([t["total"] for t in T_r], kind="stable") if T_r else None
        tots = [t["total"] for t in (T_r or []) + (T_h or [])]
        x_pw = 1.25 * max(tots) + 0.5 if tots else None
        for pi, (T, cell, ens) in enumerate(panels):
            ax = axes[ei, pi]
            if T is None or (not ens and order_pw is None):
                pending(ax, titles[pi] if ei == 0 else "")
            else:
                order = np.argsort([t["total"] for t in T], kind="stable") if ens else order_pw
                draw_timelines(ax, T, order, cell[1]["ops"], ens, 1.25 * max(t["total"] for t in T) + 0.5 if ens else x_pw)
                if ei == 0:
                    ax.set_title(titles[pi], color=DEC_INK)
            if pi == 0:
                ax.set_ylabel(f"{EQ_NAMES[eq]}\ntest instance", color=DEC_INK2)
            if ei == len(eqs) - 1:
                ax.set_xlabel("cost (corrector calls)", color=DEC_INK2)
        del pw, en
    fig.legend(handles=decision_handles(["jacobi", "jacobi_0.67", "gs"]), loc="lower center", ncol=5, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(f"{OUT}/ca_router_decisions_overview.png", bbox_inches="tight", dpi=200)
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
            for pol, lab, st in [("classical", "solver only", "-"), ("hints15", "HINTS ($\\tau{=}15$)", ":"),
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
    if "dec" in which:
        fig_decisions()
        fig_decisions_overview()
    if "conv" in which and R:
        fig_convergence(R)
    print("figures written to", OUT)
