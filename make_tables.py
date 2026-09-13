"""Generate the LaTeX tables of the cost-aware wall-clock study from
results/*.json (written by bench.py, bench_baselines.py, bench_seeds.py,
check_assumptions.py) into paper/costaware_tables.tex.

Per equation <eq> in {poisson, conv, aniso} (all grids side by side):
  \\catime<eq>     time to 1e-3 / h^2 / 1e-8 for every pairing and policy, plus the
                   classical baselines (multigrid, Krylov) -- one consolidated table
  \\caspeed<eq>    paired median speedup of the router with one-sided Wilcoxon p-values
  \\causage<eq>    iterations and corrector calls to h^2
  \\cacosts<eq>    measured per-iteration costs and macro-action sizes
  \\caauc<eq>      iteration-based metrics (AUC / final error over T iterations), 128^2
  \\caseeds<eq>    five-seed router retraining trials, 128^2
Cross-equation:
  \\camainall      main-text table: HINTS / best tau / router at h^2 on all grids
  \\caensnest      nested ensembles {J} c {J, dJ} c {J, dJ, GS} (monotonicity / separation)
  \\caens, \\caensusage, \\caensbig   earlier ensemble runs (128^2)
  \\caoverheads, \\caamort           per-operation costs vs N, training amortisation
  \\caassump, \\caassumpB            verification of the theory assumptions
plus summary macros (\\caSp..., \\caNumCells..., \\caEns...) used in the text.
"""

import glob
import json
import math
import os
import re

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

SOLVER_NAMES = {"jacobi": "Jacobi", "jacobi_0.67": "Jacobi (0.67)", "gs": "GS",
                "ssor": "SymGS", "sor_1.5": "SOR (1.5)", "mg": "Multigrid"}
SOLVER_ORDER = ["jacobi", "jacobi_0.67", "gs", "ssor", "sor_1.5"]
PAIRINGS = SOLVER_ORDER + ["mg"]
POL_NAMES = {"classical": "Solver only", "hints25": "HINTS ($\\tau{=}25$)", "best": "HINTS (best $\\tau$)", "oneshot": "One-shot schedule",
             "hints5": "HINTS ($\\tau{=}5$)", "hints10": "HINTS ($\\tau{=}10$)",
             "hints50": "HINTS ($\\tau{=}50$)", "greedy": "Greedy oracle (Alg.~1)",
             "oracle": "Cost-aware oracle", "router": "Learned router (ours)"}
EQS = ["Poisson", "ConvDiff", "AnisoDiff"]
EQ_SUF = {"Poisson": "poisson", "ConvDiff": "conv", "AnisoDiff": "aniso"}
EQ_NAMES = {"Poisson": "Poisson", "ConvDiff": "Convection--diffusion", "AnisoDiff": "Anisotropic diffusion"}
BASE_NAMES = {"mg": "Multigrid V(2,2) alone", "cg": "CG", "pcg_ssor": "PCG (SymGS)", "pcg_mg": "PCG (multigrid)",
              "bicgstab": "BiCGSTAB", "bicgstab_mg": "BiCGSTAB (multigrid)", "gmres": "GMRES(20)"}
BASE_ORDER = ["mg", "cg", "bicgstab", "pcg_ssor", "pcg_mg", "bicgstab_mg", "gmres"]
NS = [128, 256, 512]
GRID_SUF = {128: "", 256: "B", 512: "C"}

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
MAIN_N = int(os.environ.get("MAIN_N", "128"))
OUT_TEX = os.environ.get("OUT_TEX", "paper/costaware_tables.tex")


# ----------------------------------------------------------------- loading
USED_FILES = []


def load(pattern=None, tag=""):
    """Benchmark outputs with the given file tag ('' = main runs; '_oneshot', '_seed73' = auxiliary runs)."""
    pattern = pattern or f"{RESULTS_DIR}/*.json"
    R = {}
    for path in sorted(glob.glob(pattern)):
        d = json.load(open(path))
        if "args" not in d or "ensemble" not in d["args"]:
            continue
        if (d["args"].get("tag") or "") != tag:
            continue
        USED_FILES.append(path)
        a = d["args"]
        for gkey, g in d["groups"].items():
            R[(a["equation"], a["N"], gkey, bool(a["ensemble"]))] = (d, g)
    return R


def holm_mask(pvals, alpha=0.05):
    """Boolean mask of the p-values that are significant after a Holm correction."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    ok = np.zeros(n, dtype=bool)
    for rank, idx in enumerate(order):
        if p[idx] <= alpha / (n - rank):
            ok[idx] = True
        else:
            break
    return ok


def tkey(d, tol):
    for k in d["tols"]:
        if abs(k - tol) <= 1e-9 * max(1.0, tol) or abs(k - tol) < 0.02 * tol:
            return f"{k:.6g}"
    raise KeyError(tol)


def times(rows, key, field="t_live"):
    return np.array([np.inf if r["tol"][key][field] is None else r["tol"][key][field] for r in rows])


def iters(rows, key):
    return np.array([np.inf if r["tol"][key]["iters"] is None else r["tol"][key]["iters"] for r in rows])


def times_lb(rows, key, field="t_live"):
    """Censored runs enter at the time spent up to the iteration cap (a lower bound;
    conservative on the baseline side of a paired test)."""
    tot = "t_total_live" if field == "t_live" else "t_total_wu"
    return np.array([r[tot] if r["tol"][key][field] is None else r["tol"][key][field] for r in rows])


def fmt_time(x):
    if not np.isfinite(x):
        return "--"
    if x < 1e-3:
        return f"{x*1e6:.0f}\\,$\\mu$s"
    if x < 1.0:
        return f"{x*1e3:.2f}\\,ms" if x < 0.01 else f"{x*1e3:.1f}\\,ms"
    return f"{x:.2f}\\,s"


def paired_speedup(base, ours):
    """Censoring-aware paired median speedup base/ours (inf-safe)."""
    both = np.isfinite(base) & np.isfinite(ours)
    r = np.full(len(base), np.nan)
    r[both] = base[both] / ours[both]
    r[np.isfinite(base) & ~np.isfinite(ours)] = 0.0
    r[~np.isfinite(base) & np.isfinite(ours)] = np.inf
    r[~np.isfinite(base) & ~np.isfinite(ours)] = 1.0
    return float(np.median(r)), r


def fmt_sp(sp):
    if not np.isfinite(sp):
        return "$>10^{3}\\times$"
    if sp >= 100:
        return f"{sp:.0f}$\\times$"
    if sp >= 10:
        return f"{sp:.1f}$\\times$"
    return f"{sp:.2f}$\\times$"


def wilcoxon_p(base, ours):
    """One-sided paired Wilcoxon on log ratios (alternative: ours faster), censoring-aware."""
    _, r = paired_speedup(base, ours)
    ok = np.isfinite(r) & (r > 0)
    if ok.sum() < 8 or np.allclose(r[ok], 1.0):
        return 1.0
    return float(wilcoxon(np.log(r[ok]), alternative="greater").pvalue)


def ttest_p(base, ours):
    ok = np.isfinite(base) & np.isfinite(ours)
    if ok.sum() < 8 or np.allclose(base[ok], ours[ok]):
        return 1.0
    return float(ttest_rel(np.log(base[ok]), np.log(ours[ok]), alternative="greater").pvalue)


def pstr(pv):
    if pv < 1e-10:
        return "$<10^{-10}$"
    if pv < 1e-3:
        return f"$10^{{{int(np.floor(np.log10(pv)))}}}$"
    return f"{pv:.3f}"


def time_cell(rows, key, bold=False, italic=False):
    """Median time with censoring marks; majority-censored -> lower bound."""
    ts = times(rows, key)
    cens = int((~np.isfinite(ts)).sum())
    if cens > len(ts) / 2:
        cap = np.median([r.get("t_total_live", np.nan) for r in rows])
        s = f"$>${fmt_time(cap)}$^{{\\dagger {cens}}}$" if np.isfinite(cap) else "--"
    else:
        s = fmt_time(np.median(ts)) + (f"$^{{\\dagger {cens}}}$" if cens else "")
    if bold:
        s = f"\\textbf{{{s}}}"
    if italic:
        s = f"\\textit{{{s}}}"
    return s


def sp_cell(base, ours, bold_thresh=1.10, bold=None):
    """'speedup (p)' with the one-sided p-value in the direction of the median.
    bold=None: bold if speedup >= bold_thresh and p < 0.01; bold=True/False: forced (Holm pass)."""
    s, sp, p = sp_parts(base, ours)
    if bold is None:
        bold = sp >= bold_thresh and p < 0.01
    return f"\\textbf{{{s}}}" if (bold and sp >= 1.0) else s


def sp_parts(base, ours):
    sp, _ = paired_speedup(base, ours)
    if sp >= 1.0:
        p = wilcoxon_p(base, ours)
        return f"{fmt_sp(sp)} ({pstr(p)})", sp, p
    p = wilcoxon_p(ours, base)
    return f"{fmt_sp(sp)} (slower, {pstr(p)})", sp, p


def rng_macro(out, name, vals):
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return
    out.append(f"\\newcommand{{\\{name}Min}}{{{fmt_sp(min(vals))}}}")
    out.append(f"\\newcommand{{\\{name}Max}}{{{fmt_sp(max(vals))}}}")


def pending(out, name, msg="results pending"):
    out.append(f"\\newcommand{{\\{name}}}{{\\begin{{tabular}}{{c}}({msg})\\end{{tabular}}}}")


# ------------------------------------------------------------------- main
def main():
    R = load()
    Ro = load(tag="_oneshot")     # static one-shot schedule (corrector once, then the solver)
    Rh = load(tag="_seed73")      # held-out confirmation runs (fresh test seed, 3 timed replays)
    out = []
    Bf = {}
    for path in sorted(glob.glob(f"{RESULTS_DIR}/baselines_*.json")):
        d = json.load(open(path))
        Bf[(d["args"]["equation"], d["args"]["N"])] = d
    Sf = {}
    for path in sorted(glob.glob(f"{RESULTS_DIR}/seeds_*.json")):
        d = json.load(open(path))
        Sf[(d["args"]["equation"], d["args"]["N"])] = d

    def cell(eq, N, spec):
        k = [q for q in R if q[0] == eq and q[1] == N and q[2] == spec and not q[3]]
        return R[k[0]] if k else None

    def base_times(d, m, tol):
        key = tkey(d, tol)
        return np.array([np.inf if r["tol"][key]["t_live"] is None else r["tol"][key]["t_live"] for r in d["methods"][m]])

    def base_rows_present(eq, N):
        d = Bf.get((eq, N))
        return [m for m in BASE_ORDER if d and m in d["methods"] and d["methods"][m]]

    def best_tau(P, key):
        taus = [p for p in P if p.startswith("hints")]
        return min(taus, key=lambda p: np.median(times(P[p], key)))

    def kry_name(eq):
        return "pcg_mg" if eq in ("Poisson", "AnisoDiff") else "bicgstab_mg"

    grids = {eq: [N for N in NS if any(cell(eq, N, s) for s in PAIRINGS)] for eq in EQS}

    # ================================================================ per-equation tables
    for eq in EQS:
        suf = EQ_SUF[eq]
        Ns = grids[eq]
        if not Ns:
            for name in ["catime", "caspeed", "causage", "cacosts", "caauc", "caseeds"]:
                pending(out, name + suf)
            continue
        tols_lab = ["$10^{-3}$", "$h^2$", "$10^{-8}$"]

        # ---------------------------------------------------------- consolidated times
        out.append(f"\\newcommand{{\\catime{suf}}}{{")
        out.append("\\begin{tabular}{ll" + "ccc" * len(Ns) + "}\n\\toprule")
        out.append("& & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{${N}\\times{N}$}}" for N in Ns) + " \\\\ "
                   + "".join(f"\\cmidrule(lr){{{3+3*i}-{5+3*i}}}" for i in range(len(Ns))))
        out.append("Pairing & Method & " + " & ".join(" & ".join(f"$\\varepsilon{{=}}{t[1:-1]}$" for t in tols_lab) for _ in Ns) + " \\\\ \\midrule")
        n_col = 2 + 3 * len(Ns)
        for spec in PAIRINGS:
            have = {N: cell(eq, N, spec) for N in Ns}
            if not any(have.values()):
                continue
            pols = ["classical", "hints25", "best", "oneshot", "greedy", "oracle", "router"]
            for pi, pol in enumerate(pols):
                row = [f"$\\{{\\mathrm{{NO}}, \\text{{{SOLVER_NAMES[spec]}}}\\}}$" if pi == 0 else "", POL_NAMES.get(pol, pol)]
                for N in Ns:
                    dg = have[N]
                    if dg is None:
                        row += ["--"] * 3
                        continue
                    d, g = dg
                    P = g["policies"]
                    for tol in [1e-3, d["h2"], 1e-8]:
                        key = tkey(d, tol)
                        if pol == "oneshot":
                            ko = [q for q in Ro if q[0] == eq and q[1] == N and q[2] == spec and not q[3]]
                            row.append(time_cell(Ro[ko[0]][1]["policies"]["oneshot"], tkey(Ro[ko[0]][0], tol)) if ko else "--")
                        elif pol == "best":
                            bt = best_tau(P, key)
                            row.append(time_cell(P[bt], key) + f"$_{{\\tau{{=}}{bt[5:]}}}$")
                        elif pol in P:
                            t_r = np.median(times(P["router"], key))
                            comp = [np.median(times(P[p_], key)) for p_ in P if p_.startswith("hints") or p_ == "classical"]
                            bold = pol == "router" and all(t_r <= c_ for c_ in comp)
                            row.append(time_cell(P[pol], key, bold=bold, italic=(pol == "oracle")))
                        else:
                            row.append("--")
                out.append(" & ".join(row) + " \\\\")
            out.append("\\midrule")
        bases = sorted(set(m for N in Ns for m in base_rows_present(eq, N)), key=BASE_ORDER.index)
        if bases:
            out.append(f"\\multicolumn{{{n_col}}}{{l}}{{\\emph{{Classical baselines without corrector (same stencil, same accounting)}}}} \\\\ \\midrule")
            for m in bases:
                row = ["", BASE_NAMES[m]]
                for N in Ns:
                    d = Bf.get((eq, N))
                    if d is None or m not in d["methods"] or not d["methods"][m]:
                        row += ["--"] * 3
                        continue
                    for tol in [1e-3, d["h2"], 1e-8]:
                        ts = base_times(d, m, tol)
                        cens = int((~np.isfinite(ts)).sum())
                        row.append(fmt_time(np.median(ts)) + (f"$^{{\\dagger {cens}}}$" if cens else ""))
                out.append(" & ".join(row) + " \\\\")
            out.append("\\midrule")
        out[-1] = "\\bottomrule\n\\end{tabular}}"

        # ---------------------------------------------------------- speedups with p-values
        has_base = any((eq, N) in Bf for N in Ns)
        nc = 5 if has_base else 3
        # pass 1: collect every comparison (speedup text, speedup, p); pass 2: Holm-corrected bolding
        comps = []   # (row_index, col_index, text, sp, p)
        rows_spec = []
        for spec in PAIRINGS:
            have = {N: cell(eq, N, spec) for N in Ns}
            if not any(have.values()):
                continue
            for ti, (tol_f, tlab) in enumerate([("h2", "$h^2$"), (1e-8, "$10^{-8}$")]):
                row = [f"$\\{{\\mathrm{{NO}}, \\text{{{SOLVER_NAMES[spec]}}}\\}}$" if ti == 0 else "", tlab]
                for N in Ns:
                    dg = have[N]
                    if dg is None:
                        row += ["--"] * nc
                        continue
                    d, g = dg
                    P = g["policies"]
                    tol = d["h2"] if tol_f == "h2" else tol_f
                    key = tkey(d, tol)
                    t_r = times(P["router"], key)
                    bases = [times_lb(P["hints25"], key), times_lb(P[best_tau(P, key)], key)]
                    ko = [q for q in Ro if q[0] == eq and q[1] == N and q[2] == spec and not q[3]]
                    bases.append(times_lb(Ro[ko[0]][1]["policies"]["oneshot"], tkey(Ro[ko[0]][0], tol)) if ko else None)
                    if has_base:
                        db = Bf.get((eq, N))
                        for m_ in ["mg", kry_name(eq)]:
                            bases.append(base_times(db, m_, tol) if (db is not None and m_ in db["methods"] and db["methods"][m_]) else None)
                    for b_ in bases:
                        if b_ is None:
                            row.append("--")
                        else:
                            s, sp, p = sp_parts(b_, t_r)
                            comps.append((len(rows_spec), len(row), s, sp, p))
                            row.append(s)
                rows_spec.append(row)
        sig = holm_mask([c[4] for c in comps]) if comps else []
        for (ri, ci, s, sp, p), ok in zip(comps, sig):
            if ok and sp >= 1.10:
                rows_spec[ri][ci] = f"\\textbf{{{s}}}"
        out.append(f"\\newcommand{{\\caspeed{suf}}}{{")
        out.append("\\begin{tabular}{ll" + "c" * nc * len(Ns) + "}\n\\toprule")
        out.append("& & " + " & ".join(f"\\multicolumn{{{nc}}}{{c}}{{${N}\\times{N}$}}" for N in Ns) + " \\\\ "
                   + "".join(f"\\cmidrule(lr){{{3+nc*i}-{2+nc*(i+1)}}}" for i in range(len(Ns))))
        out.append("Pairing & $\\varepsilon$ & " + " & ".join("vs.\\ HINTS-25 & vs.\\ best $\\tau$ & vs.\\ one-shot" + (" & vs.\\ multigrid & vs.\\ MG-Krylov" if has_base else "") for _ in Ns) + " \\\\ \\midrule")
        for row in rows_spec:
            out.append(" & ".join(row) + " \\\\")
        out.append("\\bottomrule\n\\end{tabular}}")

        # ---------------------------------------------------------- usage
        out.append(f"\\newcommand{{\\causage{suf}}}{{")
        out.append("\\begin{tabular}{l" + "ccc" * len(Ns) + "}\n\\toprule")
        out.append("& " + " & ".join(f"\\multicolumn{{3}}{{c}}{{${N}\\times{N}$}}" for N in Ns) + " \\\\ "
                   + "".join(f"\\cmidrule(lr){{{2+3*i}-{4+3*i}}}" for i in range(len(Ns))))
        out.append("Pairing & " + " & ".join("HINTS-25 & oracle & router" for _ in Ns) + " \\\\ \\midrule")
        for spec in PAIRINGS:
            have = {N: cell(eq, N, spec) for N in Ns}
            if not any(have.values()):
                continue
            row = [f"$\\{{\\mathrm{{NO}}, \\text{{{SOLVER_NAMES[spec]}}}\\}}$"]
            for N in Ns:
                dg = have[N]
                if dg is None:
                    row += ["--"] * 3
                    continue
                d, g = dg
                key = tkey(d, d["h2"])
                P = g["policies"]
                for pol in ["hints25", "oracle", "router"]:
                    it = iters(P[pol], key)
                    nno = np.array([np.nan if r["tol"][key]["no_calls"] is None else r["tol"][key]["no_calls"] for r in P[pol]])
                    row.append(f"{np.median(it):.0f} ({np.nanmedian(nno):.0f})" if np.isfinite(np.median(it)) else "--")
            out.append(" & ".join(row) + " \\\\")
        out.append("\\bottomrule\n\\end{tabular}}")

        # ---------------------------------------------------------- costs
        out.append(f"\\newcommand{{\\cacosts{suf}}}{{")
        out.append("\\begin{tabular}{l" + "ccc" * len(Ns) + "}\n\\toprule")
        out.append("& " + " & ".join(f"\\multicolumn{{3}}{{c}}{{${N}\\times{N}$}}" for N in Ns) + " \\\\ "
                   + "".join(f"\\cmidrule(lr){{{2+3*i}-{4+3*i}}}" for i in range(len(Ns))))
        out.append("Pairing & " + " & ".join("$c_{\\text{solver}}$ & $c_{\\mathrm{NO}}$ & $m_{\\text{solver}}$" for _ in Ns) + " \\\\ \\midrule")
        for spec in PAIRINGS:
            have = {N: cell(eq, N, spec) for N in Ns}
            if not any(have.values()):
                continue
            row = [f"$\\{{\\mathrm{{NO}}, \\text{{{SOLVER_NAMES[spec]}}}\\}}$"]
            for N in Ns:
                dg = have[N]
                if dg is None:
                    row += ["--"] * 3
                    continue
                d, g = dg
                c = g["costs"]
                row += [fmt_time(c[spec]), fmt_time(c["no"]), str(g["m"][0])]
            out.append(" & ".join(row) + " \\\\")
        out.append("\\bottomrule\n\\end{tabular}}")

        # ---------------------------------------------------------- iteration metrics (128^2)
        out.append(f"\\newcommand{{\\caauc{suf}}}{{")
        out.append("\\begin{tabular}{lccc}\n\\toprule")
        out.append("Method & $\\|e^{(T)}_h\\|/\\|u_h\\|$ & AUC & $p$ \\\\ \\midrule")
        T = None
        for spec in SOLVER_ORDER:
            dg = cell(eq, MAIN_N, spec)
            if dg is None:
                continue
            d, g = dg
            T = d["args"]["T"]
            P = g["policies"]
            out.append(f"\\multicolumn{{4}}{{c}}{{$\\{{\\mathrm{{NO}}, \\text{{{SOLVER_NAMES[spec]}}}\\}}$}} \\\\ \\midrule")
            auc_r = np.array([r["auc_T"] for r in P["router"]])
            for pol in ["classical", "hints25", "router", "oracle"]:
                err = np.array([r["err_T"] for r in P[pol]])
                auc = np.array([r["auc_T"] for r in P[pol]])

                def ms(x):
                    return f"{np.mean(x):.2e} ({np.std(x, ddof=1) / math.sqrt(len(x)):.1e})"
                cells = [ms(err), ms(auc)]
                if pol in ("classical", "hints25"):
                    p = ttest_p(auc, auc_r) if not np.allclose(auc, auc_r) else 1.0
                    cells.append(pstr(p))
                else:
                    cells.append("-")
                if pol == "router" and all(np.mean(auc) <= np.mean(np.array([r["auc_T"] for r in P[q]])) for q in ["classical", "hints25"]):
                    cells = [f"\\textbf{{{c}}}" for c in cells[:2]] + cells[2:]
                if pol == "oracle":
                    cells = [f"\\textit{{{c}}}" for c in cells]
                out.append(" & ".join([POL_NAMES[pol]] + cells) + " \\\\")
            out.append("\\midrule")
        out[-1] = "\\bottomrule\n\\end{tabular}}"
        if eq == "Poisson" and T is not None:
            out.append(f"\\newcommand{{\\caT}}{{{T}}}")
            out.append(f"\\newcommand{{\\caN}}{{{MAIN_N}}}")

        # ---------------------------------------------------------- seeds (128^2, work units)
        sd = Sf.get((eq, MAIN_N))
        if sd:
            out.append(f"\\newcommand{{\\caseeds{suf}}}{{")
            out.append("\\begin{tabular}{lcccc}\n\\toprule")
            out.append("Pairing & time to $h^2$ over 5 seeds & identical decisions & speedup vs.\\ HINTS-25 & seeds with $p{<}0.01$ \\\\ \\midrule")
            for spec in SOLVER_ORDER:
                dg = cell(eq, MAIN_N, spec)
                if dg is None or spec not in sd["groups"] or not sd["groups"][spec]:
                    continue
                d, g = dg
                P = g["policies"]
                key = tkey(d, d["h2"])
                if not all("t_wu" in blk["rows"][0]["tol"][key] for blk in sd["groups"][spec].values()):
                    continue
                th = times(P["hints25"], key, field="t_wu")
                tb = times(P[best_tau(P, key)], key, field="t_wu")
                it_main = iters(P["router"], key)
                meds, sps, nsig, agree = [], [], 0, []
                for s_, blk in sd["groups"][spec].items():
                    tr_ = np.array([np.inf if r["tol"][key]["t_wu"] is None else r["tol"][key]["t_wu"] for r in blk["rows"]])
                    it_s = np.array([np.inf if r["tol"][key]["iters"] is None else r["tol"][key]["iters"] for r in blk["rows"]])
                    meds.append(np.median(tr_))
                    sps.append(paired_speedup(th, tr_)[0])
                    agree.append(float(np.mean(it_s == it_main)))
                    if wilcoxon_p(th, tr_) < 0.01 and wilcoxon_p(tb, tr_) < 0.01:
                        nsig += 1
                out.append(" & ".join([f"$\\{{\\mathrm{{NO}}, \\text{{{SOLVER_NAMES[spec]}}}\\}}$",
                                       f"{np.mean(meds)*1e3:.2f} $\\pm$ {np.std(meds)*1e3:.2f}\\,ms",
                                       f"{100*np.mean(agree):.0f}\\%", f"{min(sps):.2f}--{max(sps):.2f}$\\times$",
                                       f"{nsig}/{len(meds)}"]) + " \\\\")
            out.append("\\bottomrule\n\\end{tabular}}")
        else:
            pending(out, "caseeds" + suf)

    # ================================================================ main-text table (all grids)
    out.append("\\newcommand{\\camainall}{")
    out.append("\\begin{tabular}{ll" + "ccc" * len(NS) + "}\n\\toprule")
    out.append("& & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{${N}\\times{N}$}}" for N in NS) + " \\\\ "
               + "".join(f"\\cmidrule(lr){{{3+3*i}-{5+3*i}}}" for i in range(len(NS))))
    out.append("Equation & Pairing & " + " & ".join("HINTS-25 & best $\\tau$ & router (ours)" for _ in NS) + " \\\\ \\midrule")
    for eq in EQS:
        first = True
        any_row = False
        for spec in PAIRINGS:
            have = {N: cell(eq, N, spec) for N in NS}
            if not any(have.values()):
                continue
            row = [{"Poisson": "Poisson", "ConvDiff": "ConvDiff", "AnisoDiff": "AnisoDiff"}[eq] if first else "", SOLVER_NAMES[spec]]
            for N in NS:
                dg = have[N]
                if dg is None:
                    row += ["--"] * 3
                    continue
                d, g = dg
                P = g["policies"]
                key = tkey(d, d["h2"])
                t_r = times(P["router"], key)
                bt = best_tau(P, key)
                sp_h = paired_speedup(times(P["hints25"], key), t_r)[0]
                sp_b = paired_speedup(times(P[bt], key), t_r)[0]
                row.append(time_cell(P["hints25"], key))
                row.append(time_cell(P[bt], key) + f"$_{{\\tau{{=}}{bt[5:]}}}$")
                rc = time_cell(P["router"], key, bold=(sp_h >= 1 and sp_b >= 1))
                row.append(rc + f" ({fmt_sp(sp_h)}\\,/\\,{fmt_sp(sp_b)})")
            out.append(" & ".join(row) + " \\\\")
            first = False
            any_row = True
        # classical baselines: multigrid alone and the multigrid-preconditioned Krylov method
        for m in ["mg", kry_name(eq)]:
            if not any((eq, N) in Bf and m in Bf[(eq, N)]["methods"] and Bf[(eq, N)]["methods"][m] for N in NS):
                continue
            row = ["", {"mg": "Multigrid alone", "pcg_mg": "PCG (MG)", "bicgstab_mg": "BiCGSTAB (MG)"}.get(m, BASE_NAMES[m]) + " (no corrector)"]
            for N in NS:
                d = Bf.get((eq, N))
                if d is None or m not in d["methods"] or not d["methods"][m]:
                    row += ["--"] * 3
                    continue
                ts = base_times(d, m, d["h2"])
                cens = int((~np.isfinite(ts)).sum())
                row += ["", "", fmt_time(np.median(ts)) + (f"$^{{\\dagger {cens}}}$" if cens else "")]
            out.append(" & ".join(row) + " \\\\")
        if any_row and eq != EQS[-1]:
            out.append("\\midrule")
    out.append("\\bottomrule\n\\end{tabular}}")

    # ================================================================ summary macros per grid
    for N_ in NS:
        SUF = GRID_SUF[N_]
        summ = {"Solver": [], "Hints": [], "Best": [], "SolverDeep": [], "HintsDeep": [], "BestDeep": [], "OracleRatio": []}
        for eq in EQS:
            for spec in SOLVER_ORDER:
                dg = cell(eq, N_, spec)
                if dg is None:
                    continue
                d, g = dg
                P = g["policies"]
                for tol, suf_ in [(d["h2"], ""), (1e-8, "Deep")]:
                    key = tkey(d, tol)
                    t_r = times(P["router"], key)
                    summ["Solver" + suf_].append(paired_speedup(times(P["classical"], key), t_r)[0])
                    summ["Hints" + suf_].append(paired_speedup(times(P["hints25"], key), t_r)[0])
                    summ["Best" + suf_].append(paired_speedup(times(P[best_tau(P, key)], key), t_r)[0])
                key = tkey(d, d["h2"])
                summ["OracleRatio"].append(np.median(times(P["router"], key)) / np.median(times(P["oracle"], key)))
        if not summ["Solver"]:
            continue
        for name, vals in summ.items():
            rng_macro(out, "caSp" + name + SUF, vals)
        out.append(f"\\newcommand{{\\caNumCells{SUF}}}{{{len(summ['Solver'])}}}")
        out.append(f"\\newcommand{{\\caCellsRouterBeatsBest{SUF}}}{{{sum(v >= 1.0 for v in summ['Best'])}}}")
        out.append(f"\\newcommand{{\\caCellsRouterBeatsHints{SUF}}}{{{sum(v >= 1.0 for v in summ['Hints'])}}}")
    # baselines at 128^2 for the text
    vs_mg, vs_kry, vs_mg_ens = [], [], []
    for (eq, N), d in Bf.items():
        if N != MAIN_N:
            continue
        pw = {s_: cell(eq, N, s_) for s_ in SOLVER_ORDER if cell(eq, N, s_)}
        if not pw:
            continue
        best_s = min(pw, key=lambda s_: np.median(times(pw[s_][1]["policies"]["router"], tkey(pw[s_][0], d["h2"]))))
        dd, g = pw[best_s]
        t_r = times(g["policies"]["router"], tkey(dd, dd["h2"]))
        if "mg" in d["methods"] and d["methods"]["mg"]:
            vs_mg.append(paired_speedup(base_times(d, "mg", dd["h2"]), t_r)[0])
        if kry_name(eq) in d["methods"] and d["methods"][kry_name(eq)]:
            vs_kry.append(paired_speedup(base_times(d, kry_name(eq), dd["h2"]), t_r)[0])
        mgp = cell(eq, N, "mg")
        if mgp:
            vs_mg_ens.append(paired_speedup(times(mgp[1]["policies"]["classical"], tkey(mgp[0], dd["h2"])),
                                            times(mgp[1]["policies"]["router"], tkey(mgp[0], dd["h2"])))[0])
    for name, vals in [("caVsMg", vs_mg), ("caVsKrylov", vs_kry), ("caVsMgEns", vs_mg_ens)]:
        rng_macro(out, name, vals)

    # ================================================================ ensembles
    ens_keys = [k for k in R if k[3]]
    ens_ratios, ens_vs_solver = [], []

    def wname(members):
        return "$\\{" + ", ".join(SOLVER_NAMES[s] for s in members) + "\\}$"

    # ---- nested ensembles with same-session pairwise baselines (router@s / oracle@s)
    nested = [k for k in ens_keys if any(p.startswith("router@") for p in R[k][1]["policies"])]
    if nested:
        out.append("\\newcommand{\\caensnest}{")
        out.append("\\begin{tabular}{lll" + "ccc" * 3 + "cc}\n\\toprule")
        out.append("& & & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{$\\varepsilon = {t}$}}" for t in ["h^2", "10^{-6}", "10^{-8}"])
                   + " & \\multicolumn{2}{c}{$\\varepsilon = 10^{-8}$: vs.\\ best single solver} \\\\")
        out.append("\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}\\cmidrule(lr){10-12}\\cmidrule(lr){13-14}")
        out.append("Equation & $N$ & $\\mathcal{W}$ & " + " & ".join("router & router (WU) & oracle (WU)" for _ in range(3))
                   + " & router & oracle \\\\ \\midrule")
        nest_stats = []
        for eq in EQS:
            for N in NS:
                ks = sorted([k for k in nested if k[0] == eq and k[1] == N], key=lambda k: len(k[2].split("+")))
                if not ks:
                    continue
                # singles: every member's own pairwise router / oracle, from the largest ensemble's run
                d, g = R[ks[-1]]
                P = g["policies"]
                members_all = ks[-1][2].split("+")
                rows_ = []
                for s_ in members_all:
                    if f"router@{s_}" in P:
                        rows_.append(([s_], P[f"router@{s_}"], P[f"oracle@{s_}"], d))
                for k in ks:
                    dk, gk = R[k]
                    rows_.append((k[2].split("+"), gk["policies"]["router"], gk["policies"]["oracle"], dk))
                # best single (by median WU at 1e-8) among the singles
                singles = [r_ for r_ in rows_ if len(r_[0]) == 1]
                key8 = tkey(d, 1e-8)
                best_single = min(singles, key=lambda r_: np.median(times(r_[1], tkey(r_[3], 1e-8), field="t_wu")))
                first = True
                for members, Rr, Ro, dd in rows_:
                    row = [EQ_NAMES[eq] if first else "", f"${N}^2$" if first else "", wname(members)]
                    for tol in [dd["h2"], 1e-6, 1e-8]:
                        key = tkey(dd, tol)
                        row.append(time_cell(Rr, key))
                        tw = times(Rr, key, field="t_wu")
                        cens = int((~np.isfinite(tw)).sum())
                        row.append(fmt_time(np.median(tw)) + (f"$^{{\\dagger {cens}}}$" if cens else ""))
                        two = times(Ro, key, field="t_wu")
                        row.append("\\textit{" + fmt_time(np.median(two)) + "}")
                    tb = times(best_single[1], tkey(best_single[3], 1e-8), field="t_wu")
                    tbo = times(best_single[2], tkey(best_single[3], 1e-8), field="t_wu")
                    if len(members) > 1:
                        row.append(sp_cell(tb, times(Rr, key8, field="t_wu")))
                        row.append(sp_cell(tbo, times(Ro, key8, field="t_wu")))
                        keyh = tkey(dd, dd["h2"])
                        bsh = min(singles, key=lambda r_: np.median(times(r_[1], tkey(r_[3], r_[3]["h2"]), field="t_wu")))
                        tbh = times(bsh[1], tkey(bsh[3], bsh[3]["h2"]), field="t_wu")
                        nest_stats.append((eq, N, members, paired_speedup(tb, times(Rr, key8, field="t_wu"))[0],
                                           wilcoxon_p(tb, times(Rr, key8, field="t_wu")),
                                           paired_speedup(tbh, times(Rr, keyh, field="t_wu"))[0]))
                    else:
                        row += ["(best single)" if members == best_single[0] else "--", "--"]
                    out.append(" & ".join(row) + " \\\\")
                    first = False
                out.append("\\midrule")
        out[-1] = "\\bottomrule\n\\end{tabular}}"
        if nest_stats:
            wins = [s for s in nest_stats if s[3] >= 1.05 and s[4] < 0.01]
            out.append(f"\\newcommand{{\\caNestNum}}{{{len(nest_stats)}}}")
            out.append(f"\\newcommand{{\\caNestWins}}{{{len(wins)}}}")
            out.append(f"\\newcommand{{\\caNestLosses}}{{{sum(1 for s in nest_stats if s[3] < 0.95)}}}")
            rng_macro(out, "caNestSp", [s[3] for s in (wins or nest_stats)])
            out.append(f"\\newcommand{{\\caNestMinRatio}}{{{fmt_sp(min(s[3] for s in nest_stats))}}}")
            out.append(f"\\newcommand{{\\caNestMaxRatio}}{{{fmt_sp(max(s[3] for s in nest_stats))}}}")
            out.append(f"\\newcommand{{\\caNestMinRatioH}}{{{fmt_sp(min(s[5] for s in nest_stats))}}}")
            out.append(f"\\newcommand{{\\caNestMaxRatioH}}{{{fmt_sp(max(s[5] for s in nest_stats))}}}")
    else:
        pending(out, "caensnest")
        for name, val in [("caNestNum", "--"), ("caNestWins", "--"), ("caNestLosses", "--"), ("caNestSpMin", "--"), ("caNestSpMax", "--"), ("caNestMinRatio", "--"), ("caNestMaxRatio", "--"), ("caNestMinRatioH", "--"), ("caNestMaxRatioH", "--")]:
            out.append(f"\\newcommand{{\\{name}}}{{{val}}}")

    # ---- oracle-level screening of all subsets (screen_ensembles.py)
    Scr = {}
    for path in sorted(glob.glob(f"{RESULTS_DIR}/screen_*.json")):
        d = json.load(open(path))
        Scr[(d["args"]["equation"], d["args"]["N"])] = d
    if Scr:
        out.append("\\newcommand{\\caensscreen}{")
        out.append("\\begin{tabular}{llcccccccc}\n\\toprule")
        out.append("& & \\multicolumn{3}{c}{$\\varepsilon = h^2$} & \\multicolumn{4}{c}{$\\varepsilon = 10^{-8}$} \\\\ \\cmidrule(lr){3-5}\\cmidrule(lr){6-9}")
        out.append("Equation & $N$ & best single & best ensemble & ratio & best single & best ensemble & ratio & $\\{\\text{Jacobi}, \\text{Jacobi (0.67)}\\}$ \\\\ \\midrule")
        for eq in EQS:
            first = True
            for N in NS:
                d = Scr.get((eq, N))
                if d is None:
                    continue
                cells = []
                for tol in [d["h2"], 1e-8]:
                    tk = f"{tol:.6g}"
                    med = {k: np.median([r[tk]["t_wu"] or np.inf for r in S["rows"]]) for k, S in d["sets"].items()}
                    singles = {k: v for k, v in med.items() if "+" not in k}
                    bs = min(singles, key=singles.get)
                    multi = {k: v for k, v in med.items() if "+" in k}
                    bm = min(multi, key=multi.get)
                    ratio = singles[bs] / multi[bm]
                    cells += [f"{SOLVER_NAMES[bs]} ({fmt_time(singles[bs])})", wname(bm.split("+")) + f" ({fmt_time(multi[bm])})",
                              (f"\\textbf{{{fmt_sp(ratio)}}}" if ratio >= 1.05 else fmt_sp(ratio))]
                    if tol == 1e-8:
                        jj = med.get("jacobi+jacobi_0.67")
                        cells.append(fmt_sp(singles[bs] / jj) if jj is not None else "--")
                out.append(" & ".join([EQ_NAMES[eq] if first else "", f"${N}^2$"] + cells) + " \\\\")
                first = False
            if not first and eq != EQS[-1]:
                out.append("\\midrule")
        out[-1] = "\\bottomrule\n\\end{tabular}}"
    else:
        pending(out, "caensscreen")

    # ---- earlier 128^2 ensembles (pairwise baselines from the separate pairwise runs)
    old_ens = [k for k in ens_keys if k not in nested]
    if old_ens:
        out.append("\\newcommand{\\caens}{")
        out.append("\\begin{tabular}{llcccc}\n\\toprule")
        out.append("Equation & $\\mathcal{W}$ & Best solver only & Best pairwise router & "
                   "Router$(\\mathrm{NO}\\cup\\mathcal{W})$ & Oracle$(\\mathrm{NO}\\cup\\mathcal{W})$ \\\\ \\midrule")
        for eq in EQS:
            first = True
            for k in sorted([k for k in old_ens if k[0] == eq], key=lambda k: len(k[2].split("+"))):
                d, g = R[k]
                N = k[1]
                key = tkey(d, d["h2"])
                P = g["policies"]
                members = k[2].split("+")
                cls, lbs, pw = {}, {}, {}
                for s_ in members:
                    dg = cell(eq, N, s_)
                    if dg:
                        kk = tkey(dg[0], d["h2"])
                        cls[s_] = np.median(times(dg[1]["policies"]["classical"], kk))
                        lbs[s_] = np.median(times_lb(dg[1]["policies"]["classical"], kk))
                        pw[s_] = times(dg[1]["policies"]["router"], kk)
                if not pw:
                    continue
                bc = min(cls, key=cls.get)
                bc_cell = f"{fmt_time(cls[bc])} ({SOLVER_NAMES[bc]})" if np.isfinite(cls[bc]) else f"$>${fmt_time(min(lbs.values()))}$^{{\\dagger}}$ (none)"
                bp = min(pw, key=lambda s_: np.median(pw[s_]))
                t_ens = times(P["router"], key)
                sp_pw, _ = paired_speedup(pw[bp], t_ens)
                ens_ratios.append(np.median(pw[bp]) / np.median(t_ens))
                ens_vs_solver.append(cls[bc] / np.median(t_ens))
                out.append(" & ".join([eq if first else "", wname(members), bc_cell,
                                       f"{fmt_time(np.median(pw[bp]))} ({SOLVER_NAMES[bp]})",
                                       f"{fmt_time(np.median(t_ens))} ({fmt_sp(sp_pw)} vs pairwise)",
                                       "\\textit{" + fmt_time(np.median(times(P['oracle'], key))) + "}"]) + " \\\\")
                first = False
            if eq != EQS[-1] and any(k[0] == EQS[EQS.index(eq)+1] for k in old_ens):
                out.append("\\midrule")
        out.append("\\bottomrule\n\\end{tabular}}")
        out.append("\\newcommand{\\caensusage}{")
        out.append("\\begin{tabular}{llcccccc}\n\\toprule")
        out.append("Equation & $\\mathcal{W}$ & Jacobi & GS & SymGS & Jacobi (0.67) & SOR (1.5) & DeepONet \\\\ \\midrule")
        for eq in EQS:
            first = True
            for k in sorted([k for k in old_ens if k[0] == eq], key=lambda k: len(k[2].split("+"))):
                d, g = R[k]
                members = k[2].split("+")
                fracs = np.array([r.get("op_frac", [np.nan] * (len(members) + 1)) for r in g["policies"]["router"]])
                cells = []
                for s in ["jacobi", "gs", "ssor", "jacobi_0.67", "sor_1.5"]:
                    if s in members:
                        j = members.index(s)
                        cells.append(f"{np.nanmean(fracs[:, j]):.3f} ({np.nanstd(fracs[:, j]):.3f})")
                    else:
                        cells.append("-")
                cells.append(f"{np.nanmean(fracs[:, -1]):.3f} ({np.nanstd(fracs[:, -1]):.3f})")
                out.append(" & ".join([eq if first else "", wname(members)] + cells) + " \\\\")
                first = False
            if eq != EQS[-1] and any(k[0] == EQS[EQS.index(eq)+1] for k in old_ens):
                out.append("\\midrule")
        out.append("\\bottomrule\n\\end{tabular}}")
        # p-values
        out.append("\\newcommand{\\castatsens}{")
        out.append("\\begin{tabular}{llcccc}\n\\toprule")
        out.append("Equation & $\\mathcal{W}$ & vs.\\ best solver only & vs.\\ best pairwise router (faster) & vs.\\ best pairwise router (slower) & vs.\\ oracle$(\\mathrm{NO}\\cup\\mathcal{W})$ (slower) \\\\ \\midrule")
        for eq in EQS:
            first = True
            for k in sorted([k for k in old_ens if k[0] == eq], key=lambda k: len(k[2].split("+"))):
                d, g = R[k]
                N = k[1]
                key = tkey(d, d["h2"])
                members = k[2].split("+")
                t_e = times(g["policies"]["router"], key)
                t_o = times(g["policies"]["oracle"], key)
                cls, pw = {}, {}
                for s_ in members:
                    dg = cell(eq, N, s_)
                    if dg:
                        cls[s_] = times_lb(dg[1]["policies"]["classical"], tkey(dg[0], d["h2"]))
                        pw[s_] = times(dg[1]["policies"]["router"], tkey(dg[0], d["h2"]))
                if not pw:
                    continue
                bc = min(cls, key=lambda s_: np.median(cls[s_]))
                bp = min(pw, key=lambda s_: np.median(pw[s_]))
                out.append(" & ".join([eq if first else "", wname(members), pstr(wilcoxon_p(cls[bc], t_e)),
                                       pstr(wilcoxon_p(pw[bp], t_e)), pstr(wilcoxon_p(t_e, pw[bp])),
                                       pstr(wilcoxon_p(t_e, t_o))]) + " \\\\")
                first = False
            if eq != EQS[-1] and any(k[0] == EQS[EQS.index(eq)+1] for k in old_ens):
                out.append("\\midrule")
        out.append("\\bottomrule\n\\end{tabular}}")
    else:
        for name in ["caens", "caensusage", "castatsens"]:
            pending(out, name)

    # ---- larger ensemble routers (capacity control)
    big_paths = sorted(glob.glob("results_ens_big/*_ens_*.json"))
    if big_paths and old_ens:
        def _agree(path):
            if not os.path.exists(path):
                return None
            m = re.findall(r"agreement with oracle ([0-9.]+)%", open(path).read())
            return float(m[-1]) if m else None
        rows_big = []
        for path in big_paths:
            base_path = os.path.join(RESULTS_DIR, os.path.basename(path))
            if not os.path.exists(base_path):
                continue
            db, dr = json.load(open(path)), json.load(open(base_path))
            eq, N_ = db["args"]["equation"], db["args"]["N"]
            for grp in db["groups"]:
                key = tkey(db, db["h2"])
                Rb, Rr, Ro = (db["groups"][grp]["policies"]["router"], dr["groups"][grp]["policies"]["router"],
                              dr["groups"][grp]["policies"]["oracle"])
                t_b = times(Rb, key, field="t_wu") - iters(Rb, key) * db["groups"][grp]["router_decision_cost"]
                t_r = times(Rr, key, field="t_wu") - iters(Rr, key) * dr["groups"][grp]["router_decision_cost"]
                t_o = times(Ro, key, field="t_wu")
                same = np.mean([(a["tol"][key]["iters"], a["tol"][key]["no_calls"]) == (b["tol"][key]["iters"], b["tol"][key]["no_calls"])
                                for a, b in zip(Rr, Rb)])
                members = grp.split("+")
                rows_big.append((eq, len(members), members, t_r, t_b, t_o,
                                 _agree(f"logs/routers_ens_{eq}_{N_}_{grp}.log"), _agree(f"logs/routers_ensbig_{eq}_{N_}_{grp}.log"), same))
        out.append("\\newcommand{\\caensbig}{")
        out.append("\\begin{tabular}{llccccccc}\n\\toprule")
        out.append("Equation & $\\mathcal{W}$ & default router & larger router & oracle & larger\\,/\\,default (medians) & same decisions & $p$ (larger faster\\,/\\,slower) & agreement (default\\,/\\,larger) \\\\ \\midrule")
        for eq in EQS:
            first = True
            sel = sorted([r for r in rows_big if r[0] == eq], key=lambda r: r[1])
            for (_, _, members, t_r, t_b, t_o, ag_r, ag_b, same) in sel:
                sp = np.median(t_b) / np.median(t_r)
                ag = ("--" if ag_r is None else f"{ag_r:.0f}\\%") + " / " + ("--" if ag_b is None else f"{ag_b:.0f}\\%")
                out.append(" & ".join([eq if first else "", wname(members), fmt_time(np.median(t_r)), fmt_time(np.median(t_b)),
                                       "\\textit{" + fmt_time(np.median(t_o)) + "}", f"{sp:.3f}$\\times$", f"{100*same:.0f}\\%",
                                       f"{pstr(wilcoxon_p(t_r, t_b))} / {pstr(wilcoxon_p(t_b, t_r))}", ag]) + " \\\\")
                first = False
            if sel and eq != EQS[-1] and any(r[0] == EQS[EQS.index(eq)+1] for r in rows_big):
                out.append("\\midrule")
        out.append("\\bottomrule\n\\end{tabular}}")
        sps = [np.median(r[4]) / np.median(r[3]) for r in rows_big]
        gaps = [np.median(r[4]) / np.median(r[5]) for r in rows_big]
        gaps0 = [np.median(r[3]) / np.median(r[5]) for r in rows_big]
        out.append(f"\\newcommand{{\\caEnsBigRatioMin}}{{{min(sps):.2f}$\\times$}}")
        out.append(f"\\newcommand{{\\caEnsBigRatioMax}}{{{max(sps):.2f}$\\times$}}")
        out.append(f"\\newcommand{{\\caEnsBigGapMax}}{{{100*(max(gaps)-1):.0f}\\%}}")
        out.append(f"\\newcommand{{\\caEnsGapWuMax}}{{{100*(max(gaps0)-1):.0f}\\%}}")
        out.append(f"\\newcommand{{\\caNumEnsBig}}{{{len(rows_big)}}}")
        out.append(f"\\newcommand{{\\caEnsBigSameMin}}{{{100*min(r[8] for r in rows_big):.0f}\\%}}")
        out.append(f"\\newcommand{{\\caEnsBigSameMax}}{{{100*max(r[8] for r in rows_big):.0f}\\%}}")
        out.append(f"\\newcommand{{\\caEnsBigNFaster}}{{{sum(wilcoxon_p(r[3], r[4]) < 0.05 for r in rows_big)}}}")
        out.append(f"\\newcommand{{\\caEnsBigNSlower}}{{{sum(wilcoxon_p(r[4], r[3]) < 0.05 for r in rows_big)}}}")
    else:
        pending(out, "caensbig")

    # ================================================================ overheads / amortisation
    op = f"{RESULTS_DIR}/overheads.json"
    if os.path.exists(op):
        O = json.load(open(op))
        out.append("\\newcommand{\\caoverheads}{")
        out.append("\\begin{tabular}{lccccc}\n\\toprule")
        Ns_o = sorted(int(k) for k in O["per_op"])
        out.append("Operation & " + " & ".join(f"$N={n}$" for n in Ns_o) + " \\\\ \\midrule")
        rows = [("Jacobi iteration", "jacobi"), ("GS iteration", "gs"), ("SymGS iteration", "ssor"),
                ("Multigrid V(2,2) cycle", "mg"),
                ("DeepONet corrector call", "corrector"), ("Router decision (ours)", "router_decision"),
                ("LSTM router decision (\\Cref{sec:experiments})", "lstm_router_decision")]

        def fmt_us(v):
            if v is None or not isinstance(v, (int, float)):
                return "--"
            return f"{v*1e6:.0f}\\,$\\mu$s" if v < 1e-3 else f"{v*1e3:.2f}\\,ms"
        for lab, key in rows:
            cells = []
            for n in Ns_o:
                v = O["per_op"][str(n)].get(key)
                if key == "mg" and n < 64:
                    v = None
                cells.append(fmt_us(v))
            out.append(lab + " & " + " & ".join(cells) + " \\\\")
        out.append("\\bottomrule\n\\end{tabular}}")
        tr = O["training"]
        out.append("\\newcommand{\\caamort}{")
        out.append("\\begin{tabular}{llccccc}\n\\toprule")
        out.append("Equation & $N$ & Corrector data + fit & Router training (GS pairing) & Saving per solve vs HINTS ($\\tau{=}25$) & Break-even solves (router) & Saving per solve vs solver only \\\\ \\midrule")
        for eq in EQS:
            first = True
            for N in NS:
                c = tr.get(f"corrector_{eq}_{N}")
                rt = tr.get(f"router_{eq}_{N}_gs")
                dg = cell(eq, N, "gs")
                if dg is None or c is None:
                    continue
                d, g = dg
                P = g["policies"]
                key = tkey(d, d["h2"])
                t_r = np.median(times(P["router"], key)); t_h = np.median(times(P["hints25"], key)); t_c = np.median(times(P["classical"], key))
                sav_h = t_h - t_r; sav_c = t_c - t_r
                be = (rt["train_s"] / sav_h) if (rt and sav_h > 0) else np.inf
                out.append(" & ".join([EQ_NAMES[eq] if first else "", f"${N}^2$", f"{c['data_s'] + c['fit_s']:.0f}\\,s",
                                       f"{rt['train_s']:.0f}\\,s" if rt else "--", fmt_time(sav_h) if sav_h > 0 else "--",
                                       f"{be:.0f}" if np.isfinite(be) else "--", fmt_time(sav_c)]) + " \\\\")
                first = False
            if not first and eq != EQS[-1]:
                out.append("\\midrule")
        out[-1] = "\\bottomrule\n\\end{tabular}}"
        n128 = O["per_op"].get("128", {})
        if n128.get("lstm_router_decision") and n128.get("jacobi"):
            out.append(f"\\newcommand{{\\caLstmOverJacobi}}{{{n128['lstm_router_decision']/n128['jacobi']:.0f}}}")
            out.append(f"\\newcommand{{\\caLstmMs}}{{{n128['lstm_router_decision']*1e3:.1f}}}")
    else:
        pending(out, "caoverheads")
        pending(out, "caamort")

    # ================================================================ theory assumptions
    Af = {}
    for path in sorted(glob.glob(f"{RESULTS_DIR}/assumptions_*.json")):
        d = json.load(open(path))
        Af[(d["args"]["equation"], d["args"]["N"])] = d
    if Af:
        def f6(x, nd=4):
            if x is None:
                return "--"
            if x == 0:
                return "0"
            return f"{x:.{nd}f}" if 1e-3 <= abs(x) < 1e3 else f"{x:.1e}"

        def sci(x):
            return "--" if x is None else (f"{x:.1e}" if x != 0 else "0")
        out.append("\\newcommand{\\caassump}{")
        out.append("\\begin{tabular}{llcccccccc}\n\\toprule")
        out.append("Equation & Operation & $m_j$ & $\\|I - C_j\\mathcal{L}_h\\|_2$ & $\\|I - C_j\\mathcal{L}_h\\|_{A}$ & $\\|(I - C_j\\mathcal{L}_h)^{m_j}\\|_{A}$ & $\\rho(I - C_j\\mathcal{L}_h)$ & $\\sigma_{\\min}$ & $\\|C_j(0)\\|$ & $\\|[G_j, G_{\\mathrm{NO}}]\\|$ \\\\ \\midrule")
        for eq in EQS:
            for N in NS:
                d = Af.get((eq, N))
                if d is None:
                    continue
                first = True
                no = d["ops"]["no"]
                out.append(" & ".join([f"{EQ_NAMES[eq]}, ${N}^2$", "DeepONet corrector", "1", f6(no["rho2"], 6), f6(no.get("rhoA"), 6), f6(no.get("rhoA"), 6),
                                       f6(no["rho_spec"], 6), f"{sci(no['band_max'])} (band)", sci(no["zero"]), "--"]) + " \\\\")
                for spec in PAIRINGS:
                    r_ = d["ops"].get(spec)
                    if r_ is None:
                        continue
                    if "rho_symbol_nonyquist" in r_:   # Fourier-diagonal: exact values on the Nyquist-free subspace
                        rho2 = rhoA = rho_spec = r_["rho_symbol_nonyquist"]
                        rhoA_m = rho2 ** r_["m"]
                        sig = r_["sigma_min_symbol"]
                    else:
                        rho2, rhoA, rhoA_m, rho_spec, sig = r_["rho2"], r_.get("rhoA"), r_.get("rhoA_macro"), r_["rho_spec"], r_["sigma_min"]
                    out.append(" & ".join(["", SOLVER_NAMES[spec], str(r_["m"]), f6(rho2, 6), f6(rhoA, 6), f6(rhoA_m, 6),
                                           f6(rho_spec, 6), sci(sig), sci(r_["zero"]), sci(r_["comm"])]) + " \\\\")
                out.append("\\midrule")
        out[-1] = "\\bottomrule\n\\end{tabular}}"
        out.append("\\newcommand{\\caassumpB}{")
        out.append("\\begin{tabular}{llccccccc}\n\\toprule")
        out.append("Equation & Ensemble & $T$ & $\\sum_i \\rho_{O_i}^2$ & $\\alpha(O)$ (Prop.~\\ref{th:weaklyalphasupermodular}) & $\\hat\\alpha$ max & $\\hat\\alpha$ median & $\\max_t\\max_j \\tilde c_j/\\|e^{(t)}\\|^2$ & $\\min_t\\max_j \\tilde c_j/\\|e^{(t)}\\|^2$ \\\\ \\midrule")
        for eq in EQS:
            for N in NS:
                d = Af.get((eq, N))
                if d is None or not d.get("paths"):
                    continue
                first = True
                for key, pth in d["paths"].items():
                    rows = pth["rows"]
                    members = key.split("+")
                    Ts = [r["T"] for r in rows]
                    # alpha(O) with the energy-norm constants of the chosen macro-actions (rho_NO = 1)
                    rhoA = {}
                    for j, o in enumerate(pth["ops"]):
                        v = d["ops"][o]
                        if o == "no":
                            rhoA[o] = 1.0
                        elif "rho_symbol_nonyquist" in v:
                            rhoA[o] = v["rho_symbol_nonyquist"] ** pth["m"][j]
                        else:
                            rhoA[o] = v.get("rhoA_macro") if pth["m"][j] == v["m"] else (v.get("rhoA") or v["rho2"]) ** pth["m"][j]
                    for r in rows:
                        s2 = sum(rhoA[pth["ops"][j]] ** 2 for j in r["steps"])
                        r["sum_rho2"] = s2
                        r["alpha_bound"] = max(4.0 / (r["T"] - s2), 1.0) if r["T"] - s2 > 0 else float("inf")
                    ab = [r["alpha_bound"] for r in rows]
                    n_inf = sum(1 for a in ab if not np.isfinite(a))
                    fin = [a for a in ab if np.isfinite(a)]
                    ab_s = ("$\\infty$" if not fin else f"{np.median(fin):.0f}") + (f" ($\\infty$ on {n_inf}/{len(ab)})" if 0 < n_inf < len(ab) else "")
                    out.append(" & ".join([f"{EQ_NAMES[eq]}, ${N}^2$" if first else "", wname(members),
                                           f"{np.median(Ts):.0f}", f"{np.median([r['sum_rho2'] for r in rows]):.3f}", ab_s,
                                           f"{max(r['alpha_hat_max'] or 0 for r in rows):.3f}",
                                           f"{np.median([r['alpha_hat_median'] or 0 for r in rows]):.3f}",
                                           f"{max(r['Ebar_rel'] for r in rows):.3f}", f"{min(r['Emin_rel'] for r in rows):.1e}"]) + " \\\\")
                    first = False
                out.append("\\midrule")
        out[-1] = "\\bottomrule\n\\end{tabular}}"
        # text macros: ranges over all settings
        def _rhoA(v):
            return v["rho_symbol_nonyquist"] ** v["m"] if "rho_symbol_nonyquist" in v else (v.get("rhoA_macro") or 0)

        def _spec(v):
            return v["rho_symbol_nonyquist"] if "rho_symbol_nonyquist" in v else v["rho_spec"]
        rhoA_max = max(_rhoA(v) for d in Af.values() for k, v in d["ops"].items() if k != "no")
        spec_max = max(_spec(v) for d in Af.values() for k, v in d["ops"].items() if k != "no")
        rho2_gs = max(v["rho2"] for d in Af.values() for k, v in d["ops"].items() if k in ("gs", "ssor", "sor_1.5"))
        band_max = max(d["ops"]["no"]["band_max"] for d in Af.values())
        ah_max = max((r["alpha_hat_max"] or 0) for d in Af.values() for p in d.get("paths", {}).values() for r in p["rows"])
        ah_med = float(np.median([(r["alpha_hat_median"] or 0) for d in Af.values() for p in d.get("paths", {}).values() for r in p["rows"]]))
        ah_h2 = [(r["alpha_hat_max"] or 0) for d in Af.values() for p in d.get("paths_h2", {}).values() for r in p["rows"]]
        ab_fin = [r["alpha_bound"] for d in Af.values() for p in d.get("paths", {}).values() for r in p["rows"] if np.isfinite(r["alpha_bound"])]  # recomputed above with energy-norm constants
        out.append(f"\\newcommand{{\\caAlphaHatMed}}{{{ah_med:.2f}}}")
        out.append(f"\\newcommand{{\\caAlphaHatMaxH}}{{{(max(ah_h2) if ah_h2 else float('nan')):.3f}}}")
        out.append(f"\\newcommand{{\\caAlphaBoundMin}}{{{(min(ab_fin) if ab_fin else float('nan')):.0f}}}")
        out.append(f"\\newcommand{{\\caAlphaBoundMax}}{{{(max(ab_fin) if ab_fin else float('nan')):.0f}}}")
        out.append(f"\\newcommand{{\\caRhoAMax}}{{{rhoA_max:.6f}}}")
        out.append(f"\\newcommand{{\\caRhoSpecMax}}{{{spec_max:.6f}}}")
        out.append(f"\\newcommand{{\\caRhoTwoGsMax}}{{{rho2_gs:.2f}}}")
        out.append(f"\\newcommand{{\\caBandMax}}{{{band_max:.1e}}}".replace("e-0", "\\times 10^{-").replace("e-", "\\times 10^{-") + ("}" if "10^{" in f"{band_max:.1e}".replace("e-0", "\\times 10^{-") else ""))
        out.append(f"\\newcommand{{\\caAlphaHatMax}}{{{ah_max:.3f}}}")
    else:
        pending(out, "caassump")
        pending(out, "caassumpB")

    # ================================================================ held-out confirmation (fresh seed, 3 timed replays)
    if Rh:
        out.append("\\newcommand{\\caheldout}{")
        out.append("\\begin{tabular}{lllcccc}\n\\toprule")
        out.append("& & & \\multicolumn{2}{c}{main test set (seed 72)} & \\multicolumn{2}{c}{held-out test set (seed 73)} \\\\ \\cmidrule(lr){4-5}\\cmidrule(lr){6-7}")
        out.append("Equation & $N$ & Pairing / ensemble & vs.\\ HINTS-25 or best single & vs.\\ best $\\tau$ or oracle & vs.\\ HINTS-25 or best single & vs.\\ best $\\tau$ or oracle \\\\ \\midrule")
        for eq in EQS:
            first = True
            for N in NS:
                keys = sorted([k for k in Rh if k[0] == eq and k[1] == N], key=lambda k: (k[3], PAIRINGS.index(k[2]) if k[2] in PAIRINGS else 99, len(k[2])))
                for k in keys:
                    dh, gh = Rh[k]
                    if k not in R:
                        continue
                    dm, gm = R[k]
                    cells = []
                    for d_, g_ in [(dm, gm), (dh, gh)]:
                        P = g_["policies"]
                        if not k[3]:
                            key = tkey(d_, d_["h2"])
                            t_r = times(P["router"], key)
                            cells.append(sp_cell(times_lb(P["hints25"], key), t_r))
                            cells.append(sp_cell(times_lb(P[best_tau(P, key)], key), t_r))
                        else:
                            key = tkey(d_, 1e-8)
                            t_r = times(P["router"], key, field="t_wu")
                            singles = {p_: times(P[p_], key, field="t_wu") for p_ in P if p_.startswith("router@")}
                            if singles:
                                bs = min(singles, key=lambda p_: np.median(singles[p_]))
                                cells.append(sp_cell(singles[bs], t_r))
                            else:
                                cells.append("--")
                            cells.append(sp_cell(times(P["oracle"], key, field="t_wu"), t_r) if "oracle" in P else "--")
                    lab = (f"$\\{{\\mathrm{{NO}}, \\text{{{SOLVER_NAMES[k[2]]}}}\\}}$" if not k[3] else wname(k[2].split("+")) + " ($10^{-8}$, work units)")
                    out.append(" & ".join([EQ_NAMES[eq] if first else "", f"${N}^2$", lab] + cells) + " \\\\")
                    first = False
            if not first and eq != EQS[-1]:
                out.append("\\midrule")
        out[-1] = "\\bottomrule\n\\end{tabular}}" if out[-1] == "\\midrule" else out[-1]
        if not out[-1].endswith("\\end{tabular}}"):
            out.append("\\bottomrule\n\\end{tabular}}")
    else:
        pending(out, "caheldout")

    # ================================================================ manifest of every artifact used
    import hashlib
    man = {}
    for path in sorted(set(USED_FILES + glob.glob(f"{RESULTS_DIR}/baselines_*.json") + glob.glob(f"{RESULTS_DIR}/seeds_*.json")
                          + glob.glob(f"{RESULTS_DIR}/assumptions_*.json") + glob.glob(f"{RESULTS_DIR}/screen_*.json")
                          + glob.glob("results_ens_big/*.json") + [f"{RESULTS_DIR}/overheads.json", f"{RESULTS_DIR}/discretization_error.json"])):
        if os.path.exists(path):
            man[path] = hashlib.sha256(open(path, "rb").read()).hexdigest()
    json.dump(man, open(os.path.join(os.path.dirname(OUT_TEX) or ".", "manifest.json"), "w"), indent=1)

    # ================================================================ placeholders / summary macros
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", "\n".join(out)))
    for name, val in [("caLstmMs", "--"), ("caLstmOverJacobi", "--"), ("caVsMgMin", "--"), ("caVsMgMax", "--"),
                      ("caVsKrylovMin", "--"), ("caVsKrylovMax", "--"), ("caVsMgEnsMin", "--"), ("caVsMgEnsMax", "--"),
                      ("caRhoAMax", "--"), ("caRhoSpecMax", "--"), ("caRhoTwoGsMax", "--"), ("caBandMax", "--"), ("caAlphaHatMax", "--"),
                      ("caAlphaHatMed", "--"), ("caAlphaHatMaxH", "--"), ("caAlphaBoundMin", "--"), ("caAlphaBoundMax", "--")]:
        if name not in defined:
            out.append(f"\\newcommand{{\\{name}}}{{{val}}}")
    if ens_ratios:
        out.append(f"\\newcommand{{\\caEnsVsPairMin}}{{{fmt_sp(min(ens_ratios))}}}")
        out.append(f"\\newcommand{{\\caEnsVsPairMax}}{{{fmt_sp(max(ens_ratios))}}}")
        out.append(f"\\newcommand{{\\caEnsVsSolverMin}}{{{fmt_sp(min(ens_vs_solver))}}}")
        out.append(f"\\newcommand{{\\caEnsVsSolverMax}}{{{fmt_sp(max(ens_vs_solver))}}}")
        out.append(f"\\newcommand{{\\caNumEns}}{{{len(ens_ratios)}}}")
    else:
        for name in ["caEnsVsPairMin", "caEnsVsPairMax", "caEnsVsSolverMin", "caEnsVsSolverMax", "caNumEns"]:
            out.append(f"\\newcommand{{\\{name}}}{{--}}")
    for SUF in ["", "B", "C"]:
        for name in ["caSpSolver", "caSpHints", "caSpBest", "caSpSolverDeep", "caSpHintsDeep", "caSpBestDeep", "caSpOracleRatio"]:
            for mm in ["Min", "Max"]:
                if name + SUF + mm not in defined:
                    out.append(f"\\newcommand{{\\{name}{SUF}{mm}}}{{--}}")
        for name in ["caNumCells", "caCellsRouterBeatsBest", "caCellsRouterBeatsHints"]:
            if name + SUF not in defined:
                out.append(f"\\newcommand{{\\{name}{SUF}}}{{--}}")

    os.makedirs("paper", exist_ok=True)
    with open(OUT_TEX, "w") as fh:
        fh.write("\n".join(out) + "\n")
    print("wrote", OUT_TEX)


if __name__ == "__main__":
    main()
