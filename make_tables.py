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
                "ssor": "SymGS", "sor_1.5": "SOR (1.5)", "mg": "Multigrid", "linegs": "Line GS", "mg_line": "Multigrid (line GS)"}
SOLVER_ORDER = ["jacobi", "jacobi_0.67", "gs", "ssor", "sor_1.5", "linegs"]
PAIRINGS = SOLVER_ORDER + ["mg"]
HREF = "hints15"     # the published 2-D period of HINTS (proportion 1/15); tau = 25 is its 1-D period
POL_NAMES = {"classical": "Solver only", "hints25": "HINTS ($\\tau{=}25$)", "hints15": "HINTS ($\\tau{=}15$)", "best": "Best fixed schedule",
             "bestdecay": "Best residual-decay rule", "oneshot": "One-shot schedule",
             "hints5": "HINTS ($\\tau{=}5$)", "hints10": "HINTS ($\\tau{=}10$)",
             "hints50": "HINTS ($\\tau{=}50$)", "greedy": "Greedy oracle (Alg.~1)",
             "oracle": "Cost-aware oracle", "router": "Learned router (ours)"}
EQS = ["Poisson", "ConvDiff", "AnisoDiff", "VarCoeff"]
EQ_SUF = {"Poisson": "poisson", "ConvDiff": "conv", "AnisoDiff": "aniso", "VarCoeff": "var"}
EQ_NAMES = {"Poisson": "Poisson", "ConvDiff": "Convection--diffusion", "AnisoDiff": "Anisotropic diffusion", "VarCoeff": "Variable-coefficient diffusion"}
BASE_NAMES = {"mg": "Multigrid V(2,2) alone (point GS)", "mg_line": "Multigrid V(2,2) alone (line GS)", "cg": "CG", "pcg_ssor": "PCG (SymGS)", "pcg_mg": "PCG (multigrid)",
              "bicgstab": "BiCGSTAB", "bicgstab_mg": "BiCGSTAB (multigrid)", "gmres": "GMRES(20)",
              "fft": "FFT direct solve (exact; constant coefficients only)", "lu": "Sparse LU direct solve (cached factorisation)"}
BASE_ORDER = ["mg", "mg_line", "cg", "bicgstab", "pcg_ssor", "pcg_mg", "bicgstab_mg", "gmres", "fft", "lu"]
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
    """Two-sided paired Wilcoxon signed-rank test on the log time ratios (the direction is reported
    separately from the sign of the median ratio). Censored runs enter at their time-to-cap on both
    sides (callers pass times_lb), so a router failure counts against the router instead of being dropped."""
    _, r = paired_speedup(base, ours)
    ok = np.isfinite(r) & (r > 0)
    if ok.sum() < 8 or np.allclose(r[ok], 1.0):
        return 1.0
    return float(wilcoxon(np.log(r[ok]), alternative="two-sided").pvalue)


def ttest_p(base, ours):
    ok = np.isfinite(base) & np.isfinite(ours)
    if ok.sum() < 8 or np.allclose(base[ok], ours[ok]):
        return 1.0
    return float(ttest_rel(np.log(base[ok]), np.log(ours[ok]), alternative="greater").pvalue)


def pstr(pv):
    """p-value to one significant digit (never rounded down to a smaller power of ten)."""
    if pv < 1e-10:
        return "$<10^{-10}$"
    if pv < 1e-3:
        e = int(np.floor(np.log10(pv)))
        mant = pv / 10 ** e
        if round(mant) >= 10:
            e += 1; mant = 1.0
        return f"${mant:.0f}\\times10^{{{e}}}$" if round(mant) > 1 else f"$10^{{{e}}}$"
    return f"{pv:.3f}"


def boot_ci(r, B=2000, seed=0):
    """Bootstrap 95% interval of the paired median ratio (over instances)."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 4:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(r), size=(B, len(r)))
    meds = np.median(r[idx], axis=1)
    return (float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5)))


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


def sp_parts(base, ours, ci=False, n_fail=0, lb=False):
    """'speedup [CI] (p)' with the two-sided p-value; 'slower' marks a median ratio below one;
    n_fail router runs that did not reach the tolerance are marked with a dagger (they enter the
    ratio at the router's time-to-cap); lb marks a baseline that is censored on most instances
    (the ratio is then a lower bound)."""
    sp, r = paired_speedup(base, ours)
    p = wilcoxon_p(base, ours)
    s = fmt_sp(sp)
    if lb:
        s = "$\\geq$" + s
    if ci:
        lo, hi = boot_ci(r)
        if np.isfinite(lo):
            s += f" [{fmt_sp(lo)[:-8]}, {fmt_sp(hi)[:-8]}]"
    if n_fail:
        s += f"$^{{\\dagger {n_fail}}}$"
    if sp >= 1.0:
        return f"{s} ({pstr(p)})", sp, p
    return f"{s} (slower, {pstr(p)})", sp, p


def sp_rows(rows_b, rows_r, key, ci=True):
    """sp_parts for two row lists at a tolerance key: censoring on both sides at the time-to-cap,
    router failures counted and marked, majority-censored baselines marked as lower bounds."""
    tb, tr = times_lb(rows_b, key), times_lb(rows_r, key)
    n_fail = int(sum(1 for r in rows_r if r["tol"][key]["t_live"] is None))
    n_cb = int(sum(1 for r in rows_b if r["tol"][key]["t_live"] is None))
    return sp_parts(tb, tr, ci=ci, n_fail=n_fail, lb=(n_cb > len(rows_b) / 2))


def rng_macro(out, name, vals):
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        out.append(f"\\newcommand{{\\{name}Min}}{{{PENDING}}}")
        out.append(f"\\newcommand{{\\{name}Max}}{{{PENDING}}}")
        return
    out.append(f"\\newcommand{{\\{name}Min}}{{{fmt_sp(min(vals))}}}")
    out.append(f"\\newcommand{{\\{name}Max}}{{{fmt_sp(max(vals))}}}")


PENDING = "\\textbf{[pending]}"


def pending(out, name, msg="results pending"):
    out.append(f"\\newcommand{{\\{name}}}{{\\begin{{tabular}}{{c}}\\textbf{{[{msg}: confirmatory run in progress]}}\\end{{tabular}}}}")


# ------------------------------------------------------------------- main
def main():
    R = load()                                        # confirmatory runs (compiled kernels, test seed 73)
    Rdev = load(pattern="results_dev/*.json")         # development runs (numpy kernels, test seed 72), archived
    out = []
    Bf = {}
    for path in sorted(glob.glob(f"{RESULTS_DIR}/baselines_*.json")):      # legacy separate-session files
        d = json.load(open(path))
        Bf[(d["args"]["equation"], d["args"]["N"])] = d
    # classical baselines timed inside the pairwise sessions ('base:<method>' policies): the table
    # rows come from one designated session per (equation, grid) -- the first pairing in PAIRINGS
    # order -- and every paired comparison uses the baseline of the router's own session
    BASE_SESSION = {}
    for (eq_, N_, spec_, ens_), (d_, g_) in sorted(R.items(), key=lambda kv: (kv[0][0], kv[0][1], PAIRINGS.index(kv[0][2]) if kv[0][2] in PAIRINGS else 99)):
        if ens_ or (eq_, N_) in BASE_SESSION:
            continue
        bl = {p_[5:]: rows for p_, rows in g_["policies"].items() if p_.startswith("base:") and rows}
        if bl:
            BASE_SESSION[(eq_, N_)] = spec_
            Bf[(eq_, N_)] = {"h2": d_["h2"], "tols": d_["tols"], "methods": bl, "session": spec_,
                             "lu_factorization_s": d_.get("lu_factorization_s")}
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

    def bl_rows(g, db, m):
        """Baseline rows for a paired comparison: the router's own session if it timed the baseline,
        else the designated session's rows."""
        rows = g["policies"].get("base:" + m)
        if rows:
            return rows
        if db is not None and m in db["methods"] and db["methods"][m]:
            return db["methods"][m]
        return None

    def best_decay(P, key):
        cands = [p for p in P if p.startswith("decay")]
        return min(cands, key=lambda p: np.median(times(P[p], key))) if cands else None

    SCHED_LABEL = {"oneshot": "one-shot"}

    def sched_label(p):
        if p == "oneshot":
            return "\\text{one-shot}"
        if p.startswith("decay"):
            return f"\\theta{{=}}{p[5:]}"
        if p.startswith("phints"):
            return f"\\tau{{=}}{p[6:]},\\,t_0{{=}}0"
        return f"\\tau{{=}}{p[5:]}"

    def best_tau(P, key):
        """Best fixed schedule for this cell and tolerance: HINTS (any tau), phase-shifted HINTS, one-shot."""
        cands = [p for p in P if p.startswith("hints") or p.startswith("phints") or p == "oneshot"]
        return min(cands, key=lambda p: np.median(times(P[p], key)))

    def kry_name(eq):
        return "bicgstab_mg" if eq == "ConvDiff" else "pcg_mg"

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
            pols = ["classical", HREF, "best", "bestdecay", "oneshot", "greedy", "oracle", "router"]
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
                        if pol == "best":
                            bt = best_tau(P, key)
                            row.append(time_cell(P[bt], key) + f"$_{{{sched_label(bt)}}}$")
                        elif pol == "bestdecay":
                            bd = best_decay(P, key)
                            row.append(time_cell(P[bd], key) + f"$_{{{sched_label(bd)}}}$" if bd else "--")
                        elif pol in P:
                            t_r = np.median(times(P["router"], key))
                            # bold: the router's median is the smallest among all deployable methods
                            # (solver alone, every fixed schedule, the one-shot and residual-decay rules)
                            comp = [np.median(times(P[p_], key)) for p_ in P
                                    if p_ == "classical" or p_.startswith(("hints", "phints", "decay")) or p_ == "oneshot"]
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
        nc = 7 if has_base else 4
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
                    bd = best_decay(P, key)
                    bases = [P[HREF], P[best_tau(P, key)], P[bd] if bd else None, P["oneshot"] if "oneshot" in P else None]
                    if has_base:
                        db = Bf.get((eq, N))
                        for m_ in ["mg", kry_name(eq), "fft" if eq != "VarCoeff" else "lu"]:
                            bases.append(bl_rows(g, db, m_))
                    for b_ in bases:
                        if b_ is None:
                            row.append("--")
                        else:
                            s, sp, p = sp_rows(b_, P["router"], key)
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
        out.append("Pairing & $\\varepsilon$ & " + " & ".join("vs.\\ HINTS-15 & vs.\\ best schedule & vs.\\ best decay rule & vs.\\ one-shot" + ((" & vs.\\ multigrid & vs.\\ MG-Krylov & vs.\\ " + ("sparse LU" if eq == "VarCoeff" else "FFT solve")) if has_base else "") for _ in Ns) + " \\\\ \\midrule")
        for row in rows_spec:
            out.append(" & ".join(row) + " \\\\")
        out.append("\\bottomrule\n\\end{tabular}}")

        # ---------------------------------------------------------- usage
        out.append(f"\\newcommand{{\\causage{suf}}}{{")
        out.append("\\begin{tabular}{l" + "ccc" * len(Ns) + "}\n\\toprule")
        out.append("& " + " & ".join(f"\\multicolumn{{3}}{{c}}{{${N}\\times{N}$}}" for N in Ns) + " \\\\ "
                   + "".join(f"\\cmidrule(lr){{{2+3*i}-{4+3*i}}}" for i in range(len(Ns))))
        out.append("Pairing & " + " & ".join("HINTS-15 & oracle & router" for _ in Ns) + " \\\\ \\midrule")
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
                for pol in [HREF, "oracle", "router"]:
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
        out.append("\\begin{tabular}{lcc}\n\\toprule")
        out.append("Method & AUC & $p$ \\\\ \\midrule")
        T = None
        for spec in SOLVER_ORDER:
            dg = cell(eq, MAIN_N, spec)
            if dg is None:
                continue
            d, g = dg
            T = d["args"]["T"]
            P = g["policies"]
            out.append(f"\\multicolumn{{3}}{{c}}{{$\\{{\\mathrm{{NO}}, \\text{{{SOLVER_NAMES[spec]}}}\\}}$}} \\\\ \\midrule")
            auc_r = np.array([r["auc_T"] for r in P["router"]])
            for pol in ["classical", HREF, "router", "oracle"]:
                err = np.array([r["err_T"] for r in P[pol]])
                auc = np.array([r["auc_T"] for r in P[pol]])

                def ms(x):
                    return f"{np.mean(x):.2e} ({np.std(x, ddof=1) / math.sqrt(len(x)):.1e})"
                cells = [ms(auc)]
                if pol in ("classical", HREF):
                    p = ttest_p(auc, auc_r) if not np.allclose(auc, auc_r) else 1.0
                    cells.append(pstr(p))
                else:
                    cells.append("-")
                if pol == "router" and all(np.mean(auc) <= np.mean(np.array([r["auc_T"] for r in P[q]])) for q in ["classical", HREF]):
                    cells = [f"\\textbf{{{c}}}" for c in cells[:1]] + cells[1:]
                if pol == "oracle":
                    cells = [f"\\textit{{{c}}}" for c in cells]
                out.append(" & ".join([POL_NAMES[pol]] + cells) + " \\\\")
            out.append("\\midrule")
        out[-1] = "\\bottomrule\n\\end{tabular}}"
        if eq == "Poisson" and T is not None:
            out.append(f"\\newcommand{{\\caT}}{{{T}}}")
            out.append(f"\\newcommand{{\\caN}}{{{MAIN_N}}}")

        # ---------------------------------------------------------- seeds (work units), every grid with seed trials
        if any((eq, N_) in Sf for N_ in NS):
            out.append(f"\\newcommand{{\\caseeds{suf}}}{{")
            out.append("\\begin{tabular}{llcccc}\n\\toprule")
            out.append("$N$ & Pairing & time to $h^2$ over 5 seeds & identical decisions & speedup vs.\\ HINTS-15 & seeds with $p{<}0.01$ \\\\ \\midrule")
            for N_ in NS:
                sd = Sf.get((eq, N_))
                if not sd:
                    continue
                for spec in SOLVER_ORDER:
                    dg = cell(eq, N_, spec)
                    if dg is None or spec not in sd["groups"] or not sd["groups"][spec]:
                        continue
                    d, g = dg
                    P = g["policies"]
                    key = tkey(d, d["h2"])
                    if not all("t_wu" in blk["rows"][0]["tol"][key] for blk in sd["groups"][spec].values()):
                        continue
                    th = times(P[HREF], key, field="t_wu")
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
                    out.append(" & ".join([f"${N_}^2$", f"$\\{{\\mathrm{{NO}}, \\text{{{SOLVER_NAMES[spec]}}}\\}}$",
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
    out.append("Equation & Pairing & " + " & ".join("HINTS-15 & best schedule & router (ours)" for _ in NS) + " \\\\ \\midrule")
    for eq in EQS:
        first = True
        any_row = False
        for spec in PAIRINGS:
            have = {N: cell(eq, N, spec) for N in NS}
            if not any(have.values()):
                continue
            row = [{"Poisson": "Poisson", "ConvDiff": "ConvDiff", "AnisoDiff": "AnisoDiff", "VarCoeff": "VarCoeff"}[eq] if first else "", SOLVER_NAMES[spec]]
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
                sp_h = paired_speedup(times_lb(P[HREF], key), times_lb(P["router"], key))[0]
                sp_b = paired_speedup(times_lb(P[bt], key), times_lb(P["router"], key))[0]
                row.append(time_cell(P[HREF], key))
                row.append(time_cell(P[bt], key) + f"$_{{{sched_label(bt)}}}$")
                rc = time_cell(P["router"], key, bold=(sp_h >= 1 and sp_b >= 1))
                row.append(rc + f" ({fmt_sp(sp_h)}\\,/\\,{fmt_sp(sp_b)})")
            out.append(" & ".join(row) + " \\\\")
            first = False
            any_row = True
        # classical baselines: multigrid alone and the multigrid-preconditioned Krylov method
        for m in ["mg", kry_name(eq), "fft", "lu"]:
            if not any_row or not any((eq, N) in Bf and m in Bf[(eq, N)]["methods"] and Bf[(eq, N)]["methods"][m] for N in NS):
                continue
            row = ["", {"mg": "Multigrid alone (no corrector)", "pcg_mg": "PCG (MG) (no corrector)", "bicgstab_mg": "BiCGSTAB (MG) (no corrector)", "fft": "FFT direct solve (exact)", "lu": "Sparse LU direct solve"}.get(m, BASE_NAMES[m])]
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
        summ = {"Solver": [], "Hints": [], "HintsTF": [], "Best": [], "Decay": [], "SolverDeep": [], "HintsDeep": [], "HintsTFDeep": [], "BestDeep": [], "DecayDeep": [], "OracleRatio": []}
        for eq in EQS:
            for spec in SOLVER_ORDER:
                dg = cell(eq, N_, spec)
                if dg is None:
                    continue
                d, g = dg
                P = g["policies"]
                for tol, suf_ in [(d["h2"], ""), (1e-8, "Deep")]:
                    key = tkey(d, tol)
                    t_r = times_lb(P["router"], key)
                    summ["Solver" + suf_].append(paired_speedup(times_lb(P["classical"], key), t_r)[0])
                    summ["Hints" + suf_].append(paired_speedup(times_lb(P[HREF], key), t_r)[0])
                    summ["HintsTF" + suf_].append(paired_speedup(times_lb(P["hints25"], key), t_r)[0])
                    summ["Best" + suf_].append(paired_speedup(times_lb(P[best_tau(P, key)], key), t_r)[0])
                    bd_ = best_decay(P, key)
                    if bd_:
                        summ["Decay" + suf_].append(paired_speedup(times_lb(P[bd_], key), t_r)[0])
                key = tkey(d, d["h2"])
                summ["OracleRatio"].append(np.median(times(P["router"], key)) / np.median(times(P["oracle"], key)))
        if not summ["Solver"]:
            continue
        for name, vals in summ.items():
            rng_macro(out, "caSp" + name + SUF, vals)
        out.append(f"\\newcommand{{\\caNumCells{SUF}}}{{{len(summ['Solver'])}}}")
        out.append(f"\\newcommand{{\\caCellsRouterBeatsBest{SUF}}}{{{sum(v >= 1.0 for v in summ['Best'])}}}")
        out.append(f"\\newcommand{{\\caCellsRouterBeatsHints{SUF}}}{{{sum(v >= 1.0 for v in summ['Hints'])}}}")
        out.append(f"\\newcommand{{\\caCellsRouterBeatsBestDeep{SUF}}}{{{sum(v >= 1.0 for v in summ['BestDeep'])}}}")
        out.append(f"\\newcommand{{\\caCellsRouterBeatsHintsDeep{SUF}}}{{{sum(v >= 1.0 for v in summ['HintsDeep'])}}}")
        out.append(f"\\newcommand{{\\caCellsRouterWithinBest{SUF}}}{{{sum(v >= 0.9 for v in summ['Best'])}}}")
        out.append(f"\\newcommand{{\\caCellsRouterBeatsDecay{SUF}}}{{{sum(v >= 1.0 for v in summ['Decay'])}}}")
        out.append(f"\\newcommand{{\\caCellsRouterBeatsDecayDeep{SUF}}}{{{sum(v >= 1.0 for v in summ['DecayDeep'])}}}")
        # multigrid alone and MG-preconditioned Krylov, at h^2 and 1e-8, every stationary pairing of this grid
        vm, vmd, vk, vkd = [], [], [], []
        for eq in EQS:
            db = Bf.get((eq, N_))
            if db is None:
                continue
            for spec in SOLVER_ORDER:
                dg = cell(eq, N_, spec)
                if dg is None:
                    continue
                d, g = dg
                for tol, (lm, lk) in [(d["h2"], (vm, vk)), (1e-8, (vmd, vkd))]:
                    key = tkey(d, tol)
                    t_r = times_lb(g["policies"]["router"], key)
                    rm_ = bl_rows(g, db, "mg")
                    if rm_:
                        lm.append(paired_speedup(times_lb(rm_, key), t_r)[0])
                    rk_ = bl_rows(g, db, kry_name(eq))
                    if rk_:
                        lk.append(paired_speedup(times_lb(rk_, key), t_r)[0])
        rng_macro(out, "caVsMgAll" + SUF, vm); rng_macro(out, "caVsMgDeep" + SUF, vmd)
        rng_macro(out, "caVsKrylovAll" + SUF, vk); rng_macro(out, "caVsKrylovDeep" + SUF, vkd)
        # the {NO, multigrid} router against multigrid alone
        vme, vmed = [], []
        for eq in EQS:
            dg = cell(eq, N_, "mg")
            db = Bf.get((eq, N_))
            if dg is None or db is None or "mg" not in db["methods"] or not db["methods"]["mg"]:
                continue
            d, g = dg
            rm_ = bl_rows(g, db, "mg")
            vme.append(paired_speedup(times_lb(rm_, tkey(d, d["h2"])), times_lb(g["policies"]["router"], tkey(d, d["h2"])))[0])
            vmed.append(paired_speedup(times_lb(rm_, tkey(d, 1e-8)), times_lb(g["policies"]["router"], tkey(d, 1e-8)))[0])
        rng_macro(out, "caVsMgEnsAll" + SUF, vme); rng_macro(out, "caVsMgEnsDeep" + SUF, vmed)
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
        kh = tkey(dd, dd["h2"])
        t_r = times_lb(g["policies"]["router"], kh)
        rm_ = bl_rows(g, d, "mg")
        if rm_:
            vs_mg.append(paired_speedup(times_lb(rm_, kh), t_r)[0])
        rk_ = bl_rows(g, d, kry_name(eq))
        if rk_:
            vs_kry.append(paired_speedup(times_lb(rk_, kh), t_r)[0])
        mgp = cell(eq, N, "mg")
        if mgp:
            vs_mg_ens.append(paired_speedup(times(mgp[1]["policies"]["classical"], tkey(mgp[0], dd["h2"])),
                                            times(mgp[1]["policies"]["router"], tkey(mgp[0], dd["h2"])))[0])
    for name, vals in [("caVsMg", vs_mg), ("caVsKrylov", vs_kry), ("caVsMgEns", vs_mg_ens)]:
        rng_macro(out, name, vals)
    fft_ratio = []
    for (eq, N), d in Bf.items():
        pw = {s_: cell(eq, N, s_) for s_ in PAIRINGS if cell(eq, N, s_)}
        if not pw or "fft" not in d["methods"] or not d["methods"]["fft"]:
            continue
        best_s = min(pw, key=lambda s_: np.median(times(pw[s_][1]["policies"]["router"], tkey(pw[s_][0], d["h2"]))))
        dd, g = pw[best_s]
        rf_ = bl_rows(g, d, "fft")
        fft_ratio.append(paired_speedup(times_lb(g["policies"]["router"], tkey(dd, dd["h2"])), times_lb(rf_, tkey(dd, dd["h2"])))[0])
    rng_macro(out, "caFftRatio", fft_ratio)

    # ================================================================ in-situ overheads, drift, charged fraction, live/WU, macro sizes
    def num_macro(name, vals, fmt):
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        out.append(f"\\newcommand{{\\{name}Min}}{{{(fmt % min(vals)) if vals else PENDING}}}")
        out.append(f"\\newcommand{{\\{name}Max}}{{{(fmt % max(vals)) if vals else PENDING}}}")
    dec_us, gap_us, dfrac, dfrac8, charged, livewu, drifts, waits, kchk, ndec = [], [], [], [], [], [], [], 0, [], []
    for (eq_, N_, spec_, ens_), (d_, g_) in R.items():
        if ens_:
            continue
        P = g_["policies"]
        for tol_, lst in [(d_["h2"], dfrac), (1e-8, dfrac8)]:
            key = tkey(d_, tol_)
            rr = P["router"]
            td = np.array([r["tol"][key].get("t_dec") if r["tol"][key].get("t_dec") is not None else np.nan for r in rr])
            nd = np.array([r["tol"][key].get("n_dec") if r["tol"][key].get("n_dec") is not None else np.nan for r in rr], dtype=float)
            tl = times(rr, key)
            ok = np.isfinite(td) & np.isfinite(nd) & (nd > 0) & np.isfinite(tl)
            if ok.any():
                lst.append(float(np.median(td[ok] / tl[ok])) * 100)
                if tol_ == d_["h2"]:
                    dec_us.append(float(np.median(td[ok] / nd[ok])) * 1e6)
                    ndec.append(float(np.median(nd[ok])))
                    to = times(P["oracle"], key)
                    both = ok & np.isfinite(to)
                    gap_us.append(float(np.median(tl[both] - to[both])) * 1e6)
                    tw = times(rr, key, field="t_wu")
                    livewu.append(float(np.median(tl[ok] / tw[ok])))
        ch = [r["t_total_live"] / r["t_outer_total"] for r in P["router"] if r.get("t_outer_total")]
        if ch:
            charged.append(float(np.median(ch)) * 100)
        for dr in g_.get("drift", []):
            drifts.append(abs(dr["ratio"] - 1.0) * 100); waits += int(dr["retries"])
        for k_, v_ in g_.items():
            if k_.startswith("krylov_check:"):
                kchk.append(v_["max_ratio_timed_over_untimed_error"])
    num_macro("caDecInSitu", dec_us, "%.0f")
    num_macro("caRouterOracleGap", gap_us, "%.0f")
    num_macro("caDecFrac", dfrac, "%.1f")
    num_macro("caDecFracDeep", dfrac8, "%.1f")
    num_macro("caChargedFrac", charged, "%.0f")
    num_macro("caLiveWu", livewu, "%.2f")
    num_macro("caNDec", ndec, "%.0f")
    out.append(f"\\newcommand{{\\caDriftMax}}{{{('%.1f' % max(drifts)) if drifts else PENDING}}}")
    out.append(f"\\newcommand{{\\caDriftWaits}}{{{waits}}}")
    out.append(f"\\newcommand{{\\caKrylovCheckMax}}{{{('%.2f' % max(kchk)) if kchk else PENDING}}}")
    for N_ in NS:
        SUF = GRID_SUF[N_]
        for spec_, nm_ in [("jacobi", "Jacobi"), ("gs", "Gs"), ("mg", "Mg"), ("ssor", "SymGs")]:
            dg = cell("Poisson", N_, spec_) or next((cell(e_, N_, spec_) for e_ in EQS if cell(e_, N_, spec_)), None)
            if dg:
                out.append(f"\\newcommand{{\\caM{nm_}{SUF}}}{{{dg[1]['m'][0]}}}")
                c_ = dg[1]["costs"]
                out.append(f"\\newcommand{{\\caCost{nm_}{SUF}}}{{{fmt_time(c_[spec_])}}}")
                out.append(f"\\newcommand{{\\caCostNo{SUF}}}{{{fmt_time(c_['no'])}}}")
                out.append(f"\\newcommand{{\\caCostRes{SUF}}}{{{fmt_time(c_['_residual'])}}}")

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
                                           paired_speedup(tbh, times(Rr, keyh, field="t_wu"))[0],
                                           paired_speedup(tbo, times(Ro, key8, field="t_wu"))[0]))
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
            out.append(f"\\newcommand{{\\caNestOrMinRatio}}{{{fmt_sp(min(s[6] for s in nest_stats))}}}")
            out.append(f"\\newcommand{{\\caNestOrMaxRatio}}{{{fmt_sp(max(s[6] for s in nest_stats))}}}")
            out.append(f"\\newcommand{{\\caNestOrWins}}{{{sum(1 for s in nest_stats if s[6] >= 1.05)}}}")
    else:
        pending(out, "caensnest")
        for name, val in [("caNestNum", PENDING), ("caNestWins", PENDING), ("caNestLosses", PENDING), ("caNestSpMin", PENDING), ("caNestSpMax", PENDING), ("caNestMinRatio", PENDING), ("caNestMaxRatio", PENDING), ("caNestMinRatioH", PENDING), ("caNestMaxRatioH", PENDING), ("caNestOrMinRatio", PENDING), ("caNestOrMaxRatio", PENDING), ("caNestOrWins", PENDING)]:
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
        out.append("Equation & $N$ & Corrector data + fit & Router training (GS pairing) & Saving per solve vs HINTS ($\\tau{=}15$) & Break-even solves (router) & Saving per solve vs solver only \\\\ \\midrule")
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
                t_r = np.median(times(P["router"], key)); t_h = np.median(times(P[HREF], key)); t_c = np.median(times(P["classical"], key))
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
        out.append("Equation & Ensemble & $T$ & $T - \\sum_i \\rho_{O_i}^2$ & $\\alpha(O)$ (Prop.~\\ref{th:weaklyalphasupermodular}): median [min, max] & $\\hat\\alpha$ max & $\\hat\\alpha$ median & $\\max_t\\max_j \\tilde c_j/\\|e^{(t)}\\|^2$ & $\\min_t\\max_j \\tilde c_j/\\|e^{(t)}\\|^2$ \\\\ \\midrule")
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
                        r["alpha_bound"] = max(1.0 / (r["T"] - s2), 1.0) if r["T"] - s2 > 0 else float("inf")
                    ab = [r["alpha_bound"] for r in rows]
                    n_inf = sum(1 for a in ab if not np.isfinite(a))
                    fin = [a for a in ab if np.isfinite(a)]
                    def fa(x):
                        return f"{x:.2f}" if x < 100 else f"{x:.0f}"
                    ab_s = ("$\\infty$" if not fin else f"{fa(np.median(fin))} [{fa(min(fin))}, {fa(max(fin))}]") + (f" ($\\infty$ on {n_inf}/{len(ab)})" if 0 < n_inf < len(ab) else "")
                    gap = [r["T"] - r["sum_rho2"] for r in rows]
                    out.append(" & ".join([f"{EQ_NAMES[eq]}, ${N}^2$" if first else "", wname(members),
                                           f"{np.median(Ts):.0f}", f"{np.median(gap):.3f}", ab_s,
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

    # ================================================================ exact short-horizon verification of Theorem 4.1
    Tf = {}
    for path in sorted(glob.glob(f"{RESULTS_DIR}/theorem_*.json")):
        d = json.load(open(path))
        Tf[(d["args"]["equation"], d["args"]["N"])] = d
    if Tf:
        out.append("\\newcommand{\\catheorem}{")
        out.append("\\begin{tabular}{llcccccccccc}\n\\toprule")
        out.append("Equation & Ensemble & $K$ & $T$ & $n$ & $\\mu$ & $\\alpha(O)$ greedy prefixes & $\\alpha(O)$ all $|S| < T$ & bound holds & $g(S^T)/g(O)$ med (max) & greedy $= O$ & $g(O)$ at floor / rule $=$ Alg.~1 / violations \\\\ \\midrule")
        for eq in EQS:
            for N in NS:
                d = Tf.get((eq, N))
                if d is None:
                    continue
                first = True
                for key, G in d["groups"].items():
                    rows = G["rows"]
                    members = key.split("+")
                    out.append(" & ".join([f"{EQ_NAMES[eq]}, ${N}^2$" if first else "", wname(members), str(rows[0]["K"]), str(rows[0]["T"]), str(len(rows)),
                                           f"{max(r['mu'] for r in rows):.3f}", f"{max(r['alpha_greedy'] for r in rows):.3f}",
                                           f"{max(r['alpha_all'] for r in rows):.3f}", f"{sum(r['holds'] for r in rows)}/{len(rows)}",
                                           f"{np.median([r['greedy_over_opt'] for r in rows]):.3f} ({max(r['greedy_over_opt'] for r in rows):.2f})",
                                           f"{sum(r['greedy'] == r['O'] for r in rows)}/{len(rows)}",
                                           f"{sum(r.get('opt_at_floor', False) for r in rows)} / {sum(r.get('deployed_eq_plain', True) for r in rows)} / {sum(r.get('viol_greedy', 0) + r.get('viol_all', 0) for r in rows)}"]) + " \\\\")
                    first = False
                out.append("\\midrule")
        out[-1] = "\\bottomrule\n\\end{tabular}}"
        allrows = [r for d in Tf.values() for G in d["groups"].values() for r in G["rows"]]
        out.append(f"\\newcommand{{\\caThmMuMax}}{{{max(r['mu'] for r in allrows):.3f}}}")
        out.append(f"\\newcommand{{\\caThmHolds}}{{{sum(r['holds'] for r in allrows)}/{len(allrows)}}}")
        out.append(f"\\newcommand{{\\caThmFloor}}{{{sum(r.get('opt_at_floor', False) for r in allrows)}}}")
        out.append(f"\\newcommand{{\\caThmPlainDiff}}{{{sum(not r.get('deployed_eq_plain', True) for r in allrows)}}}")
        out.append(f"\\newcommand{{\\caThmViol}}{{{sum(r.get('viol_greedy', 0) + r.get('viol_all', 0) for r in allrows)}}}")
        out.append(f"\\newcommand{{\\caThmMaxRatio}}{{{max(r['greedy_over_opt'] for r in allrows):.1f}}}")
    else:
        pending(out, "catheorem")
        for nm_ in ["caThmMuMax", "caThmHolds", "caThmFloor", "caThmPlainDiff", "caThmViol", "caThmMaxRatio"]:
            out.append(f"\\newcommand{{\\{nm_}}}{{{PENDING}}}")

    # ================================================================ decision-granularity ablation (unit = corrector / 4)
    Ru4 = load(tag="_u4")
    if Ru4:
        out.append("\\newcommand{\\cagranularity}{")
        out.append("\\begin{tabular}{llcccccccc}\n\\toprule")
        out.append("& & \\multicolumn{4}{c}{unit = one corrector call} & \\multicolumn{4}{c}{unit = corrector call / 4} \\\\ \\cmidrule(lr){3-6}\\cmidrule(lr){7-10}")
        out.append("Equation & Pairing & oracle $h^2$ & router $h^2$ & router $10^{-8}$ & vs.\\ best schedule ($10^{-8}$) & oracle $h^2$ & router $h^2$ & router $10^{-8}$ & vs.\\ best schedule ($10^{-8}$) \\\\ \\midrule")
        for eq in EQS:
            first = True
            for spec in PAIRINGS:
                k = (eq, 128, spec, False)
                if k not in Ru4 or k not in R:
                    continue
                cells = []
                for d_, g_ in [R[k], Ru4[k]]:
                    P = g_["policies"]
                    key2, key8 = tkey(d_, d_["h2"]), tkey(d_, 1e-8)
                    Pm = R[k][1]["policies"]   # fixed schedules from the main run (same instances)
                    cells += [time_cell(P["oracle"], key2, italic=True), time_cell(P["router"], key2), time_cell(P["router"], key8),
                              sp_cell(times_lb(Pm[best_tau(Pm, key8)], key8, field="t_wu"), times(P["router"], key8, field="t_wu"))]
                out.append(" & ".join([EQ_NAMES[eq] if first else "", SOLVER_NAMES[spec]] + cells) + " \\\\")
                first = False
            if not first and eq != EQS[-1]:
                out.append("\\midrule")
        if out[-1] == "\\midrule":
            out.pop()
        out.append("\\bottomrule\n\\end{tabular}}")
    else:
        pending(out, "cagranularity")

    # ================================================================ development vs confirmatory runs
    if Rdev and R:
        out.append("\\newcommand{\\caheldout}{")
        out.append("\\begin{tabular}{lllcccc}\n\\toprule")
        out.append("& & & \\multicolumn{2}{c}{development (seed 72, numpy kernels)} & \\multicolumn{2}{c}{confirmatory (seed 73, compiled kernels)} \\\\ \\cmidrule(lr){4-5}\\cmidrule(lr){6-7}")
        out.append("Equation & $N$ & Pairing / ensemble & vs.\\ HINTS-25 or best single & vs.\\ best schedule or oracle & vs.\\ HINTS-25 or best single & vs.\\ best schedule or oracle \\\\ \\midrule")
        for eq in EQS:
            first = True
            for N in NS:
                keys = sorted([k for k in R if k[0] == eq and k[1] == N and k in Rdev], key=lambda k: (k[3], PAIRINGS.index(k[2]) if k[2] in PAIRINGS else 99, len(k[2])))
                for k in keys:
                    cells = []
                    for d_, g_ in [Rdev[k], R[k]]:
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
                            cells.append(sp_cell(singles[min(singles, key=lambda p_: np.median(singles[p_]))], t_r) if singles else "--")
                            cells.append(sp_cell(times(P["oracle"], key, field="t_wu"), t_r) if "oracle" in P else "--")
                    lab = (f"$\\{{\\mathrm{{NO}}, \\text{{{SOLVER_NAMES[k[2]]}}}\\}}$" if not k[3] else wname(k[2].split("+")) + " ($10^{-8}$, work units)")
                    out.append(" & ".join([EQ_NAMES[eq] if first else "", f"${N}^2$", lab] + cells) + " \\\\")
                    first = False
            if not first and eq != EQS[-1]:
                out.append("\\midrule")
        if out[-1] == "\\midrule":
            out.pop()
        out.append("\\bottomrule\n\\end{tabular}}")
    else:
        pending(out, "caheldout")

    # ================================================================ discretisation-error study (confirmatory instances)
    dpath = f"{RESULTS_DIR}/discretization_error.json"
    if os.path.exists(dpath):
        D = json.load(open(dpath))
        meds = [v["median"] / v["h2"] for v in D.values()]
        mins = [min(v["rel"]) / v["h2"] for v in D.values()]
        out.append(f"\\newcommand{{\\caDiscMedMin}}{{{min(meds):.1f}}}")
        out.append(f"\\newcommand{{\\caDiscMedMax}}{{{max(meds):.0f}}}")
        out.append(f"\\newcommand{{\\caDiscMinHsq}}{{{min(mins):.1f}}}")
        out.append(f"\\newcommand{{\\caDiscN}}{{{sum(len(v['rel']) for v in D.values())}}}")
    else:
        for nm_ in ["caDiscMedMin", "caDiscMedMax", "caDiscMinHsq", "caDiscN"]:
            out.append(f"\\newcommand{{\\{nm_}}}{{{PENDING}}}")

    # ================================================================ manifest of every artifact used
    import hashlib
    man = {}
    for path in sorted(set(USED_FILES + glob.glob(f"{RESULTS_DIR}/baselines_*.json") + glob.glob(f"{RESULTS_DIR}/seeds_*.json")
                          + glob.glob(f"{RESULTS_DIR}/assumptions_*.json") + glob.glob(f"{RESULTS_DIR}/screen_*.json") + glob.glob(f"{RESULTS_DIR}/theorem_*.json")
                          + glob.glob("results_ens_big/*.json") + [f"{RESULTS_DIR}/overheads.json", f"{RESULTS_DIR}/discretization_error.json"])):
        if os.path.exists(path):
            man[path] = hashlib.sha256(open(path, "rb").read()).hexdigest()
    json.dump(man, open(os.path.join(os.path.dirname(OUT_TEX) or ".", "manifest.json"), "w"), indent=1)

    # ================================================================ placeholders / summary macros
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", "\n".join(out)))
    for name, val in [("caT", "300"), ("caN", "128"), ("caLstmMs", PENDING), ("caLstmOverJacobi", PENDING), ("caVsMgMin", PENDING), ("caVsMgMax", PENDING),
                      ("caVsKrylovMin", PENDING), ("caVsKrylovMax", PENDING), ("caVsMgEnsMin", PENDING), ("caVsMgEnsMax", PENDING),
                      ("caFftRatioMin", PENDING), ("caFftRatioMax", PENDING),
                      ("caRhoAMax", PENDING), ("caRhoSpecMax", PENDING), ("caRhoTwoGsMax", PENDING), ("caBandMax", PENDING), ("caAlphaHatMax", PENDING),
                      ("caAlphaHatMed", PENDING), ("caAlphaHatMaxH", PENDING), ("caAlphaBoundMin", PENDING), ("caAlphaBoundMax", PENDING)]:
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
            out.append(f"\\newcommand{{\\{name}}}{{{PENDING}}}")
    for SUF in ["", "B", "C"]:
        for name in ["caSpSolver", "caSpHints", "caSpHintsTF", "caSpBest", "caSpDecay", "caSpSolverDeep", "caSpHintsDeep", "caSpHintsTFDeep", "caSpBestDeep", "caSpDecayDeep", "caSpOracleRatio",
                     "caVsMgAll", "caVsMgDeep", "caVsKrylovAll", "caVsKrylovDeep", "caVsMgEnsAll", "caVsMgEnsDeep"]:
            for mm in ["Min", "Max"]:
                if name + SUF + mm not in defined:
                    out.append(f"\\newcommand{{\\{name}{SUF}{mm}}}{{{PENDING}}}")
        for name in ["caNumCells", "caCellsRouterBeatsBest", "caCellsRouterBeatsHints", "caCellsRouterBeatsBestDeep", "caCellsRouterBeatsHintsDeep", "caCellsRouterWithinBest",
                     "caCellsRouterBeatsDecay", "caCellsRouterBeatsDecayDeep",
                     "caMJacobi", "caMGs", "caMMg", "caMSymGs", "caCostJacobi", "caCostGs", "caCostMg", "caCostSymGs", "caCostNo", "caCostRes"]:
            if name + SUF not in defined:
                out.append(f"\\newcommand{{\\{name}{SUF}}}{{{PENDING}}}")

    os.makedirs("paper", exist_ok=True)
    with open(OUT_TEX, "w") as fh:
        fh.write("\n".join(out) + "\n")
    print("wrote", OUT_TEX)


if __name__ == "__main__":
    main()
