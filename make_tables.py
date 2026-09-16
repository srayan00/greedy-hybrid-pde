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
# pairwise cells a grid must have before its grid-wide summary macros are emitted (all four equations at 128^2 and
# 256^2, the two isotropic ones at 512^2); until then the macros print as pending rather than summarising a subset
EXPECTED_CELLS = {128: 24, 256: 24, 512: 12}

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
    """Time to the tolerance per instance; inf = censored (did not reach it within the cap);
    nan = the run is invalid (a baseline whose timed pass failed its cross-check) and is excluded."""
    return np.array([np.nan if not r.get("valid", True) else (np.inf if r["tol"][key][field] is None else r["tol"][key][field]) for r in rows])


def iters(rows, key):
    return np.array([np.inf if r["tol"][key]["iters"] is None else r["tol"][key]["iters"] for r in rows])


def _lab2(p):
    if p == "oneshot":
        return "one-shot"
    if p.startswith("phints"):
        return f"$\\tau{{=}}{p[6:]}$ (first call at $0$)"
    if p.startswith("decay"):
        return f"$\\theta{{=}}{p[5:]}$"
    return f"$\\tau{{=}}{p[5:]}$"


def seq_agree(rows_a, rows_b, key):
    """Percentage of instances on which two policies executed the same operation sequence up to the
    tolerance crossing (same crossing iteration and identical operations before it). Instances on which
    both are censored are skipped; one censored counts as disagreement. nan if the rows carry no
    operation sequences (cells produced before they were stored)."""
    if not rows_a or not rows_b or any("op_rle" not in r for r in rows_a) or any("op_rle" not in r for r in rows_b):
        return np.nan
    agree, n = 0, 0
    for ra, rb in zip(rows_a, rows_b):
        ia, ib = ra["tol"][key]["iters"], rb["tol"][key]["iters"]
        if ia is None and ib is None:
            continue
        n += 1
        if ia is None or ib is None or ia != ib:
            continue
        oa = np.repeat([o for o, c in ra["op_rle"]], [c for o, c in ra["op_rle"]])[:ia]
        ob = np.repeat([o for o, c in rb["op_rle"]], [c for o, c in rb["op_rle"]])[:ib]
        agree += int(len(oa) == len(ob) and np.array_equal(oa, ob))
    return 100.0 * agree / n if n else np.nan


def times_lb(rows, key, field="t_live"):
    """Censored runs enter at the time spent up to the iteration cap (a lower bound;
    conservative on the baseline side of a paired test)."""
    tot = "t_total_live" if field == "t_live" else "t_total_wu"
    return np.array([np.nan if not r.get("valid", True) else (r[tot] if r["tol"][key][field] is None else r["tol"][key][field]) for r in rows])


def fmt_time(x):
    if not np.isfinite(x):
        return "--"
    if x < 1e-3:
        return f"{x*1e6:.0f}\\,$\\mu$s"
    if x < 1.0:
        return f"{x*1e3:.2f}\\,ms" if x < 0.01 else f"{x*1e3:.1f}\\,ms"
    return f"{x:.2f}\\,s"


def paired_speedup(base, ours):
    """Paired median speedup base/ours. Conventions: a censored (inf) 'ours' with a finite base gives
    ratio 0 (counted against the router); a censored base with finite 'ours' gives inf unless the
    caller has already replaced the base by its time-to-cap (times_lb), which makes the ratio a lower
    bound; both censored gives 1 (a tie); nan on either side (invalid run) drops the pair."""
    keep = ~(np.isnan(base) | np.isnan(ours))
    base, ours = base[keep], ours[keep]
    both = np.isfinite(base) & np.isfinite(ours)
    r = np.full(len(base), np.nan)
    r[both] = base[both] / ours[both]
    r[np.isfinite(base) & ~np.isfinite(ours)] = 0.0
    r[~np.isfinite(base) & np.isfinite(ours)] = np.inf
    r[~np.isfinite(base) & ~np.isfinite(ours)] = 1.0
    return (float(np.median(r)) if len(r) else np.nan), r


def fmt_sp(sp):
    if np.isnan(sp):
        return "--"
    if np.isinf(sp):
        return "$>10^{3}\\times$"
    if sp == 0:
        return "0$\\times$"
    if sp >= 100:
        return f"{sp:.0f}$\\times$"
    if sp >= 10:
        return f"{sp:.1f}$\\times$"
    return f"{sp:.2f}$\\times$"


def wilcoxon_p(base, ours):
    """Two-sided paired Wilcoxon signed-rank test on the log time ratios of two time arrays (legacy path;
    the row-based helpers pair_ratios / wilcoxon_d classify censoring before substitution and are used
    for every table and macro). A ratio of 0 (router censored) enters as the most extreme negative rank."""
    _, r = paired_speedup(base, ours)
    r = r[~np.isnan(r)]
    fin = np.isfinite(r) & (r > 0)
    if len(r) < 8 or np.allclose(r[fin], 1.0) and (fin.sum() == len(r)):
        return 1.0
    d = np.empty(len(r))
    d[fin] = np.log(r[fin])
    ext = (np.abs(d[fin]).max() if fin.any() else 0.0) + 1.0
    d[(r == 0)] = -ext            # router failure: the most extreme negative difference
    d[np.isinf(r)] = ext          # baseline failure without a cap time: the most extreme positive one
    if np.allclose(d, 0.0):
        return 1.0
    return float(wilcoxon(d, alternative="two-sided").pvalue)


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
    r = r[~np.isnan(r)]
    if len(r) < 4:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(r), size=(B, len(r)))
    meds = np.median(r[idx], axis=1)
    return (float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5)))


def time_cell(rows, key, bold=False, italic=False, field="t_live"):
    """Median time with censoring marks: with any censored run the median of the mixed vector
    (reached times, and the time spent up to the cap for the censored runs) is reported, a lower
    bound marked >=; invalid runs are dropped and counted (times mark)."""
    ts = times(rows, key, field)
    n_inv = int(np.isnan(ts).sum())
    ts = ts[~np.isnan(ts)]
    cens = int(np.isinf(ts).sum())
    if cens > 0:
        lb = times_lb(rows, key, field)
        lb = lb[~np.isnan(lb)]
        s = f"$\\geq${fmt_time(np.median(lb))}$^{{\\dagger {cens}}}$" if len(lb) else "--"
    else:
        s = fmt_time(np.median(ts)) if len(ts) else "--"
    if n_inv:
        s += f"$^{{\\times {n_inv}}}$"
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
        if np.isfinite(lo) and np.isfinite(hi):
            s += f" [{fmt_sp(lo)[:-8]}, {fmt_sp(hi)[:-8]}]"
        elif np.isfinite(lo):
            s += f" [{fmt_sp(lo)[:-8]}, $>10^{{3}}$]"
    if n_fail:
        s += f"$^{{\\dagger {n_fail}}}$"
    if sp >= 1.0:
        return f"{s} ({pstr(p)})", sp, p
    return f"{s} (slower, {pstr(p)})", sp, p


def pair_ratios(rows_b, rows_r, key, field="t_live"):
    """Per-instance ratios base/router with the censoring classes of Appendix E.1, decided from the
    censoring masks before any substitution: both reached -> t_b/t_r; base censored only -> cap_b/t_r
    (a lower bound); router censored only -> 0 (counted against the router; enters the test as the most
    extreme negative rank); both censored -> 1 (a tie carrying no information); an invalid run on either
    side -> the pair is dropped. Returns the ratios r (nan = dropped), the log-differences d for the
    test (nan = dropped) and the class counts."""
    tb_f, tr_f = times(rows_b, key, field), times(rows_r, key, field)
    tb_lb = times_lb(rows_b, key, field)
    n = min(len(tb_f), len(tr_f))
    r = np.full(n, np.nan)
    cls = []
    for i in range(n):
        if np.isnan(tb_f[i]) or np.isnan(tr_f[i]):
            cls.append("invalid"); continue
        bc, rc = np.isinf(tb_f[i]), np.isinf(tr_f[i])
        if bc and rc:
            r[i] = 1.0; cls.append("both")
        elif rc:
            r[i] = 0.0; cls.append("router")
        elif bc:
            r[i] = tb_lb[i] / tr_f[i]; cls.append("base")
        else:
            r[i] = tb_f[i] / tr_f[i]; cls.append("ok")
    # both-censored ties carry no information: they are excluded from the median and from the test
    both_ = np.array([c_ == "both" for c_ in cls], dtype=bool)
    r[both_] = np.nan
    ok = ~np.isnan(r)
    d = np.full(n, np.nan)
    fin = ok & (r > 0) & np.isfinite(r)
    if fin.any():
        d[fin] = np.log(r[fin])
    ext = (np.abs(d[fin]).max() if fin.any() else 0.0) + 1.0
    d[ok & (r == 0)] = -ext
    counts = {c: cls.count(c) for c in ("ok", "base", "router", "both", "invalid")}
    counts["informative"] = int(ok.sum())
    return r, d, counts


def wilcoxon_d(d):
    """Two-sided Wilcoxon signed-rank test on the log-differences of pair_ratios (nan dropped)."""
    d = d[~np.isnan(d)]
    if len(d) < 8 or np.allclose(d, 0.0):
        return 1.0
    return float(wilcoxon(d, alternative="two-sided").pvalue)


def ratio_rows(rows_b, rows_r, key, field="t_live"):
    """Paired median ratio base/router with the censoring classes of pair_ratios."""
    r, _, _ = pair_ratios(rows_b, rows_r, key, field)
    r = r[~np.isnan(r)]
    return float(np.median(r)) if len(r) else np.nan


def p_rows(rows_b, rows_r, key, field="t_live"):
    return wilcoxon_d(pair_ratios(rows_b, rows_r, key, field)[1])


def sp_rows(rows_b, rows_r, key, ci=True, field="t_live"):
    """'ratio [CI] (p)' for two row lists at a tolerance key, with the censoring classes of
    pair_ratios: >= marks a ratio that is a lower bound (baseline censored on most instances, no router
    failure), a dagger counts router failures (ratio 0), a double dagger counts both-censored ties."""
    r, d, c = pair_ratios(rows_b, rows_r, key, field)
    rr = r[~np.isnan(r)]
    if len(rr) < 4:
        return "--" + class_marks(c), np.nan, 1.0
    sp = float(np.median(rr))
    p = wilcoxon_d(d)
    s = fmt_sp(sp)
    # the median ratio is a lower bound on the true ratio only when every substitution pushed it
    # downwards: baseline caps present, no router failure (ratio 0) and no both-censored tie in play
    if c["base"] > 0 and c["router"] == 0 and c["both"] == 0:
        s = "$\\geq$" + s
    if ci:
        lo, hi = boot_ci(rr)
        if np.isfinite(lo) and np.isfinite(hi):
            s += f" [{fmt_sp(lo)[:-8]}, {fmt_sp(hi)[:-8]}]"
        elif np.isfinite(lo):
            s += f" [{fmt_sp(lo)[:-8]}, $>10^{{3}}$]"
    s += class_marks(c)
    return (f"{s} ({pstr(p)})" if sp >= 1.0 else f"{s} (slower, {pstr(p)})"), sp, p


def class_marks(c):
    """Superscript counts of the censoring classes: dagger = router-only failures (ratio 0), double
    dagger = both censored (excluded ties), section = baseline-only censored (entered at the cap),
    times = invalid pairs (dropped)."""
    s = ""
    if c.get("router"):
        s += f"$^{{\\dagger {c['router']}}}$"
    if c.get("both"):
        s += f"$^{{\\ddagger {c['both']}}}$"
    if c.get("base"):
        s += f"$^{{\\S {c['base']}}}$"
    if c.get("invalid"):
        s += f"$^{{\\times {c['invalid']}}}$"
    return s


def ratio_marks(rows_b, rows_r, key, field="t_live"):
    """Compact 'ratio + class marks' for descriptive tables (no interval, no test)."""
    r, d, c = pair_ratios(rows_b, rows_r, key, field)
    rr = r[~np.isnan(r)]
    if len(rr) < 4:
        return "--" + class_marks(c), np.nan
    sp = float(np.median(rr))
    s = fmt_sp(sp)
    if c["base"] > 0 and c["router"] == 0 and c["both"] == 0:
        s = "$\\geq$" + s
    return s + class_marks(c), sp


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
        return times(d["methods"][m], key)

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

    _best_cache = {}

    def best_tau(P, key):
        """The strongest fixed schedule of this cell and tolerance against the router: among HINTS (any tau),
        phase-shifted HINTS and one-shot, the schedule with the smallest paired median ratio to the router (the
        best-by-median-time schedule can be a weak paired competitor when medians are close); without a router
        column, the schedule with the smallest median time."""
        ck = (id(P), key)
        if ck not in _best_cache:
            cands = [p for p in P if p.startswith("hints") or p.startswith("phints") or p == "oneshot"]
            if "router" in P:
                def paired(p):
                    r, _, _ = pair_ratios(P[p], P["router"], key)
                    ok = ~np.isnan(r)
                    return float(np.median(r[ok])) if ok.any() else np.inf
                _best_cache[ck] = min(cands, key=paired)
            else:
                _best_cache[ck] = min(cands, key=lambda p: np.median(times(P[p], key)))
        return _best_cache[ck]

    _dev_cache = {}

    def dev_sched(eq, N, spec, key):
        """The schedule selected on the development instances (select_schedule.py) for this cell and tolerance,
        or None if the selection file is absent."""
        path = f"{RESULTS_DIR}/schedule_dev_{eq}_{N}.json"
        if path not in _dev_cache:
            try:
                _dev_cache[path] = json.load(open(path))
            except (OSError, ValueError):
                _dev_cache[path] = None
        dd = _dev_cache[path]
        return dd["groups"][spec]["selected"].get(key) if dd and spec in dd["groups"] else None

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
                        row.append(time_cell(d["methods"][m], tkey(d, tol)))
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
                        sp_h_ = ratio_rows(P[HREF], blk["rows"], key, field="t_wu")
                        sp_b_ = ratio_rows(P[best_tau(P, key)], blk["rows"], key, field="t_wu")
                        sps.append(sp_h_)
                        agree.append(seq_agree(blk["rows"], P["router"], key))
                        if sp_h_ > 1.0 and p_rows(P[HREF], blk["rows"], key, field="t_wu") < 0.01 and sp_b_ > 1.0 and p_rows(P[best_tau(P, key)], blk["rows"], key, field="t_wu") < 0.01:
                            nsig += 1
                    out.append(" & ".join([f"${N_}^2$", f"$\\{{\\mathrm{{NO}}, \\text{{{SOLVER_NAMES[spec]}}}\\}}$",
                                           f"{np.mean(meds)*1e3:.2f} $\\pm$ {np.std(meds)*1e3:.2f}\\,ms",
                                           (f"{np.mean(agree):.0f}\\%" if all(np.isfinite(agree)) else PENDING), f"{min(sps):.2f}--{max(sps):.2f}$\\times$",
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
                s_h, sp_h = ratio_marks(P[HREF], P["router"], key)
                s_b, sp_b = ratio_marks(P[bt], P["router"], key)
                row.append(time_cell(P[HREF], key))
                row.append(time_cell(P[bt], key) + f"$_{{{sched_label(bt)}}}$")
                rc = time_cell(P["router"], key, bold=(sp_h >= 1 and sp_b >= 1))
                row.append(rc + f" ({s_h}\\,/\\,{s_b})")
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
                row += ["", "", time_cell(d["methods"][m], tkey(d, d["h2"]))]
            out.append(" & ".join(row) + " \\\\")
        if any_row and eq != EQS[-1]:
            out.append("\\midrule")
    out.append("\\bottomrule\n\\end{tabular}}")

    # ================================================================ summary macros per grid
    agree_incomplete = set()
    for N_ in NS:
        SUF = GRID_SUF[N_]
        best_h2_sched = []
        summ_specs = []
        summ = {"Solver": [], "Hints": [], "HintsTF": [], "Best": [], "Decay": [], "Oneshot": [], "SolverDeep": [], "HintsDeep": [], "HintsTFDeep": [], "BestDeep": [], "DecayDeep": [], "OneshotDeep": [], "OracleRatio": [],
                "AgreeOracle": [], "AgreeBest": [], "AgreeOracleDeep": [], "OneCall": [],
                "DevSched": [], "DevSchedDeep": [], "DevSchedSame": [], "DevSchedSameDeep": []}
        for eq in EQS:
            for spec in PAIRINGS:
                dg = cell(eq, N_, spec)
                if dg is None:
                    continue
                d, g = dg
                P = g["policies"]
                summ_specs.append(spec)
                for tol, suf_ in [(d["h2"], ""), (1e-8, "Deep")]:
                    key = tkey(d, tol)
                    t_r = times(P["router"], key)
                    summ["Solver" + suf_].append(ratio_rows(P["classical"], P["router"], key))
                    summ["Hints" + suf_].append(ratio_rows(P[HREF], P["router"], key))
                    summ["HintsTF" + suf_].append(ratio_rows(P["hints25"], P["router"], key))
                    summ["Best" + suf_].append(ratio_rows(P[best_tau(P, key)], P["router"], key))
                    dsel = dev_sched(eq, N_, spec, key)
                    if dsel and dsel in P:      # the deployable baseline: the schedule selected on the development instances
                        summ["DevSched" + suf_].append(ratio_rows(P[dsel], P["router"], key))
                        summ["DevSchedSame" + suf_].append(float(dsel == best_tau(P, key)))
                    if "oneshot" in P:
                        summ["Oneshot" + suf_].append(ratio_rows(P["oneshot"], P["router"], key))
                    bd_ = best_decay(P, key)
                    if bd_:
                        summ["Decay" + suf_].append(ratio_rows(P[bd_], P["router"], key))
                key = tkey(d, d["h2"])
                summ["OracleRatio"].append(np.median(times(P["router"], key)) / np.median(times(P["oracle"], key)))
                # agreement of the executed decision sequence up to the tolerance crossing (from the
                # run-length-encoded operation sequences stored with every row; nan if a cell predates them)
                a_o, a_b = seq_agree(P["router"], P["oracle"], key), seq_agree(P["router"], P[best_tau(P, key)], key)
                a_o8 = seq_agree(P["router"], P["oracle"], tkey(d, 1e-8))
                for nm_, v_ in [("AgreeOracle", a_o), ("AgreeBest", a_b), ("AgreeOracleDeep", a_o8)]:
                    if np.isfinite(v_):
                        summ[nm_].append(v_)
                    else:
                        agree_incomplete.add(SUF)     # a cell without stored sequences: the range stays pending
                best_h2_sched.append(best_tau(P, key))
                # instances on which the router makes exactly one corrector call to h^2
                nc_ = np.array([r["tol"][key]["no_calls"] if r["tol"][key]["no_calls"] is not None else -1 for r in P["router"]])
                summ["OneCall"].append(100 * float(np.mean(nc_ == 1)))
        if not summ["Solver"] or len(summ["Solver"]) < EXPECTED_CELLS.get(N_, 0):
            continue          # incomplete grid: every summary macro of this grid is defined as pending below
        for name, vals in summ.items():
            if name.startswith("DevSchedSame"):
                out.append(f"\\newcommand{{\\ca{name}{SUF}}}{{{(f'{int(sum(vals))}' if vals else PENDING)}}}")
                continue
            if name.startswith("Agree") or name == "OneCall":
                ok_ = bool(vals) and not (name.startswith("Agree") and SUF in agree_incomplete)
                out.append(f"\\newcommand{{\\ca{name}{SUF}Min}}{{{(f'{min(vals):.0f}' if ok_ else PENDING)}}}")
                out.append(f"\\newcommand{{\\ca{name}{SUF}Max}}{{{(f'{max(vals):.0f}' if ok_ else PENDING)}}}")
            else:
                rng_macro(out, "caSp" + name + SUF, vals)
        # the pairing with the smallest solver-alone ratio, and the best fixed schedules at h^2
        sol_pairs = [(v, spec) for v, spec in zip(summ["Solver"], summ_specs)]
        out.append(f"\\newcommand{{\\caSpSolverMinPairing{SUF}}}{{{SOLVER_NAMES[min(sol_pairs)[1]] if sol_pairs else PENDING}}}")
        cnt_h = {}
        for p in best_h2_sched:
            cnt_h[p] = cnt_h.get(p, 0) + 1
        out.append(f"\\newcommand{{\\caBestHTwoSchedList{SUF}}}{{{', '.join(f'{_lab2(p)} ({n})' for p, n in sorted(cnt_h.items(), key=lambda kv: -kv[1])) if cnt_h else PENDING}}}")
        out.append(f"\\newcommand{{\\caCellsRouterBeatsOneshotDeep{SUF}}}{{{sum(v > 1.0 for v in summ['OneshotDeep'])}}}")
        out.append(f"\\newcommand{{\\caCellsRouterBeatsHintsTFDeep{SUF}}}{{{sum(v > 1.0 for v in summ['HintsTFDeep'])}}}")
        # best fixed schedule at 1e-8 per pairing (labels), for the text
        bl_, short_ = [], []
        for eq in EQS:
            for spec in PAIRINGS:
                dg = cell(eq, N_, spec)
                if dg is None:
                    continue
                d, g = dg
                bl_.append(best_tau(g["policies"], tkey(d, 1e-8)))
                # does the best schedule at 1e-8 call the corrector more often than one macro-action of the paired
                # solver allows the router to (period shorter than the macro size)?
                m_ = re.match(r"p?hints(\d+)$", bl_[-1])
                short_.append(int(m_.group(1)) < int(g["m"][0]) if m_ else False)
        # the router's ratio against the best schedule at 1e-8, split by that criterion (same cell order as summ)
        for flag_, nm_ in [(True, "caShortPeriod"), (False, "caLongPeriod")]:
            vals_ = [v for v, s_ in zip(summ["BestDeep"], short_) if s_ == flag_]
            out.append(f"\\newcommand{{\\{nm_}Cells{SUF}}}{{{len(vals_) if vals_ else PENDING}}}")
            out.append(f"\\newcommand{{\\{nm_}Wins{SUF}}}{{{sum(v > 1.0 for v in vals_) if vals_ else PENDING}}}")
            out.append(f"\\newcommand{{\\{nm_}Med{SUF}}}{{{fmt_sp(float(np.median(vals_))) if vals_ else PENDING}}}")
            rng_macro(out, nm_ + SUF, vals_)
        lab_ = {"oneshot": "one-shot"}
        def _lab(p):
            if p == "oneshot":
                return "one-shot"
            if p.startswith("phints"):
                return f"$\\tau{{=}}{p[6:]}$ (first call at $0$)"
            return f"$\\tau{{=}}{p[5:]}$"
        cnt_ = {}
        for p in bl_:
            cnt_[p] = cnt_.get(p, 0) + 1
        out.append(f"\\newcommand{{\\caBestDeepSchedList{SUF}}}{{{', '.join(f'{_lab(p)} ({n})' for p, n in sorted(cnt_.items(), key=lambda kv: -kv[1]))}}}")
        out.append(f"\\newcommand{{\\caNumCells{SUF}}}{{{len(summ['Solver'])}}}")
        out.append(f"\\newcommand{{\\caCellsRouterBeatsBest{SUF}}}{{{sum(v > 1.0 for v in summ['Best'])}}}")
        out.append(f"\\newcommand{{\\caCellsRouterBeatsDevSched{SUF}}}{{{sum(v > 1.0 for v in summ['DevSched']) if summ['DevSched'] else PENDING}}}")
        out.append(f"\\newcommand{{\\caCellsRouterBeatsDevSchedDeep{SUF}}}{{{sum(v > 1.0 for v in summ['DevSchedDeep']) if summ['DevSchedDeep'] else PENDING}}}")
        out.append(f"\\newcommand{{\\caNumDevSched{SUF}}}{{{len(summ['DevSched']) if summ['DevSched'] else PENDING}}}")
        out.append(f"\\newcommand{{\\caCellsRouterBeatsHints{SUF}}}{{{sum(v > 1.0 for v in summ['Hints'])}}}")
        out.append(f"\\newcommand{{\\caCellsRouterBeatsBestDeep{SUF}}}{{{sum(v > 1.0 for v in summ['BestDeep'])}}}")
        out.append(f"\\newcommand{{\\caCellsRouterBeatsHintsDeep{SUF}}}{{{sum(v > 1.0 for v in summ['HintsDeep'])}}}")
        out.append(f"\\newcommand{{\\caCellsRouterWithinBest{SUF}}}{{{sum(v >= 0.9 for v in summ['Best'])}}}")
        out.append(f"\\newcommand{{\\caCellsRouterBeatsDecay{SUF}}}{{{sum(v > 1.0 for v in summ['Decay'])}}}")
        out.append(f"\\newcommand{{\\caCellsRouterBeatsDecayDeep{SUF}}}{{{sum(v > 1.0 for v in summ['DecayDeep'])}}}")
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
                        lm.append(ratio_rows(rm_, g["policies"]["router"], key))
                    rk_ = bl_rows(g, db, kry_name(eq))
                    if rk_:
                        lk.append(ratio_rows(rk_, g["policies"]["router"], key))
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
            vme.append(ratio_rows(rm_, g["policies"]["router"], tkey(d, d["h2"])))
            vmed.append(ratio_rows(rm_, g["policies"]["router"], tkey(d, 1e-8)))
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
            vs_mg.append(ratio_rows(rm_, g["policies"]["router"], kh))
        rk_ = bl_rows(g, d, kry_name(eq))
        if rk_:
            vs_kry.append(ratio_rows(rk_, g["policies"]["router"], kh))
        mgp = cell(eq, N, "mg")
        if mgp:
            vs_mg_ens.append(ratio_rows(mgp[1]["policies"]["classical"], mgp[1]["policies"]["router"], tkey(mgp[0], dd["h2"])))
    for name, vals in [("caVsMg", vs_mg), ("caVsKrylov", vs_kry), ("caVsMgEns", vs_mg_ens)]:
        rng_macro(out, name, vals)
    fft_ratio = []
    for (eq, N), d in Bf.items():
        if N != MAIN_N:
            continue
        pw = {s_: cell(eq, N, s_) for s_ in PAIRINGS if cell(eq, N, s_)}
        if not pw or "fft" not in d["methods"] or not d["methods"]["fft"]:
            continue
        best_s = min(pw, key=lambda s_: np.median(times(pw[s_][1]["policies"]["router"], tkey(pw[s_][0], d["h2"]))))
        dd, g = pw[best_s]
        rf_ = bl_rows(g, d, "fft")
        fft_ratio.append(ratio_rows(g["policies"]["router"], rf_, tkey(dd, dd["h2"])))
    rng_macro(out, "caFftRatio", fft_ratio)

    def num_macro(name, vals, fmt):
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        out.append(f"\\newcommand{{\\{name}Min}}{{{(fmt % min(vals)) if vals else PENDING}}}")
        out.append(f"\\newcommand{{\\{name}Max}}{{{(fmt % max(vals)) if vals else PENDING}}}")

    # ================================================================ AUC summary macros (128^2 cells), corrector-time fraction
    auc_h, auc_s, auc_p, nofrac = [], [], [], []
    for (eq_, N_, spec_, ens_), (d_, g_) in R.items():
        if ens_ or N_ != MAIN_N:
            continue
        P = g_["policies"]
        ar = np.array([r["auc_T"] for r in P["router"]])
        for pol_, lst in [(HREF, auc_h), ("classical", auc_s)]:
            a_ = np.array([r["auc_T"] for r in P[pol_]])
            lst.append(float(np.mean(a_) / np.mean(ar)))
            auc_p.append(ttest_p(a_, ar))
        key = tkey(d_, d_["h2"])
        c_no = g_["costs"]["no"]
        fr = [r["tol"][key]["no_calls"] * c_no / r["tol"][key]["t_live"] for r in P["router"] if r["tol"][key]["t_live"] and r["tol"][key]["no_calls"] is not None]
        if fr:
            nofrac.append(100 * float(np.median(fr)))
    num_macro("caAucHintsRatio", auc_h, "%.0f")
    num_macro("caAucSolverRatio", auc_s, "%.0f")
    # largest AUC-test p-value as a power-of-ten bound for use inside math ($p < \caAucPMax$)
    if auc_p:
        pm_ = max(max(auc_p), 1e-10)
        out.append(f"\\newcommand{{\\caAucPMax}}{{10^{{{int(np.ceil(np.log10(pm_)))}}}}}")
    else:
        out.append(f"\\newcommand{{\\caAucPMax}}{{{PENDING}}}")
    num_macro("caNoFrac", nofrac, "%.0f")

    # ================================================================ in-situ overheads, drift, charged fraction, live/WU, macro sizes
    dec_us, gap_us, dfrac, dfrac8, charged, livewu, drifts, waits, kchk, ndec = [], [], [], [], [], [], [], 0, [], []
    n_retimed, n_flagged, kinv = 0, 0, 0
    pre_protocol = set()     # cells produced before the validity flag / in-cell re-timing existed
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
            drifts.append(abs(dr["ratio"] - 1.0) * 100); waits += int(dr["retries"]) + int(dr.get("retries_first", 0))
            n_retimed += int(bool(dr.get("retimed"))); n_flagged += int(dr["ratio"] > 1.10)
        for k_, v_ in g_.items():
            if k_.startswith("krylov_check:"):
                kchk.append(v_["max_ratio_timed_over_untimed_error"])
        for p_, rows_ in P.items():
            if p_.startswith("base:"):
                kinv += int(sum(1 for r in rows_ if not r.get("valid", True)))
                if rows_ and "valid" not in rows_[0]:
                    pre_protocol.add((eq_, N_, spec_))
        if g_.get("drift") and "ref_us" not in g_["drift"][0]:
            pre_protocol.add((eq_, N_, spec_))
    num_macro("caDecInSitu", dec_us, "%.0f")
    num_macro("caRouterOracleGap", gap_us, "%.0f")
    num_macro("caDecFrac", dfrac, "%.1f")
    num_macro("caDecFracDeep", dfrac8, "%.1f")
    num_macro("caChargedFrac", charged, "%.0f")
    num_macro("caLiveWu", livewu, "%.2f")
    num_macro("caNDec", ndec, "%.0f")
    out.append(f"\\newcommand{{\\caDriftMax}}{{{('%.1f' % max(drifts)) if drifts else PENDING}}}")
    out.append(f"\\newcommand{{\\caDriftWaits}}{{{waits}}}")
    out.append(f"\\newcommand{{\\caDriftRetimed}}{{{n_retimed if not pre_protocol else PENDING}}}")
    out.append(f"\\newcommand{{\\caDriftFlagged}}{{{n_flagged if not pre_protocol else PENDING}}}")
    out.append(f"\\newcommand{{\\caDriftN}}{{{len(drifts)}}}")
    out.append(f"\\newcommand{{\\caKrylovCheckMax}}{{{('%.2f' % max(kchk)) if kchk else PENDING}}}")
    # the validity count and the re-timing counts are only meaningful once every cell carries them
    out.append(f"\\newcommand{{\\caKrylovInvalid}}{{{kinv if not pre_protocol else PENDING}}}")
    out.append(f"\\newcommand{{\\caPreProtocolCells}}{{{len(pre_protocol)}}}")
    for N_ in NS:
        SUF = GRID_SUF[N_]
        for spec_, nm_ in [("jacobi", "Jacobi"), ("gs", "Gs"), ("mg", "Mg"), ("ssor", "SymGs")]:
            dg = cell("Poisson", N_, spec_) or next((cell(e_, N_, spec_) for e_ in EQS if cell(e_, N_, spec_)), None)
            if dg:
                out.append(f"\\newcommand{{\\caM{nm_}{SUF}}}{{{dg[1]['m'][0]}}}")
                if spec_ == "jacobi" and SUF == "":
                    dgd_ = cell("Poisson", N_, "jacobi_0.67") or next((cell(e_, N_, "jacobi_0.67") for e_ in EQS if cell(e_, N_, "jacobi_0.67")), None)
                    mj_ = (dgd_[1]["m"][0] if dgd_ else dg[1]["m"][0])
                    out.append(f"\\newcommand{{\\caMJacobiDamped}}{{{mj_}}}")
                    def sci_tex(x):
                        e_ = int(np.floor(np.log10(x))); mant_ = x / 10 ** e_
                        return f"{mant_:.0f}\\times 10^{{{e_}}}"
                    out.append(f"\\newcommand{{\\caJacDampMacro}}{{{sci_tex(0.665 ** mj_)}}}")
                    out.append(f"\\newcommand{{\\caJacNyqMacro}}{{{sci_tex(0.34 ** mj_)}}}")
                c_ = dg[1]["costs"]
                out.append(f"\\newcommand{{\\caCost{nm_}{SUF}}}{{{fmt_time(c_[spec_])}}}")
                if f"caCostNo{SUF}" not in "\n".join(out[-400:]):
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
        nest_comps = []   # (line index in out, column, text, speedup, p) for the Holm pass
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
                        row.append(time_cell(Rr, key, field="t_wu"))
                        row.append(time_cell(Ro, key, field="t_wu", italic=True))
                    tb = times(best_single[1], tkey(best_single[3], 1e-8), field="t_wu")
                    tbo = times(best_single[2], tkey(best_single[3], 1e-8), field="t_wu")
                    if len(members) > 1:
                        s1, sp1, p1 = sp_rows(best_single[1], Rr, key8, ci=False, field="t_wu")
                        s2, sp2, p2 = sp_rows(best_single[2], Ro, key8, ci=False, field="t_wu")
                        ci1 = len(nest_comps)
                        nest_comps.append((len(out), len(row), s1, sp1, p1)); row.append(s1)
                        nest_comps.append((len(out), len(row), s2, sp2, p2)); row.append(s2)
                        keyh = tkey(dd, dd["h2"])
                        bsh = min(singles, key=lambda r_: np.median(times(r_[1], tkey(r_[3], r_[3]["h2"]), field="t_wu")))
                        tbh = times(bsh[1], tkey(bsh[3], bsh[3]["h2"]), field="t_wu")
                        nest_stats.append((eq, N, members, sp1, p1, ratio_rows(bsh[1], Rr, keyh, field="t_wu"), sp2, ci1))
                    else:
                        row += ["(best single)" if members == best_single[0] else "--", "--"]
                    out.append(" & ".join(row) + " \\\\")
                    first = False
                out.append("\\midrule")
        out[-1] = "\\bottomrule\n\\end{tabular}}"
        # Holm-corrected bolding over all comparisons of the nested table (speedup >= 1.10)
        sig_n = holm_mask([c[4] for c in nest_comps]) if nest_comps else []
        nest_sig = {}
        for (li, ci_, s_, sp_, p_), ok_ in zip(nest_comps, sig_n):
            nest_sig[(li, ci_)] = ok_
            if ok_ and sp_ >= 1.10:
                cells_ = out[li].rstrip(" \\\\").split(" & ")
                cells_[ci_] = f"\\textbf{{{s_}}}"
                out[li] = " & ".join(cells_) + " \\\\"
        # per grid (suffix "" = 128^2, "B" = 256^2): a win is a cell whose table entry is bold (ratio >= 1.10 and
        # Holm-significant over the table); a grid's macros are emitted once every equation's ensemble cells exist
        # (the two nested sets and, at 128^2, the four larger sets), so that a range never summarises a subset
        NEST_EXPECTED = {128: 4 * 2 + 3 * 4, 256: 4 * 2}
        for N_n in NS:
            SUFn = GRID_SUF[N_n]
            st_ = [s for s in nest_stats if s[1] == N_n]
            if not st_ or len(st_) < NEST_EXPECTED.get(N_n, 0):
                continue
            wins = [s for s in st_ if s[3] >= 1.10 and bool(sig_n[s[7]])]
            out.append(f"\\newcommand{{\\caNestNum{SUFn}}}{{{len(st_)}}}")
            out.append(f"\\newcommand{{\\caNestWins{SUFn}}}{{{len(wins)}}}")
            out.append(f"\\newcommand{{\\caNestLosses{SUFn}}}{{{sum(1 for s in st_ if s[3] < 0.95)}}}")
            rng_macro(out, "caNestSp" + SUFn, [s[3] for s in (wins or st_)])
            out.append(f"\\newcommand{{\\caNestMinRatio{SUFn}}}{{{fmt_sp(min(s[3] for s in st_))}}}")
            out.append(f"\\newcommand{{\\caNestMaxRatio{SUFn}}}{{{fmt_sp(max(s[3] for s in st_))}}}")
            out.append(f"\\newcommand{{\\caNestMinRatioH{SUFn}}}{{{fmt_sp(min(s[5] for s in st_))}}}")
            out.append(f"\\newcommand{{\\caNestMaxRatioH{SUFn}}}{{{fmt_sp(max(s[5] for s in st_))}}}")
            out.append(f"\\newcommand{{\\caNestOrMinRatio{SUFn}}}{{{fmt_sp(min(s[6] for s in st_))}}}")
            out.append(f"\\newcommand{{\\caNestOrMaxRatio{SUFn}}}{{{fmt_sp(max(s[6] for s in st_))}}}")
            out.append(f"\\newcommand{{\\caNestOrWins{SUFn}}}{{{sum(1 for s in st_ if s[6] >= 1.05)}}}")
        if nest_stats:
            pass
    else:
        pending(out, "caensnest")
    for SUFn in ["", "B", "C"]:
        for base_, tail_ in [("caNestNum", ""), ("caNestWins", ""), ("caNestLosses", ""), ("caNestSp", "Min"), ("caNestSp", "Max"), ("caNestMinRatio", ""), ("caNestMaxRatio", ""),
                             ("caNestMinRatioH", ""), ("caNestMaxRatioH", ""), ("caNestOrMinRatio", ""), ("caNestOrMaxRatio", ""), ("caNestOrWins", "")]:
            name = f"{base_}{SUFn}{tail_}"      # the grid suffix precedes Min/Max, as rng_macro writes it
            if not any(l.startswith(f"\\newcommand{{\\{name}}}") for l in out):
                out.append(f"\\newcommand{{\\{name}}}{{{PENDING}}}")

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
        be_list, be_kind = [], []
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
                be_list.append(be)
                be_kind.append("nosaving" if sav_h <= 0 else ("notrain" if not rt else "ok"))
                out.append(" & ".join([EQ_NAMES[eq] if first else "", f"${N}^2$", f"{c['data_s'] + c['fit_s']:.0f}\\,s",
                                       f"{rt['train_s']:.0f}\\,s" if rt else "--", fmt_time(sav_h) if sav_h > 0 else "--",
                                       f"{be:.0f}" if np.isfinite(be) else "--", fmt_time(sav_c)]) + " \\\\")
                first = False
            if not first and eq != EQS[-1]:
                out.append("\\midrule")
        out[-1] = "\\bottomrule\n\\end{tabular}}"
        fin_be = [b for b in be_list if np.isfinite(b)]
        out.append(f"\\newcommand{{\\caBreakEvenMin}}{{{(f'{min(fin_be):.0f}' if fin_be else PENDING)}}}")
        out.append(f"\\newcommand{{\\caBreakEvenMax}}{{{(f'{max(fin_be):.0f}' if fin_be else PENDING)}}}")
        out.append(f"\\newcommand{{\\caBreakEvenNone}}{{{be_kind.count('nosaving')}}}")
        out.append(f"\\newcommand{{\\caBreakEvenNoTrain}}{{{be_kind.count('notrain')}}}")
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
        out.append("Equation & Ensemble & $K$ & $T$ & $n$ & $\\mu$ & $\\alpha(O)$ greedy prefixes & $\\alpha(O)$ all $|S| < T$ & bound holds & $g(S^T)/g(O)$ med (max) & greedy $= O$ & $g(O)$ at floor / rule $=$ Alg.~1 / violations (greedy prefixes, all prefixes) / on-path premises verified \\\\ \\midrule")
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
                                           f"{sum(r.get('opt_at_floor', False) for r in rows)} / {sum(r.get('deployed_eq_plain', True) for r in rows)} / {sum(r.get('viol_greedy', 0) for r in rows)}, {sum(r.get('viol_all', 0) for r in rows)} / {sum(r.get('premises_verified', False) for r in rows)}"]) + " \\\\")
                    first = False
                out.append("\\midrule")
        out[-1] = "\\bottomrule\n\\end{tabular}}"
        allrows = [r for d in Tf.values() for G in d["groups"].values() for r in G["rows"]]
        out.append(f"\\newcommand{{\\caThmMuMax}}{{{max(r['mu'] for r in allrows):.3f}}}")
        out.append(f"\\newcommand{{\\caThmHolds}}{{{sum(r['holds'] for r in allrows)}/{len(allrows)}}}")
        out.append(f"\\newcommand{{\\caThmFloor}}{{{sum(r.get('opt_at_floor', False) for r in allrows)}}}")
        out.append(f"\\newcommand{{\\caThmPlainDiff}}{{{sum(not r.get('deployed_eq_plain', True) for r in allrows)}}}")
        out.append(f"\\newcommand{{\\caThmViol}}{{{sum(r.get('viol_greedy', 0) for r in allrows)}}}")
        out.append(f"\\newcommand{{\\caThmViolAll}}{{{sum(r.get('viol_all', 0) for r in allrows)}}}")
        res_rows = [r for r in allrows if not r.get('opt_at_floor', False)] or allrows
        out.append(f"\\newcommand{{\\caThmMaxRatio}}{{{math.ceil(10 * max(r['greedy_over_opt'] for r in res_rows)) / 10:.1f}}}")
        out.append(f"\\newcommand{{\\caThmResolved}}{{{len([r for r in allrows if not r.get('opt_at_floor', False)])}}}")
        out.append(f"\\newcommand{{\\caThmAlphaGreedyMax}}{{{max(r['alpha_greedy'] for r in allrows):.3f}}}")
        out.append(f"\\newcommand{{\\caThmAlphaAllMax}}{{{max(r['alpha_all'] for r in allrows):.3f}}}")
        out.append(f"\\newcommand{{\\caThmPremises}}{{{sum(r.get('premises_verified', False) for r in allrows)}/{len(allrows)}}}")
        out.append(f"\\newcommand{{\\caThmExpand}}{{{sum(r.get('expand_greedy', 0) for r in allrows)}}}")
        out.append(f"\\newcommand{{\\caThmExpandAll}}{{{sum(r.get('expand_all', 0) for r in allrows)}}}")
        out.append(f"\\newcommand{{\\caThmClip}}{{{sum(r.get('clip_changed_path', False) for r in allrows)}}}")
        out.append(f"\\newcommand{{\\caThmN}}{{{len(allrows)}}}")
    else:
        pending(out, "catheorem")
        for nm_ in ["caThmMuMax", "caThmHolds", "caThmFloor", "caThmPlainDiff", "caThmViol", "caThmViolAll", "caThmMaxRatio", "caThmResolved", "caThmAlphaGreedyMax", "caThmAlphaAllMax", "caThmPremises", "caThmExpand", "caThmExpandAll", "caThmClip", "caThmN"]:
            out.append(f"\\newcommand{{\\{nm_}}}{{{PENDING}}}")

    # ================================================================ decision-granularity ablation (unit = corrector / 4)
    Ru4 = load(tag="_u4")
    if Ru4:
        out.append("\\newcommand{\\cagranularity}{")
        out.append("\\begin{tabular}{llcccccccc}\n\\toprule")
        out.append("& & \\multicolumn{4}{c}{unit = one corrector call} & \\multicolumn{4}{c}{unit = corrector call / 4} \\\\ \\cmidrule(lr){3-6}\\cmidrule(lr){7-10}")
        out.append("Equation & Pairing & oracle $h^2$ & router $h^2$ & router $10^{-8}$ & vs.\\ best schedule ($10^{-8}$) & oracle $h^2$ & router $h^2$ & router $10^{-8}$ & vs.\\ best schedule ($10^{-8}$) \\\\ \\midrule")
        gran_comps = []
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
                    Pm = R[k][1]["policies"]   # fixed schedules from the main run (same instances, work units)
                    s_, sp_, p_ = sp_rows(Pm[best_tau(Pm, key8)], P["router"], key8, ci=False, field="t_wu")
                    cells += [time_cell(P["oracle"], key2, italic=True), time_cell(P["router"], key2), time_cell(P["router"], key8), s_]
                    gran_comps.append((len(out), 1 + len(cells), s_, sp_, p_))
                out.append(" & ".join([EQ_NAMES[eq] if first else "", SOLVER_NAMES[spec]] + cells) + " \\\\")
                first = False
            if not first and eq != EQS[-1]:
                out.append("\\midrule")
        sig_g = holm_mask([c[4] for c in gran_comps]) if gran_comps else []
        for (li, ci_, s_, sp_, p_), ok_ in zip(gran_comps, sig_g):
            if ok_ and sp_ >= 1.10:
                cells_ = out[li].rstrip(" \\\\").split(" & ")
                cells_[ci_] = f"\\textbf{{{s_}}}"
                out[li] = " & ".join(cells_) + " \\\\"
        if out[-1] == "\\midrule":
            out.pop()
        out.append("\\bottomrule\n\\end{tabular}}")
    else:
        pending(out, "cagranularity")

    # ================================================================ development vs confirmatory runs
    if Rdev and R:
        held_comps = []
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
                    row_start = len(held_comps)
                    for d_, g_ in [Rdev[k], R[k]]:
                        P = g_["policies"]
                        if not k[3]:
                            key = tkey(d_, d_["h2"])
                            for rows_b in [P["hints25"], P[best_tau(P, key)]]:
                                s_, sp_, p_ = sp_rows(rows_b, P["router"], key, ci=False)
                                held_comps.append([None, 3 + len(cells), s_, sp_, p_]); cells.append(s_)
                        else:
                            key = tkey(d_, 1e-8)
                            singles = {p_: P[p_] for p_ in P if p_.startswith("router@")}
                            if singles:
                                bs_ = min(singles, key=lambda p_: np.median(times(singles[p_], key, field="t_wu")))
                                s_, sp_, p_ = sp_rows(singles[bs_], P["router"], key, ci=False, field="t_wu")
                                held_comps.append([None, 3 + len(cells), s_, sp_, p_]); cells.append(s_)
                            else:
                                cells.append("--")
                            if "oracle" in P:
                                s_, sp_, p_ = sp_rows(P["oracle"], P["router"], key, ci=False, field="t_wu")
                                held_comps.append([None, 3 + len(cells), s_, sp_, p_]); cells.append(s_)
                            else:
                                cells.append("--")
                    lab = (f"$\\{{\\mathrm{{NO}}, \\text{{{SOLVER_NAMES[k[2]]}}}\\}}$" if not k[3] else wname(k[2].split("+")) + " ($10^{-8}$, work units)")
                    out.append(" & ".join([EQ_NAMES[eq] if first else "", f"${N}^2$", lab] + cells) + " \\\\")
                    for kk in range(row_start, len(held_comps)):
                        held_comps[kk][0] = len(out) - 1
                    first = False
            if not first and eq != EQS[-1]:
                out.append("\\midrule")
        sig_h = holm_mask([c[4] for c in held_comps]) if held_comps else []
        for (li, ci_, s_, sp_, p_), ok_ in zip(held_comps, sig_h):
            if ok_ and sp_ >= 1.10:
                cells_ = out[li].rstrip(" \\\\").split(" & ")
                cells_[ci_] = f"\\textbf{{{s_}}}"
                out[li] = " & ".join(cells_) + " \\\\"
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
            man[path] = {"sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
            try:
                dj_ = json.load(open(path))
                if isinstance(dj_, dict) and "provenance" in dj_:
                    man[path]["git_commit"] = dj_["provenance"].get("git_commit"); man[path]["git_dirty"] = dj_["provenance"].get("git_dirty")
                    if "retime_provenance" in dj_:
                        man[path]["retime_git_commit"] = dj_["retime_provenance"].get("git_commit")
            except Exception:
                pass
    json.dump(man, open(os.path.join(os.path.dirname(OUT_TEX) or ".", "manifest.json"), "w"), indent=1)

    # ================================================================ placeholders / summary macros
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", "\n".join(out)))
    for name, val in [("caT", "300"), ("caN", "128"), ("caLstmMs", PENDING), ("caLstmOverJacobi", PENDING), ("caVsMgMin", PENDING), ("caVsMgMax", PENDING),
                      ("caVsKrylovMin", PENDING), ("caVsKrylovMax", PENDING), ("caVsMgEnsMin", PENDING), ("caVsMgEnsMax", PENDING),
                      ("caFftRatioMin", PENDING), ("caFftRatioMax", PENDING),
                      ("caRhoAMax", PENDING), ("caRhoSpecMax", PENDING), ("caRhoTwoGsMax", PENDING), ("caBandMax", PENDING), ("caAlphaHatMax", PENDING),
                      ("caAlphaHatMed", PENDING), ("caAlphaHatMaxH", PENDING), ("caAlphaBoundMin", PENDING), ("caAlphaBoundMax", PENDING),
                      ("caJacDampMacro", PENDING), ("caJacNyqMacro", PENDING), ("caBreakEvenMin", PENDING), ("caBreakEvenMax", PENDING), ("caBreakEvenNone", PENDING),
                      ("caBreakEvenNoTrain", PENDING), ("caMJacobiDamped", PENDING)]:
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
        for name in ["caSpSolver", "caSpHints", "caSpHintsTF", "caSpBest", "caSpDecay", "caSpOneshot", "caSpSolverDeep", "caSpHintsDeep", "caSpHintsTFDeep", "caSpBestDeep", "caSpDecayDeep", "caSpOneshotDeep", "caSpOracleRatio",
                     "caAgreeOracle", "caAgreeBest", "caAgreeOracleDeep", "caOneCall",
                     "caVsMgAll", "caVsMgDeep", "caVsKrylovAll", "caVsKrylovDeep", "caVsMgEnsAll", "caVsMgEnsDeep",
                     "caShortPeriod", "caLongPeriod", "caSpDevSched", "caSpDevSchedDeep"]:
            for mm in ["Min", "Max"]:
                if name + SUF + mm not in defined:
                    out.append(f"\\newcommand{{\\{name}{SUF}{mm}}}{{{PENDING}}}")
        for name in ["caNumCells", "caCellsRouterBeatsBest", "caCellsRouterBeatsHints", "caCellsRouterBeatsBestDeep", "caCellsRouterBeatsHintsDeep", "caCellsRouterWithinBest",
                     "caCellsRouterBeatsDecay", "caCellsRouterBeatsDecayDeep", "caCellsRouterBeatsOneshotDeep", "caCellsRouterBeatsHintsTFDeep", "caBestDeepSchedList", "caBestHTwoSchedList", "caSpSolverMinPairing",
                     "caMJacobi", "caMGs", "caMMg", "caMSymGs", "caCostJacobi", "caCostGs", "caCostMg", "caCostSymGs", "caCostNo", "caCostRes",
                     "caShortPeriodCells", "caShortPeriodWins", "caShortPeriodMed", "caLongPeriodCells", "caLongPeriodWins", "caLongPeriodMed",
                     "caDevSchedSame", "caDevSchedSameDeep", "caCellsRouterBeatsDevSched", "caCellsRouterBeatsDevSchedDeep", "caNumDevSched"]:
            if name + SUF not in defined:
                out.append(f"\\newcommand{{\\{name}{SUF}}}{{{PENDING}}}")

    os.makedirs("paper", exist_ok=True)
    with open(OUT_TEX, "w") as fh:
        fh.write("\n".join(out) + "\n")
    print("wrote", OUT_TEX)


if __name__ == "__main__":
    main()
