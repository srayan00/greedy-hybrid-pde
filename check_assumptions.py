"""Numerical verification of the assumptions behind Theorem 4.1 / Propositions
4.2-4.3 / Theorem 5.1 for the settings of the wall-clock study.

Per (equation, N) and per operation j (classical solvers, their cost-equalised
macro-actions, the DeepONet corrector), on the mean-free subspace:
  * rho2      = ||I - C_j L||_2   (Lipschitz constant of the error propagation map;
                power iteration on G^T G; exact from the symbol for Jacobi)
  * rho_spec  = spectral radius of I - C_j L (ARPACK)
  * sigma_min = smallest observed contraction ||G x||/||x|| (exact null vector for
                GS/SOR/SymGS: the error at the first grid point is annihilated by
                one sweep; min over Fourier modes for the others)
  * zero      = ||C_j(0)||  (zero preservation)
  * comm      = ||G_j G_NO - G_NO G_j||/||x|| on random vectors (departure from a
                shared eigenbasis, Proposition 4.3; 0 for Fourier-diagonal maps)
and for the corrector the in-band / out-of-band per-mode contraction.
Along the cost-aware oracle path to h^2 (8 test instances): T, sum rho^2,
alpha(O) of Proposition 4.2, the empirical supermodularity ratio of
eq. (alpha_supermod) with S = oracle prefix and S' = oracle suffix, and the
normalised bounds of Theorem 5.1.

  python check_assumptions.py --equation Poisson --N 128
writes results/assumptions_<eq>_<N>.json
"""

import argparse
import json
import os
import time

import numpy as np
import scipy.sparse.linalg as spla
import torch

from fast_pde import FastStencilPDE, GRF2D, make_solver, FastJacobi, FastGaussSeidel, FastSOR, FastSSOR, FastMultigrid
from corrector import DeepONetCorrector
from hybrid import Env

p = argparse.ArgumentParser()
p.add_argument("--equation", default="Poisson")
p.add_argument("--N", type=int, default=128)
p.add_argument("--solvers", default=None)
p.add_argument("--n_inst", type=int, default=8)
p.add_argument("--n_modes", type=int, default=4000)
p.add_argument("--seed", type=int, default=72)
p.add_argument("--ckp_dir", default="./checkpoints")
p.add_argument("--out_dir", default="./results")
p.add_argument("--path_tol", type=float, default=1e-8, help="horizon of the oracle path (relative error)")
p.add_argument("--paths_only", action="store_true", help="reuse the stored norms and recompute only the oracle-path quantities")
args = p.parse_args()
torch.set_num_threads(1)
rng = np.random.default_rng(0)

N = args.N
n = N * N
pde = FastStencilPDE(N, equation=args.equation)
pdeT = FastStencilPDE(N, equation=args.equation, b_vec=(-pde.b1, -pde.b2), aniso_eps=pde.aniso_eps)  # A^T (central convection)
corrector = DeepONetCorrector(f"{args.ckp_dir}/deeponet_{args.equation}_{N}_best.pth", threads=1)
costs_all = json.load(open(f"{args.ckp_dir}/costs_{args.equation}_{N}.json"))
solvers = args.solvers.split(",") if args.solvers else [k for k in costs_all if "+" not in k]
h2 = 1.0 / N ** 2
zeros = np.zeros((1, N, N))


def demean(x):
    return x - x.mean()


# ---------------------------------------------------------------- linear maps
def apply_A(x):
    return pde.apply_A(x)


def apply_AT(x):
    return pdeT.apply_A(x)


# sanity: <A x, y> = <x, A^T y>
_x, _y = rng.standard_normal((1, N, N)), rng.standard_normal((1, N, N))
assert abs((apply_A(_x) * _y).sum() - (_x * apply_AT(_y)).sum()) < 1e-8 * abs((apply_A(_x) * _y).sum()) + 1e-12


def C_apply(sv, r):
    return sv.step(zeros, None, r)


def C_apply_T(sv, r):
    """C^T r for the splu-based solvers (transposed triangular solves)."""
    rr = np.ascontiguousarray(r.reshape(-1, 1))
    if isinstance(sv, FastJacobi):
        return (sv.weight / sv.pde.diag) * r
    if isinstance(sv, MGmap):
        return sv.apply_T(r)
    if isinstance(sv, (FastGaussSeidel, FastSOR)):
        return sv.lu.solve(rr, trans="T").reshape(r.shape)
    if isinstance(sv, FastSSOR):
        y = sv._lu_up.solve(rr, trans="T")
        y = sv._diag[:, None] * y
        y = sv._lu_low.solve(np.ascontiguousarray(y), trans="T")
        return sv._scale * y.reshape(r.shape)
    raise NotImplementedError


from fast_pde import restrict_fw, prolong_bilinear


class MGmap:
    """The V-cycle of FastMultigrid as a linear map r -> e and its transpose.
    With S = (D+L)^{-1} (GS), Q = sum_{i<nu} (I - S A)^i S (nu smoothing steps from
    zero), the level-l map is V_l = (I - S A)^{nu2} [Q + P V_{l+1} R (I - A Q)] + Q,
    whose transpose follows by reversing the composition; R = c P^T (c calibrated)."""

    def __init__(self, mg, pde, pdeT):
        self.mg = mg
        self.lv = []
        for (lp, sm) in mg.levels:
            lpT = FastStencilPDE(lp.N, equation=lp.equation, a=lp.a, b_vec=(-lp.b1, -lp.b2), aniso_eps=lp.aniso_eps)
            self.lv.append((lp, lpT, sm))
        self.coarseT = FastStencilPDE(mg.coarse.N, equation=mg.coarse.equation, a=mg.coarse.a,
                                      b_vec=(-mg.coarse.b1, -mg.coarse.b2), aniso_eps=mg.coarse.aniso_eps)
        # calibrate R = c P^T on the finest level
        Nl = mg.levels[0][0].N
        x = rng.standard_normal((1, Nl, Nl)); y = rng.standard_normal((1, Nl // 2, Nl // 2))
        self.c = float((restrict_fw(x) * y).sum() / (x * prolong_bilinear(y)).sum())

    def _Q(self, lp, sm, r, nu):
        e = np.zeros_like(r)
        for _ in range(nu):
            e = sm.step(e, r)
        return e

    def _QT(self, lp, lpT, sm, y, nu):
        # Q^T y = sum_i S^T (I - A^T S^T)^i y
        acc = np.zeros_like(y)
        z = y.copy()
        for i in range(nu):
            acc = acc + C_apply_T(sm, z)
            z = z - lpT.apply_A(C_apply_T(sm, z))
        return acc

    def apply(self, r):
        return self.mg._vcycle(0, r)

    def apply_T(self, y, lvl=0):
        if lvl == len(self.lv):
            return self.coarseT.solve_direct(y)
        lp, lpT, sm = self.lv[lvl]
        nu1, nu2 = self.mg.nu1, self.mg.nu2
        # y1 = ((I - S A)^{nu2})^T y = (I - A^T S^T)^{nu2} y
        y1 = y.copy()
        for _ in range(nu2):
            y1 = y1 - lpT.apply_A(C_apply_T(sm, y1))
        # [Q + P V R (I - A Q)]^T y1 = Q^T y1 + (I - Q^T A^T) R^T V^T P^T y1
        w = self.c * restrict_fw(y1)                      # P^T y1 = c R y1
        w = self.apply_T(w, lvl + 1)                       # V_{l+1}^T
        w = prolong_bilinear(w) / self.c                   # R^T w = P w / c
        w = w - self._QT(lp, lpT, sm, lpT.apply_A(w), nu1)
        return self._QT(lp, lpT, sm, y1, nu1) + w + self._QT(lp, lpT, sm, y, nu2)


class NOmap:
    """The corrector as a linear map and its transpose (P B R; B = folded matrix)."""

    def __init__(self, corr):
        self.c = corr
        self.tr = corr.transfer
        self.A = corr.A_lin.numpy().astype(np.float64)
        self.scale = corr.in_scale * corr.inv_scale

    def apply(self, r):
        return self.c.correct(r)

    def apply_T(self, r):
        rc = self.tr.restrict(r)
        yc = (rc.reshape(1, -1) @ self.A.T).reshape(rc.shape) * self.scale
        return self.tr.prolong(yc)


no_map = NOmap(corrector)
# adjoint check of the corrector transpose: <C x, y> = <x, C^T y>
_x, _y = demean(rng.standard_normal((1, N, N))), demean(rng.standard_normal((1, N, N)))
adj_err = abs((no_map.apply(_x) * _y).sum() - (_x * no_map.apply_T(_y)).sum()) / abs((no_map.apply(_x) * _y).sum())


def make_G(kind, sv=None, m=1):
    """(G, G^T) for the error propagation map of m consecutive applications."""
    if kind == "no":
        def G(x):
            for _ in range(m):
                x = x - no_map.apply(apply_A(x))
            return x

        def GT(x):
            for _ in range(m):
                x = x - apply_AT(no_map.apply_T(x))
            return x
    else:
        def G(x):
            for _ in range(m):
                x = x - (sv.apply(apply_A(x)) if isinstance(sv, MGmap) else C_apply(sv, apply_A(x)))
            return x

        def GT(x):
            for _ in range(m):
                x = x - apply_AT(C_apply_T(sv, x))
            return x
    return G, GT


_kx = np.fft.fftfreq(N) * N
_KX, _KY = np.meshgrid(_kx, _kx[: N // 2 + 1], indexing="ij")
_sym = pde.ax * (2 - 2 * np.cos(2 * np.pi * _KX / N)) + pde.ay * (2 - 2 * np.cos(2 * np.pi * _KY / N))  # symmetric part of A (x h^2)
_sym[0, 0] = 1.0
_sq = np.sqrt(_sym)


def A_half(x, sign=+1):
    xh = np.fft.rfft2(x)
    xh = xh * (_sq if sign > 0 else 1.0 / _sq)
    xh[..., 0, 0] = 0.0
    return np.fft.irfft2(xh, s=(N, N))


def norm2(G, GT, iters=300, tol=1e-6):
    """||G||_2 on the mean-free subspace by power iteration on G^T G."""
    x = demean(rng.standard_normal((1, N, N)))
    x /= np.linalg.norm(x)
    lam = 0.0
    for k in range(iters):
        y = demean(GT(demean(G(x))))
        lam_new = float(np.sqrt(max((x * y).sum(), 0.0)))
        x = y / max(np.linalg.norm(y), 1e-300)
        if abs(lam_new - lam) < tol * max(lam_new, 1e-30) and k > 10:
            lam = lam_new
            break
        lam = lam_new
    return lam


def normA(G, GT, iters=300, tol=1e-6):
    """||A_s^{1/2} G A_s^{-1/2}||_2 on the mean-free subspace (energy norm of the
    symmetric part of A): power iteration on the similarity-transformed map."""
    Gs = lambda x: A_half(G(A_half(x, -1)), +1)
    GsT = lambda x: A_half(GT(A_half(x, +1)), -1)
    return norm2(Gs, GsT, iters, tol)


def spectral_radius(G, k=3):
    op = spla.LinearOperator((n, n), matvec=lambda v: demean(G(demean(v.reshape(1, N, N)))).ravel(), dtype=np.float64)
    try:
        w = spla.eigs(op, k=k, which="LM", tol=1e-6, maxiter=3000, return_eigenvectors=False)
        return float(np.max(np.abs(w)))
    except Exception as e:  # ARPACK failure: fall back to a plain power iteration
        x = demean(rng.standard_normal((1, N, N)))
        for _ in range(500):
            y = demean(G(x))
            r_ = np.linalg.norm(y) / np.linalg.norm(x)
            x = y / np.linalg.norm(y)
        return float(r_)


def fourier_modes(n_modes):
    """Real Fourier modes (kx, ky) -> arrays; all modes if n^2/2 <= n_modes."""
    kx = np.fft.fftfreq(N) * N
    ks = [(int(a), int(b)) for a in kx for b in kx if not (a == 0 and b == 0)]
    if len(ks) > n_modes:
        idx = rng.choice(len(ks), n_modes, replace=False)
        ks = [ks[i] for i in idx]
    X, Y = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    for (a, b) in ks:
        phi = np.cos(2 * np.pi * (a * X + b * Y) / N)[None]
        nrm = np.linalg.norm(phi)
        if nrm < 1e-12:
            phi = np.sin(2 * np.pi * (a * X + b * Y) / N)[None]
            nrm = np.linalg.norm(phi)
        yield (a, b), phi / nrm


def in_band(k):
    a, b = k
    return abs(a) <= corrector.n_cx // 2 - 1 and abs(b) <= corrector.n_cy // 2 - 1


out = {"args": vars(args), "h2": h2, "corrector": {"n_cx": corrector.n_cx, "n_cy": corrector.n_cy, "adjoint_check": adj_err}, "ops": {}}
t0 = time.time()
_out_path = f"{args.out_dir}/assumptions_{args.equation}_{N}.json"
if args.paths_only and os.path.exists(_out_path):
    out = json.load(open(_out_path))
    out["args"]["path_tol"] = args.path_tol
    solvers_norms = []
else:
    solvers_norms = None

# ------------------------------------------------------------ the corrector
G_no, GT_no = make_G("no")
band_c, out_c, leak = [], [], []
for k, phi in (fourier_modes(args.n_modes) if solvers_norms is None else []):
    y = G_no(phi)
    coef = float((y * phi).sum())
    (band_c if in_band(k) else out_c).append(abs(coef))
    leak.append(float(np.linalg.norm(y - coef * phi)))
if solvers_norms is None:
  out["ops"]["no"] = {
    "rho2": norm2(G_no, GT_no), "rhoA": normA(G_no, GT_no), "rho_spec": spectral_radius(G_no),
    "zero": float(np.linalg.norm(no_map.apply(zeros))),
    "band_max": float(max(band_c)), "band_mean": float(np.mean(band_c)),
    "outband_min": float(min(out_c)), "outband_max": float(max(out_c)), "leak_max": float(max(leak)),
    "n_band_modes": len(band_c), "n_out_modes": len(out_c)}
  print(f"[{args.equation} N={N}] corrector: rho2 {out['ops']['no']['rho2']:.6f} rho_spec {out['ops']['no']['rho_spec']:.6f} "
      f"band max {max(band_c):.2e} out-of-band [{min(out_c):.6f}, {max(out_c):.6f}] leak {max(leak):.2e} adjoint err {adj_err:.1e} ({time.time()-t0:.0f}s)", flush=True)

# ------------------------------------------------------------ classical solvers
X, Y = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
rand_vecs = [demean(rng.standard_normal((1, N, N))) for _ in range(6)]
for spec in (solvers if solvers_norms is None else []):
    t1 = time.time()
    sv = make_solver(pde, spec) if spec != "mg" else FastMultigrid(pde)
    m = max(1, int(round(costs_all[spec]["no"] / costs_all[spec][spec])))
    if spec == "mg":
        mgm = MGmap(sv, pde, pdeT)
        _x, _y = demean(rng.standard_normal((1, N, N))), demean(rng.standard_normal((1, N, N)))
        mg_adj = abs((mgm.apply(_x) * _y).sum() - (_x * mgm.apply_T(_y)).sum()) / abs((mgm.apply(_x) * _y).sum())
        print(f"  multigrid adjoint check: relative error {mg_adj:.1e}", flush=True)
        G, GT = make_G("mg", mgm, 1)
        Gm, GTm = make_G("mg", mgm, m)
    else:
        G, GT = make_G(spec, sv, 1)
        Gm, GTm = make_G(spec, sv, m)
    row = {"m": m, "zero": float(np.linalg.norm(C_apply(sv, zeros)))}
    row["rho2"] = norm2(G, GT)
    row["rhoA"] = normA(G, GT)
    row["rho2_macro"] = norm2(Gm, GTm) if m > 1 else row["rho2"]
    row["rhoA_macro"] = normA(Gm, GTm) if m > 1 else row["rhoA"]
    row["rho_spec"] = spectral_radius(G)
    # exact symbol for (damped) Jacobi: lambda(k) = 1 - w (1 - mu(k)),
    # mu = (ax cos tx + ay cos ty) / (ax + ay); also the Nyquist-free value
    if spec.startswith("jacobi"):
        w = sv.weight
        kx = np.fft.fftfreq(N) * N
        KX, KY = np.meshgrid(kx, kx, indexing="ij")
        mu = (pde.ax * np.cos(2 * np.pi * KX / N) + pde.ay * np.cos(2 * np.pi * KY / N)) / (pde.ax + pde.ay)
        lam = np.abs(1 - w * (1 - mu))
        lam[0, 0] = 0.0
        nyq = (np.abs(KX) == N // 2) | (np.abs(KY) == N // 2)
        row["rho_symbol"] = float(lam.max())
        row["rho_symbol_nonyquist"] = float(lam[~nyq].max())
        row["sigma_min_symbol"] = float(lam[~((KX == 0) & (KY == 0))].min())
    # smallest observed contraction
    if isinstance(sv, (FastGaussSeidel, FastSOR, FastSSOR)):
        d0 = np.zeros((1, N, N)); d0[0, 0, 0] = 1.0
        row["sigma_min"] = float(np.linalg.norm(G(d0)) / np.linalg.norm(d0))
        row["sigma_min_note"] = "unit error at the first grid point"
    else:
        vals = [float(np.linalg.norm(G(phi))) for k, phi in fourier_modes(min(args.n_modes, 2000))]
        row["sigma_min"] = float(min(vals))
        row["sigma_min_note"] = "min over Fourier modes"
        row["fourier_max"] = float(max(vals))
    # commutator with the corrector map (Proposition 4.3)
    row["comm"] = float(max(np.linalg.norm(G(G_no(x)) - G_no(G(x))) / np.linalg.norm(x) for x in rand_vecs))
    out["ops"][spec] = row
    print(f"  {spec:12s} m={m:2d} rho2 {row['rho2']:.6f} rhoA {row['rhoA']:.6f} "
          f"macro rho2 {row['rho2_macro']:.6f} rhoA {row['rhoA_macro']:.6f} rho_spec {row['rho_spec']:.6f} "
          f"sigma_min {row['sigma_min']:.2e} zero {row['zero']:.1e} comm {row['comm']:.2e}"
          + (f" symbol {row['rho_symbol']:.6f}/{row['rho_symbol_nonyquist']:.6f}" if "rho_symbol" in row else "")
          + f" ({time.time()-t1:.0f}s)", flush=True)
    json.dump(out, open(f"{args.out_dir}/assumptions_{args.equation}_{N}.json", "w"), indent=1)

# ------------------------------------------------ along the oracle path (per pairing and full ensemble)
f = GRF2D(N, rng=np.random.default_rng(args.seed)).sample(args.n_inst)
u_true = pde.solve_direct(f)


def oracle_path(env, fi, ui):
    """Cost-aware greedy oracle to h^2 on one instance; returns the quantities of
    Prop. 4.2 / Thm 5.1 along the path."""
    u = np.zeros_like(fi)
    un = np.linalg.norm(ui)
    e = demean(ui - u)          # the constant mode is the null space of the periodic operator (as in hybrid.run_untimed)
    g = [float(np.linalg.norm(e) ** 2)]
    steps, ctil = [], []
    K = env.K
    for t in range(400):
        rel = np.linalg.norm(e) / un
        if rel <= args.path_tol:
            break
        r = pde.residual(u, fi)
        cand, errs = [], []
        for j in range(K):
            uj = u.copy()
            for _ in range(env.m[j]):
                rj = pde.residual(uj, fi)
                uj = uj + (env.corrector.correct(rj) if j == env.no_index else C_apply(env.solvers[j], rj))
            cand.append(uj)
            errs.append(float(np.linalg.norm(demean(ui - uj))))
        ctil.append([x ** 2 for x in errs])
        j = int(np.argmin(env.macro_score(np.linalg.norm(e), np.array(errs))))
        steps.append(j)
        u = cand[j]
        e = demean(ui - u)
        g.append(float(np.linalg.norm(e) ** 2))
    T = len(steps)
    # empirical supermodularity ratio: S = prefix t, S' = suffix
    ratios = []
    for t in range(T):
        lhs = g[t] - g[T]
        rhs = sum(g[t] - ctil[t][steps[i]] for i in range(t, T))
        if rhs > 0:
            ratios.append(lhs / rhs)
    ebar = max(max(c) / gg for c, gg in zip(ctil, g[:-1]))
    emin = min(max(c) / gg for c, gg in zip(ctil, g[:-1]))
    return {"T": T, "steps": steps, "alpha_hat_max": max(ratios) if ratios else None,
            "alpha_hat_median": float(np.median(ratios)) if ratios else None, "Ebar_rel": ebar, "Emin_rel": emin}


_pkey = "paths" if args.path_tol < h2 else "paths_h2"
out[_pkey] = {}
solvers = [s for s in solvers if s in out["ops"]]   # operations whose norms were computed
groups = [[s] for s in solvers]
if "jacobi" in solvers and "jacobi_0.67" in solvers:
    groups.append(["jacobi", "jacobi_0.67"])
if N == 128:
    groups.append([s for s in solvers if s != "mg"])
for grp in groups:
    key = "+".join(grp)
    if key in costs_all:
        costs = costs_all[key]
    else:
        costs = {s: costs_all[s][s] for s in grp}
        costs["no"] = float(np.mean([costs_all[s]["no"] for s in grp]))
        costs["_residual"] = float(np.mean([costs_all[s]["_residual"] for s in grp]))
    env = Env(pde, grp, corrector, costs=costs)
    rows = [oracle_path(env, f[i:i + 1], u_true[i:i + 1]) for i in range(args.n_inst)]
    rho = {}
    for j, o in enumerate(env.ops):
        if o == "no":
            rho[o] = out["ops"]["no"]["rho2"]
        else:
            rho[o] = out["ops"][o]["rho2_macro"] if env.m[j] == out["ops"][o]["m"] else out["ops"][o]["rho2"] ** env.m[j]
    alphas = []
    for r_ in rows:
        s2 = sum(rho[env.ops[j]] ** 2 for j in r_["steps"])
        r_["sum_rho2"] = s2
        r_["alpha_bound"] = max(4.0 / (r_["T"] - s2), 1.0) if r_["T"] - s2 > 0 else float("inf")
        alphas.append(r_["alpha_bound"])
    out[_pkey][key] = {"ops": env.ops, "m": env.m, "rho_macro": rho, "rows": rows}
    print(f"  path {key:40s} T med {np.median([r_['T'] for r_ in rows]):.0f} alpha(O) med {np.median(alphas):.2f} "
          f"alpha_hat max {max(r_['alpha_hat_max'] or 0 for r_ in rows):.3f} med {np.median([r_['alpha_hat_median'] or 0 for r_ in rows]):.3f} "
          f"Ebar {max(r_['Ebar_rel'] for r_ in rows):.3f} Emin {min(r_['Emin_rel'] for r_ in rows):.2e}", flush=True)
    json.dump(out, open(f"{args.out_dir}/assumptions_{args.equation}_{N}.json", "w"), indent=1)
print("saved", f"{args.out_dir}/assumptions_{args.equation}_{N}.json", f"({time.time()-t0:.0f}s)")
