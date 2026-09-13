"""Strong classical baselines on the same fast stencil, with the same
time-to-tolerance accounting as bench.py:

  fft         direct FFT solve (the reference solver of the constant-coefficient
              periodic problem; one call)
  mg          geometric multigrid V(2,2) with GS smoothing (stationary; also
              usable as an ensemble member through fast_pde.make_solver)
  cg          conjugate gradients (Poisson only)
  pcg_ssor    CG preconditioned by symmetric GS
  pcg_mg      CG preconditioned by a symmetric V-cycle (forward/backward GS)
  bicgstab    BiCGSTAB (ConvDiff), bicgstab_mg with the V-cycle preconditioner
  gmres       restarted GMRES(20), gmres_mg with the V-cycle preconditioner

Krylov methods are scipy.sparse.linalg implementations driven by the O(N^2)
stencil matvec and the numpy preconditioners; every iteration's iterate is
recorded once (untimed) to find the first iteration below each tolerance, and
the timed run then executes exactly that many iterations.

  python bench_baselines.py --equation Poisson --N 128 --n_test 64
"""

import time

import numpy as np
import scipy.sparse.linalg as spla

from fast_pde import (FastStencilPDE, FastGaussSeidel, FastSSOR, FastMultigrid, FFTDirect,
                      demean, l2, restrict_fw, prolong_bilinear)


class FastGaussSeidelBackward:
    """u <- u + (D + U)^{-1} (f - A u): backward lexicographic sweep."""

    def __init__(self, pde):
        import scipy.sparse as sp
        self.pde = pde
        A = pde.sparse_A()
        self.lu = spla.splu(sp.triu(A).tocsc(), permc_spec="NATURAL")

    def step(self, u, f, r=None):
        if r is None:
            r = self.pde.residual(u, f)
        B = r.shape[0] if r.ndim == 3 else 1
        N = self.pde.N
        du = self.lu.solve(np.ascontiguousarray(r.reshape(B, N * N).T))
        return u + du.T.reshape(r.shape)


class SymmetricMultigrid(FastMultigrid):
    """V-cycle with forward GS pre-smoothing and backward GS post-smoothing:
    a symmetric positive definite preconditioner for CG when A is SPD."""

    def __init__(self, pde, nu1=2, nu2=2, n_coarsest=32):
        super().__init__(pde, nu1=nu1, nu2=nu2, n_coarsest=n_coarsest, smoother="gs")
        self.back = [FastGaussSeidelBackward(lp) for (lp, _) in self.levels]
        self.name = "mg_sym"

    def _vcycle(self, lvl, r):
        if lvl == len(self.levels):
            return self.coarse.solve_direct(r)
        lp, sm = self.levels[lvl]
        e = np.zeros_like(r)
        for _ in range(self.nu1):
            e = sm.step(e, r)
        rc = restrict_fw(lp.residual(e, r))
        e = e + prolong_bilinear(self._vcycle(lvl + 1, rc))
        for _ in range(self.nu2):
            e = self.back[lvl].step(e, r)
        return e


KRYLOV = {"cg", "pcg_ssor", "pcg_mg", "bicgstab", "bicgstab_mg", "gmres", "gmres_mg"}


class SparseLUDirect:
    """Sparse direct solve (SuperLU) of the periodic system with one pinned unknown; the
    factorisation is computed once and its time recorded, each solve is timed separately."""

    def __init__(self, pde):
        import scipy.sparse as sp
        self.pde = pde
        t0 = time.perf_counter_ns()
        A = pde.sparse_A().tocsr()
        self.lu = spla.splu(A[1:, 1:].tocsc())
        self.factor_s = (time.perf_counter_ns() - t0) * 1e-9

    def solve(self, f):
        N = self.pde.N
        ff = f.reshape(-1, N * N)
        u = np.zeros_like(ff)
        u[:, 1:] = self.lu.solve(np.ascontiguousarray(ff[:, 1:].T)).T
        u = u - u.mean(axis=1, keepdims=True)
        return u.reshape(f.shape)


def make_krylov(pde, method):
    """Returns (scipy solver function, preconditioner LinearOperator or None, kwargs)."""
    N = pde.N
    n = N * N

    def matvec(x):
        return pde.apply_A(x.reshape(N, N)).ravel()

    A = spla.LinearOperator((n, n), matvec=matvec, dtype=np.float64)
    M = None
    if method.endswith("_ssor"):
        ss = FastSSOR(pde)
        zero = np.zeros((1, N, N))
        M = spla.LinearOperator((n, n), matvec=lambda x: ss.step(zero, None, x.reshape(1, N, N)).ravel(), dtype=np.float64)
    elif method.endswith("_mg"):
        mg = SymmetricMultigrid(pde) if method.startswith("pcg") else FastMultigrid(pde)
        M = spla.LinearOperator((n, n), matvec=lambda x: mg._vcycle(0, x.reshape(1, N, N)).ravel(), dtype=np.float64)
    if method.startswith("cg") or method.startswith("pcg"):
        fn = spla.cg
        kw = {}
    elif method.startswith("bicgstab"):
        fn = spla.bicgstab
        kw = {}
    elif method.startswith("gmres"):
        fn = spla.gmres
        kw = {"restart": 20, "callback_type": "x"}
    else:
        raise ValueError(method)
    return A, M, fn, kw


class _Done(Exception):
    pass


def run_krylov_untimed(pde, f, u_truth, method, tols, max_iter=5000, err_stop=1e-9):
    """Per-iteration true relative error of the Krylov iterates (untimed).
    For GMRES(20) the callback fires once per restart cycle, so its iteration
    unit is one cycle of 20 inner iterations (time_krylov uses the same unit)."""
    A, M, fn, kw = make_krylov(pde, method)
    N = pde.N
    un = max(float(l2(demean(u_truth))[0]), 1e-300)
    errs = [float(l2(demean(-u_truth))[0]) / un]
    stop_at = min(min(tols), err_stop)

    def cb(xk):
        e = float(l2(demean(xk.reshape(1, N, N) - u_truth))[0]) / un
        errs.append(e)
        if e <= stop_at or not np.isfinite(e):
            raise _Done
    try:
        fn(A, f.ravel(), M=M, rtol=1e-300, atol=0.0, maxiter=max_iter, callback=cb, **kw)
    except _Done:
        pass
    errs = np.asarray(errs)
    hit = {}
    for tol in tols:
        idx = np.flatnonzero(errs <= tol)
        hit[tol] = int(idx[0]) if len(idx) else None
    return errs, hit


def time_krylov(pde, f, method, n_iter, reps=1):
    """Wall-clock time of exactly n_iter iterations (no error diagnostics)."""
    A, M, fn, kw = make_krylov(pde, method)
    if n_iter is None or n_iter <= 0:
        return 0.0 if n_iter == 0 else np.inf
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn(A, f.ravel(), M=M, rtol=1e-300, atol=0.0, maxiter=n_iter, **kw)
        ts.append((time.perf_counter_ns() - t0) * 1e-9)
    return float(np.median(ts))
