"""Fast periodic constant-coefficient PDE utilities for wall-clock benchmarking.

The dense-matrix implementations in pde.py / numerical_solver.py are convenient
for training on small grids but make every classical iteration an O(N^4) dense
matvec, which would unfairly slow the classical baselines in any wall-clock
comparison. This module provides:

  * FastStencilPDE  - O(N^2) stencil application of the same 5-point
                      discretizations built by pde.py (Poisson / ConvDiff,
                      periodic BCs, constant coefficients), an FFT direct solver
                      (exact solution of the discrete system, used as ground
                      truth), and cheap residuals.
  * Jacobi / GS / SOR / SSOR one-step iterations matching numerical_solver.py
                      (validated against the dense implementations at N=31 by
                      validate_fast_pde.py).
  * GRF2D           - numpy port of data_generation.GaussianRandomFieldHierarchical
                      (same spectral law, same sampling of alpha/beta/gamma),
                      extended to even grid sizes.

Everything is batched over the leading axis and works on (B, N, N) float64
arrays.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ---------------------------------------------------------------------------
# Compiled kernels (stencil.c -> libstencil.so); numpy/scipy fallback if absent
# ---------------------------------------------------------------------------
import ctypes as _ct
import os as _os
_LIB = None
try:
    _p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "libstencil.so")
    if _os.path.exists(_p) and _os.environ.get("STENCIL_NUMPY", "0") != "1":
        _LIB = _ct.CDLL(_p)
        _dp = _ct.POINTER(_ct.c_double)
        _LIB.apply_A.argtypes = [_dp, _dp, _ct.c_int, _ct.c_int, _ct.c_double, _ct.c_double, _ct.c_double, _ct.c_double]
        _LIB.residual.argtypes = [_dp, _dp, _dp, _ct.c_int, _ct.c_int, _ct.c_double, _ct.c_double, _ct.c_double, _ct.c_double]
        _LIB.residual_norm2.argtypes = [_dp, _dp, _dp, _ct.c_int, _ct.c_int, _ct.c_double, _ct.c_double, _ct.c_double, _ct.c_double]
        _LIB.residual_norm2.restype = _ct.c_double
        _LIB.jacobi.argtypes = [_dp, _dp, _dp, _ct.c_int, _ct.c_int, _ct.c_double, _ct.c_double, _ct.c_double, _ct.c_double, _ct.c_double]
        _LIB.sor_sweep.argtypes = [_dp, _dp, _ct.c_int, _ct.c_int, _ct.c_double, _ct.c_double, _ct.c_double, _ct.c_double, _ct.c_double, _ct.c_int]
        _LIB.ssor_sweep.argtypes = [_dp, _dp, _ct.c_int, _ct.c_int, _ct.c_double, _ct.c_double, _ct.c_double, _ct.c_double, _ct.c_double]
        _LIB.apply_A_var.argtypes = [_dp, _dp, _ct.c_int, _ct.c_int, _dp, _dp, _dp, _dp, _dp]
        _LIB.residual_var.argtypes = [_dp, _dp, _dp, _ct.c_int, _ct.c_int, _dp, _dp, _dp, _dp, _dp]
        _LIB.residual_norm2_var.argtypes = [_dp, _dp, _dp, _ct.c_int, _ct.c_int, _dp, _dp, _dp, _dp, _dp]
        _LIB.residual_norm2_var.restype = _ct.c_double
        _LIB.jacobi_var.argtypes = [_dp, _dp, _dp, _ct.c_int, _ct.c_int, _dp, _dp, _dp, _dp, _dp, _ct.c_double]
        _LIB.sor_sweep_var.argtypes = [_dp, _dp, _ct.c_int, _ct.c_int, _dp, _dp, _dp, _dp, _dp, _ct.c_double, _ct.c_int]
        _LIB.ssor_sweep_var.argtypes = [_dp, _dp, _ct.c_int, _ct.c_int, _dp, _dp, _dp, _dp, _dp, _ct.c_double]
        _LIB.restrict_fw.argtypes = [_dp, _dp, _ct.c_int, _ct.c_int]
        _LIB.prolong_bilinear.argtypes = [_dp, _dp, _ct.c_int, _ct.c_int]
except OSError:
    _LIB = None
COMPILED = _LIB is not None


def _ptr(a):
    return a.ctypes.data_as(_ct.POINTER(_ct.c_double))


def _c64(a):
    a = np.asarray(a, dtype=np.float64)
    return a if a.flags["C_CONTIGUOUS"] else np.ascontiguousarray(a)


class FastStencilPDE:
    """-a * Lap(u) + b . grad(u) = f on [0,1]^2, periodic, uniform N x N grid.

    Matches the discretization of pde.py: second-order central differences for
    both diffusion and advection, grid x = linspace(0, 1, N+1)[:-1], h = 1/N.
    """

    def __init__(self, N, equation="Poisson", a=1.0, b_vec=(20.0, 20.0), aniso_eps=0.01, coef=None, coef_seed=0, contrast=10.0):
        assert equation in ("Poisson", "ConvDiff", "AnisoDiff", "VarCoeff")
        self.N = N
        self.equation = equation
        self.a = a
        # anisotropic diffusion: -eps u_xx - u_yy = f (diffusion tensor diag(eps, 1))
        self.ax, self.ay = (a * aniso_eps, a) if equation == "AnisoDiff" else (a, a)
        self.aniso_eps = aniso_eps
        self.b1, self.b2 = (0.0, 0.0) if equation != "ConvDiff" else b_vec
        self.h = 1.0 / N
        self._lu_cache = {}
        self._direct_lu = None
        if equation == "VarCoeff":
            # -div(a grad u) = f with a smooth, fixed random coefficient field a(x) = exp(kappa z(x)),
            # z a band-limited GRF (|k| <= 4) scaled to [-1, 1], kappa = ln(contrast)/2 (max a / min a = contrast)
            self.coef = coef if coef is not None else self.random_coefficient(N, coef_seed, contrast)
            self.coef_seed, self.contrast = coef_seed, contrast
            self.ax = self.ay = None
            self._build_var_stencil()
            self._symbol = None
        else:
            self.coef = None
            self.diag = 2.0 * (self.ax + self.ay) / self.h ** 2
            self._symbol = self._build_symbol()

    _coef_norm = {}

    @staticmethod
    def random_coefficient(N, seed=0, contrast=10.0, kmax=4):
        """Smooth random coefficient field on the periodic grid (deterministic in seed and N,
        independent of the grid size for a given seed: the same Fourier modes are sampled on every grid)."""
        rng = np.random.default_rng(1000 + seed)
        z = np.zeros((N, N))
        x = np.arange(N) / N
        X, Y = np.meshgrid(x, x, indexing="ij")
        for kx in range(-kmax, kmax + 1):
            for ky in range(-kmax, kmax + 1):
                if kx == 0 and ky == 0:
                    continue
                amp = rng.standard_normal(2) / (1.0 + kx ** 2 + ky ** 2)
                z += amp[0] * np.cos(2 * np.pi * (kx * X + ky * Y)) + amp[1] * np.sin(2 * np.pi * (kx * X + ky * Y))
        # normalise by the maximum of the same trigonometric polynomial on a fixed fine reference
        # grid (1024^2), so that the coefficient is one grid-independent function a(x) whose samples
        # on every grid agree (the earlier per-grid normalisation changed a by ~1e-4 between grids)
        key = (seed, kmax)
        if key not in FastStencilPDE._coef_norm:
            rng2 = np.random.default_rng(1000 + seed)
            M = 1024
            xr = np.arange(M) / M
            XR, YR = np.meshgrid(xr, xr, indexing="ij")
            zr = np.zeros((M, M))
            for kx in range(-kmax, kmax + 1):
                for ky in range(-kmax, kmax + 1):
                    if kx == 0 and ky == 0:
                        continue
                    amp = rng2.standard_normal(2) / (1.0 + kx ** 2 + ky ** 2)
                    zr += amp[0] * np.cos(2 * np.pi * (kx * XR + ky * YR)) + amp[1] * np.sin(2 * np.pi * (kx * XR + ky * YR))
            FastStencilPDE._coef_norm[key] = float(np.abs(zr).max())
        z = z / FastStencilPDE._coef_norm[key]
        return np.exp(0.5 * np.log(contrast) * z)

    def _build_var_stencil(self):
        """Face-averaged (arithmetic mean) coefficients of the conservative 5-point discretisation."""
        a, h2 = self.coef, self.h ** 2
        aw = 0.5 * (a + np.roll(a, 1, axis=0)); ae = 0.5 * (a + np.roll(a, -1, axis=0))
        as_ = 0.5 * (a + np.roll(a, 1, axis=1)); an = 0.5 * (a + np.roll(a, -1, axis=1))
        self.cw, self.ce, self.cs, self.cn = (-aw / h2, -ae / h2, -as_ / h2, -an / h2)
        self.diag = (aw + ae + as_ + an) / h2
        self._stencil = [np.ascontiguousarray(x) for x in (self.cw, self.ce, self.cs, self.cn, self.diag)]

    # -- operator ------------------------------------------------------------
    def apply_A(self, u):
        """u: (..., N, N). First grid axis is i (x), second is j (y)."""
        if self.equation == "VarCoeff":
            if COMPILED and u.dtype == np.float64 and u.ndim in (2, 3):
                uc = _c64(u); out = np.empty_like(uc)
                _LIB.apply_A_var(_ptr(uc), _ptr(out), 1 if uc.ndim == 2 else uc.shape[0], self.N, *[_ptr(s) for s in self._stencil])
                return out
            return (self.diag * u + self.cw * np.roll(u, 1, axis=-2) + self.ce * np.roll(u, -1, axis=-2)
                    + self.cs * np.roll(u, 1, axis=-1) + self.cn * np.roll(u, -1, axis=-1))
        if COMPILED and u.dtype == np.float64 and u.ndim in (2, 3):
            uc = _c64(u)
            out = np.empty_like(uc)
            B = 1 if uc.ndim == 2 else uc.shape[0]
            _LIB.apply_A(_ptr(uc), _ptr(out), B, self.N, self.ax, self.ay, self.b1, self.b2)
            return out
        h = self.h
        up_i = np.roll(u, -1, axis=-2)   # u_{i+1,j}
        dn_i = np.roll(u, 1, axis=-2)    # u_{i-1,j}
        up_j = np.roll(u, -1, axis=-1)   # u_{i,j+1}
        dn_j = np.roll(u, 1, axis=-1)    # u_{i,j-1}
        if self.ax == self.ay:
            out = self.ax * (4.0 * u - up_i - dn_i - up_j - dn_j) / h ** 2
        else:
            out = (self.ax * (2.0 * u - up_i - dn_i) + self.ay * (2.0 * u - up_j - dn_j)) / h ** 2
        if self.b1 or self.b2:
            out = out + self.b1 * (up_i - dn_i) / (2 * h) + self.b2 * (up_j - dn_j) / (2 * h)
        return out

    def residual(self, u, f):
        if self.equation == "VarCoeff":
            if COMPILED and u.dtype == np.float64 and u.ndim in (2, 3) and f.shape == u.shape:
                uc, fc = _c64(u), _c64(f); r = np.empty_like(uc)
                _LIB.residual_var(_ptr(uc), _ptr(fc), _ptr(r), 1 if uc.ndim == 2 else uc.shape[0], self.N, *[_ptr(s) for s in self._stencil])
                return r
            return f - self.apply_A(u)
        if COMPILED and u.dtype == np.float64 and u.ndim in (2, 3) and f.shape == u.shape:
            uc, fc = _c64(u), _c64(f)
            r = np.empty_like(uc)
            B = 1 if uc.ndim == 2 else uc.shape[0]
            _LIB.residual(_ptr(uc), _ptr(fc), _ptr(r), B, self.N, self.ax, self.ay, self.b1, self.b2)
            return r
        return f - self.apply_A(u)

    def residual_norm(self, u, f):
        """Residual r = f - A u and its L2 norm (shape (B,)), computed in one pass by the
        compiled kernel for a single instance (the stopping test every method pays)."""
        single = (u.ndim == 2) or (u.ndim == 3 and u.shape[0] == 1)
        if COMPILED and single and u.dtype == np.float64 and f.shape == u.shape:
            uc, fc = _c64(u), _c64(f)
            r = np.empty_like(uc)
            if self.equation == "VarCoeff":
                acc = _LIB.residual_norm2_var(_ptr(uc), _ptr(fc), _ptr(r), 1, self.N, *[_ptr(s) for s in self._stencil])
            else:
                acc = _LIB.residual_norm2(_ptr(uc), _ptr(fc), _ptr(r), 1, self.N, self.ax, self.ay, self.b1, self.b2)
            return r, np.array([acc ** 0.5]) if u.ndim == 3 else np.array(acc ** 0.5)
        r = self.residual(u, f)
        return r, l2(r)

    # -- FFT direct solve (exact solution of the discrete system) -------------
    def _build_symbol(self):
        N, h = self.N, self.h
        k = np.fft.fftfreq(N) * N          # integer wavenumbers
        theta_x = 2 * np.pi * k[:, None] / N
        theta_y = 2 * np.pi * k[None, :] / N
        sym = (self.ax * (2 - 2 * np.cos(theta_x)) + self.ay * (2 - 2 * np.cos(theta_y))) / h ** 2
        sym = sym.astype(np.complex128)
        if self.b1 or self.b2:
            sym = sym + 1j * (self.b1 * np.sin(theta_x) + self.b2 * np.sin(theta_y)) / h
        return sym

    def solve_direct(self, f):
        """Exact solution of A u = f (f mean-free; the returned solution is mean-free): FFT
        diagonalization for constant coefficients, a cached sparse LU with one pinned unknown otherwise."""
        if self.equation == "VarCoeff":
            if self._direct_lu is None:
                A = self.sparse_A().tocsr()
                self._direct_lu = spla.splu(A[1:, 1:].tocsc())
            ff = np.asarray(f).reshape(-1, self.N * self.N)
            u = np.zeros_like(ff)
            u[:, 1:] = self._direct_lu.solve(np.ascontiguousarray(ff[:, 1:].T)).T
            u = u - u.mean(axis=1, keepdims=True)
            return u.reshape(np.asarray(f).shape)
        fhat = np.fft.fft2(f, axes=(-2, -1))
        sym = self._symbol.copy()
        sym[0, 0] = 1.0
        uhat = fhat / sym
        uhat[..., 0, 0] = 0.0
        return np.real(np.fft.ifft2(uhat, axes=(-2, -1)))

    # -- sparse matrix (for GS / SOR triangular solves) ------------------------
    def sparse_A(self):
        N, h = self.N, self.h
        ax, ay = self.ax, self.ay
        idx = np.arange(N * N).reshape(N, N)
        rows, cols, vals = [], [], []

        def add(nbr_idx, val):
            rows.append(idx.ravel())
            cols.append(nbr_idx.ravel())
            vals.append(np.full(N * N, val) if np.isscalar(val) else np.asarray(val).ravel())

        if self.equation == "VarCoeff":
            add(idx, self.diag)
            add(np.roll(idx, 1, axis=0), self.cw); add(np.roll(idx, -1, axis=0), self.ce)
            add(np.roll(idx, 1, axis=1), self.cs); add(np.roll(idx, -1, axis=1), self.cn)
            return sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N * N, N * N)).tocsr()
        add(idx, self.diag)
        add(np.roll(idx, 1, axis=0), -ax / h ** 2 - self.b1 / (2 * h))    # (i-1, j)
        add(np.roll(idx, -1, axis=0), -ax / h ** 2 + self.b1 / (2 * h))   # (i+1, j)
        add(np.roll(idx, 1, axis=1), -ay / h ** 2 - self.b2 / (2 * h))    # (i, j-1)
        add(np.roll(idx, -1, axis=1), -ay / h ** 2 + self.b2 / (2 * h))   # (i, j+1)
        A = sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(N * N, N * N),
        ).tocsr()
        return A

    def _get_lower_solver(self, kind, omega=1.0):
        """Cached splu factorization of the (lower triangular) sweep matrix."""
        key = (kind, omega)
        if key not in self._lu_cache:
            A = self.sparse_A()
            if kind == "gs":
                M = sp.tril(A).tocsc()
            elif kind == "sor":
                D = sp.diags(A.diagonal())
                L = sp.tril(A, k=-1)
                M = ((1.0 / omega) * D + L).tocsc()
            else:
                raise ValueError(kind)
            self._lu_cache[key] = spla.splu(M, permc_spec="NATURAL")
        return self._lu_cache[key]


# ---------------------------------------------------------------------------
# One-step classical iterations (match numerical_solver.py conventions)
# ---------------------------------------------------------------------------

class FastJacobi:
    """u <- u + w * D^{-1} (f - A u). D is a constant scalar here."""

    def __init__(self, pde: FastStencilPDE, weight=1.0):
        self.pde = pde
        self.weight = weight
        self.name = "jacobi" if weight == 1.0 else f"jacobi_{weight:g}"

    def step(self, u, f, r=None):
        if r is None:
            if COMPILED and u.dtype == np.float64 and u.ndim in (2, 3) and f is not None:
                uc, fc = _c64(u), _c64(f)
                out = np.empty_like(uc)
                B = 1 if uc.ndim == 2 else uc.shape[0]
                if self.pde.equation == "VarCoeff":
                    _LIB.jacobi_var(_ptr(uc), _ptr(fc), _ptr(out), B, self.pde.N, *[_ptr(s) for s in self.pde._stencil], self.weight)
                else:
                    _LIB.jacobi(_ptr(uc), _ptr(fc), _ptr(out), B, self.pde.N, self.pde.ax, self.pde.ay, self.pde.b1, self.pde.b2, self.weight)
                return out
            r = self.pde.residual(u, f)
        return u + (self.weight / self.pde.diag) * r


def _c_sweep(pde, u, f, r, omega, symmetric=False, forward=1):
    """In-place SOR / symmetric-SOR sweep in C on a copy of u (needs f; if only r is
    available, f is reconstructed as A u + r)."""
    uc = _c64(u).copy()
    if f is None:
        f = pde.apply_A(u) + r
    fc = _c64(f)
    B = 1 if uc.ndim == 2 else uc.shape[0]
    if pde.equation == "VarCoeff":
        st = [_ptr(s) for s in pde._stencil]
        if symmetric:
            _LIB.ssor_sweep_var(_ptr(uc), _ptr(fc), B, pde.N, *st, omega)
        else:
            _LIB.sor_sweep_var(_ptr(uc), _ptr(fc), B, pde.N, *st, omega, forward)
        return uc
    if symmetric:
        _LIB.ssor_sweep(_ptr(uc), _ptr(fc), B, pde.N, pde.ax, pde.ay, pde.b1, pde.b2, omega)
    else:
        _LIB.sor_sweep(_ptr(uc), _ptr(fc), B, pde.N, pde.ax, pde.ay, pde.b1, pde.b2, omega, forward)
    return uc


class FastGaussSeidel:
    """u <- u + (D + L)^{-1} (f - A u), lexicographic ordering as in pde.py."""

    def __init__(self, pde: FastStencilPDE):
        self.pde = pde
        self.lu = None if COMPILED else pde._get_lower_solver("gs")
        self.name = "gs"

    def step(self, u, f, r=None):
        if COMPILED and u.dtype == np.float64 and u.ndim in (2, 3):
            return _c_sweep(self.pde, u, f, r, 1.0)
        if r is None:
            r = self.pde.residual(u, f)
        B = r.shape[0] if r.ndim == 3 else 1
        N = self.pde.N
        rr = r.reshape(B, N * N).T  # (N^2, B) for a single multi-rhs solve
        du = self.lu.solve(np.ascontiguousarray(rr))
        return u + du.T.reshape(r.shape)


class FastSOR:
    """u <- u + omega * (D + omega L)^{-1} (f - A u)."""

    def __init__(self, pde: FastStencilPDE, omega=1.5):
        self.pde = pde
        self.omega = omega
        self.lu = None if COMPILED else pde._get_lower_solver("sor", omega)
        self.name = f"sor_{omega:g}"

    def step(self, u, f, r=None):
        if COMPILED and u.dtype == np.float64 and u.ndim in (2, 3):
            return _c_sweep(self.pde, u, f, r, self.omega)
        if r is None:
            r = self.pde.residual(u, f)
        B = r.shape[0] if r.ndim == 3 else 1
        N = self.pde.N
        rr = r.reshape(B, N * N).T
        du = self.lu.solve(np.ascontiguousarray(rr))
        return u + du.T.reshape(r.shape)


class FastSSOR:
    """Symmetric SOR: u <- u + M^{-1} (f - A u) with
    M = (omega/(2-omega)) (D/omega + L) D^{-1} (D/omega + U); omega=1 is
    symmetric Gauss-Seidel (SymGS)."""

    def __init__(self, pde: FastStencilPDE, omega=1.0):
        self.pde = pde
        self.omega = omega
        if not COMPILED:
            A = pde.sparse_A()
            D = A.diagonal()
            Dm = sp.diags(D / omega)
            self._diag = D
            self._lu_low = spla.splu((Dm + sp.tril(A, k=-1)).tocsc(), permc_spec="NATURAL")
            self._lu_up = spla.splu((Dm + sp.triu(A, k=1)).tocsc(), permc_spec="NATURAL")
        self._scale = (2.0 - omega) / omega
        self.name = "ssor" if omega == 1.0 else f"ssor_{omega:g}"

    def step(self, u, f, r=None):
        if COMPILED and u.dtype == np.float64 and u.ndim in (2, 3):
            return _c_sweep(self.pde, u, f, r, self.omega, symmetric=True)
        if r is None:
            r = self.pde.residual(u, f)
        B = r.shape[0] if r.ndim == 3 else 1
        N = self.pde.N
        rr = np.ascontiguousarray(r.reshape(B, N * N).T)
        y = self._lu_low.solve(rr)
        y = self._diag[:, None] * y
        y = self._lu_up.solve(np.ascontiguousarray(y))
        return u + self._scale * y.T.reshape(r.shape)


def make_solver(pde, spec):
    """spec strings as used by train_router.py: jacobi, jacobi_0.67, gs,
    sor_1.5, ssor (SymGS)."""
    parts = spec.split("_")
    if parts[0] == "jacobi":
        return FastJacobi(pde, float(parts[1]) if len(parts) > 1 else 1.0)
    if parts[0] == "gs":
        return FastGaussSeidel(pde)
    if parts[0] == "sor":
        return FastSOR(pde, float(parts[1]) if len(parts) > 1 else 1.0)
    if parts[0] == "ssor":
        return FastSSOR(pde, float(parts[1]) if len(parts) > 1 else 1.0)
    raise ValueError(f"unknown solver spec {spec}")


SOLVER_NAMES = {"jacobi": "Jacobi", "jacobi_0.67": "Jacobi (0.67)", "gs": "GS",
                "ssor": "SymGS", "sor_1.5": "SOR (1.5)", "mg": "Multigrid"}


# ---------------------------------------------------------------------------
# Hierarchical GRF sampling (numpy port of data_generation.py)
# ---------------------------------------------------------------------------

class GRF2D:
    """Zero-mean hierarchical Gaussian random field on the periodic unit square.

    Covariance operator alpha * (-Lap + beta I)^(-gamma) with
      alpha ~ LogUniform(alpha_min, alpha_max)
      beta  ~ LogUniform(beta_min, beta_max)
      gamma ~ Uniform(gamma_list)
    sampled independently per field, exactly as in
    data_generation.GaussianRandomFieldHierarchical (dim=2). For even N the
    Nyquist modes are excluded so the field is a real trigonometric polynomial
    with |k| < N/2 in both directions.
    """

    def __init__(self, N, alpha=(0.01, 100.0), beta=(0.1, 1000.0),
                 gamma_list=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0), rng=None):
        self.N = N
        self.k_max = (N - 1) // 2
        self.alpha_min, self.alpha_max = alpha
        self.beta_min, self.beta_max = beta
        self.gamma_list = np.asarray(gamma_list, dtype=np.float64)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        kx = np.fft.fftfreq(N) * N                # 0..N/2-1, (-N/2), ..-1
        ky = np.arange(0, N // 2 + 1)
        self.kx, self.ky = np.meshgrid(kx, ky, indexing="ij")
        self.mask = (np.abs(self.kx) <= self.k_max) & (np.abs(self.ky) <= self.k_max)

    def sample(self, n, gamma=None, return_params=False):
        """n fields. Every field is drawn sequentially (its (alpha, beta, gamma), then its noise), so
        the first k fields of sample(n) are exactly sample(k) for the same generator state. The
        noise is real white noise filtered in Fourier space, which gives exactly the stated
        covariance with the Hermitian symmetry of a real field (an earlier version drew
        independent complex coefficients on the half-spectrum, which halved the variance of the
        modes on the k_y = 0 and k_y = N/2 lines)."""
        rng = self.rng
        N = self.N
        fields = np.empty((n, N, N))
        al = np.empty(n); be = np.empty(n); ga = np.empty(n)
        for i in range(n):
            al[i] = np.exp(rng.uniform(np.log(self.alpha_min), np.log(self.alpha_max)))
            be[i] = np.exp(rng.uniform(np.log(self.beta_min), np.log(self.beta_max)))
            ga[i] = rng.choice(self.gamma_list) if gamma is None else float(gamma)
            psd = np.sqrt(al[i]) * (4 * np.pi ** 2 * (self.kx ** 2 + self.ky ** 2) + be[i]) ** (-ga[i] / 2)
            w = np.fft.rfft2(rng.standard_normal((N, N)), norm="ortho")   # unit variance per mode, Hermitian
            w[0, 0] = 0.0
            w = w * self.mask
            field = np.fft.irfft2(psd * w, s=(N, N), norm="ortho")
            fields[i] = field - field.mean()
        if return_params:
            return fields, {"alpha": al, "beta": be, "gamma": ga}
        return fields


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def demean(u):
    return u - u.mean(axis=(-2, -1), keepdims=True)


def l2(u):
    """Batched L2 norm over the grid axes."""
    return np.sqrt(np.sum(u ** 2, axis=(-2, -1)))


# ---------------------------------------------------------------------------
# Geometric multigrid (periodic, even N), usable as a stationary solver step
# ---------------------------------------------------------------------------

def restrict_fw(r):
    """Full-weighting restriction (..., N, N) -> (..., N/2, N/2), periodic."""
    N = r.shape[-1]
    if COMPILED and r.dtype == np.float64 and r.ndim in (2, 3):
        rc = _c64(r)
        B = 1 if rc.ndim == 2 else rc.shape[0]
        out = np.empty(rc.shape[:-2] + (N // 2, N // 2))
        _LIB.restrict_fw(_ptr(rc), _ptr(out), B, N)
        return out
    up = np.roll(r, -1, axis=-2); dn = np.roll(r, 1, axis=-2)
    lf = np.roll(r, 1, axis=-1); rt = np.roll(r, -1, axis=-1)
    s = (4.0 * r + 2.0 * (up + dn + lf + rt)
         + np.roll(up, 1, axis=-1) + np.roll(up, -1, axis=-1)
         + np.roll(dn, 1, axis=-1) + np.roll(dn, -1, axis=-1)) / 16.0
    return s[..., ::2, ::2]


def prolong_bilinear(c):
    """Bilinear prolongation (..., n, n) -> (..., 2n, 2n), periodic."""
    n = c.shape[-1]
    if COMPILED and c.dtype == np.float64 and c.ndim in (2, 3):
        cc_ = _c64(c)
        B = 1 if cc_.ndim == 2 else cc_.shape[0]
        out = np.empty(cc_.shape[:-2] + (2 * n, 2 * n))
        _LIB.prolong_bilinear(_ptr(cc_), _ptr(out), B, n)
        return out
    out = np.empty(c.shape[:-2] + (2 * n, 2 * n))
    cr = np.roll(c, -1, axis=-2)          # c[i+1, j]
    cc = np.roll(c, -1, axis=-1)          # c[i, j+1]
    crc = np.roll(cr, -1, axis=-1)        # c[i+1, j+1]
    out[..., 0::2, 0::2] = c
    out[..., 1::2, 0::2] = 0.5 * (c + cr)
    out[..., 0::2, 1::2] = 0.5 * (c + cc)
    out[..., 1::2, 1::2] = 0.25 * (c + cr + cc + crc)
    return out


class FastMultigrid:
    """V(nu1, nu2)-cycle with lexicographic Gauss-Seidel smoothing, full
    weighting / bilinear transfers, rediscretised coarse operators and an FFT
    direct solve on the coarsest grid (n_coarsest x n_coarsest). One `step`
    applies one V-cycle to the residual equation, i.e. it is the stationary
    iteration u <- u + M_MG^{-1} (f - A u)."""

    def __init__(self, pde: FastStencilPDE, nu1=2, nu2=2, n_coarsest=32, smoother="gs"):
        self.pde = pde
        self.nu1, self.nu2 = nu1, nu2
        self.levels = []
        N = pde.N
        coef = pde.coef
        def level_pde(n, c):
            if pde.equation == "VarCoeff":
                return FastStencilPDE(n, equation="VarCoeff", coef=c, coef_seed=pde.coef_seed, contrast=pde.contrast)
            return FastStencilPDE(n, equation=pde.equation, a=pde.a, b_vec=(pde.b1, pde.b2), aniso_eps=pde.aniso_eps)
        while N > n_coarsest:
            assert N % 2 == 0
            lp = level_pde(N, coef)
            sm = FastGaussSeidel(lp) if smoother == "gs" else FastJacobi(lp, 0.8)
            self.levels.append((lp, sm))
            N //= 2
            if coef is not None:   # rediscretised coarse operator: block-averaged coefficient
                coef = coef.reshape(N, 2, N, 2).mean(axis=(1, 3))
        self.coarse = level_pde(N, coef)
        self.name = "mg"
        self.n_levels = len(self.levels) + 1

    def _vcycle(self, lvl, r):
        """Approximate solution e of A_lvl e = r (r mean-free), starting from 0."""
        if lvl == len(self.levels):
            return self.coarse.solve_direct(r)
        lp, sm = self.levels[lvl]
        e = np.zeros_like(r)
        for _ in range(self.nu1):
            e = sm.step(e, r)
        rc = restrict_fw(lp.residual(e, r))
        e = e + prolong_bilinear(self._vcycle(lvl + 1, rc))
        for _ in range(self.nu2):
            e = sm.step(e, r)
        return e

    def step(self, u, f, r=None):
        if r is None:
            r = self.pde.residual(u, f)
        return u + self._vcycle(0, r)


class FFTDirect:
    """Direct solve by FFT diagonalisation (the reference solver for the
    constant-coefficient periodic problem); one `step` solves exactly."""

    def __init__(self, pde: FastStencilPDE):
        self.pde = pde
        self.name = "fft"

    def step(self, u, f, r=None):
        if r is None:
            r = self.pde.residual(u, f)
        return u + self.pde.solve_direct(r)


_MAKE_SOLVER_BASIC = make_solver


def make_solver(pde, spec):
    if spec == "mg":
        return FastMultigrid(pde)
    if spec.startswith("mg_"):           # e.g. mg_1_1 (nu1, nu2)
        _, a, b = spec.split("_")
        return FastMultigrid(pde, nu1=int(a), nu2=int(b))
    if spec == "fft":
        return FFTDirect(pde)
    return _MAKE_SOLVER_BASIC(pde, spec)


SOLVER_NAMES.update({"mg": "Multigrid V(2,2)", "fft": "FFT direct"})
