"""Fast periodic constant-coefficient PDE utilities for wall-clock benchmarking.

The dense-matrix implementations in pde.py / numerical_solver.py are convenient for
training but make every classical iteration an O(N^4) dense matvec, which would
unfairly slow the classical baselines in any wall-clock comparison. This module
provides:

  * FastStencilPDE  - O(N^2) stencil application of the same 5-point discretizations
                      built by pde.py (Poisson / ConvDiff, periodic BCs, constant
                      coefficients), an FFT direct solver (exact solution of the
                      discrete system, used as ground truth), and cheap residuals.
  * Jacobi / GS / SOR / SSOR one-step iterations matching numerical_solver.py
                      (validated against the dense implementations at N=31).
  * GRF2D           - numpy port of data_generation.GaussianRandomFieldHierarchical
                      (same spectral law, same sampling of alpha/beta/gamma).

Everything is batched over the leading axis and works on (B, N, N) float64 arrays.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class FastStencilPDE:
    """-a * Lap(u) + b . grad(u) + c * u = f on [0,1]^2, periodic, uniform N x N grid.

    Matches the discretization of pde.py: second-order central differences for both
    diffusion and advection, grid x = linspace(0, 1, N+1)[:-1] so h = 1/N.
    """

    def __init__(self, N, equation="Poisson", a=1.0, b_vec=(20.0, 20.0), reaction=0.0):
        assert equation in ("Poisson", "ConvDiff")
        self.N = N
        self.equation = equation
        self.a = a
        self.b1, self.b2 = (0.0, 0.0) if equation == "Poisson" else b_vec
        self.reaction = reaction if equation == "ConvDiff" else 0.0
        self.h = 1.0 / N
        self.diag = 4.0 * a / self.h ** 2 + self.reaction
        self._symbol = self._build_symbol()
        self._lu_cache = {}

    # -- operator ------------------------------------------------------------
    def apply_A(self, u):
        """u: (..., N, N). First grid axis is i (x), second is j (y)."""
        h, a = self.h, self.a
        up_i = np.roll(u, -1, axis=-2)   # u_{i+1,j}
        dn_i = np.roll(u, 1, axis=-2)    # u_{i-1,j}
        up_j = np.roll(u, -1, axis=-1)   # u_{i,j+1}
        dn_j = np.roll(u, 1, axis=-1)    # u_{i,j-1}
        out = a * (4.0 * u - up_i - dn_i - up_j - dn_j) / h ** 2
        if self.b1 or self.b2:
            out = out + self.b1 * (up_i - dn_i) / (2 * h) + self.b2 * (up_j - dn_j) / (2 * h)
        if self.reaction:
            out = out + self.reaction * u
        return out

    def residual(self, u, f):
        return f - self.apply_A(u)

    # -- FFT direct solve (exact solution of the discrete system) -------------
    def _build_symbol(self):
        N, h = self.N, self.h
        k = np.fft.fftfreq(N) * N          # integer wavenumbers 0..k,-k..-1
        theta_x = 2 * np.pi * k[:, None] / N
        theta_y = 2 * np.pi * k[None, :] / N
        sym = self.a * (4 - 2 * np.cos(theta_x) - 2 * np.cos(theta_y)) / h ** 2
        sym = sym.astype(np.complex128)
        if self.b1 or self.b2:
            sym = sym + 1j * (self.b1 * np.sin(theta_x) + self.b2 * np.sin(theta_y)) / h
        sym = sym + self.reaction
        return sym

    def solve_direct(self, f):
        """Exact solution of the discrete system A u = f via FFT diagonalization.

        For reaction == 0 the operator has a constant null space; f must be
        (numerically) mean-free and the returned solution is mean-free.
        """
        fhat = np.fft.fft2(f, axes=(-2, -1))
        sym = self._symbol.copy()
        if abs(self.reaction) < 1e-14:
            sym[0, 0] = 1.0  # avoid divide-by-zero; mode is zeroed below
        uhat = fhat / sym
        if abs(self.reaction) < 1e-14:
            uhat[..., 0, 0] = 0.0
        return np.real(np.fft.ifft2(uhat, axes=(-2, -1)))

    # -- sparse matrix (for GS / SOR triangular solves) ------------------------
    def sparse_A(self):
        N, h, a = self.N, self.h, self.a
        idx = np.arange(N * N).reshape(N, N)
        rows, cols, vals = [], [], []

        def add(nbr_idx, val):
            rows.append(idx.ravel())
            cols.append(nbr_idx.ravel())
            vals.append(np.full(N * N, val))

        add(idx, self.diag)
        add(np.roll(idx, 1, axis=0), -a / h ** 2 - self.b1 / (2 * h))    # (i-1, j)
        add(np.roll(idx, -1, axis=0), -a / h ** 2 + self.b1 / (2 * h))   # (i+1, j)
        add(np.roll(idx, 1, axis=1), -a / h ** 2 - self.b2 / (2 * h))    # (i, j-1)
        add(np.roll(idx, -1, axis=1), -a / h ** 2 + self.b2 / (2 * h))   # (i, j+1)
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

    name = "jacobi"

    def __init__(self, pde: FastStencilPDE, weight=1.0):
        self.pde = pde
        self.weight = weight
        self.name = "jacobi" if weight == 1.0 else f"jacobi_{weight:g}"

    def step(self, u, f, r=None):
        if r is None:
            r = self.pde.residual(u, f)
        return u + (self.weight / self.pde.diag) * r


class FastGaussSeidel:
    """u <- u + (D + L)^{-1} (f - A u), lexicographic ordering as in pde.py."""

    name = "gs"

    def __init__(self, pde: FastStencilPDE):
        self.pde = pde
        self.lu = pde._get_lower_solver("gs")

    def step(self, u, f, r=None):
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
        self.lu = pde._get_lower_solver("sor", omega)
        self.name = f"sor_{omega:g}"

    def step(self, u, f, r=None):
        if r is None:
            r = self.pde.residual(u, f)
        B = r.shape[0] if r.ndim == 3 else 1
        N = self.pde.N
        rr = r.reshape(B, N * N).T
        du = self.lu.solve(np.ascontiguousarray(rr))
        return u + du.T.reshape(r.shape)


class FastSSOR:
    """One SSOR preconditioner step: u <- u + M^{-1} (f - A u) with

        M = (ω/(2-ω)) (D/ω + L) D^{-1} (D/ω + U)

    (standard form; paper's torch SSOR has a known scalar/+ bug we do not copy).
    Default ω=1 is symmetric Gauss–Seidel.
    """

    def __init__(self, pde, omega=1.0):
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        if not (0.0 < omega < 2.0):
            raise ValueError(f"SSOR omega must be in (0, 2), got {omega}")
        self.pde = pde
        self.omega = float(omega)
        self.name = "ssor" if abs(self.omega - 1.0) < 1e-14 else f"ssor_{self.omega:g}"
        self.scale = self.omega / (2.0 - self.omega)
        A = pde.sparse_A().tocsr()
        D = sp.diags(A.diagonal())
        L = sp.tril(A, k=-1)
        U = sp.triu(A, k=1)
        self.diag = np.ascontiguousarray(A.diagonal(), dtype=np.float64)
        self.lu_lower = spla.splu(((1.0 / self.omega) * D + L).tocsc(), permc_spec="NATURAL")
        self.lu_upper = spla.splu(((1.0 / self.omega) * D + U).tocsc(), permc_spec="NATURAL")

    def step(self, u, f, r=None):
        if r is None:
            r = self.pde.residual(u, f)
        B = r.shape[0] if r.ndim == 3 else 1
        N = self.pde.N
        rr = np.ascontiguousarray(r.reshape(B, N * N).T)  # (N^2, B)
        # c * Ml * D^{-1} * Mu * du = r
        y = self.lu_lower.solve(rr / self.scale)          # Ml y = r/c
        z = self.diag[:, None] * y                         # z = D y
        du = self.lu_upper.solve(np.ascontiguousarray(z))  # Mu du = z
        return u + du.T.reshape(r.shape)


def make_solver(pde, spec):
    """spec: jacobi, jacobi_0.67, gs, sor_1.5, ssor, ssor_1.5, mg."""
    parts = spec.split("_")
    if parts[0] == "jacobi":
        return FastJacobi(pde, float(parts[1]) if len(parts) > 1 else 1.0)
    if parts[0] == "gs":
        return FastGaussSeidel(pde)
    if parts[0] == "sor":
        return FastSOR(pde, float(parts[1]) if len(parts) > 1 else 1.0)
    if parts[0] == "ssor":
        return FastSSOR(pde, float(parts[1]) if len(parts) > 1 else 1.0)
    if parts[0] == "mg":
        from fast_multigrid import FastMultigrid

        return FastMultigrid(pde)
    raise ValueError(f"unknown solver spec {spec}")


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
    data_generation.GaussianRandomFieldHierarchical (dim=2).
    """

    def __init__(self, N, alpha=(0.01, 100.0), beta=(0.1, 1000.0),
                 gamma_list=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0), rng=None):
        assert N % 2 == 1, "grid size must be odd (matches the paper's setup)"
        self.N = N
        self.k_max = N // 2
        self.alpha_min, self.alpha_max = alpha
        self.beta_min, self.beta_max = beta
        self.gamma_list = np.asarray(gamma_list, dtype=np.float64)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        kx = np.concatenate([np.arange(0, self.k_max + 1), np.arange(-self.k_max, 0)])
        ky = np.arange(0, self.k_max + 1)
        self.kx, self.ky = np.meshgrid(kx, ky, indexing="ij")

    def sample(self, n, gamma=None):
        rng = self.rng
        alpha = np.exp(rng.uniform(np.log(self.alpha_min), np.log(self.alpha_max), n))
        beta = np.exp(rng.uniform(np.log(self.beta_min), np.log(self.beta_max), n))
        if gamma is None:
            gamma = rng.choice(self.gamma_list, size=n)
        else:
            gamma = np.full(n, gamma, dtype=np.float64)
        psd = (np.sqrt(alpha)[:, None, None]
               * (4 * np.pi ** 2 * (self.kx ** 2 + self.ky ** 2)[None] + beta[:, None, None])
               ** (-gamma[:, None, None] / 2))
        z = rng.standard_normal((n,) + self.kx.shape) + 1j * rng.standard_normal((n,) + self.kx.shape)
        z[:, 0, 0] = 0.0
        field = np.fft.irfft2(psd * z, s=(self.N, self.N), norm="ortho")
        return field - field.mean(axis=(-2, -1), keepdims=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def demean(u):
    return u - u.mean(axis=(-2, -1), keepdims=True)


def l2(u):
    """Batched L2 norm over the grid axes."""
    return np.sqrt(np.sum(u ** 2, axis=(-2, -1)))
