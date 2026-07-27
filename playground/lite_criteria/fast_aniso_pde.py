"""Fast periodic anisotropic diffusion: -eps_x u_xx - eps_y u_yy = f.

Matches playground/flops_regimes/aniso_pde.py for a ≡ 1 (constant tensor K).
Still FFT-diagonalizable, so scale-equivariant training stays cheap.
"""

from __future__ import annotations

import numpy as np

# Reuse Jacobi / GRF / helpers from the Poisson fast_pde module
from fast_pde import (  # noqa: F401
    FastGaussSeidel,
    FastJacobi,
    FastSOR,
    GRF2D,
    demean,
    l2,
    make_solver,
)


class FastAnisoPDE:
    """-eps_x u_xx - eps_y u_yy = f on [0,1]^2, periodic, uniform N x N grid."""

    def __init__(self, N, eps_x=0.01, eps_y=1.0):
        assert eps_x > 0 and eps_y > 0
        self.N = N
        self.eps_x = float(eps_x)
        self.eps_y = float(eps_y)
        self.equation = "AnisoDiff"
        self.h = 1.0 / N
        # Diagonal of the 5-point stencil (a≡1)
        self.diag = 2.0 * (self.eps_x + self.eps_y) / self.h ** 2
        self._symbol = self._build_symbol()
        self._lu_cache = {}

    def apply_A(self, u):
        h = self.h
        up_i = np.roll(u, -1, axis=-2)
        dn_i = np.roll(u, 1, axis=-2)
        up_j = np.roll(u, -1, axis=-1)
        dn_j = np.roll(u, 1, axis=-1)
        return (
            self.eps_x * (2.0 * u - up_i - dn_i) / h ** 2
            + self.eps_y * (2.0 * u - up_j - dn_j) / h ** 2
        )

    def residual(self, u, f):
        return f - self.apply_A(u)

    def _build_symbol(self):
        N, h = self.N, self.h
        k = np.fft.fftfreq(N) * N
        theta_x = 2 * np.pi * k[:, None] / N
        theta_y = 2 * np.pi * k[None, :] / N
        sym = (
            self.eps_x * (2.0 - 2.0 * np.cos(theta_x)) / h ** 2
            + self.eps_y * (2.0 - 2.0 * np.cos(theta_y)) / h ** 2
        ).astype(np.complex128)
        return sym

    def solve_direct(self, f):
        fhat = np.fft.fft2(f, axes=(-2, -1))
        sym = self._symbol.copy()
        sym[0, 0] = 1.0
        uhat = fhat / sym
        uhat[..., 0, 0] = 0.0
        return np.real(np.fft.ifft2(uhat, axes=(-2, -1)))

    def sparse_A(self):
        import scipy.sparse as sp

        N, h = self.N, self.h
        idx = np.arange(N * N).reshape(N, N)
        rows, cols, vals = [], [], []

        def add(nbr_idx, val):
            rows.append(idx.ravel())
            cols.append(nbr_idx.ravel())
            vals.append(np.full(N * N, val))

        add(idx, self.diag)
        add(np.roll(idx, 1, axis=0), -self.eps_x / h ** 2)
        add(np.roll(idx, -1, axis=0), -self.eps_x / h ** 2)
        add(np.roll(idx, 1, axis=1), -self.eps_y / h ** 2)
        add(np.roll(idx, -1, axis=1), -self.eps_y / h ** 2)
        return sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(N * N, N * N),
        ).tocsr()

    def _get_lower_solver(self, kind, omega=1.0):
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

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
