"""Fast 2-level Galerkin multigrid V-cycle for periodic constant-coeff PDEs.

Works for Poisson, ConvDiff, and AnisoDiff via the PDE's `sparse_A` / `residual` /
constant `diag`. Used as a `solver.step(u, f, r=None)`-compatible classical
iteration for the wall-clock hybrid pipeline.

Design:
  * pre/post smoothing with weighted Jacobi (omega=0.8);
  * full-weighting periodic restriction R (weights sum to 1), bilinear
    prolongation P = 4 R^T;
  * Galerkin coarse operator Ac = R A P (valid for any grid ratio, so the
    odd 31 -> 16 periodic coarsening is fine);
  * reaction-free operators have a constant null space -> pin coarse DOF 0 and
    keep the iterate mean-free.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class FastMultigrid:
    name = "mg"

    def __init__(self, pde, nu1=2, nu2=2, omega=0.8):
        eq = getattr(pde, "equation", None)
        if eq not in ("Poisson", "ConvDiff", "AnisoDiff"):
            raise ValueError(f"MG supports Poisson/ConvDiff/AnisoDiff, got {eq}")
        self.pde = pde
        self.equation = eq
        self.N = pde.N
        self.nu1 = nu1
        self.nu2 = nu2
        self.omega = omega
        self.diag = float(pde.diag)
        self.singular = abs(getattr(pde, "reaction", 0.0)) < 1e-14
        self.A = pde.sparse_A().tocsr()
        self.R = self._build_restriction(pde.N).tocsr()
        self.P = (4.0 * self.R.T).tocsr()
        Ac = (self.R @ self.A @ self.P).tolil()
        if self.singular:
            Ac[0, :] = 0.0
            Ac[0, 0] = 1.0
        self.Ac = Ac.tocsc()
        self.lu = spla.splu(self.Ac, permc_spec="NATURAL")
        self.Nc = (pde.N + 1) // 2

    def _build_restriction(self, N):
        Nc = (N + 1) // 2
        rows, cols, vals = [], [], []

        def fidx(i, j):
            return (i % N) * N + (j % N)

        stencil = [
            (0, 0, 0.25),
            (-1, 0, 0.125), (1, 0, 0.125), (0, -1, 0.125), (0, 1, 0.125),
            (-1, -1, 0.0625), (1, -1, 0.0625), (-1, 1, 0.0625), (1, 1, 0.0625),
        ]
        for I in range(Nc):
            for J in range(Nc):
                ci = I * Nc + J
                i, j = 2 * I, 2 * J
                for di, dj, w in stencil:
                    rows.append(ci)
                    cols.append(fidx(i + di, j + dj))
                    vals.append(w)
        return sp.coo_matrix((vals, (rows, cols)), shape=(Nc * Nc, N * N))

    def _smooth(self, u, f, n):
        for _ in range(n):
            r = self.pde.residual(u, f)
            u = u + (self.omega / self.diag) * r
        return u

    def step(self, u, f, r=None):
        N, Nc = self.N, self.Nc
        B = u.shape[0] if u.ndim == 3 else 1
        u = self._smooth(u, f, self.nu1)
        r = self.pde.residual(u, f)
        rc = self.R @ r.reshape(B, N * N).T  # (Nc^2, B)
        if self.singular:
            rc[0, :] = 0.0
        ec = self.lu.solve(np.ascontiguousarray(rc))  # (Nc^2, B)
        e = (self.P @ ec).T.reshape(u.shape)
        u = u + e
        u = self._smooth(u, f, self.nu2)
        if self.singular:
            u = u - u.mean(axis=(-2, -1), keepdims=True)
        return u


# Back-compat alias
FastMultigridPoisson = FastMultigrid
