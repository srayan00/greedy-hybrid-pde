"""Validate fast_pde.py against the dense reference implementations in pde.py /
numerical_solver.py at N=31 (Poisson and ConvDiff, periodic)."""

import numpy as np
import torch

from fast_pde import (FastStencilPDE, FastJacobi, FastGaussSeidel, FastSOR, FastSSOR,
                      GRF2D, demean, l2)
from pde import PoissonEquation2D, ConvectionDiffusion2D
from numerical_solver import (WeightedJacobiSolver, GaussSeidelSolver,
                              SuccessiveOverRelaxationSolver,
                              SymmetricSuccessiveOverRelaxationSolver)

torch.manual_seed(0)
N = 31
x = torch.linspace(0, 1, N + 1)[:-1]
grf = GRF2D(N, rng=np.random.default_rng(1))
f_np = grf.sample(2)                      # (2, N, N), zero-mean
f_t = torch.tensor(f_np.reshape(2, -1), dtype=torch.float64)

for eq_name in ["Poisson", "ConvDiff"]:
    print(f"=== {eq_name} ===")
    if eq_name == "Poisson":
        ref = PoissonEquation2D(a_func=lambda a, b: 1.0, f_func=f_t, boundary="Periodic",
                                x=x.double(), y=x.double(), solve=False)
    else:
        ref = ConvectionDiffusion2D(a_func=lambda a, b: 1.0, f_func=f_t, b_vec=(20.0, 20.0),
                                    boundary="Periodic", x=x.double(), y=x.double(), solve=False)
    ref.A = ref.A.double()
    ref.b = ref.b.double()
    A_dense = ref.A.numpy()
    fast = FastStencilPDE(N, equation=eq_name)

    u_test = grf.sample(2)
    Au_fast = fast.apply_A(u_test)
    Au_dense = (A_dense @ u_test.reshape(2, -1).T).T.reshape(2, N, N)
    print("  apply_A max err:", np.abs(Au_fast - Au_dense).max())

    u_direct = fast.solve_direct(f_np)
    u_lstsq = np.linalg.lstsq(A_dense, f_np.reshape(2, -1).T, rcond=None)[0].T.reshape(2, N, N)
    u_lstsq = demean(u_lstsq)
    print("  direct-vs-lstsq rel err:", (l2(u_direct - u_lstsq) / l2(u_lstsq)).max())
    print("  direct residual rel:", (l2(fast.residual(u_direct, f_np)) / l2(f_np)).max())

    u0 = torch.tensor(grf.sample(2).reshape(2, -1), dtype=torch.float64)
    u0_np = u0.numpy().reshape(2, N, N)
    pairs = [
        (FastJacobi(fast, 1.0), WeightedJacobiSolver(ref, weight=1.0), "jacobi"),
        (FastJacobi(fast, 0.67), WeightedJacobiSolver(ref, weight=0.67), "jacobi_0.67"),
        (FastGaussSeidel(fast), GaussSeidelSolver(ref), "gs"),
        (FastSOR(fast, 1.5), SuccessiveOverRelaxationSolver(ref, omega=1.5), "sor_1.5"),
    ]
    try:
        pairs.append((FastSSOR(fast, 1.0), SymmetricSuccessiveOverRelaxationSolver(ref, omega=1.0), "ssor"))
    except Exception as e:  # signature differences
        print("  (ssor reference unavailable:", e, ")")
    tol = 1e-6   # the dense reference assembles its matrix in single precision
    rel_A = np.abs(Au_fast - Au_dense).max() / np.abs(Au_dense).max()
    assert rel_A < tol, f"{eq_name}: apply_A relative error {rel_A:.2e} exceeds {tol:.0e}"
    for fast_solver, dense_solver, tag in pairs:
        u1_fast = fast_solver.step(u0_np, f_np)
        u1_dense = dense_solver.iteration(u0.clone()).numpy().reshape(2, N, N)
        rel = np.abs(u1_fast - u1_dense).max() / max(np.abs(u1_dense - u0_np).max(), 1e-300)
        print(f"  {tag} one-step max err: {np.abs(u1_fast - u1_dense).max():.3e} (relative to the update: {rel:.3e})")
        assert rel < tol, f"{eq_name}/{tag}: one-step relative error {rel:.2e} exceeds {tol:.0e}"

print("all checks passed")
