"""Empirical discretisation error of the second-order scheme on the test forcing distribution:
the exact discrete solution on the N x N grid is compared with the exact discrete solution on
the 4N x 4N grid for the same (band-limited) forcing, restricted to the coarse nodes.
Writes results/discretization_error.json (relative errors per instance, equation and grid)."""
import json, numpy as np
from fast_pde import FastStencilPDE, GRF2D
out = {}
for eq in ["Poisson", "ConvDiff", "AnisoDiff", "VarCoeff"]:
    for N in ([128] if eq == "VarCoeff" else [128, 256]):
        n = 32
        f = GRF2D(N, rng=np.random.default_rng(72)).sample(n)     # the test forcing (seed 72)
        fh = np.fft.fft2(f); M = 4 * N
        Fh = np.zeros((n, M, M), dtype=complex)
        idx = (np.fft.fftfreq(N) * N).astype(int) % M
        Fh[np.ix_(range(n), idx, idx)] = fh * (M / N) ** 2
        F = np.real(np.fft.ifft2(Fh))
        uN = FastStencilPDE(N, equation=eq).solve_direct(f)
        uM = FastStencilPDE(M, equation=eq).solve_direct(F)[:, ::4, ::4]
        rel = np.linalg.norm((uN - uM).reshape(n, -1), axis=1) / np.linalg.norm(uM.reshape(n, -1), axis=1)
        out[f"{eq}_{N}"] = {"h2": 1.0 / N ** 2, "rel": rel.tolist(), "median": float(np.median(rel)), "min": float(rel.min()), "max": float(rel.max())}
        print(f"{eq:9s} N={N}: median {np.median(rel):.2e} ({np.median(rel)/N**-2:.1f} h^2), range [{rel.min():.1e}, {rel.max():.1e}]")
json.dump(out, open("results/discretization_error.json", "w"), indent=1)
