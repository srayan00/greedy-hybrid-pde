"""Empirical discretisation error of the second-order scheme on the confirmatory test forcing:
the exact discrete solution on the N x N grid is compared with the exact discrete solution on the
4N x 4N grid for the same (band-limited) forcing, restricted to the coarse nodes. The instances are
exactly the benchmark's (seed 73; 64, 32 and 16 instances at 128^2, 256^2 and 512^2; the sampler
draws instances sequentially, so a prefix of the benchmark's instances is reproduced by the same
seed). For variable-coefficient diffusion the coefficient a(x) is one grid-independent function
(sampled on every grid), so the comparison measures discretisation error only.
Writes results/discretization_error.json (relative errors per instance, equation and grid)."""
import json, sys, numpy as np
from fast_pde import FastStencilPDE, GRF2D
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 73
NT = {128: 64, 256: 32, 512: 16}
out = {}
for eq in ["Poisson", "ConvDiff", "AnisoDiff", "VarCoeff"]:
    for N in ([128, 256] if eq in ("VarCoeff", "AnisoDiff") else [128, 256, 512]):
        n = NT[N] if not (eq == "AnisoDiff" and N == 256) else 16     # the anisotropic 256^2 benchmark uses 16 instances
        f = GRF2D(N, rng=np.random.default_rng(seed)).sample(n)
        fh = np.fft.fft2(f); M = 4 * N
        Fh = np.zeros((n, M, M), dtype=complex)
        idx = (np.fft.fftfreq(N) * N).astype(int) % M
        Fh[np.ix_(range(n), idx, idx)] = fh * (M / N) ** 2
        F = np.real(np.fft.ifft2(Fh))
        uN = FastStencilPDE(N, equation=eq).solve_direct(f)
        uM = FastStencilPDE(M, equation=eq).solve_direct(F)[:, ::4, ::4]
        rel = np.linalg.norm((uN - uM).reshape(n, -1), axis=1) / np.linalg.norm(uM.reshape(n, -1), axis=1)
        out[f"{eq}_{N}"] = {"h2": 1.0 / N ** 2, "n": n, "seed": seed, "rel": rel.tolist(), "median": float(np.median(rel)),
                            "min": float(rel.min()), "max": float(rel.max())}
        print(f"{eq:9s} N={N}: n={n} median {np.median(rel):.2e} ({np.median(rel)/N**-2:.1f} h^2), range [{rel.min():.1e} ({rel.min()/N**-2:.1f} h^2), {rel.max():.1e}]", flush=True)
json.dump(out, open("results/discretization_error.json", "w"), indent=1)
