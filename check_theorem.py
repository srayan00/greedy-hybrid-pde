"""Exact verification of Theorem 4.1 (with postfix slack mu) for short horizons on the
experimental solver ensembles: the optimal sequence O of T macro-actions is found by
exhaustive search, and the quantities the proof uses are computed exactly:

  mu_O      = max_{t<T} g(S^t + O) / g(O)          (postfix slack along the greedy prefixes)
  alpha_g   = max_{t<T} [g(S^t) - g(S^t + O)] / sum_i [g(S^t) - g(S^t + O_i)]   (greedy prefixes)
  alpha_all = the same maximum over every sequence S with |S| <= T-1            (all short prefixes)
  bound     = mu (1 - phi) g(O) + phi g(empty),  phi = (1 - 1/(alpha T))^T
and whether g(S^T) <= bound.  g(S) = ||e_S||_2^2 (mean-free), macro-actions as in the benchmark.

  python check_theorem.py --equation Poisson --N 128
writes results/theorem_<eq>_<N>.json
"""
import argparse, itertools, json, time
import math
import numpy as np, torch
from fast_pde import FastStencilPDE, GRF2D, demean
from corrector import DeepONetCorrector
from hybrid import Env

p = argparse.ArgumentParser()
p.add_argument("--equation", default="Poisson")
p.add_argument("--N", type=int, default=128)
p.add_argument("--n_inst", type=int, default=8)
p.add_argument("--seed", type=int, default=72)
p.add_argument("--ckp_dir", default="./checkpoints")
p.add_argument("--out_dir", default="./results")
args = p.parse_args()
torch.set_num_threads(1)
N = args.N
pde = FastStencilPDE(N, equation=args.equation)
corr = DeepONetCorrector(f"{args.ckp_dir}/deeponet_{args.equation}_{N}_best.pth", threads=1)
cache = json.load(open(f"{args.ckp_dir}/costs_{args.equation}_{N}.json"))
solvers = [s for s in ["jacobi", "jacobi_0.67", "gs", "ssor", "sor_1.5", "mg"] if s in cache]
f = GRF2D(N, rng=np.random.default_rng(args.seed)).sample(args.n_inst)
u_true = pde.solve_direct(f)
# horizons short enough that every sequence stays well above double-precision roundoff
groups = [([s], 4) for s in solvers]
if "jacobi" in solvers and "jacobi_0.67" in solvers:
    groups.append((["jacobi", "jacobi_0.67"], 4))
if "gs" in solvers and "sor_1.5" in solvers:
    groups.append((["gs", "sor_1.5"], 4))          # two non-commuting sweeps
if "gs" in solvers and "mg" in solvers:
    groups.append((["gs", "mg"], 4))
stat = [s for s in solvers if s != "mg"]
if len(stat) >= 5:
    groups.append((stat, 3))
FLOOR_REL = 1e-11   # errors below FLOOR_REL * ||u|| are at the level of roundoff and are clipped


def costs_for(grp):
    key = "+".join(grp)
    if key in cache:
        return cache[key]
    c = {s: cache[s][s] for s in grp}
    c["no"] = float(np.mean([cache[s]["no"] for s in grp])); c["_residual"] = float(np.mean([cache[s]["_residual"] for s in grp]))
    return c


def apply_macro(env, j, u, fi):
    for _ in range(env.m[j]):
        r = pde.residual(u, fi)
        u = env.apply_op(j, u, fi, r)
    return u


out = {"args": vars(args), "groups": {}}
for grp, T in groups:
    env = Env(pde, grp, corr, costs=costs_for(grp))
    K = env.K
    t0 = time.time()
    rows = []
    for i in range(min(args.n_inst, 4 if K > 4 else args.n_inst)):
        fi, ui = f[i:i + 1], u_true[i:i + 1]
        # tree of all sequences up to length T (states and objective values)
        states = {(): np.zeros_like(fi)}
        floor = (FLOOR_REL * np.linalg.norm(ui)) ** 2
        gg = lambda e: float(max(np.linalg.norm(demean(e)) ** 2, floor))
        g = {(): gg(ui - states[()])}
        for depth in range(T):
            for S in [s for s in list(states) if len(s) == depth]:
                for j in range(K):
                    S2 = S + (j,)
                    states[S2] = apply_macro(env, j, states[S], fi)
                    g[S2] = gg(ui - states[S2])
        leaves = [S for S in g if len(S) == T]
        O = min(leaves, key=lambda S: g[S])
        # greedy: the deployed cost-aware rule (argmin of the per-macro-action score, which is the
        # plain error of Alg. 1 for every action not dearer than the unit, exponent 1, and the
        # per-unit-cost error for dearer actions); also recorded: whether it differs from plain Alg. 1
        S_g = ()
        S_plain = ()
        for t in range(T):
            e0 = math.sqrt(g[S_g])
            sc = env.macro_score(e0, [math.sqrt(g[S_g + (j,)]) for j in range(K)])
            S_g = S_g + (int(np.argmin(sc)),)
            S_plain = S_plain + (min(range(K), key=lambda j: g[S_plain + (j,)]),)
        greedy_prefixes = [S_g[:t] for t in range(T)]

        def g_concat(S, seq):
            u = states[S]
            for j in seq:
                u = apply_macro(env, j, u, fi)
            return gg(ui - u)

        viol = {"greedy": 0, "all": 0}      # prefixes with lhs > 0 and rhs <= 0: the inequality fails for every alpha

        def ratio(S, which):
            lhs = g[S] - g_concat(S, O)
            rhs = sum(g[S] - g_concat(S, (Oi,)) for Oi in O)
            if rhs > 0:
                return lhs / rhs
            if lhs > 1e-12 * max(g[S], floor):
                viol[which] += 1
            return None                      # lhs <= 0 with rhs <= 0: holds trivially
        mu = max(g_concat(S, O) / g[O] for S in greedy_prefixes)
        a_g = [ratio(S, "greedy") for S in greedy_prefixes]
        a_g = [x for x in a_g if x is not None]
        alpha_g = max(max(a_g), 1.0) if a_g else 1.0
        a_all = [ratio(S, "all") for S in g if len(S) <= T - 1]
        a_all = [x for x in a_all if x is not None]
        alpha_all = max(max(a_all), 1.0) if a_all else 1.0
        phi = (1 - 1 / (alpha_g * T)) ** T
        bound = mu * (1 - phi) * g[O] + phi * g[()]
        rows.append({"T": T, "K": K, "O": list(O), "greedy": list(S_g), "g0": g[()], "gO": g[O], "gS": g[S_g], "floor": floor,
                     "mu": mu, "alpha_greedy": alpha_g, "alpha_all": alpha_all, "phi": phi, "bound": bound,
                     "holds": bool(g[S_g] <= bound * (1 + 1e-9)), "greedy_over_opt": g[S_g] / g[O],
                     "greedy_plain": list(S_plain), "deployed_eq_plain": bool(S_plain == S_g),
                     "opt_at_floor": bool(g[O] <= floor * (1 + 1e-9)), "viol_greedy": viol["greedy"], "viol_all": viol["all"],
                     "macro_exp": list(env.macro_exp)})
    key = "+".join(grp)
    out["groups"][key] = {"ops": env.ops, "m": env.m, "T": T, "rows": rows}
    print(f"[{args.equation} N={N}] {key:35s} K={K} T={T}: mu max {max(r['mu'] for r in rows):.3f} | alpha greedy max {max(r['alpha_greedy'] for r in rows):.3f} | alpha all max {max(r['alpha_all'] for r in rows):.3f} "
          f"| bound holds {sum(r['holds'] for r in rows)}/{len(rows)} | g(S^T)/g(O) median {np.median([r['greedy_over_opt'] for r in rows]):.3f} max {max(r['greedy_over_opt'] for r in rows):.3f} | greedy=O on {sum(r['greedy']==r['O'] for r in rows)} ({time.time()-t0:.0f}s)", flush=True)
    json.dump(out, open(f"{args.out_dir}/theorem_{args.equation}_{N}.json", "w"), indent=1)
print("saved")
