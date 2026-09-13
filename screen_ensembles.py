"""Oracle-level screening of solver ensembles: for every subset W (|W| <= max_size)
of the member family, run the cost-aware greedy oracle over NO u W on the test
instances and record the work-unit time to h^2 and 1e-8 (deterministic, timer-free).
The ratio (best pairwise oracle) / (ensemble oracle) is the ceiling of what a
router over W can gain over the best single numerical solver.

  python screen_ensembles.py --equation Poisson --N 128 --members jacobi,jacobi_0.67,gs,ssor,sor_1.5,mg
writes results/screen_<eq>_<N>.json
"""

import argparse
import itertools
import json
import os
import time

import numpy as np
import torch

from fast_pde import FastStencilPDE, GRF2D
from corrector import DeepONetCorrector
from hybrid import Env, run_untimed, time_to_tol, work_units

p = argparse.ArgumentParser()
p.add_argument("--equation", default="Poisson")
p.add_argument("--N", type=int, default=128)
p.add_argument("--members", default="jacobi,jacobi_0.67,gs,ssor,sor_1.5,mg")
p.add_argument("--max_size", type=int, default=3)
p.add_argument("--n_inst", type=int, default=16)
p.add_argument("--seed", type=int, default=72)
p.add_argument("--max_ops", type=int, default=2000)
p.add_argument("--ckp_dir", default="./checkpoints")
p.add_argument("--out_dir", default="./results")
args = p.parse_args()
torch.set_num_threads(1)

pde = FastStencilPDE(args.N, equation=args.equation)
corrector = DeepONetCorrector(f"{args.ckp_dir}/deeponet_{args.equation}_{args.N}_best.pth", threads=1)
f = GRF2D(args.N, rng=np.random.default_rng(args.seed)).sample(args.n_inst)
u = pde.solve_direct(f)
h2 = 1.0 / args.N ** 2
tols = [h2, 1e-8]

# per-operation costs from the cached pairwise measurements (same N); a member
# without a cache entry for this equation borrows the Poisson cost (same stencil)
cache = json.load(open(f"{args.ckp_dir}/costs_{args.equation}_{args.N}.json"))
fallback = json.load(open(f"{args.ckp_dir}/costs_Poisson_{args.N}.json"))
members = [m for m in args.members.split(",") if m in cache or m in fallback]
unit = {}
for m in members:
    src = cache.get(m) or fallback[m]
    unit[m] = src[m]
no_cost = float(np.mean([v["no"] for k, v in cache.items() if "no" in v and "+" not in k]))
res_cost = float(np.mean([v["_residual"] for k, v in cache.items() if "_residual" in v and "+" not in k]))
print(f"{args.equation} N={args.N}: members {members}; NO {no_cost*1e6:.0f}us residual {res_cost*1e6:.0f}us", flush=True)

out = {"args": vars(args), "h2": h2, "members": members, "costs": {**unit, "no": no_cost, "_residual": res_cost}, "sets": {}}
path = f"{args.out_dir}/screen_{args.equation}_{args.N}.json"
for size in range(1, args.max_size + 1):
    for W in itertools.combinations(members, size):
        key = "+".join(W)
        t0 = time.time()
        costs = {m: unit[m] for m in W}
        costs.update({"no": no_cost, "_residual": res_cost})
        env = Env(pde, list(W), corrector, costs=costs)
        rows = []
        for i in range(args.n_inst):
            tr = run_untimed(env, f[i:i + 1], u[i:i + 1], "oracle", max_ops=args.max_ops, err_stop=1e-9)
            t = work_units(env, tr, "oracle")
            tt = time_to_tol(tr, t, tols)
            ops = np.asarray(tr["op"])
            row = {"n_ops": int(len(ops))}
            for tol in tols:
                tw, it = tt[tol]
                k = int(it) if np.isfinite(it) else len(ops)
                frac = [float(np.mean(ops[:k] == j)) if k > 0 else 0.0 for j in range(env.K)]
                row[f"{tol:.6g}"] = {"t_wu": None if not np.isfinite(tw) else float(tw),
                                     "iters": None if not np.isfinite(it) else int(it), "op_frac": frac}
            rows.append(row)
        out["sets"][key] = {"ops": env.ops, "m": env.m, "rows": rows}
        med = [np.median([r[f"{tol:.6g}"]["t_wu"] or np.inf for r in rows]) for tol in tols]
        fr = np.mean([r[f"{h2:.6g}"]["op_frac"] for r in rows], axis=0)
        print(f"  {key:45s} WU to h2 {med[0]*1e3:8.3f} ms | to 1e-8 {med[1]*1e3:8.3f} ms | usage to h2 "
              + " ".join(f"{o}:{x:.2f}" for o, x in zip(env.ops, fr)) + f"  ({time.time()-t0:.0f}s)", flush=True)
        json.dump(out, open(path, "w"))

# summary: ceiling of every |W| >= 2 set over its best member pair
print("\n=== ceiling (best pairwise oracle / ensemble oracle), paired median over instances ===")
for key, S in out["sets"].items():
    W = key.split("+")
    if len(W) < 2:
        continue
    for tol in tols:
        tk = f"{tol:.6g}"
        te = np.array([r[tk]["t_wu"] or np.inf for r in S["rows"]])
        best, best_m = None, None
        for m in W:
            tm = np.array([r[tk]["t_wu"] or np.inf for r in out["sets"][m]["rows"]])
            if best is None or np.median(tm) < np.median(best):
                best, best_m = tm, m
        print(f"  {key:45s} tol {tk:>10s}: ratio-of-medians {np.median(best)/np.median(te):.3f}  paired {np.median(best/te):.3f}  (best pair {best_m})")
print("saved", path)
