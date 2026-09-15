"""Multi-trial robustness: retrain the router of every pairing with several
independent seeds and re-benchmark the router policy on the same test
instances (all other policies are seed-independent).

  python bench_seeds.py --equation Poisson --N 128 --seeds 1,2,3,4,5
writes results/seeds_<eq>_<N>.json with per-seed, per-instance time-to-tol.
"""

import argparse
import gc
import json
import os
import time

import numpy as np
import torch

from fast_pde import FastStencilPDE, GRF2D
from corrector import DeepONetCorrector
from hybrid import Env, FeatureState, run_untimed, run_timed, time_to_tol, work_units
from router import fit_router

p = argparse.ArgumentParser()
p.add_argument("--equation", default="Poisson")
p.add_argument("--N", type=int, default=128)
p.add_argument("--solvers", default="jacobi,jacobi_0.67,gs,ssor,sor_1.5")
p.add_argument("--seeds", default="1,2,3,4,5")
p.add_argument("--n_test", type=int, default=64)
p.add_argument("--seed", type=int, default=72)
p.add_argument("--tols", default="1e-2,1e-3,h2,1e-5,1e-6,1e-8")
p.add_argument("--max_ops", type=int, default=60000)
p.add_argument("--router_inst", type=int, default=128)
p.add_argument("--router_max_epochs", type=int, default=600)
p.add_argument("--ckp_dir", default="./checkpoints")
p.add_argument("--out_dir", default="./results")
args = p.parse_args()

torch.set_num_threads(1)
h2 = 1.0 / args.N ** 2
tols = [h2 if t == "h2" else float(t) for t in args.tols.split(",")]
pde = FastStencilPDE(args.N, equation=args.equation)
corrector = DeepONetCorrector(f"{args.ckp_dir}/deeponet_{args.equation}_{args.N}_best.pth", threads=1)
f_test = GRF2D(args.N, rng=np.random.default_rng(args.seed)).sample(args.n_test)
u_truth = pde.solve_direct(f_test)
costs_all = json.load(open(f"{args.ckp_dir}/costs_{args.equation}_{args.N}.json"))
seeds = [int(s) for s in args.seeds.split(",")]
out_path = f"{args.out_dir}/seeds_{args.equation}_{args.N}.json"
res = json.load(open(out_path)) if os.path.exists(out_path) else {"args": vars(args), "tols": tols, "h2": h2, "groups": {}}

for spec in args.solvers.split(","):
    env = Env(pde, [spec], corrector, costs=costs_all[spec])
    _ = env.solvers[0].step(np.zeros_like(f_test[:1]), f_test[:1])
    res["groups"].setdefault(spec, {})
    for sd in seeds:
        if str(sd) in res["groups"][spec]:
            continue
        torch.set_num_threads(8)
        t0 = time.time()
        router = fit_router(env, n_inst=args.router_inst, seed=1000 * sd + 7, dagger_rounds=2,
                            max_epochs=args.router_max_epochs, verbose=False)
        train_s = time.time() - t0
        router.save(f"{args.ckp_dir}/router_{args.equation}_{args.N}_{spec}_seed{sd}.pth")
        torch.set_num_threads(1)
        rows = []
        for i in range(args.n_test):
            f1, u1 = f_test[i:i + 1], u_truth[i:i + 1]
            tr = run_untimed(env, f1, u1, "router", max_ops=args.max_ops, err_stop=1e-9, router=router)
            gc.collect(); gc.disable()
            t, _ = run_timed(env, f1, tr, "router", router=router)
            gc.enable()
            tt = time_to_tol(tr, t, tols)
            t_wu = work_units(env, tr, "router", router_cost=5e-6)
            tw = time_to_tol(tr, t_wu, tols)
            ops_ = np.asarray(tr["op"], dtype=int)
            if len(ops_):
                brk_ = np.flatnonzero(np.diff(ops_)) + 1
                starts_ = np.concatenate([[0], brk_]); ends_ = np.concatenate([brk_, [len(ops_)]])
                op_rle = [[int(ops_[s_]), int(e_ - s_)] for s_, e_ in zip(starts_, ends_)]
            else:
                op_rle = []
            rows.append({"n_ops": int(len(tr["op"])), "n_no": int(tr["n_no"]), "op_rle": op_rle,
                         "tol": {f"{tol:.6g}": {"iters": None if not np.isfinite(tt[tol][1]) else int(tt[tol][1]),
                                                "t_live": None if not np.isfinite(tt[tol][0]) else float(tt[tol][0]),
                                                "t_wu": None if not np.isfinite(tw[tol][0]) else float(tw[tol][0])}
                                 for tol in tols}})
        res["groups"][spec][str(sd)] = {"train_s": train_s, "rows": rows}
        med = np.median([r["tol"][f"{h2:.6g}"]["t_live"] or np.inf for r in rows])
        print(f"[{args.equation} N={args.N} {spec} seed {sd}] router median time-to-h2 {med*1e3:.3f} ms "
              f"(train {train_s:.0f}s)", flush=True)
        json.dump(res, open(out_path, "w"))
print("saved", out_path)
