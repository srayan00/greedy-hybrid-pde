"""Selection of the best fixed schedule on the development instances (seed 72), for a deployable fixed-schedule
baseline: the schedule a practitioner would pick without seeing the test instances. Every schedule of the family
(HINTS with tau in {2, 5, 10, 15, 25, 50}, phase-shifted HINTS with tau in {5, 10, 15, 25, 50}, one-shot) is run
untimed on the development instances of an equation and grid with the cached per-operation costs, and its work-unit
time to every tolerance is recorded; the selected schedule per tolerance is the one with the smallest median (a
censored run counts as infinite). Schedules need no router and no timing, so this is cheap and deterministic.

  python select_schedule.py --equation Poisson --N 128 --solvers jacobi,gs
writes results/schedule_dev_<eq>_<N>.json
"""

import argparse
import json
import os

import numpy as np
import torch

from fast_pde import FastStencilPDE, GRF2D
from corrector import DeepONetCorrector
from hybrid import Env, run_untimed, work_units, time_to_tol

SCHEDULES = ["hints2", "hints5", "hints10", "hints15", "hints25", "hints50",
             "phints5", "phints10", "phints15", "phints25", "phints50", "oneshot"]

p = argparse.ArgumentParser()
p.add_argument("--equation", default="Poisson")
p.add_argument("--N", type=int, default=128)
p.add_argument("--solvers", default="jacobi,jacobi_0.67,gs,ssor,sor_1.5,mg")
p.add_argument("--n_test", type=int, default=64)
p.add_argument("--seed", type=int, default=72, help="development seed (the confirmatory instances use 73)")
p.add_argument("--max_ops", type=int, default=60000)
p.add_argument("--err_stop", type=float, default=1e-9)
p.add_argument("--tols", default="1e-2,1e-3,h2,1e-5,1e-6,1e-8")
p.add_argument("--b_vel", type=float, default=1.0)
p.add_argument("--ckp_dir", default="./checkpoints")
p.add_argument("--out_dir", default="./results")
args = p.parse_args()

torch.set_num_threads(1)
h2 = 1.0 / args.N ** 2
tols = [h2 if t == "h2" else float(t) for t in args.tols.split(",")]
pde = FastStencilPDE(args.N, equation=args.equation, b_vec=(args.b_vel, args.b_vel))
corrector = DeepONetCorrector(f"{args.ckp_dir}/deeponet_{args.equation}_{args.N}_best.pth", threads=1)
f_dev = GRF2D(args.N, rng=np.random.default_rng(args.seed)).sample(args.n_test)
u_dev = pde.solve_direct(f_dev)
costs_all = json.load(open(f"{args.ckp_dir}/costs_{args.equation}_{args.N}.json"))

out = {"args": vars(args), "tols": tols, "h2": h2, "schedules": SCHEDULES, "groups": {}}
for spec in args.solvers.split(","):
    env = Env(pde, [spec], corrector, costs=costs_all[spec])
    env.costs = costs_all[spec]
    env.set_macro_sizes(unit="no")
    _ = env.solvers[0].step(np.zeros_like(f_dev[:1]), f_dev[:1])
    wu = {s: {f"{t:.6g}": [] for t in tols} for s in SCHEDULES}
    for i in range(args.n_test):
        f1, u1 = f_dev[i:i + 1], u_dev[i:i + 1]
        for s in SCHEDULES:
            tr = run_untimed(env, f1, u1, s, max_ops=args.max_ops, err_stop=args.err_stop)
            tt = time_to_tol(tr, work_units(env, tr, s), tols)
            for t in tols:
                wu[s][f"{t:.6g}"].append(None if not np.isfinite(tt[t][0]) else float(tt[t][0]))
    selected = {}
    for t in tols:
        k = f"{t:.6g}"
        med = {s: float(np.median([v if v is not None else np.inf for v in wu[s][k]])) for s in SCHEDULES}
        selected[k] = min(med, key=med.get)
    out["groups"][spec] = {"work_units": wu, "selected": selected, "n_dev": args.n_test}
    print(f"{args.equation} N={args.N} {spec:12s} selected: " + ", ".join(f"{k}: {v}" for k, v in selected.items()), flush=True)
    del env
path = f"{args.out_dir}/schedule_dev_{args.equation}_{args.N}.json"
json.dump(out, open(path, "w"))
print("saved", path)
