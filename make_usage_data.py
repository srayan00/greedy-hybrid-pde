"""Untimed decision traces of router / cost-aware oracle / HINTS on all test
instances (for the corrector-usage and decision-raster figures). Uses the
same test seed, corrector, routers and cached costs as bench.py.

  python make_usage_data.py --equation Poisson --N 128
writes results/usage_<eq>_<N>.json with, per pairing and policy, the first
T operations of every test instance.
"""

import argparse
import json
import os

import numpy as np
import torch

from fast_pde import FastStencilPDE, GRF2D
from corrector import DeepONetCorrector
from hybrid import Env, run_untimed
from router import Router

p = argparse.ArgumentParser()
p.add_argument("--equation", default="Poisson")
p.add_argument("--N", type=int, default=128)
p.add_argument("--solvers", default="jacobi,jacobi_0.67,gs,ssor,sor_1.5")
p.add_argument("--n_test", type=int, default=64)
p.add_argument("--seed", type=int, default=72)
p.add_argument("--T", type=int, default=300)
p.add_argument("--policies", default="router,oracle,hints15,hints25,hints5")
p.add_argument("--ckp_dir", default="./checkpoints")
p.add_argument("--out_dir", default="./results")
args = p.parse_args()

torch.set_num_threads(4)
pde = FastStencilPDE(args.N, equation=args.equation)
corrector = DeepONetCorrector(f"{args.ckp_dir}/deeponet_{args.equation}_{args.N}_best.pth", threads=4)
f_test = GRF2D(args.N, rng=np.random.default_rng(args.seed)).sample(args.n_test)
u_truth = pde.solve_direct(f_test)
costs = json.load(open(f"{args.ckp_dir}/costs_{args.equation}_{args.N}.json"))
out = {"args": vars(args), "groups": {}}
for spec in args.solvers.split(","):
    env = Env(pde, [spec], corrector, costs=costs[spec])
    router = Router.load(f"{args.ckp_dir}/router_{args.equation}_{args.N}_{spec}.pth")
    g = {"ops": env.ops, "m": env.m, "policies": {}}
    for pol in args.policies.split(","):
        seqs = []
        for i in range(args.n_test):
            tr = run_untimed(env, f_test[i:i + 1], u_truth[i:i + 1], pol, max_ops=args.T,
                             err_stop=1e-9, router=router)
            seqs.append(tr["op"].tolist())
        g["policies"][pol] = seqs
        print(f"  {spec} {pol}: mean corrector calls in first {args.T} its = "
              f"{np.mean([sum(1 for o in s_ if o == env.no_index) for s_ in seqs]):.2f}", flush=True)
    out["groups"][spec] = g
os.makedirs(args.out_dir, exist_ok=True)
path = f"{args.out_dir}/usage_{args.equation}_{args.N}.json"
json.dump(out, open(path, "w"))
print("saved", path)
