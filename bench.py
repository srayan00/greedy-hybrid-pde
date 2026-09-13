"""Wall-clock + iteration-count benchmark of hybrid policies.

Pairwise mode (default): for every classical solver in --solvers, the ensemble
is {solver, corrector}; policies classical / hints<tau> / greedy / oracle /
router are compared.

Ensemble mode (--ensemble): one ensemble {all solvers, corrector}; policies
classical:<k> for every member, greedy, oracle, router.

Example:
  python bench.py --equation Poisson --N 128 --solvers jacobi,gs --n_test 64
"""

import argparse
import gc
import json
import os
import time

import numpy as np
import torch

from fast_pde import FastStencilPDE, GRF2D, demean, l2
from corrector import DeepONetCorrector
from hybrid import Env, FeatureState, measure_costs, run_untimed, run_timed, time_to_tol, work_units
from router import Router, fit_router

parser = argparse.ArgumentParser()
parser.add_argument("--equation", default="Poisson", choices=["Poisson", "ConvDiff", "AnisoDiff"])
parser.add_argument("--N", type=int, default=128)
parser.add_argument("--solvers", default="jacobi,jacobi_0.67,gs,ssor,sor_1.5")
parser.add_argument("--ensemble", action="store_true")
parser.add_argument("--n_test", type=int, default=64)
parser.add_argument("--seed", type=int, default=72)
parser.add_argument("--b_vel", type=float, default=20.0)
parser.add_argument("--policies", default="classical,hints25,hints5,hints10,hints50,greedy,oracle,router")
parser.add_argument("--tols", default="1e-2,1e-3,h2,1e-5,1e-6,1e-8")
parser.add_argument("--T", type=int, default=300, help="horizon for AUC / final error")
parser.add_argument("--max_ops", type=int, default=60000)
parser.add_argument("--err_stop", type=float, default=1e-9)
parser.add_argument("--ckp", default=None)
parser.add_argument("--ckp_dir", default="./checkpoints")
parser.add_argument("--router_tag", default="")
parser.add_argument("--retrain_router", action="store_true")
parser.add_argument("--router_inst", type=int, default=128)
parser.add_argument("--router_seed", type=int, default=555)
parser.add_argument("--dagger_rounds", type=int, default=2)
parser.add_argument("--router_max_epochs", type=int, default=3000)
parser.add_argument("--router_hidden", type=int, default=64)
parser.add_argument("--router_epochs", type=int, default=150)
parser.add_argument("--router_err_stop", type=float, default=1e-9)
parser.add_argument("--timed_reps", type=int, default=1)
parser.add_argument("--keep_curves", type=int, default=4, help="instances whose full error curves are stored")
parser.add_argument("--out_dir", default="./results")
parser.add_argument("--tag", default="")
parser.add_argument("--remeasure_costs", action="store_true")
parser.add_argument("--train_only", action="store_true", help="measure costs, train routers, exit")
parser.add_argument("--measure_only", action="store_true", help="measure and cache costs only")
parser.add_argument("--with_pairwise", action="store_true",
                    help="ensemble mode: also evaluate every member's pairwise router/oracle (same instances, same session)")
parser.add_argument("--rate", action="store_true",
                    help="per-iteration form of the cost-aware rule (policies rate / router_rate)")
args = parser.parse_args()

torch.set_num_threads(1)
os.makedirs(args.out_dir, exist_ok=True)
h2 = 1.0 / args.N ** 2
tols = [h2 if t == "h2" else float(t) for t in args.tols.split(",")]
specs = args.solvers.split(",")
policies = args.policies.split(",")

pde = FastStencilPDE(args.N, equation=args.equation, b_vec=(args.b_vel, args.b_vel))
ckp = args.ckp or f"{args.ckp_dir}/deeponet_{args.equation}_{args.N}_best.pth"
corrector = DeepONetCorrector(ckp, threads=1)
grf = GRF2D(args.N, rng=np.random.default_rng(args.seed))
f_test, params = grf.sample(args.n_test, return_params=True)
u_truth = pde.solve_direct(f_test)
_ = corrector.correct(f_test[:1])  # warm-up

groups = [specs] if args.ensemble else [[s] for s in specs]
def provenance(ckp_path):
    """git commit / dirty state, package versions, checkpoint hash, time and run id."""
    import hashlib, platform, subprocess, uuid, datetime, scipy
    def sh(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None
    ck = hashlib.sha256(open(ckp_path, "rb").read()).hexdigest()[:16] if os.path.exists(ckp_path) else None
    return {"git_commit": sh(["git", "rev-parse", "HEAD"]), "git_dirty": bool(sh(["git", "status", "--porcelain"])),
            "numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__, "python": platform.python_version(),
            "platform": platform.platform(), "corrector_sha256": ck, "started": datetime.datetime.now().isoformat(timespec="seconds"),
            "run_id": uuid.uuid4().hex}


results = {"args": vars(args), "tols": tols, "h2": h2, "groups": {}, "provenance": provenance(ckp),
           "test_params": {k: v.tolist() for k, v in params.items()}}

for group in groups:
    gkey = "+".join(group)
    print(f"\n=== {args.equation} N={args.N} ensemble {gkey} ===", flush=True)
    env = Env(pde, group, corrector)
    for j in range(len(group)):
        _ = env.solvers[j].step(np.zeros_like(f_test[:1]), f_test[:1])  # warm splu paths
    cost_path = f"{args.ckp_dir}/costs_{args.equation}_{args.N}.json"
    cached = json.load(open(cost_path)) if os.path.exists(cost_path) else {}
    if gkey in cached and not args.remeasure_costs:
        costs = cached[gkey]
        print(f"  using cached costs from {cost_path}")
    elif args.ensemble and all(s_ in cached for s_ in group) and not args.remeasure_costs:
        # assemble the ensemble's costs from the cached pairwise measurements so that the
        # macro-action sizes of every member are identical to those of its pairwise run
        costs = {s_: cached[s_][s_] for s_ in group}
        costs["no"] = float(np.mean([cached[s_]["no"] for s_ in group]))
        costs["_residual"] = float(np.mean([cached[s_]["_residual"] for s_ in group]))
        cached[gkey] = costs
        json.dump(cached, open(cost_path, "w"), indent=1)
        print(f"  ensemble costs assembled from the pairwise cache {cost_path}")
    else:
        gc.collect()
        gc.disable()
        costs = measure_costs(env, f_test)
        gc.enable()
        cached[gkey] = costs
        json.dump(cached, open(cost_path, "w"), indent=1)
    env.costs = costs
    env.set_macro_sizes()
    print("  per-iteration costs: " + ", ".join(f"{k} {v*1e6:.0f}us" for k, v in costs.items() if k != "_spread")
          + f" | spread {costs.get('_spread')} | macro sizes {dict(zip(env.ops, env.m))}", flush=True)
    if args.measure_only:
        continue

    pol_list = list(policies)
    if args.ensemble:
        # member-solver baselines are taken from the pairwise runs (same test
        # instances); HINTS does not apply to ensembles
        pol_list = [p for p in pol_list if p in ("greedy", "oracle", "router")]

    router = None
    t_dec = 0.0
    rpol = "router_rate" if args.rate else "router"
    if rpol in pol_list:
        rpath = f"{args.ckp_dir}/router_{args.equation}_{args.N}_{gkey}{'_rate' if args.rate else ''}{args.router_tag}.pth"
        if os.path.exists(rpath) and not args.retrain_router:
            router = Router.load(rpath)
            print(f"  loaded router {rpath}")
        else:
            torch.set_num_threads(8)
            t0 = time.time()
            router = fit_router(env, n_inst=args.router_inst, seed=args.router_seed,
                                dagger_rounds=args.dagger_rounds, rate=args.rate,
                                max_epochs=(args.max_ops if args.rate else args.router_max_epochs),
                                err_stop=args.router_err_stop, hidden=args.router_hidden,
                                epochs=args.router_epochs)
            router.save(rpath, meta={"costs": costs, "m": env.m, "ops": env.ops})
            torch.set_num_threads(1)
            print(f"  trained router in {time.time()-t0:.0f}s -> {rpath}", flush=True)
        # decision cost (features + inference)
        fs = FeatureState(env.K, env.no_index)
        reps = []
        for k in range(300):
            t0 = time.perf_counter_ns()
            x = fs.features(1e-3)
            d = router.decide(x)
            fs.update(d, 1 if args.rate else env.m[d])
            reps.append(time.perf_counter_ns() - t0)
        t_dec = float(np.median(reps)) * 1e-9
        print(f"  router decision cost {t_dec*1e6:.1f}us", flush=True)

    if args.train_only:
        continue
    gres = {"costs": costs, "m": env.m, "ops": env.ops, "router_decision_cost": t_dec,
            "policies": {p: [] for p in pol_list}, "curves": {p: [] for p in pol_list}}
    # same-session pairwise baselines for ensembles: every member's own router and oracle
    # (pairwise costs / routers from the pairwise runs), evaluated on the same instances
    runs = {p: (env, router, p, t_dec) for p in pol_list}   # name -> (env, router, base policy, decision cost)
    if args.ensemble and args.with_pairwise:
        for s_ in group:
            rp_ = f"{args.ckp_dir}/router_{args.equation}_{args.N}_{s_}.pth"
            if s_ not in cached or not os.path.exists(rp_):
                print(f"  (no pairwise router/costs for {s_}; skipped)", flush=True)
                continue
            env_s = Env(pde, [s_], corrector, costs=cached[s_])
            _ = env_s.solvers[0].step(np.zeros_like(f_test[:1]), f_test[:1])
            r_s = Router.load(rp_)
            fs_ = FeatureState(env_s.K, env_s.no_index)
            reps_ = []
            for k in range(300):
                t0 = time.perf_counter_ns()
                d = r_s.decide(fs_.features(1e-3))
                fs_.update(d, env_s.m[d])
                reps_.append(time.perf_counter_ns() - t0)
            runs[f"router@{s_}"] = (env_s, r_s, "router", float(np.median(reps_)) * 1e-9)
            runs[f"oracle@{s_}"] = (env_s, None, "oracle", 0.0)
            gres["policies"][f"router@{s_}"] = []; gres["policies"][f"oracle@{s_}"] = []
            gres["curves"][f"router@{s_}"] = []; gres["curves"][f"oracle@{s_}"] = []
            gres.setdefault("pairwise", {})[s_] = {"costs": cached[s_], "m": env_s.m, "ops": env_s.ops,
                                                    "router_decision_cost": runs[f"router@{s_}"][3]}
    names = list(runs.keys())
    t_start = time.time()
    for i in range(args.n_test):
        f1, u1 = f_test[i:i + 1], u_truth[i:i + 1]
        traces = {}
        for p in names:
            env_, router_, base_, _ = runs[p]
            traces[p] = run_untimed(env_, f1, u1, base_, max_ops=args.max_ops, err_stop=args.err_stop,
                                    router=router_)
        # timed replays, order rotated per instance
        gc.collect()
        gc.disable()
        times = {p: [] for p in names}
        for rep in range(args.timed_reps):
            order = names[(i + rep) % len(names):] + names[:(i + rep) % len(names)]
            for p in order:
                env_, router_, base_, _ = runs[p]
                t, u_end = run_timed(env_, f1, traces[p], base_, router=router_)
                times[p].append(t)
        gc.enable()
        for p in names:
            env, router, base_, t_dec_ = runs[p]
            tr = traces[p]
            t_live = np.median(np.stack(times[p]), axis=0)
            t_wu = work_units(env, tr, base_, router_cost=t_dec_)
            tt_live = time_to_tol(tr, t_live, tols)
            tt_wu = time_to_tol(tr, t_wu, tols)
            e = tr["rel_err"]
            eT = e[1:args.T + 1]
            if len(eT) < args.T:
                eT = np.concatenate([eT, np.full(args.T - len(eT), e[-1])])
            no_cum = np.cumsum(tr["op"] == env.no_index) if env.no_index is not None else np.zeros(len(tr["op"]))
            idx_h2 = np.flatnonzero(e <= h2)
            n_h2 = int(idx_h2[0]) if len(idx_h2) else len(tr["op"])
            ops_h2 = tr["op"][:max(n_h2, 1)]
            op_frac = [float((ops_h2 == k).mean()) for k in range(env.K)]
            ops_T = tr["op"][:args.T]
            op_frac_T = [float((ops_T == k).mean()) if len(ops_T) else 0.0 for k in range(env.K)]
            row = {
                "op_frac": op_frac, "op_frac_T": op_frac_T,
                "n_ops": int(len(tr["op"])), "n_no": int(tr["n_no"]),
                "final_rel_err": float(e[-1]),
                "auc_T": float(eT.sum()), "err_T": float(eT[-1]),
                "no_calls_T": int(no_cum[min(args.T, len(no_cum)) - 1]) if len(no_cum) else 0,
                "t_total_live": float(t_live[-1]), "t_total_wu": float(t_wu[-1]),
                "tol": {},
            }
            for tol in tols:
                tl, il = tt_live[tol]
                tw, iw = tt_wu[tol]
                nno = int(no_cum[il - 1]) if np.isfinite(il) and il > 0 else (0 if np.isfinite(il) else None)
                row["tol"][f"{tol:.6g}"] = {"iters": None if not np.isfinite(il) else int(il),
                                            "t_live": None if not np.isfinite(tl) else float(tl),
                                            "t_wu": None if not np.isfinite(tw) else float(tw),
                                            "no_calls": nno}
            gres["policies"][p].append(row)
            if i < args.keep_curves:
                gres["curves"][p].append({"rel_err": e[:min(len(e), 5000)].tolist(),
                                          "op": tr["op"][:5000].tolist(),
                                          "t_live": t_live[:5001].tolist()})
        env, router = runs[pol_list[0]][0], runs[pol_list[0]][1]
        if (i + 1) % 8 == 0 or i == args.n_test - 1:
            msg = " | ".join(
                f"{p}: {np.median([r['tol'][f'{h2:.6g}']['t_live'] or np.inf for r in gres['policies'][p]])*1e3:.1f}ms"
                for p in names)
            print(f"  [{i+1:3d}/{args.n_test}] median time-to-h2  {msg}   ({time.time()-t_start:.0f}s)", flush=True)
    results["groups"][gkey] = gres
    out = f"{args.out_dir}/{args.equation}_{args.N}_{'ens_' if args.ensemble else ''}{gkey}{args.tag}.json"
    with open(out, "w") as fh:
        json.dump(results if args.ensemble else {**results, "groups": {gkey: gres}}, fh)
    print(f"  saved {out}", flush=True)
