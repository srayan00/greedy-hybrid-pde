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
import hashlib as _hl

import numpy as np
import torch

from fast_pde import FastStencilPDE, GRF2D, demean, l2
from corrector import DeepONetCorrector
from hybrid import Env, FeatureState, measure_costs, run_untimed, run_timed, time_to_tol, work_units, is_macro_policy
from baselines import make_baseline, default_baselines
from fast_pde import COMPILED
from router import Router, fit_router

parser = argparse.ArgumentParser()
parser.add_argument("--equation", default="Poisson", choices=["Poisson", "ConvDiff", "AnisoDiff", "VarCoeff"])
parser.add_argument("--N", type=int, default=128)
parser.add_argument("--solvers", default="jacobi,jacobi_0.67,gs,ssor,sor_1.5")
parser.add_argument("--ensemble", action="store_true")
parser.add_argument("--n_test", type=int, default=64)
parser.add_argument("--seed", type=int, default=72)
parser.add_argument("--b_vel", type=float, default=20.0)
parser.add_argument("--policies", default="classical,hints2,hints5,hints10,hints15,hints25,hints50,phints5,phints10,phints15,phints25,phints50,oneshot,decay0.9,decay0.95,decay0.98,greedy,oracle,router")
parser.add_argument("--baselines", default="auto", help="classical methods without corrector timed in the same replay loop (comma list, 'auto' = per-equation default, 'none')")
parser.add_argument("--max_iter_baseline", type=int, default=5000, help="iteration cap of the classical baselines (units of work)")
parser.add_argument("--retime_all", action="store_true", help="with --retime_only: re-time every instance, not only the drift-flagged ones")
parser.add_argument("--retime_only", action="store_true", help="re-time the drift-flagged instances of an existing result file (same routers and costs) and re-save it")
parser.add_argument("--drift_tol", type=float, default=0.15, help="drift guard: re-time a reference operation before every instance and wait while it is more than this fraction slower than the session's reference (10th percentile of the measurements so far)")
parser.add_argument("--drift_wait", type=float, default=120.0, help="drift guard: maximum wait per instance in the main pass (seconds); the end-of-cell re-timing pass waits up to 2.5x longer")
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
parser.add_argument("--unit_frac", type=float, default=1.0,
                    help="unit of cost as a fraction of one corrector call (1 = the corrector call; smaller = finer decisions, the corrector then scored per unit cost)")
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

base_names = [] if args.baselines == "none" else (default_baselines(args.equation) if args.baselines == "auto" else args.baselines.split(","))
if args.ensemble or args.train_only or args.measure_only:
    base_names = []
baselines = {m: make_baseline(pde, m, max_iter=args.max_iter_baseline, err_stop=args.err_stop) for m in base_names}


def ref_time(n=5):
    """Reference operation for the drift guard: one corrector call on the first test residual."""
    r0 = pde.residual(np.zeros_like(f_test[:1]), f_test[:1])
    ts = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        corrector.correct(r0)
        ts.append(time.perf_counter_ns() - t0)
    return float(np.median(ts))


ref_hist = []


def drift_reference():
    """The session's reference time: the 10th percentile of all reference measurements so far
    (robust to noise, unlike a running minimum, and tightening only as genuinely faster states
    are observed)."""
    return float(np.percentile(ref_hist, 10)) if len(ref_hist) >= 5 else float(np.median(ref_hist))


def drift_guard(tol, max_wait_s):
    """Waits (in 5 s steps, at most max_wait_s) while the reference operation is more than `tol`
    slower than the session's reference (a transient load); a faster machine is not waited for
    (paired comparisons are within an instance). Returns (ratio, retries, ref_cmp, ref_abs) in ns;
    every measurement enters the reference history."""
    retries = 0
    while True:
        ref_cmp = drift_reference()
        ref_abs = ref_time(9)
        ratio = ref_abs / ref_cmp
        if ratio <= 1.0 + tol or retries * 5 >= max_wait_s:
            ref_hist.append(ref_abs)
            return ratio, retries, ref_cmp, ref_abs
        retries += 1
        time.sleep(5)


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
    lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libstencil.so")
    lk = hashlib.sha256(open(lib, "rb").read()).hexdigest()[:16] if os.path.exists(lib) else None
    return {"git_commit": sh(["git", "rev-parse", "HEAD"]), "git_dirty": bool(sh(["git", "status", "--porcelain"])),
            "compiled_kernels": bool(COMPILED), "libstencil_sha256": lk,
            "numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__, "python": platform.python_version(),
            "platform": platform.platform(), "corrector_sha256": ck, "started": datetime.datetime.now().isoformat(timespec="seconds"),
            "run_id": uuid.uuid4().hex}


results = {"args": vars(args), "tols": tols, "h2": h2, "groups": {}, "provenance": provenance(ckp),
           "test_params": {k: v.tolist() for k, v in params.items()}, "baselines": base_names,
           "lu_factorization_s": (baselines["lu"].factor_s if "lu" in baselines else None)}

measured_all = False
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
        # one shared measurement per (equation, grid): every operation of this invocation is timed in
        # the same interleaved session (11 blocks of 100 repetitions), so that every pairing and
        # ensemble uses the same corrector cost and consistent macro-action sizes (the corrector's
        # 64 MB matrix--vector product is memory-bound and its cost is the noisiest of all)
        call = cached.get("__all__")
        if call is None or any(s_ not in call for s_ in group) or (args.remeasure_costs and not measured_all):
            env_all = Env(pde, specs, corrector)
            for j_ in range(len(specs)):
                _ = env_all.solvers[j_].step(np.zeros_like(f_test[:1]), f_test[:1])
            gc.collect()
            gc.disable()
            call = measure_costs(env_all, f_test, reps=100, blocks=11)
            gc.enable()
            cached["__all__"] = call
            measured_all = True
            print("  shared per-iteration costs: " + ", ".join(f"{k} {v*1e6:.0f}us" for k, v in call.items() if not isinstance(v, dict))
                  + f" | spread {call.get('_spread')}", flush=True)
        costs = {s_: call[s_] for s_ in group}
        costs.update({"no": call["no"], "_residual": call["_residual"],
                      "_min": {k: call["_min"][k] for k in list(group) + ["no", "_residual"]},
                      "_spread": {k: call["_spread"][k] for k in list(group) + ["no", "_residual"]}})
        cached[gkey] = costs
        json.dump(cached, open(cost_path, "w"), indent=1)
    env.costs = costs
    env.set_macro_sizes(unit="no" if args.unit_frac == 1.0 else args.unit_frac * costs["no"])
    print("  per-iteration costs: " + ", ".join(f"{k} {v*1e6:.0f}us" for k, v in costs.items() if not isinstance(v, dict))
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
    rpath = f"{args.ckp_dir}/router_{args.equation}_{args.N}_{gkey}{'_rate' if args.rate else ''}{args.router_tag}.pth"
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
    gres = {"costs": costs, "m": env.m, "ops": env.ops, "macro_exp": env.macro_exp, "router_decision_cost": t_dec,
            "router_sha256": (_hl.sha256(open(rpath, "rb").read()).hexdigest()[:16] if os.path.exists(rpath) else None),
            "policies": {p: [] for p in pol_list}, "curves": {p: [] for p in pol_list}, "drift": []}
    for bn in base_names:
        gres["policies"][f"base:{bn}"] = []
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
    bnames = [f"base:{bn}" for bn in base_names]
    all_names = names + bnames
    ref_hist.clear()
    for _ in range(5):
        ref_hist.append(ref_time(9))
    t_start = time.time()
    def time_instance(i, max_wait_s=None):
        """Untimed traces, drift guard and timed replays of test instance i; returns the result
        rows of every policy and baseline, the stored curves and the drift record."""
        f1, u1 = f_test[i:i + 1], u_truth[i:i + 1]
        traces = {}
        for p in names:
            env_, router_, base_, _ = runs[p]
            traces[p] = run_untimed(env_, f1, u1, base_, max_ops=args.max_ops, err_stop=args.err_stop,
                                    router=router_)
        for bn in base_names:
            traces[f"base:{bn}"] = baselines[bn].untimed(f1, u1)
        # drift guard: the reference operation must be within drift_tol of its start-of-session time
        ratio, retries, ref_cmp, ref_abs = drift_guard(args.drift_tol, args.drift_wait if max_wait_s is None else max_wait_s)
        try:
            load1 = float(os.getloadavg()[0])
        except (AttributeError, OSError):
            load1 = None
        drift_rec = {"instance": i, "ratio": ratio, "retries": retries, "loadavg": load1, "t_wall": time.time(),
                     "ref_us": ref_abs * 1e-3, "ref_cmp_us": ref_cmp * 1e-3, "ref0_us": drift_reference() * 1e-3}   # ns -> us
        rows, curves = {}, {}
        # timed replays: random order over policies and baselines per (instance, replay), and an
        # untimed warm-up (one corrector call and one sweep, or the baseline's own operation)
        # before every replay so that every method starts from the same cache state (the
        # corrector's matrix is larger than the CPU caches)
        gc.collect()
        gc.disable()
        times = {p: [] for p in all_names}
        outer = {p: [] for p in all_names}
        valid = {}
        dec_times = {p: [] for p in names}
        order_rng = np.random.default_rng(10_000 + i)
        for rep in range(args.timed_reps):
            for p in order_rng.permutation(all_names):
                if p.startswith("base:"):
                    bl = baselines[p[5:]]
                    bl.warm(f1)
                    t0o = time.perf_counter_ns()
                    t = bl.timed(f1, traces[p])
                    outer[p].append((time.perf_counter_ns() - t0o) * 1e-9)
                    times[p].append(t)
                    valid[p] = valid.get(p, True) and bool(getattr(bl, "last_ok", True))
                    continue
                env_, router_, base_, _ = runs[p]
                r0 = env_.pde.residual(np.zeros_like(f1), f1)
                if env_.corrector is not None:
                    env_.corrector.correct(r0)
                env_.solvers[0].step(np.zeros_like(f1), f1, r0)
                t0o = time.perf_counter_ns()
                t, u_end, tdec = run_timed(env_, f1, traces[p], base_, router=router_, return_decision_time=True)
                outer[p].append((time.perf_counter_ns() - t0o) * 1e-9)
                times[p].append(t)
                dec_times[p].append(tdec)
        gc.enable()
        for p in bnames:
            tr = traces[p]
            t_live = np.median(np.stack(times[p]), axis=0)
            tt_live = time_to_tol(tr, t_live, tols)
            e = tr["rel_err"]
            row = {"n_ops": int(len(tr["op"])), "final_rel_err": float(e[-1]), "t_total_live": float(t_live[-1]),
                   "t_outer_total": float(np.median(outer[p])), "valid": bool(valid.get(p, True)), "tol": {}}
            for tol in tols:
                tl, il = tt_live[tol]
                row["tol"][f"{tol:.6g}"] = {"iters": None if not np.isfinite(il) else int(il),
                                            "t_live": None if not np.isfinite(tl) else float(tl), "t_wu": None}
            rows[p] = row
        for p in names:
            env, router, base_, t_dec_ = runs[p]
            tr = traces[p]
            t_live = np.median(np.stack(times[p]), axis=0)
            t_dec_live = np.median(np.stack(dec_times[p]), axis=0)
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
            ops_ = np.asarray(tr["op"], dtype=int)
            if len(ops_):
                brk_ = np.flatnonzero(np.diff(ops_)) + 1
                starts_ = np.concatenate([[0], brk_]); ends_ = np.concatenate([brk_, [len(ops_)]])
                op_rle = [[int(ops_[s_]), int(e_ - s_)] for s_, e_ in zip(starts_, ends_)]
            else:
                op_rle = []
            row = {
                "op_rle": op_rle,      # run-length-encoded operation sequence (for decision-sequence agreement)
                "op_frac": op_frac, "op_frac_T": op_frac_T,
                "n_ops": int(len(tr["op"])), "n_no": int(tr["n_no"]),
                "final_rel_err": float(e[-1]),
                "auc_T": float(eT.sum()), "err_T": float(eT[-1]),
                "no_calls_T": int(no_cum[min(args.T, len(no_cum)) - 1]) if len(no_cum) else 0,
                "t_total_live": float(t_live[-1]), "t_total_wu": float(t_wu[-1]),
                "t_outer_total": float(np.median(outer[p])), "t_dec_total": float(t_dec_live[-1]),
                "n_epochs": int(len(tr["epochs"])),
                "tol": {},
            }
            starts = np.array([s for (s, _) in tr["epochs"]], dtype=int)
            for tol in tols:
                tl, il = tt_live[tol]
                tw, iw = tt_wu[tol]
                nno = int(no_cum[il - 1]) if np.isfinite(il) and il > 0 else (0 if np.isfinite(il) else None)
                row["tol"][f"{tol:.6g}"] = {"iters": None if not np.isfinite(il) else int(il),
                                            "t_live": None if not np.isfinite(tl) else float(tl),
                                            "t_wu": None if not np.isfinite(tw) else float(tw),
                                            "no_calls": nno,
                                            "t_dec": None if not np.isfinite(il) else float(t_dec_live[int(il)]),
                                            "n_dec": None if not np.isfinite(il) else int((starts < max(int(il), 1)).sum())}
            rows[p] = row
            if i < args.keep_curves:
                curves[p] = {"rel_err": e[:min(len(e), 5000)].tolist(),
                             "op": tr["op"][:5000].tolist(),
                             "t_live": t_live[:5001].tolist()}
        return rows, curves, drift_rec

    if args.retime_only:
        # re-time the instances of an existing result file that were timed under a machine slowdown
        out = f"{args.out_dir}/{args.equation}_{args.N}_{'ens_' if args.ensemble else ''}{gkey}{args.tag}.json"
        d_old = json.load(open(out))
        gres = d_old["groups"][gkey]
        if gres.get("router_sha256") != (_hl.sha256(open(rpath, "rb").read()).hexdigest()[:16] if os.path.exists(rpath) else None):
            raise RuntimeError("router checkpoint differs from the one of the stored results; cannot re-time")
        if abs(gres["costs"]["no"] - costs["no"]) > 1e-12:
            raise RuntimeError("cost cache differs from the one of the stored results; cannot re-time")
        results["provenance"] = d_old["provenance"]
        results["retime_provenance"] = provenance(ckp)
    else:
        for i in range(args.n_test):
            rows, curves, drift_rec = time_instance(i)
            gres["drift"].append(drift_rec)
            for p, row in rows.items():
                gres["policies"][p].append(row)
            for p, cv in curves.items():
                gres["curves"][p].append(cv)
            if (i + 1) % 8 == 0 or i == args.n_test - 1:
                msg = " | ".join(
                    f"{p}: {np.median([r['tol'][f'{h2:.6g}']['t_live'] or np.inf for r in gres['policies'][p]])*1e3:.1f}ms"
                    for p in all_names)
                dr = gres["drift"][-1]
                msg += f" | drift {dr['ratio']:.3f} ({dr['retries']} waits)"
                print(f"  [{i+1:3d}/{args.n_test}] median time-to-h2  {msg}   ({time.time()-t_start:.0f}s)", flush=True)
    # re-timing pass: instances whose drift guard gave up (reference still more than drift_tol slower
    # than at the start of the session) are timed again once the machine has calmed down (the guard
    # then waits up to 15 min); the first-pass ratio is kept in the record
    flagged = [dr["instance"] for dr in gres["drift"] if dr["ratio"] > 1.0 + args.drift_tol or args.retime_all]
    if flagged:
        print(f"  re-timing {len(flagged)} instance(s) timed under a machine slowdown: {flagged}", flush=True)
    for i in flagged:
        rows, curves, drift_rec = time_instance(i, max_wait_s=2.5 * args.drift_wait)
        rec = gres["drift"][i]
        rec.update({"retimed": True, "ratio_first": rec.get("ratio_first", rec["ratio"]), "retries_first": rec.get("retries_first", rec["retries"]),
                    "ref_us_first": rec.get("ref_us_first", rec.get("ref_us")), "ref0_us_first": rec.get("ref0_us_first", rec.get("ref0_us")),
                    "ratio": drift_rec["ratio"], "retries": drift_rec["retries"], "loadavg": drift_rec["loadavg"], "t_wall": drift_rec["t_wall"],
                    "ref_us": drift_rec.get("ref_us"), "ref_cmp_us": drift_rec.get("ref_cmp_us"), "ref0_us": drift_rec.get("ref0_us")})
        for p, row in rows.items():
            gres["policies"][p][i] = row
        for p, cv in curves.items():
            gres["curves"][p][i] = cv
        print(f"    instance {i}: reference ratio {rec['ratio_first']:.2f} -> {rec['ratio']:.2f} ({drift_rec['retries']} waits)", flush=True)
    for bn, bl in baselines.items():
        chk = getattr(bl, "err_end_check", None)
        if chk:
            worst = max(c[0] / max(c[1], 1e-300) for c in chk)
            gres[f"krylov_check:{bn}"] = {"n": len(chk), "max_ratio_timed_over_untimed_error": float(worst),
                                          "n_info_nonzero": int(sum(1 for c in chk if c[2] != 0)),
                                          "n_invalid": int(sum(1 for c in chk if len(c) > 3 and not c[3]))}
            print(f"  krylov cross-check {bn}: timed/untimed final error ratio <= {worst:.3g} on {len(chk)} runs", flush=True)
    results["groups"][gkey] = gres
    out = f"{args.out_dir}/{args.equation}_{args.N}_{'ens_' if args.ensemble else ''}{gkey}{args.tag}.json"
    with open(out, "w") as fh:
        json.dump(results if args.ensemble else {**results, "groups": {gkey: gres}}, fh)
    print(f"  saved {out}", flush=True)
