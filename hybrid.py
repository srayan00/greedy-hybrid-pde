"""Cost-aware hybrid solver rollouts (single instance) with wall-clock accounting.

Action set. The ensemble is a list of *operations*: classical solvers
(jacobi, jacobi_0.67, gs, ssor, sor_1.5) and the neural corrector ("no").
Every operation has a measured per-iteration cost c_j (residual evaluation +
the update itself). The cost-aware greedy rule of the paper is Algorithm 1
applied to *cost-equalised macro-actions*: operation j is always applied
m_j = round(c_max / c_j) times in a row, so every macro-action costs
approximately the same and "minimise the error after one macro-action" is
"minimise the error per unit of wall-clock time". m_j consecutive stationary
sweeps are themselves a stationary iteration (preconditioner
sum_{i<m}(I - C L)^i C), so the theory applies verbatim.

Policies
  classical[:spec]  - always the classical solver (single-solver runs)
  hints<tau>        - the corrector every tau-th iteration (HINTS)
  greedy            - paper's Alg. 1 on single operations (cost-agnostic oracle)
  oracle            - Alg. 1 on macro-actions (cost-aware oracle, true error)
  router            - learned router on macro-actions (deployable)

Accounting
  * every executed iteration is charged (residual + update); the learned
    router additionally pays its feature/decision cost at every decision
    epoch; oracles are idealised (decisions not charged, candidate
    evaluations not charged);
  * the decision trace is produced by an untimed pass (which also records the
    true error after every iteration, benchmark-only), then replayed in a
    timed pass that executes only the chosen operations (the router re-decides
    live in the timed pass and its decisions are checked against the untimed
    trace).
"""

import math
import time

import numpy as np

from fast_pde import FastStencilPDE, demean, l2, make_solver


class Env:
    """PDE + operations + costs + macro-action sizes."""

    def __init__(self, pde: FastStencilPDE, solver_specs, corrector, costs=None, unit="no"):
        self.pde = pde
        self.N = pde.N
        self.specs = list(solver_specs)
        self.solvers = [make_solver(pde, s) for s in self.specs]
        self.corrector = corrector
        self.ops = self.specs + (["no"] if corrector is not None else [])
        self.K = len(self.ops)
        self.no_index = self.K - 1 if corrector is not None else None
        self.costs = costs  # per-iteration costs, dict op -> seconds
        self.m = None
        if costs is not None:
            self.set_macro_sizes(unit)

    def set_macro_sizes(self, unit="no"):
        """Cost-equalised macro-actions. The unit of cost is one corrector call
        (unit="no"; "max" uses the most expensive operation). Operation j is
        applied m_j = max(1, round(u / c_j)) times per decision, which
        equalises costs up to rounding for operations cheaper than the unit;
        an operation dearer than the unit (e.g. a multigrid cycle) is applied
        once and its error reduction is compared per unit of cost, i.e. the
        oracle compares ||e_j||^(u / (m_j c_j)) (exponent 1 when costs match)."""
        c = np.array([self.costs[o] for o in self.ops])
        assert np.all(c > 0)
        if unit == "max":
            cu = c.max()
        elif unit == "no":
            cu = c[self.no_index] if self.no_index is not None else c.max()
        else:
            cu = float(unit)
        self.m = [max(1, int(round(cu / cj))) for cj in c]
        self.unit_cost = cu
        mc = np.array(self.m) * c
        self.macro_exp = (cu / mc).tolist()          # per-macro-action exponent
        # exponent used by the per-iteration (rate) form of the rule
        self.rate_exp = (cu / c).tolist()

    def macro_score(self, e0, errs):
        """Cost-normalised errors after each macro-action: exp(exp_j * log(e_j/e0)).
        argmin gives the cost-aware greedy decision (Alg. 1 when all
        exponents are 1)."""
        e0 = max(e0, 1e-300)
        return [self.macro_exp[k] * math.log(max(errs[k], 1e-300) / e0) for k in range(self.K)]

    # -- single operation ----------------------------------------------------
    def apply_op(self, j, u, f, r):
        """One iteration of operation j given the current residual r."""
        if j == self.no_index:
            return u + self.corrector.correct(r)
        return self.solvers[j].step(u, f, r)

    def apply_macro(self, j, u, f, r):
        """m_j iterations of operation j. Returns (u, last residual computed)."""
        for i in range(self.m[j]):
            if i > 0:
                r = self.pde.residual(u, f)
            u = self.apply_op(j, u, f, r)
        return u


def measure_costs(env: Env, f, reps=60, warm=10, blocks=7):
    """Per-iteration cost (residual + update) of every operation, measured in
    isolation on one instance, single thread. Each operation is timed in
    `blocks` separate blocks of `reps` repetitions (interleaved across
    operations to balance drift) and the minimum over blocks of the block
    median is used, which is robust to transient slow-downs of the machine.
    Returns dict op -> seconds and the residual cost."""
    u = np.zeros_like(f[:1])
    ff = f[:1]
    r = env.pde.residual(u, ff)

    def block(fn):
        for _ in range(warm):
            fn()
        ts = np.empty(reps)
        for k in range(reps):
            t0 = time.perf_counter_ns()
            fn()
            ts[k] = time.perf_counter_ns() - t0
        return float(np.median(ts)) * 1e-9

    fns = {"_residual": (lambda: env.pde.residual(u, ff))}
    for j, op in enumerate(env.ops):
        fns[op] = (lambda j=j: env.apply_op(j, u, ff, r))
    meds = {k: [] for k in fns}
    for b in range(blocks):
        for k, fn in fns.items():
            meds[k].append(block(fn))
    out = {k: float(min(v)) for k, v in meds.items()}
    t_res = out["_residual"]
    for op in env.ops:
        out[op] = out[op] + t_res
    out["_spread"] = {k: float(max(v) / min(v)) for k, v in meds.items()}
    return out


# ---------------------------------------------------------------------------
# Router feature state (shared by rollouts and training data collection)
# ---------------------------------------------------------------------------

EMA = 0.7


def n_features(K):
    return 6 + K


class FeatureState:
    """Scalar, observable router features at decision epochs.

      0  log10 relative residual / 8
      1  per-iteration change of log10 residual over the last macro-action
      2  EMA of (1)
      3  log(1 + epoch) / 8
      4..4+K-1  one-hot of the previous macro-action (zeros at the first epoch)
      4+K  log(1 + number of corrector calls so far) / 3
      5+K  log(1 + iterations since the last corrector call) / 8
    All are O(1) given the residual norm that the stopping test computes.
    """

    def __init__(self, K, no_index):
        self.K, self.no_index = K, no_index
        self.reset()

    def reset(self):
        self.prev_log = None
        self.prev_ops = 1
        self.ema = 0.0
        self.prev_action = -1
        self.n_no = 0
        self.since_no = 10 ** 4
        self.epoch = 0
        self.x = np.zeros(n_features(self.K))

    def features(self, rel_res):
        lr = math.log10(rel_res) if rel_res > 1e-300 else -300.0
        if self.prev_log is None:
            dlr = 0.0
        else:
            dlr = max(-2.0, min(2.0, (lr - self.prev_log) / self.prev_ops))
        self.ema = EMA * self.ema + (1 - EMA) * dlr
        self.prev_log = lr
        x = self.x
        x[:] = 0.0
        x[0] = lr / 8.0
        x[1] = dlr
        x[2] = self.ema
        x[3] = math.log1p(self.epoch) / 8.0
        if self.prev_action >= 0:
            x[4 + self.prev_action] = 1.0
        x[4 + self.K] = math.log1p(self.n_no) / 3.0
        x[5 + self.K] = math.log1p(min(self.since_no, 9999)) / 8.0
        return x

    def update(self, action, n_ops):
        self.prev_action = action
        self.prev_ops = n_ops
        self.epoch += 1
        if action == self.no_index:
            self.n_no += n_ops
            self.since_no = 0
        else:
            self.since_no += n_ops


# ---------------------------------------------------------------------------
# Untimed rollout: decision trace + per-iteration true errors
# ---------------------------------------------------------------------------

def run_untimed(env: Env, f, u_truth, policy, max_ops=100000, err_stop=1e-9,
                res_floor=1e-13, router=None, single=None, explore=None, rng=None):
    """f, u_truth: (1, N, N). Returns dict with
        rel_err (n_ops+1,), rel_res (n_ops+1,)  -- state before/after each op
        op      (n_ops,)   operation index executed at each iteration
        epochs  list of (op_start_index, macro_action)   decision epochs
        n_no    number of corrector iterations
    single: for 'classical', the solver index to use (default 0).
    explore: exploration probability for oracle/greedy (training only).
    """
    pde = env.pde
    u = np.zeros_like(f)
    fn = max(float(l2(f)[0]), 1e-300)
    un = max(float(l2(demean(u_truth))[0]), 1e-300)
    rel_err, rel_res, ops, epochs = [], [], [], []
    hints_tau = int(policy[5:]) if policy.startswith("hints") else None
    if policy.startswith("classical"):
        single = int(policy.split(":")[1]) if ":" in policy else (single or 0)
    fs = FeatureState(env.K, env.no_index) if policy in ("router", "router_rate") else None
    if router is not None and policy in ("router", "router_rate"):
        router.reset()

    def record(r):
        rel_res.append(float(l2(r)[0]) / fn)
        rel_err.append(float(l2(demean(u - u_truth))[0]) / un)

    r = pde.residual(u, f)
    record(r)
    it = 0
    while it < max_ops and rel_err[-1] > err_stop and rel_res[-1] > res_floor:
        # ---------------- choose a (macro-)action
        if policy.startswith("classical"):
            j, m = single, 1
        elif hints_tau is not None:
            j = env.no_index if (it + 1) % hints_tau == 0 else 0
            m = 1
        elif policy == "oneshot":
            j, m = (env.no_index if it == 0 else 0), 1
        elif policy.startswith("phints"):          # phase-shifted HINTS: first call at t = 0
            j, m = (env.no_index if it % int(policy[6:]) == 0 else 0), 1
        elif policy == "greedy":
            errs = [float(l2(demean(env.apply_op(k, u, f, r) - u_truth))[0]) for k in range(env.K)]
            j, m = int(np.argmin(errs)), 1
            if explore is not None and rng.random() < explore:
                j = int(rng.integers(env.K))
        elif policy == "oracle":
            errs = [float(l2(demean(env.apply_macro(k, u, f, r) - u_truth))[0]) for k in range(env.K)]
            j = int(np.argmin(env.macro_score(rel_err[-1] * un, errs)))
            m = env.m[j]
            if explore is not None and rng.random() < explore:
                j = int(rng.integers(env.K))
                m = env.m[j]
        elif policy == "router":
            x = fs.features(rel_res[-1])
            j = router.decide(x)
            m = env.m[j]
        elif policy == "rate":
            # cost-aware rule evaluated every iteration: argmin_j ||e_j||^(c_max/c_j)
            errs = [float(l2(demean(env.apply_op(k, u, f, r) - u_truth))[0]) for k in range(env.K)]
            e0 = max(rel_err[-1] * un, 1e-300)
            score = [env.rate_exp[k] * math.log(max(errs[k], 1e-300) / e0) for k in range(env.K)]
            j, m = int(np.argmin(score)), 1
            if explore is not None and rng.random() < explore:
                j = int(rng.integers(env.K))
        elif policy == "router_rate":
            x = fs.features(rel_res[-1])
            j, m = router.decide(x), 1
        else:
            raise ValueError(policy)
        epochs.append((it, j))
        # ---------------- execute m iterations of op j, recording every state
        for i in range(m):
            # r always holds the residual of the current iterate
            u = env.apply_op(j, u, f, r)
            ops.append(j)
            it += 1
            r = pde.residual(u, f)
            record(r)
            if rel_err[-1] <= err_stop or rel_res[-1] <= res_floor or it >= max_ops:
                break
        if fs is not None:
            fs.update(j, i + 1)
    ops = np.asarray(ops, dtype=np.int16)
    return {"rel_err": np.asarray(rel_err), "rel_res": np.asarray(rel_res), "op": ops,
            "epochs": epochs, "n_no": int((ops == env.no_index).sum()) if env.no_index is not None else 0,
            "final_u": u}


# ---------------------------------------------------------------------------
# Timed replay: executes only the chosen operations, charging everything the
# deployed method would pay
# ---------------------------------------------------------------------------

def run_timed(env: Env, f, trace, policy, router=None):
    """Replays trace['epochs'] (macro-actions) and returns cumulative charged
    time after every iteration (n_ops+1 entries, t[0] = 0 + first residual).
    For policy == 'router' the router is executed live (its decisions are
    verified against the trace) so its feature and inference costs are paid."""
    pde = env.pde
    u = np.zeros_like(f)
    fn = max(float(l2(f)[0]), 1e-300)
    n_ops = len(trace["op"])
    t = np.empty(n_ops + 1)
    fs = FeatureState(env.K, env.no_index) if policy in ("router", "router_rate") else None
    if fs is not None:
        router.reset()
    epochs = trace["epochs"]
    t_cum = 0
    it = 0
    t0 = time.perf_counter_ns()
    r = pde.residual(u, f)
    rr = float(l2(r)[0]) / fn          # stopping test (charged, all methods)
    t_cum += time.perf_counter_ns() - t0
    t[0] = t_cum * 1e-9
    for (start, j) in epochs:
        m_planned = env.m[j] if policy in ("oracle", "router") else 1
        if fs is not None:
            t0 = time.perf_counter_ns()
            x = fs.features(rr)
            jj = router.decide(x)
            t_cum += time.perf_counter_ns() - t0
            if jj != j:
                raise RuntimeError(f"router replay mismatch at op {it}: {jj} vs {j}")
        n_exec = 0
        for i in range(m_planned):
            if it >= n_ops:
                break
            t0 = time.perf_counter_ns()
            u = env.apply_op(j, u, f, r)      # r is the residual of the current iterate
            r = pde.residual(u, f)           # residual of the new iterate (stopping test)
            rr = float(l2(r)[0]) / fn
            t_cum += time.perf_counter_ns() - t0
            it += 1
            n_exec += 1
            t[it] = t_cum * 1e-9
        if fs is not None:
            t0 = time.perf_counter_ns()
            fs.update(j, n_exec)
            t_cum += time.perf_counter_ns() - t0
    assert it == n_ops, (it, n_ops)
    return t, u


def time_to_tol(trace, t, tols, key="rel_err"):
    """First charged time (and iteration) at which trace[key] <= tol."""
    v = trace[key]
    out = {}
    for tol in tols:
        idx = np.flatnonzero(v <= tol)
        out[tol] = (float(t[idx[0]]), int(idx[0])) if len(idx) else (math.inf, math.inf)
    return out


def work_units(env: Env, trace, policy, router_cost=0.0):
    """Timer-free cost estimate: sum of measured per-iteration costs of the
    executed operations (+ decision cost per epoch for the router)."""
    c = np.array([env.costs[o] for o in env.ops])
    per_op = c[trace["op"]]
    t = np.concatenate([[env.costs["_residual"]], env.costs["_residual"] + np.cumsum(per_op)])
    if policy in ("router", "router_rate") and router_cost > 0:
        # add the decision cost at each epoch start
        add = np.zeros_like(t)
        for (start, j) in trace["epochs"]:
            add[start:] += router_cost
        t = t + add
    return t
