"""Diagnostic: is the lite router's failure an *information* bottleneck?

Collects error-greedy oracle decisions along MG/Jacobi rollouts, records
  (a) the current 7 scalar-trajectory features the LiteRouter uses, and
  (b) candidate cheap *structure-aware* features of the residual field,
then compares how well each feature set predicts the oracle decision
(grouped train/test split by instance, logistic regression + small MLP).

If (a) is near chance and (b) is well above it, the fix is richer features,
not a bigger router.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

from deeponet_corrector import DeepONetCorrector
from fast_pde import GRF2D, demean, l2, make_solver
from pde_factory import make_pde


BASE_NAMES = ["log_res", "dlog_res", "ema_dlog", "log_it", "prev_dec", "log_n_no", "log_since_no"]
STRUCT_NAMES = [
    "hf_frac",        # ||r - S r|| / ||r||, S = 5-pt average  (smoothness of residual)
    "rayleigh",       # <r, A r> / (<r,r> * diag)             (where in the spectrum)
    "inf_over_l2",    # localization
    "l1_over_l2",     # localization / sparsity
    "band_lo",        # Fourier energy in low band
    "band_mid",
    "band_hi",
    "last_no_gain",   # observed log10 residual gain of most recent NO step
    "last_cl_gain",   # observed log10 residual gain of most recent classical step
]


def smooth5(r):
    return 0.25 * (
        np.roll(r, 1, axis=-2) + np.roll(r, -1, axis=-2)
        + np.roll(r, 1, axis=-1) + np.roll(r, -1, axis=-1)
    )


def band_masks(N):
    k = np.fft.fftfreq(N) * N
    kk = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    kmax = kk.max()
    return [(kk <= kmax / 3), (kk > kmax / 3) & (kk <= 2 * kmax / 3), (kk > 2 * kmax / 3)]


def struct_features(pde, r, masks, last_no_gain, last_cl_gain):
    rn = float(l2(r)[0])
    if rn <= 0:
        return np.zeros(len(STRUCT_NAMES))
    hf = float(l2(r - smooth5(r))[0]) / rn
    Ar = pde.apply_A(r)
    rq = float((r * Ar).sum()) / (rn ** 2 * pde.diag)
    inf_l2 = float(np.abs(r).max()) / rn
    l1_l2 = float(np.abs(r).sum()) / (rn * r.size ** 0.5)
    rh = np.fft.fft2(r[0])
    e = np.abs(rh) ** 2
    tot = e.sum() + 1e-300
    bands = [float(e[m].sum() / tot) for m in masks]
    return np.array([hf, rq, inf_l2, l1_l2, bands[0], bands[1], bands[2],
                     last_no_gain, last_cl_gain])


def collect(equation, N, solver_spec, don_ckp, n_inst, max_iters, res_floor, seed, device):
    pde = make_pde(equation, N)
    solver = make_solver(pde, solver_spec)
    corr = DeepONetCorrector(don_ckp, device=device, threads=1)
    grf = GRF2D(N, rng=np.random.default_rng(seed))
    f_all = grf.sample(n_inst)
    u_all = demean(pde.solve_direct(f_all))
    masks = band_masks(N)

    Xb, Xs, y, gid = [], [], [], []
    for i in range(n_inst):
        f, ut = f_all[i:i + 1], u_all[i:i + 1]
        u = np.zeros_like(f)
        fn = float(l2(f)[0])
        un = float(l2(demean(ut))[0])
        prev_lr = None
        ema = 0.0
        n_no = 0
        since_no = 10 ** 4
        prev_dec = 0
        last_no_gain = 0.0
        last_cl_gain = 0.0
        for it in range(max_iters):
            r = pde.residual(u, f)
            rel_res = float(l2(r)[0]) / fn
            if rel_res <= res_floor:
                break
            lr = math.log10(rel_res) if rel_res > 1e-16 else -16.0
            dlr = 0.0 if prev_lr is None else max(-2.0, min(2.0, lr - prev_lr))
            ema = 0.7 * ema + 0.3 * dlr
            prev_lr = lr
            xb = np.array([lr / 8.0, dlr, ema, math.log1p(it) / 8.0, float(prev_dec),
                           math.log1p(n_no) / 3.0, math.log1p(min(since_no, 9999)) / 8.0])
            xs = struct_features(pde, r, masks, last_no_gain, last_cl_gain)

            # error-greedy oracle label
            u_c = solver.step(u, f, r)
            u_n = u + corr.correct(r)
            e_c = float(l2(demean(u_c - ut))[0])
            e_n = float(l2(demean(u_n - ut))[0])
            dec = int(e_n < e_c)

            Xb.append(xb); Xs.append(xs); y.append(dec); gid.append(i)

            u_next = u_n if dec else u_c
            new_res = float(l2(pde.residual(u_next, f))[0]) / fn
            gain = math.log10(max(rel_res, 1e-300) / max(new_res, 1e-300))
            if dec:
                last_no_gain = gain
                n_no += 1
                since_no = 0
            else:
                last_cl_gain = gain
                since_no += 1
            u = u_next
            prev_dec = dec
        _ = un
    return np.array(Xb), np.array(Xs), np.array(y), np.array(gid)


def _auc(y_true, score):
    """Rank-based ROC AUC with tie handling."""
    y_true = np.asarray(y_true)
    order = np.argsort(score, kind="mergesort")
    s = np.asarray(score)[order]
    yt = y_true[order]
    # average ranks over ties
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    n_pos = float(yt.sum())
    n_neg = float(len(yt) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[yt == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _fit_torch(Xtr, ytr, Xte, hidden=None, epochs=800, seed=0):
    import torch

    torch.manual_seed(seed)
    xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    xe = torch.tensor(Xte, dtype=torch.float32)
    d = xt.shape[1]
    if hidden is None:
        model = torch.nn.Linear(d, 1)
    else:
        model = torch.nn.Sequential(
            torch.nn.Linear(d, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(model(xt).squeeze(-1), yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return model(xe).squeeze(-1).numpy()


def eval_sets(Xb, Xs, y, gid, seed=0):
    rng = np.random.default_rng(seed)
    insts = np.unique(gid)
    insts = insts.copy()
    rng.shuffle(insts)
    n_tr = int(0.7 * len(insts))
    tr = np.isin(gid, insts[:n_tr])
    te = ~tr

    sets = {
        "base7 (current LiteRouter)": Xb,
        "struct9 (cheap residual structure)": Xs,
        "base7+struct9": np.hstack([Xb, Xs]),
    }
    out = {}
    base_rate = float(y[te].mean())
    for name, X in sets.items():
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        row = {}
        for tag, hid in (("logreg", None), ("mlp", 32)):
            sc = _fit_torch(Xtr, y[tr], Xte, hidden=hid, seed=seed)
            row[f"{tag}_auc"] = _auc(y[te], sc)
            row[f"{tag}_acc"] = float(((sc > 0).astype(int) == y[te]).mean())
        out[name] = row
    return out, base_rate, int(tr.sum()), int(te.sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--equation", default="Poisson")
    p.add_argument("--N", type=int, default=31)
    p.add_argument("--solver", default="mg")
    p.add_argument("--don_ckp", required=True)
    p.add_argument("--n_inst", type=int, default=96)
    p.add_argument("--max_iters", type=int, default=400)
    p.add_argument("--res_floor", type=float, default=1e-9)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    t0 = time.time()
    Xb, Xs, y, gid = collect(a.equation, a.N, a.solver, a.don_ckp, a.n_inst,
                             a.max_iters, a.res_floor, a.seed, a.device)
    print(f"collected {len(y)} steps over {a.n_inst} instances "
          f"({a.equation} N={a.N} solver={a.solver}) in {time.time()-t0:.0f}s")
    print(f"oracle NO rate = {y.mean():.3f}")

    # per-instance NO fraction spread = the diversity a router would need to match
    fr = np.array([y[gid == i].mean() for i in np.unique(gid)])
    print(f"oracle per-instance NO frac: mean {fr.mean():.3f} std {fr.std():.3f} "
          f"min {fr.min():.3f} max {fr.max():.3f}")

    res, base_rate, ntr, nte = eval_sets(Xb, Xs, y, gid, seed=a.seed)
    print(f"\ntest majority-class rate = {max(base_rate, 1-base_rate):.3f} "
          f"(train {ntr} / test {nte} steps)")
    print(f"\n{'feature set':>36}  {'logreg AUC':>10} {'MLP AUC':>8} {'MLP acc':>8}")
    for k, v in res.items():
        print(f"{k:>36}  {v['logreg_auc']:10.3f} {v['mlp_auc']:8.3f} {v['mlp_acc']:8.3f}")

    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"args": vars(a), "oracle_no_rate": float(y.mean()),
                       "per_inst_no_frac": fr.tolist(), "results": res,
                       "majority_rate": max(base_rate, 1 - base_rate)}, fh, indent=2)
        print("saved", a.out)


if __name__ == "__main__":
    main()
