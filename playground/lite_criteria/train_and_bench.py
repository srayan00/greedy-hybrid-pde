"""Fork DeepONet corrector + ApproxGreedy lightweight scalar router + wall-clock bench.

Corrector: same scale-equivariant DeepONet as wallclock-rebuttal
(`train_fast_deeponet.py` + `finetune_deeponet_residual.py`).

Router features mirror the wallclock-rebuttal lite router (O(1) in N). Training
uses the paper's ApproxGreedyRouterLoss soft weights:

  w_j = (sum_k e_k) - e_j
  L = sum_j w_j * (-log softmax(logits)_j)

applied online with scheduled sampling, Jacobi + DeepONet as the two experts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from deeponet_corrector import DeepONetCorrector
from fast_pde import GRF2D, demean, l2, make_solver
from pde_factory import ckp_tag, make_pde


BASE_DIM = 7
STRUCT_DIM = 9
FEAT_DIMS = {"base7": BASE_DIM, "struct": STRUCT_DIM, "both": BASE_DIM + STRUCT_DIM}
EMA = 0.7


# ---------------------------------------------------------------------------
# Cheap structure-aware residual features
#
# Motivation: the original 7 features are all functions of the scalar residual
# *norm* trajectory, so two instances at the same point on the convergence curve
# are indistinguishable. These add residual *shape* at the cost of one stencil
# apply + one FFT + a few reductions (no LSTM over full fields).
# ---------------------------------------------------------------------------

def band_masks(N: int) -> np.ndarray:
    k = np.fft.fftfreq(N) * N
    kk = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    kmax = kk.max()
    return np.stack([kk <= kmax / 3,
                     (kk > kmax / 3) & (kk <= 2 * kmax / 3),
                     kk > 2 * kmax / 3])


def struct_features_batch(pde, r, masks, last_no_gain, last_cl_gain) -> np.ndarray:
    """r: (B, N, N) -> (B, STRUCT_DIM). last_*_gain: (B,) observed log10 gains."""
    rn = l2(r)
    safe = np.maximum(rn, 1e-300)
    Ar = pde.apply_A(r)
    hf = l2(Ar) / (pde.diag * safe)                              # high-freq content
    rq = (r * Ar).sum(axis=(-2, -1)) / (pde.diag * safe ** 2)    # Rayleigh quotient
    inf_l2 = np.abs(r).max(axis=(-2, -1)) / safe                 # localization
    npts = r.shape[-1] * r.shape[-2]
    l1_l2 = np.abs(r).sum(axis=(-2, -1)) / (safe * math.sqrt(npts))
    e = np.abs(np.fft.fft2(r, axes=(-2, -1))) ** 2
    tot = e.sum(axis=(-2, -1)) + 1e-300
    bands = [(e * m).sum(axis=(-2, -1)) / tot for m in masks]
    return np.stack([hf, rq, inf_l2, l1_l2, bands[0], bands[1], bands[2],
                     last_no_gain, last_cl_gain], axis=-1)


# ---------------------------------------------------------------------------
# Lite router (numpy inference)
# ---------------------------------------------------------------------------

class _LiteRouterBase:
    """Shared feature assembly / per-instance state for numpy routers."""

    arch = "mlp"

    def __init__(self, feat_mode: str = "base7", mu=None, sd=None):
        self.feat_mode = feat_mode
        self.dim = FEAT_DIMS[feat_mode]
        self.mu = np.zeros(self.dim) if mu is None else np.asarray(mu, dtype=np.float64)
        self.sd = np.ones(self.dim) if sd is None else np.asarray(sd, dtype=np.float64)
        self._masks = None
        self.reset()

    def reset(self):
        self._prev_log_res = None
        self._ema = 0.0
        self._n_no = 0
        self._since_no = 10 ** 4
        self._last_no_gain = 0.0
        self._last_cl_gain = 0.0
        self._prev_rel = None
        self._last_dec = None

    def _base(self, rel_res: float, it: int, prev_decision: int) -> np.ndarray:
        rr = float(rel_res)
        if not math.isfinite(rr) or rr <= 1e-16:
            lr = -16.0
        else:
            lr = math.log10(rr)
        dlr = 0.0 if self._prev_log_res is None else max(-2.0, min(2.0, lr - self._prev_log_res))
        self._ema = EMA * self._ema + (1 - EMA) * dlr
        self._prev_log_res = lr
        return np.array([
            lr / 8.0,
            dlr,
            self._ema,
            math.log1p(it) / 8.0,
            float(prev_decision),
            math.log1p(self._n_no) / 3.0,
            math.log1p(min(self._since_no, 9999)) / 8.0,
        ])

    def _features(self, rel_res, it, prev_decision, r, pde) -> np.ndarray:
        # Guard against NaN/negative/overflow residuals (bad NO steps under
        # strong anisotropy can blow up the field).
        rr = float(rel_res)
        if not math.isfinite(rr) or rr < 0.0:
            rr = 0.0
        if self._prev_rel is not None and self._last_dec is not None:
            prev = float(self._prev_rel)
            if not math.isfinite(prev) or prev <= 0.0:
                prev = 1e-300
            cur = rr if rr > 0.0 else 1e-300
            ratio = prev / cur
            if (not math.isfinite(ratio)) or ratio <= 0.0:
                g = 0.0
            else:
                g = math.log10(ratio)
            g = max(-2.0, min(4.0, g))
            if self._last_dec:
                self._last_no_gain = g
            else:
                self._last_cl_gain = g
        self._prev_rel = rr

        parts = []
        if self.feat_mode in ("base7", "both"):
            parts.append(self._base(rr, it, prev_decision))
        if self.feat_mode in ("struct", "both"):
            if self._masks is None:
                self._masks = band_masks(pde.N)
            parts.append(struct_features_batch(
                pde, r, self._masks,
                np.array([self._last_no_gain]), np.array([self._last_cl_gain]))[0])
        return (np.concatenate(parts) - self.mu) / self.sd

    def _logits(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def decide(self, rel_res: float, it: int, prev_decision: int,
               r=None, pde=None) -> int:
        x = self._features(rel_res, it, prev_decision, r, pde)
        o = self._logits(x)
        d = int(o[1] > o[0])
        if d:
            self._n_no += 1
            self._since_no = 0
        else:
            self._since_no += 1
        self._last_dec = d
        return d

    def save(self, path: str, *, epoch: Optional[int] = None):
        blob = {
            "weights": [torch.tensor(w) for w in self.weights],
            "feat_mode": self.feat_mode,
            "arch": self.arch,
            "mu": torch.tensor(self.mu),
            "sd": torch.tensor(self.sd),
        }
        if epoch is not None:
            blob["epoch"] = int(epoch)
        torch.save(blob, path)


class LiteRouter(_LiteRouterBase):
    """feat_dim -> hidden -> hidden -> 2, numpy forward for cheap decisions."""

    arch = "mlp"

    def __init__(self, weights: List[np.ndarray], feat_mode: str = "base7",
                 mu=None, sd=None):
        self.weights = [np.ascontiguousarray(w, dtype=np.float64) for w in weights]
        super().__init__(feat_mode=feat_mode, mu=mu, sd=sd)

    def _logits(self, x):
        W1, b1, W2, b2, W3, b3 = self.weights
        h = np.maximum(x @ W1 + b1, 0.0)
        h = np.maximum(h @ W2 + b2, 0.0)
        return h @ W3 + b3

    @classmethod
    def load(cls, path: str) -> "LiteRouter":
        return load_router(path)


try:
    from scipy.special import expit as _sigmoid
except ImportError:  # pragma: no cover
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))


class LiteGRURouter(_LiteRouterBase):
    """GRUCell(feat_dim -> H) + linear head, numpy forward.

    Adds memory over the residual trajectory that the memoryless MLP lacks,
    at a per-decision cost comparable to the MLP (a few thousand flops).
    """

    arch = "gru"

    def __init__(self, weights: List[np.ndarray], feat_mode: str = "base7",
                 mu=None, sd=None):
        self.weights = [np.ascontiguousarray(w, dtype=np.float64) for w in weights]
        self.W_ih, self.W_hh, self.b_ih, self.b_hh, self.Wo, self.bo = self.weights
        self.H = self.W_hh.shape[1]
        super().__init__(feat_mode=feat_mode, mu=mu, sd=sd)

    def reset(self):
        super().reset()
        self.h = np.zeros(getattr(self, "H", 0))

    def _logits(self, x):
        # kept in as few numpy calls as possible: at this size (H=16) per-call
        # dispatch overhead dominates the actual arithmetic
        H, H2 = self.H, 2 * self.H
        gi = self.W_ih @ x
        gi += self.b_ih
        gh = self.W_hh @ self.h
        gh += self.b_hh
        rz = _sigmoid(gi[:H2] + gh[:H2])
        n = np.tanh(gi[H2:] + rz[:H] * gh[H2:])
        h = self.h - n            # h' = n + z * (h - n)
        h *= rz[H:]
        h += n
        self.h = h
        return h @ self.Wo + self.bo


def load_router(path: str):
    ckp = torch.load(path, map_location="cpu", weights_only=False)
    mu = ckp.get("mu")
    sd = ckp.get("sd")
    cls = LiteGRURouter if ckp.get("arch", "mlp") == "gru" else LiteRouter
    return cls(
        [w.numpy() for w in ckp["weights"]],
        feat_mode=ckp.get("feat_mode", "base7"),
        mu=None if mu is None else mu.numpy(),
        sd=None if sd is None else sd.numpy(),
    )


class TorchLiteMLP(torch.nn.Module):
    recurrent = False

    def __init__(self, feat_dim: int = BASE_DIM, hidden: int = 32):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 2),
        )

    def forward(self, x, h=None):
        return self.net(x), None

    def export_weights(self) -> List[np.ndarray]:
        ws = []
        for m in self.net:
            if isinstance(m, torch.nn.Linear):
                ws += [m.weight.detach().cpu().numpy().T.copy(), m.bias.detach().cpu().numpy().copy()]
        return ws


class TorchLiteGRU(torch.nn.Module):
    recurrent = True

    def __init__(self, feat_dim: int = BASE_DIM, hidden: int = 16):
        super().__init__()
        self.hidden = hidden
        self.cell = torch.nn.GRUCell(feat_dim, hidden)
        self.head = torch.nn.Linear(hidden, 2)

    def forward(self, x, h):
        h = self.cell(x, h)
        return self.head(h), h

    def export_weights(self) -> List[np.ndarray]:
        c = self.cell
        return [
            c.weight_ih.detach().cpu().numpy().copy(),
            c.weight_hh.detach().cpu().numpy().copy(),
            c.bias_ih.detach().cpu().numpy().copy(),
            c.bias_hh.detach().cpu().numpy().copy(),
            self.head.weight.detach().cpu().numpy().T.copy(),
            self.head.bias.detach().cpu().numpy().copy(),
        ]


def batch_features(lr, dlr, ema, it, prev_dec, n_no, since_no) -> np.ndarray:
    return np.stack(
        [
            lr / 8.0,
            np.clip(dlr, -2, 2),
            ema,
            np.full_like(lr, math.log1p(it) / 8.0),
            prev_dec,
            np.log1p(n_no) / 3.0,
            np.log1p(np.minimum(since_no, 9999)) / 8.0,
        ],
        axis=-1,
    )


def approx_greedy_loss(logits: torch.Tensor, errors: torch.Tensor,
                       normalize: bool = True) -> torch.Tensor:
    """Paper ApproxGreedyRouterLoss for a single step, batch of instances.

    logits: (B, K), errors: (B, K) L2 errors of each expert vs ground truth.

    With raw absolute errors the weights shrink with the solve (~10 orders of
    magnitude over a converging rollout), so early iterations dominate the
    gradient and late ones contribute nothing. `normalize` rescales the weights
    per step so each iteration contributes comparably while preserving the
    *relative* preference between experts.
    """
    log_scores = -F.log_softmax(logits, dim=-1)  # (B, K)
    weights = errors.sum(dim=-1, keepdim=True) - errors
    if normalize:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-30)
    return (log_scores * weights).sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# Train lite router with ApproxGreedy + scheduled sampling
# ---------------------------------------------------------------------------

def _rollout_features(pde, solver, corrector, f, u_star, *, feat_mode, max_iters,
                      res_floor, masks, rng, policy_fn):
    """Generic teacher-forced/on-policy rollout yielding (features, errors, active).

    policy_fn(logits_or_none, best, n) -> chosen actions; keeps feature assembly
    in one place so the stats warmup and the training loop cannot drift apart.
    """
    n_inst = f.shape[0]
    fn = l2(f)
    u = np.zeros_like(f)
    prev_log = None
    ema = np.zeros(n_inst)
    prev_dec = np.zeros(n_inst)
    n_no = np.zeros(n_inst)
    since_no = np.full(n_inst, 1e4)
    last_no_gain = np.zeros(n_inst)
    last_cl_gain = np.zeros(n_inst)
    prev_rel = None
    active = np.ones(n_inst, dtype=bool)

    for it in range(max_iters):
        r = pde.residual(u, f)
        rel = l2(r) / fn
        # attribute observed gain to whichever expert ran last step
        if prev_rel is not None:
            g = np.clip(np.log10(np.maximum(prev_rel, 1e-300) / np.maximum(rel, 1e-300)), -2, 4)
            last_no_gain = np.where(prev_dec > 0, g, last_no_gain)
            last_cl_gain = np.where(prev_dec > 0, last_cl_gain, g)
        prev_rel = rel

        active = active & (rel > res_floor)
        if not active.any():
            break
        lr_v = np.log10(np.maximum(rel, 1e-16))
        dlr = np.zeros(n_inst) if prev_log is None else np.clip(lr_v - prev_log, -2, 2)
        ema = EMA * ema + (1 - EMA) * dlr
        prev_log = lr_v

        parts = []
        if feat_mode in ("base7", "both"):
            parts.append(batch_features(lr_v, dlr, ema, it, prev_dec, n_no, since_no))
        if feat_mode in ("struct", "both"):
            parts.append(struct_features_batch(pde, r, masks, last_no_gain, last_cl_gain))
        feats = np.concatenate(parts, axis=-1)

        u_c = solver.step(u, f, r)
        u_n = u + corrector.correct(r)
        e_c = l2(demean(u_c - u_star))
        e_n = l2(demean(u_n - u_star))
        errors = np.stack([e_c, e_n], axis=-1)  # (B, 2): classical, NO
        idx = np.where(active)[0]
        best = (e_n[idx] < e_c[idx]).astype(np.int64)

        chosen = yield feats, errors, idx, best

        act = np.zeros(n_inst, dtype=bool)
        act[idx] = np.asarray(chosen).astype(bool)
        u = np.where(act[:, None, None], u_n, u_c)
        prev_dec = act.astype(np.float64)
        n_no = n_no + prev_dec
        since_no = np.where(act, 0.0, since_no + 1.0)
    _ = policy_fn, rng


def collect_feature_stats(pde, solver, corrector, *, feat_mode, max_iters,
                          res_floor, masks, rng, f=None, u_star=None, n_inst=64):
    """Oracle-forced warmup pass to fix input normalization for the router."""
    if f is None:
        f = GRF2D(pde.N, rng=rng).sample(n_inst)
        u_star = demean(pde.solve_direct(f))
    gen = _rollout_features(pde, solver, corrector, f, u_star, feat_mode=feat_mode,
                            max_iters=max_iters, res_floor=res_floor, masks=masks,
                            rng=rng, policy_fn=None)
    acc = []
    try:
        feats, errors, idx, best = next(gen)
        while True:
            acc.append(feats[idx])
            feats, errors, idx, best = gen.send(best)
    except StopIteration:
        pass
    X = np.concatenate(acc, axis=0)
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-8
    return mu, sd


def feature_stats_cache_path(
    data_path: str,
    solver,
    *,
    feat_mode: str,
    max_iters: int,
    res_floor: float,
    seed: int,
    n_stats: int,
) -> str:
    """Cache path keyed by fixed dataset + solver rollout settings."""
    d = os.path.dirname(data_path) or "."
    base = os.path.splitext(os.path.basename(data_path))[0]
    solver_name = getattr(solver, "name", str(solver))
    rf = f"{res_floor:.0e}".replace("+", "")
    tag = f"{base}_{solver_name}_{feat_mode}_mi{max_iters}_rf{rf}_s{seed}_n{n_stats}"
    return os.path.join(d, f"featstats_{tag}.pt")


def make_or_load_feature_stats(
    pde,
    solver,
    corrector,
    *,
    feat_mode: str,
    max_iters: int,
    res_floor: float,
    masks,
    seed: int,
    stats_path: Optional[str] = None,
    f=None,
    u_star=None,
    n_inst: int = 64,
):
    """Oracle warmup for router input normalization; cache per (dataset, solver)."""
    meta = {
        "equation": getattr(pde, "equation", None),
        "N": pde.N,
        "solver": getattr(solver, "name", str(solver)),
        "feat_mode": feat_mode,
        "max_iters": max_iters,
        "res_floor": res_floor,
        "seed": seed,
        "n_stats": n_inst,
    }
    if stats_path and os.path.exists(stats_path):
        blob = torch.load(stats_path, map_location="cpu", weights_only=False)
        for k, v in meta.items():
            if blob.get(k) != v:
                print(f"  feature-stats cache stale ({k}: {blob.get(k)!r} != {v!r}), "
                      f"recomputing ...", flush=True)
                break
        else:
            print(f"  loading cached feature stats from {stats_path}", flush=True)
            return blob["mu"], blob["sd"]

    print(f"  estimating feature stats (feats={feat_mode}, dim={FEAT_DIMS[feat_mode]}) ...",
          flush=True)
    mu, sd = collect_feature_stats(
        pde, solver, corrector, feat_mode=feat_mode, max_iters=max_iters,
        res_floor=res_floor, masks=masks, rng=np.random.default_rng(seed + 999),
        f=f, u_star=u_star, n_inst=n_inst)
    if stats_path:
        os.makedirs(os.path.dirname(stats_path) or ".", exist_ok=True)
        torch.save({**meta, "mu": mu, "sd": sd}, stats_path)
        print(f"  cached feature stats -> {stats_path}", flush=True)
    return mu, sd


def make_or_load_router_dataset(pde, *, n_train, n_val, seed, data_path=None):
    """Fixed (f, u*) set matching the paper's train_router.py protocol.

    If data_path is set and exists, load it; otherwise generate once and
    optionally cache. Returns (f_train, u_train, f_val, u_val).
    """
    if data_path and os.path.exists(data_path):
        print(f"  loading fixed router dataset from {data_path}", flush=True)
        blob = torch.load(data_path, map_location="cpu", weights_only=False)
        return (blob["f_train"], blob["u_train"], blob["f_val"], blob["u_val"])

    print(f"  generating fixed router dataset n_train={n_train} n_val={n_val} ...",
          flush=True)
    rng = np.random.default_rng(seed)
    f_all = GRF2D(pde.N, rng=rng).sample(n_train + n_val)
    # demean forcings for singular Poisson / zero-reaction ConvDiff
    singular = abs(getattr(pde, "reaction", 0.0)) < 1e-14
    if singular and getattr(pde, "equation", "") in ("Poisson", "ConvDiff"):
        f_all = demean(f_all)
    u_all = demean(pde.solve_direct(f_all))
    f_train, f_val = f_all[:n_train], f_all[n_train:]
    u_train, u_val = u_all[:n_train], u_all[n_train:]
    if data_path:
        os.makedirs(os.path.dirname(data_path) or ".", exist_ok=True)
        torch.save({
            "f_train": f_train, "u_train": u_train,
            "f_val": f_val, "u_val": u_val,
            "n_train": n_train, "n_val": n_val, "seed": seed, "N": pde.N,
            "equation": getattr(pde, "equation", None),
        }, data_path)
        print(f"  cached dataset -> {data_path}", flush=True)
    return f_train, u_train, f_val, u_val


def _train_one_batch(model, opt, pde, solver, corrector, f, u_star, *,
                     feat_mode, max_iters, res_floor, masks, rng, mu, sd,
                     normalize_loss, tbptt, tf, device, hidden):
    """One scheduled-sampling rollout + ApproxGreedy updates. Returns metrics."""
    n_inst = f.shape[0]
    gen = _rollout_features(pde, solver, corrector, f, u_star,
                            feat_mode=feat_mode, max_iters=max_iters,
                            res_floor=res_floor, masks=masks, rng=rng,
                            policy_fn=None)
    h_full = (torch.zeros(n_inst, hidden, device=device)
              if model.recurrent else None)
    window_loss = None
    window_n = 0
    ep_loss = ep_agree = ep_no = 0.0
    n_steps = 0

    def flush(window_loss, h_full):
        if window_loss is None:
            return None, h_full
        opt.zero_grad(set_to_none=True)
        window_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if h_full is not None:
            h_full = h_full.detach()
        return None, h_full

    try:
        feats, errors, idx, best = next(gen)
        while True:
            x_t = torch.tensor((feats[idx] - mu) / sd, dtype=torch.float32, device=device)
            e_t = torch.tensor(errors[idx], dtype=torch.float32, device=device)
            if model.recurrent:
                idx_t = torch.as_tensor(idx, dtype=torch.long, device=device)
                logits, h_new = model(x_t, h_full[idx_t])
                h_full = h_full.index_copy(0, idx_t, h_new)
            else:
                logits, _ = model(x_t, None)
            loss = approx_greedy_loss(logits, e_t, normalize=normalize_loss)

            window_loss = loss if window_loss is None else window_loss + loss
            window_n += 1
            ep_loss += loss.item()
            n_steps += 1
            if window_n >= (tbptt if model.recurrent else 1):
                window_loss, h_full = flush(window_loss, h_full)
                window_n = 0

            with torch.no_grad():
                pred = logits.argmax(dim=-1).cpu().numpy()
            ep_agree += float((pred == best).mean())
            ep_no += float(pred.mean())
            use_tf = rng.random(len(idx)) < tf
            chosen = np.where(use_tf, best, pred)
            feats, errors, idx, best = gen.send(chosen)
    except StopIteration:
        pass
    flush(window_loss, h_full)
    return ep_loss, ep_agree, ep_no, n_steps


def train_lite_router(
    pde,
    solver,
    corrector: DeepONetCorrector,
    *,
    n_inst: int = 128,
    max_iters: int = 300,
    epochs: int = 40,
    batches_per_epoch: int = 8,
    lr: float = 1e-3,
    seed: int = 7,
    hidden: Optional[int] = None,
    tf_start: float = 1.0,
    tf_end: float = 0.1,
    tf_decay: float = 0.95,
    res_floor: float = 1e-8,
    feat_mode: str = "base7",
    normalize_loss: bool = True,
    arch: str = "mlp",
    tbptt: int = 16,
    device: str = "cpu",
    # fixed-dataset mode (paper protocol); if n_train is set we reuse a fixed
    # (f, u*) set each epoch instead of sampling a fresh GRF every batch
    n_train: Optional[int] = None,
    n_val: int = 128,
    data_path: Optional[str] = None,
    stats_path: Optional[str] = None,
    ckpt_dir: Optional[str] = None,
    ckpt_prefix: Optional[str] = None,
    ckpt_every: int = 1,
):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    masks = band_masks(pde.N)
    feat_dim = FEAT_DIMS[feat_mode]
    if hidden is None:
        hidden = 16 if arch == "gru" else 32

    fixed = n_train is not None
    f_train = u_train = None
    if fixed:
        f_train, u_train, f_val, u_val = make_or_load_router_dataset(
            pde, n_train=n_train, n_val=n_val, seed=seed, data_path=data_path)
        # one pass over the fixed set per epoch (same #batches as online default
        # when n_train == n_inst * batches_per_epoch)
        batch_size = n_inst
        n_batches = int(math.ceil(n_train / batch_size))
        print(f"  fixed-dataset mode: n_train={n_train} batch_size={batch_size} "
              f"batches/epoch={n_batches}", flush=True)
        stats_n = min(n_inst, 64)
        stats_f, stats_u = f_train[:stats_n], u_train[:stats_n]
    else:
        n_batches = batches_per_epoch
        stats_f = stats_u = None
        stats_n = min(n_inst, 64)

    if stats_path is None and data_path:
        stats_path = feature_stats_cache_path(
            data_path, solver, feat_mode=feat_mode, max_iters=max_iters,
            res_floor=res_floor, seed=seed, n_stats=stats_n)
    mu, sd = make_or_load_feature_stats(
        pde, solver, corrector, feat_mode=feat_mode, max_iters=max_iters,
        res_floor=res_floor, masks=masks, seed=seed, stats_path=stats_path,
        f=stats_f, u_star=stats_u, n_inst=stats_n)

    if arch == "gru":
        model = TorchLiteGRU(feat_dim=feat_dim, hidden=hidden).to(device)
    else:
        model = TorchLiteMLP(feat_dim=feat_dim, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tf = tf_start
    cls = LiteGRURouter if arch == "gru" else LiteRouter

    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
    if ckpt_prefix is None and ckpt_dir:
        ckpt_prefix = os.path.join(ckpt_dir, f"lite_router_{arch}_{feat_mode}")

    for ep in range(epochs):
        model.train()
        ep_loss = ep_agree = ep_no = 0.0
        n_steps = 0
        if fixed:
            order = rng.permutation(n_train)
        for b in range(n_batches):
            if fixed:
                idx = order[b * batch_size:(b + 1) * batch_size]
                f = f_train[idx]
                u_star = u_train[idx]
            else:
                f = GRF2D(pde.N, rng=rng).sample(n_inst)
                u_star = demean(pde.solve_direct(f))
            loss_b, agree_b, no_b, steps_b = _train_one_batch(
                model, opt, pde, solver, corrector, f, u_star,
                feat_mode=feat_mode, max_iters=max_iters, res_floor=res_floor,
                masks=masks, rng=rng, mu=mu, sd=sd,
                normalize_loss=normalize_loss, tbptt=tbptt, tf=tf,
                device=device, hidden=hidden)
            ep_loss += loss_b
            ep_agree += agree_b
            ep_no += no_b
            n_steps += steps_b

        tf = max(tf_end, tf * tf_decay)
        d = max(n_steps, 1)
        print(
            f"  router ep {ep:3d}: loss {ep_loss/d:.4f}  oracle_agree {ep_agree/d:.3f}  "
            f"pred_no_frac {ep_no/d:.3f}  tf={tf:.3f}",
            flush=True,
        )
        if ckpt_prefix and ckpt_every > 0 and ((ep + 1) % ckpt_every == 0 or ep == epochs - 1):
            snap = cls(model.export_weights(), feat_mode=feat_mode, mu=mu, sd=sd)
            ep_path = f"{ckpt_prefix}_ep{ep:03d}.pth"
            os.makedirs(os.path.dirname(ep_path) or ".", exist_ok=True)
            snap.save(ep_path, epoch=ep)
            print(f"  checkpoint -> {ep_path}", flush=True)

    return cls(model.export_weights(), feat_mode=feat_mode, mu=mu, sd=sd)



# ---------------------------------------------------------------------------
# Rollouts + criteria
# ---------------------------------------------------------------------------

def run_rollout(
    pde,
    solver,
    f,
    u_truth,
    policy: str,
    corrector: Optional[DeepONetCorrector] = None,
    router: Optional[LiteRouter] = None,
    max_iters: int = 100000,
    time_cap: float = 120.0,
    res_floor: float = 1e-10,
) -> Dict[str, np.ndarray]:
    N = pde.N
    u = np.zeros((1, N, N))
    fn = float(l2(f)[0])
    un = float(l2(demean(u_truth))[0])
    t_cum = 0.0
    hints_tau = int(policy[5:]) if policy.startswith("hints") else None
    if router is not None:
        router.reset()
    prev_decision = 0
    trace = {"t": [], "rel_res": [], "rel_err": [], "decision": []}

    for it in range(max_iters):
        t0 = time.perf_counter()
        r = pde.residual(u, f)
        rel_res = float(l2(r)[0]) / fn
        t_cum += time.perf_counter() - t0
        rel_err = float(l2(demean(u - u_truth))[0]) / un
        trace["t"].append(t_cum)
        trace["rel_res"].append(rel_res)
        trace["rel_err"].append(rel_err)
        # Divergent solve (common when a bad NO call blows up Aniso Jacobi):
        # stop cleanly so bench reports inf rather than crashing in log10.
        if (not math.isfinite(rel_res)) or (not math.isfinite(rel_err)) or (not np.isfinite(u).all()):
            trace["decision"].append(-1)
            break
        if rel_res <= res_floor or t_cum > time_cap:
            trace["decision"].append(-1)
            break

        if policy == "oracle":
            # One-pass error-greedy (paper Alg. 1): run both experts once,
            # time them separately, charge only the winner. Avoids the old
            # two-pass replay (which re-executed the chosen expert) while
            # keeping reported wall-clock free of the unchosen expert's cost.
            t0 = time.perf_counter()
            u_c = solver.step(u, f, r)
            t_c = time.perf_counter() - t0
            t0 = time.perf_counter()
            u_n = u + corrector.correct(r)
            t_n = time.perf_counter() - t0
            e_c = float(l2(demean(u_c - u_truth))[0])
            e_n = float(l2(demean(u_n - u_truth))[0])
            decision = 1 if e_n < e_c else 0
            t_cum += t_n if decision else t_c
            u = u_n if decision else u_c
        else:
            if policy == "classical":
                decision = 0
            elif hints_tau is not None:
                decision = 1 if (it + 1) % hints_tau == 0 else 0
            elif policy == "router":
                t0 = time.perf_counter()
                decision = router.decide(rel_res, it, prev_decision, r=r, pde=pde)
                t_cum += time.perf_counter() - t0
            else:
                raise ValueError(policy)

            t0 = time.perf_counter()
            if decision == 1:
                u = u + corrector.correct(r)
            else:
                u = solver.step(u, f, r)
            t_cum += time.perf_counter() - t0

        prev_decision = decision
        trace["decision"].append(decision)

    return {k: np.asarray(v) for k, v in trace.items()}


def time_to_tol(trace, tols, criterion: str) -> Dict[float, Tuple[float, float]]:
    vals = trace[criterion]
    out = {}
    for tol in tols:
        idx = np.nonzero(vals <= tol)[0]
        if len(idx) == 0:
            out[tol] = (np.inf, np.inf)
        else:
            i = int(idx[0])
            out[tol] = (float(trace["t"][i]), float(i))
    return out


def diversity_stats(decision_rows: List[np.ndarray]) -> Dict[str, float]:
    """Criteria 1: diverse routing — uses both experts, varies across instances."""
    fracs = []
    switches = []
    early_no = []
    for d in decision_rows:
        d = d[d >= 0]
        if len(d) == 0:
            continue
        frac = float((d == 1).mean())
        fracs.append(frac)
        switches.append(float(np.sum(d[1:] != d[:-1]) / max(len(d) - 1, 1)))
        early = d[: max(len(d) // 5, 1)]
        early_no.append(float((early == 1).mean()))
    fracs = np.asarray(fracs)
    return {
        "mean_no_fraction": float(np.mean(fracs)) if len(fracs) else 0.0,
        "std_no_fraction": float(np.std(fracs)) if len(fracs) else 0.0,
        "frac_instances_both_experts": float(np.mean((fracs > 0.01) & (fracs < 0.99))) if len(fracs) else 0.0,
        "mean_switch_rate": float(np.mean(switches)) if len(switches) else 0.0,
        "mean_early_no_fraction": float(np.mean(early_no)) if len(early_no) else 0.0,
    }


def evaluate(
    *,
    equation: str,
    N: int,
    don_ckp: str,
    router_ckp: Optional[str],
    solver_spec: str = "jacobi",
    eps_x: float = 0.01,
    eps_y: float = 1.0,
    b_vel: float = 20.0,
    n_test: int = 32,
    hints_tau: int = 25,
    tols: Optional[List[float]] = None,
    primary_tol: Optional[float] = None,
    seed: int = 72,
    out_path: str,
    device: str = "cpu",
    max_iters: int = 100000,
    time_cap: float = 120.0,
):
    if tols is None:
        tols = [1.0 / N ** 2, 1e-3, 1e-4, 1e-5, 1e-6]
    tols = [float(t) for t in tols]
    if primary_tol is None:
        primary_tol = tols[-1]
    primary_tol = float(primary_tol)
    if primary_tol not in tols:
        tols = sorted(set(tols + [primary_tol]), reverse=True)
    pt = str(primary_tol)

    pde = make_pde(equation, N, b_vel=b_vel, eps_x=eps_x, eps_y=eps_y)
    solver = make_solver(pde, solver_spec)
    corrector = DeepONetCorrector(don_ckp, device=device, threads=1)
    torch.set_num_threads(1)
    router = load_router(router_ckp) if router_ckp else None

    grf = GRF2D(N, rng=np.random.default_rng(seed))
    f_test = grf.sample(n_test)
    u_truth = demean(pde.solve_direct(f_test))
    _ = corrector.correct(f_test[:1])
    _ = solver.step(np.zeros_like(f_test[:1]), f_test[:1])

    policies = ["classical", f"hints{hints_tau}"]
    if router is not None:
        policies.append("router")
    policies.append("oracle")

    results = {
        "args": {
            "equation": equation,
            "N": N,
            "solver": solver_spec,
            "eps_x": eps_x,
            "eps_y": eps_y,
            "n_test": n_test,
            "hints_tau": hints_tau,
            "tols": tols,
            "primary_tol": primary_tol,
            "max_iters": max_iters,
            "time_cap": time_cap,
            "router_arch": None if router is None else router.arch,
            "router_feats": None if router is None else router.feat_mode,
        },
        "policies": {},
    }

    for policy in policies:
        rows = []
        decs = []
        t0 = time.time()
        for i in range(n_test):
            tr = run_rollout(
                pde,
                solver,
                f_test[i:i + 1],
                u_truth[i:i + 1],
                policy,
                corrector=corrector,
                router=router,
                max_iters=max_iters,
                time_cap=time_cap,
                res_floor=min(tols) * 0.1,
            )
            row = {
                "res": {str(t): time_to_tol(tr, [t], "rel_res")[t] for t in tols},
                "err": {str(t): time_to_tol(tr, [t], "rel_err")[t] for t in tols},
                "n_iters": int(len(tr["decision"])),
                "n_no_calls": int((tr["decision"] == 1).sum()),
                "final_rel_res": float(tr["rel_res"][-1]),
                "final_rel_err": float(tr["rel_err"][-1]),
            }
            rows.append(row)
            decs.append(tr["decision"])
        def _finite(xs):
            return [float(x) for x in xs if x is not None and math.isfinite(float(x))]

        t_err = _finite([r["err"][pt][0] for r in rows])
        it_err = _finite([r["err"][pt][1] for r in rows])
        t_res = _finite([r["res"][pt][0] for r in rows])
        it_res = _finite([r["res"][pt][1] for r in rows])
        no_calls = [float(r["n_no_calls"]) for r in rows]
        no_fracs = []
        for r, d in zip(rows, decs):
            dd = d[d >= 0]
            if len(dd):
                no_fracs.append(float((dd == 1).mean()))
        div = diversity_stats(decs)
        block = {
            "rows": rows,
            "diversity": div,
            "n_converged_err_primary": len(t_err),
            "mean_time_err_primary": float(np.mean(t_err)) if t_err else float("inf"),
            "std_time_err_primary": float(np.std(t_err)) if t_err else float("nan"),
            "median_time_err_primary": float(np.median(t_err)) if t_err else float("inf"),
            "mean_iters_err_primary": float(np.mean(it_err)) if it_err else float("inf"),
            "std_iters_err_primary": float(np.std(it_err)) if it_err else float("nan"),
            "median_iters_err_primary": float(np.median(it_err)) if it_err else float("inf"),
            "mean_time_res_primary": float(np.mean(t_res)) if t_res else float("inf"),
            "std_time_res_primary": float(np.std(t_res)) if t_res else float("nan"),
            "median_time_res_primary": float(np.median(t_res)) if t_res else float("inf"),
            "mean_iters_res_primary": float(np.mean(it_res)) if it_res else float("inf"),
            "std_iters_res_primary": float(np.std(it_res)) if it_res else float("nan"),
            "median_iters_res_primary": float(np.median(it_res)) if it_res else float("inf"),
            "mean_no_calls": float(np.mean(no_calls)) if no_calls else 0.0,
            "std_no_calls": float(np.std(no_calls)) if no_calls else 0.0,
            "median_no_calls": float(np.median(no_calls)) if no_calls else 0.0,
            "mean_no_fraction": float(np.mean(no_fracs)) if no_fracs else 0.0,
            "std_no_fraction": float(np.std(no_fracs)) if no_fracs else 0.0,
            # keep h2 aliases when present for backward compat with prior JSON readers
            "median_time_err_h2": float(np.median([r["err"].get(str(1.0 / N ** 2), (np.inf, np.inf))[0] for r in rows])),
            "median_iters_err_h2": float(np.median([r["err"].get(str(1.0 / N ** 2), (np.inf, np.inf))[1] for r in rows])),
            "median_time_res_h2": float(np.median([r["res"].get(str(1.0 / N ** 2), (np.inf, np.inf))[0] for r in rows])),
            "median_iters_res_h2": float(np.median([r["res"].get(str(1.0 / N ** 2), (np.inf, np.inf))[1] for r in rows])),
        }
        results["policies"][policy] = block
        print(
            f"[{policy:>10s}] err@{primary_tol:g} "
            f"{block['mean_time_err_primary']*1e3:.2f}±{block['std_time_err_primary']*1e3:.2f}ms "
            f"(med {block['median_time_err_primary']*1e3:.2f}) / "
            f"{block['mean_iters_err_primary']:.0f}±{block['std_iters_err_primary']:.0f}it | "
            f"NO%={100*block['mean_no_fraction']:.1f}±{100*block['std_no_fraction']:.1f} "
            f"(#{block['mean_no_calls']:.0f}±{block['std_no_calls']:.0f}) "
            f"ok={block['n_converged_err_primary']}/{n_test} "
            f"({time.time()-t0:.0f}s)",
            flush=True,
        )

    # Criteria checklist vs classical + HINTS (primary_tol)
    if "router" in results["policies"]:
        R = results["policies"]["router"]
        C = results["policies"]["classical"]
        H = results["policies"][f"hints{hints_tau}"]
        checklist = {
            "primary_tol": primary_tol,
            "diverse_routing": {
                "pass": R["diversity"]["frac_instances_both_experts"] >= 0.5
                and R["diversity"]["std_no_fraction"] > 0.01,
                "frac_instances_both_experts": R["diversity"]["frac_instances_both_experts"],
                "std_no_fraction": R["diversity"]["std_no_fraction"],
                "mean_no_fraction": R["diversity"]["mean_no_fraction"],
                "mean_early_no_fraction": R["diversity"]["mean_early_no_fraction"],
            },
            "fewer_iterations_than_baselines": {
                "pass_res": R["median_iters_res_primary"] < min(C["median_iters_res_primary"], H["median_iters_res_primary"]),
                "pass_err": R["median_iters_err_primary"] < min(C["median_iters_err_primary"], H["median_iters_err_primary"]),
                "router_iters_res": R["median_iters_res_primary"],
                "classical_iters_res": C["median_iters_res_primary"],
                "hints_iters_res": H["median_iters_res_primary"],
                "router_iters_err": R["median_iters_err_primary"],
                "classical_iters_err": C["median_iters_err_primary"],
                "hints_iters_err": H["median_iters_err_primary"],
            },
            "less_wallclock_than_baselines": {
                "pass_res": R["median_time_res_primary"] < min(C["median_time_res_primary"], H["median_time_res_primary"]),
                "pass_err": R["median_time_err_primary"] < min(C["median_time_err_primary"], H["median_time_err_primary"]),
                "router_time_res": R["median_time_res_primary"],
                "classical_time_res": C["median_time_res_primary"],
                "hints_time_res": H["median_time_res_primary"],
                "router_time_err": R["median_time_err_primary"],
                "classical_time_err": C["median_time_err_primary"],
                "hints_time_err": H["median_time_err_primary"],
            },
        }
        results["checklist"] = checklist
        print(f"\n=== CRITERIA CHECKLIST (primary_tol={primary_tol:g}) ===", flush=True)
        for name, block in checklist.items():
            if name == "primary_tol":
                continue
            keys = [k for k in block if k.startswith("pass")]
            verdict = "PASS" if all(block[k] for k in keys) else ("PARTIAL" if any(block[k] for k in keys) else "FAIL")
            print(f"  {name}: {verdict}  { {k: block[k] for k in block} }", flush=True)

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, float) and (np.isinf(o) or np.isnan(o)):
            return str(o)
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(_clean(results), fh, indent=2)
    print("saved", out_path, flush=True)
    return results


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    def _add_pde_args(sp):
        sp.add_argument("--equation", type=str, default="Poisson",
                        choices=["Poisson", "ConvDiff", "AnisoDiff"])
        sp.add_argument("--N", type=int, default=31)
        sp.add_argument("--solver", type=str, default="jacobi")
        sp.add_argument("--eps_x", type=float, default=0.01)
        sp.add_argument("--eps_y", type=float, default=1.0)
        sp.add_argument("--b_vel", type=float, default=20.0)
        sp.add_argument("--device", type=str, default=None)
        sp.add_argument("--seed", type=int, default=7)

    pt = sub.add_parser("train_router")
    _add_pde_args(pt)
    pt.add_argument("--don_ckp", type=str, required=True)
    pt.add_argument("--out", type=str, default="./checkpoints/lite_router.pth")
    pt.add_argument("--n_inst", type=int, default=128)
    pt.add_argument("--max_iters", type=int, default=300)
    pt.add_argument("--epochs", type=int, default=40)
    pt.add_argument("--batches_per_epoch", type=int, default=8)
    pt.add_argument("--feats", type=str, default="base7", choices=list(FEAT_DIMS),
                    help="base7 = scalar residual trajectory only; struct = cheap "
                         "residual-shape features; both = concatenation")
    pt.add_argument("--raw_loss", action="store_true",
                    help="use unnormalized ApproxGreedy weights (original behavior)")
    pt.add_argument("--arch", type=str, default="mlp", choices=["mlp", "gru"],
                    help="mlp = memoryless; gru = recurrent over the residual trajectory")
    pt.add_argument("--hidden", type=int, default=None,
                    help="hidden width (default 32 for mlp, 16 for gru)")
    pt.add_argument("--tbptt", type=int, default=16,
                    help="truncated BPTT window for --arch gru")
    pt.add_argument("--n_train", type=int, default=None,
                    help="if set, train on a fixed (f,u*) dataset of this size "
                         "(paper protocol) instead of sampling a fresh GRF each batch")
    pt.add_argument("--n_val", type=int, default=128)
    pt.add_argument("--data_path", type=str, default=None,
                    help="cache path for the fixed router dataset (.pt)")
    pt.add_argument("--stats_path", type=str, default=None,
                    help="cache path for feature-stats warmup (.pt); auto-derived "
                         "from --data_path when omitted")
    pt.add_argument("--ckpt_dir", type=str, default=None,
                    help="if set, save a router checkpoint after every --ckpt_every epochs")
    pt.add_argument("--ckpt_prefix", type=str, default=None,
                    help="prefix for epoch checkpoints (default: <ckpt_dir>/lite_router_<arch>_<feats>); "
                         "writes {prefix}_epXXX.pth")
    pt.add_argument("--ckpt_every", type=int, default=1,
                    help="save epoch checkpoint every N epochs (default 1)")
    pt.set_defaults(seed=7)

    pb = sub.add_parser("bench")
    _add_pde_args(pb)
    pb.add_argument("--don_ckp", type=str, required=True)
    pb.add_argument("--router_ckp", type=str, default=None,
                    help="Omit to bench classical/HINTS/oracle only (no router).")
    pb.add_argument("--n_test", type=int, default=32)
    pb.add_argument("--hints_tau", type=int, default=25)
    pb.add_argument("--tols", type=float, nargs="+", default=None,
                    help="Error/residual tolerances to measure (default: h^-2..1e-6)")
    pb.add_argument("--primary_tol", type=float, default=None,
                    help="Tolerance used for criteria checklist (default: tightest in --tols)")
    pb.add_argument("--max_iters", type=int, default=100000)
    pb.add_argument("--time_cap", type=float, default=120.0)
    pb.add_argument("--out", type=str, default="./results/criteria.json")
    pb.set_defaults(seed=72)

    args = p.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.cmd == "train_router":
        pde = make_pde(args.equation, args.N, b_vel=args.b_vel, eps_x=args.eps_x, eps_y=args.eps_y)
        solver = make_solver(pde, args.solver)
        corr = DeepONetCorrector(args.don_ckp, device=device, threads=8)
        print(
            f"training lite router with ApproxGreedy on {device} "
            f"({args.equation} N={args.N} solver={args.solver}) ...",
            flush=True,
        )
        router = train_lite_router(
            pde,
            solver,
            corr,
            n_inst=args.n_inst,
            max_iters=args.max_iters,
            epochs=args.epochs,
            batches_per_epoch=args.batches_per_epoch,
            seed=args.seed,
            feat_mode=args.feats,
            normalize_loss=not args.raw_loss,
            arch=args.arch,
            hidden=args.hidden,
            tbptt=args.tbptt,
            n_train=args.n_train,
            n_val=args.n_val,
            data_path=args.data_path,
            stats_path=args.stats_path,
            ckpt_dir=args.ckpt_dir,
            ckpt_prefix=args.ckpt_prefix,
            ckpt_every=args.ckpt_every,
            device=device,
        )
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        router.save(args.out)
        print("saved", args.out, flush=True)
    else:
        evaluate(
            equation=args.equation,
            N=args.N,
            don_ckp=args.don_ckp,
            router_ckp=args.router_ckp,
            solver_spec=args.solver,
            eps_x=args.eps_x,
            eps_y=args.eps_y,
            b_vel=args.b_vel,
            n_test=args.n_test,
            hints_tau=args.hints_tau,
            tols=args.tols,
            primary_tol=args.primary_tol,
            out_path=args.out,
            device=device,
            seed=args.seed,
            max_iters=args.max_iters,
            time_cap=args.time_cap,
        )


if __name__ == "__main__":
    main()
