"""Coarse-grid DeepONet corrector for the wall-clock hybrid solver.

The corrector is a DeepONet (branch MLP on sensor values, trunk MLP on
coordinates, output = inner product of branch and trunk features) that acts
on the residual of the fine-grid system:

    du = C_NO(r) = ||R r|| * P( DeepONet( R r / ||R r|| ) ) / s

where
  * R is the band-limited restriction of the N x N residual to an n_c x n_c
    sensor grid (n_c = N / coarsen), implemented by an FFT truncation to the
    modes |k| < n_c / 2 (the standard "ideal low-pass + subsample" transfer);
  * P is the exact trigonometric prolongation back to the fine grid (FFT zero
    padding), so the correction is band-limited by construction: the operator
    never injects error into the high-frequency band that the classical
    smoother owns;
  * s is a fixed output scale stored in the checkpoint;
  * the normalisation by ||R r|| makes the corrector scale-equivariant, which
    is what a linear inverse must satisfy and which keeps C_NO(0) = 0 (the
    zero-preservation condition of Proposition 4.2 in the paper).

Training data are exact (residual, error) pairs of the discrete operator
(e = A^{-1} r via the FFT solver), drawn from a mixture of spectral shapes so
the network is accurate on the whole band it is responsible for.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from fast_pde import FastStencilPDE, GRF2D, demean, l2


# ---------------------------------------------------------------------------
# Transfer operators (band-limited restriction / exact prolongation)
# ---------------------------------------------------------------------------

class BandTransfer:
    """FFT-based restriction N -> (n_cx, n_cy) and prolongation back on the
    periodic grid. Modes |k_x| <= n_cx/2 - 1 and |k_y| <= n_cy/2 - 1 are
    retained (a square band when n_cx == n_cy; an anisotropic band otherwise,
    e.g. n_cx = N keeps every mode along x)."""

    def __init__(self, N, n_cx, n_cy=None):
        n_cy = n_cx if n_cy is None else n_cy
        assert N % 2 == 0 and n_cx % 2 == 0 and n_cy % 2 == 0 and n_cx <= N and n_cy <= N
        self.N, self.n_cx, self.n_cy = N, n_cx, n_cy
        self.n_c = n_cx  # backwards compatibility (square band)
        self.kmax_x = n_cx // 2 - 1
        self.kmax_y = n_cy // 2 - 1
        kx, ky = self.kmax_x, self.kmax_y
        # index sets in fft layout for the first axis, rfft layout for the last
        self.ix_fine = np.concatenate([np.arange(0, kx + 1), np.arange(N - kx, N)])
        self.ix_coarse = np.concatenate([np.arange(0, kx + 1), np.arange(n_cx - kx, n_cx)])
        self.ny = ky + 1  # last-axis (rfft) modes 0..ky

    def restrict(self, r):
        """r: (..., N, N) real -> (..., n_cx, n_cy) real (band-limited)."""
        rhat = np.fft.rfft2(r, axes=(-2, -1), norm="forward")
        sub = rhat[..., self.ix_fine, :][..., :self.ny]
        chat = np.zeros(r.shape[:-2] + (self.n_cx, self.n_cy // 2 + 1), dtype=np.complex128)
        chat[..., self.ix_coarse, :self.ny] = sub
        return np.fft.irfft2(chat, s=(self.n_cx, self.n_cy), axes=(-2, -1), norm="forward")

    def prolong(self, c):
        """c: (..., n_cx, n_cy) real -> (..., N, N) real, exact for band-limited c."""
        chat = np.fft.rfft2(c, axes=(-2, -1), norm="forward")
        fhat = np.zeros(c.shape[:-2] + (self.N, self.N // 2 + 1), dtype=np.complex128)
        sub = chat[..., self.ix_coarse, :][..., :self.ny]
        fhat[..., self.ix_fine, :self.ny] = sub
        return np.fft.irfft2(fhat, s=(self.N, self.N), axes=(-2, -1), norm="forward")


# ---------------------------------------------------------------------------
# DeepONet on the sensor grid
# ---------------------------------------------------------------------------

def fourier_features(coords, kmax, nonredundant=False, kmax_y=None):
    """coords (M, 2) in [0,1)^2 -> (M, F) features cos/sin(2 pi k . x) for
    integer k with |k_x| <= kmax, |k_y| <= kmax_y (default kmax). With
    nonredundant=True only one k of each (k, -k) pair is kept (plus the
    constant), which gives an orthogonal real basis of the band with
    F = 2 * (#half-plane modes) + 1 = (2 kmax + 1)(2 kmax_y + 1)."""
    kmax_y = kmax if kmax_y is None else kmax_y
    kx, ky = np.meshgrid(np.arange(-kmax, kmax + 1), np.arange(-kmax_y, kmax_y + 1), indexing="ij")
    K = np.stack([kx.ravel(), ky.ravel()], axis=1).astype(np.float64)  # (F/2, 2)
    if nonredundant:
        half = (K[:, 0] > 0) | ((K[:, 0] == 0) & (K[:, 1] > 0))
        Kh = K[half]
        phase = 2 * np.pi * coords @ Kh.T
        Mp = len(coords)
        feats = np.concatenate([np.ones((Mp, 1)) / np.sqrt(Mp),
                                np.cos(phase) * np.sqrt(2.0 / Mp),
                                np.sin(phase) * np.sqrt(2.0 / Mp)], axis=1)
        return feats.astype(np.float32)  # orthonormal columns
    phase = 2 * np.pi * coords @ K.T
    return np.concatenate([np.cos(phase), np.sin(phase)], axis=1).astype(np.float32)


class CoarseDeepONet(nn.Module):
    def __init__(self, n_c, p=1024, hidden=1024, layers=3, trunk_hidden=512,
                 trunk_layers=2, trunk_kmax=None, skip=True, trunk="fourier", mlp=True, n_cy=None):
        super().__init__()
        self.n_c = n_c
        self.n_cx, self.n_cy = n_c, (n_c if n_cy is None else n_cy)
        self.trunk_kind = trunk
        d_in = self.n_cx * self.n_cy
        self.use_mlp = bool(mlp)
        if trunk == "fourier":
            # fixed orthogonal Fourier basis of the band as the trunk output
            # (POD-DeepONet style fixed trunk); p is then the basis size
            kmax = trunk_kmax if trunk_kmax is not None else self.n_cx // 2 - 1
            kmax_y = self.n_cy // 2 - 1
            xs = np.linspace(0, 1, self.n_cx + 1)[:-1]
            ys = np.linspace(0, 1, self.n_cy + 1)[:-1]
            Xg, Yg = np.meshgrid(xs, ys, indexing="ij")
            coords = np.stack([Xg.ravel(), Yg.ravel()], axis=1)
            B = fourier_features(coords, kmax, nonredundant=True, kmax_y=kmax_y)
            p = B.shape[1]
            self.register_buffer("fixed_basis", torch.tensor(B))
        self.p = p
        mods = []
        last = d_in
        for _ in range(layers):
            mods += [nn.Linear(last, hidden), nn.GELU()]
            last = hidden
        mods += [nn.Linear(last, p)]
        self.branch_mlp = nn.Sequential(*mods) if self.use_mlp else None
        if self.branch_mlp is not None:
            # start the nonlinear path at zero so that training first fits the
            # linear path and the MLP only adds what the linear path cannot fit
            nn.init.zeros_(self.branch_mlp[-1].weight)
            nn.init.zeros_(self.branch_mlp[-1].bias)
        # linear skip path in the branch net (the map r -> e is linear, so a
        # linear path lets the MLP concentrate on refining it)
        self.branch_skip = nn.Linear(d_in, p, bias=False) if skip else None
        if trunk == "mlp":
            kmax = trunk_kmax if trunk_kmax is not None else n_c // 2 - 1
            xs = np.linspace(0, 1, self.n_cx + 1)[:-1]
            ys = np.linspace(0, 1, self.n_cy + 1)[:-1]
            Xg, Yg = np.meshgrid(xs, ys, indexing="ij")
            coords = np.stack([Xg.ravel(), Yg.ravel()], axis=1)
            feats = fourier_features(coords, kmax)
            self.register_buffer("trunk_in", torch.tensor(feats))
            tm = []
            last = feats.shape[1]
            for _ in range(trunk_layers):
                tm += [nn.Linear(last, trunk_hidden), nn.GELU()]
                last = trunk_hidden
            tm += [nn.Linear(last, p)]
            self.trunk = nn.Sequential(*tm)

    def basis(self):
        if self.trunk_kind == "fourier":
            return self.fixed_basis
        return self.trunk(self.trunk_in)  # (n_c^2, p)

    def branch(self, x):
        b = self.branch_mlp(x) if self.branch_mlp is not None else 0.0
        if self.branch_skip is not None:
            b = b + self.branch_skip(x)
        return b

    def forward(self, x):
        """x: (B, n_c^2) normalised sensor values -> (B, n_c^2)."""
        b = self.branch(x)            # (B, p)
        T = self.basis()              # (M, p)
        return b @ T.T


# ---------------------------------------------------------------------------
# Inference wrapper used inside the hybrid solver (CPU, single thread)
# ---------------------------------------------------------------------------

class DeepONetCorrector:
    """Loads a checkpoint; caches the trunk basis; applies the scale-equivariant
    band-limited correction to (B, N, N) float64 residuals."""

    def __init__(self, path, threads=1):
        torch.set_num_threads(threads)
        ckp = torch.load(path, map_location="cpu", weights_only=False)
        a = ckp["args"]
        self.N, self.n_c = a["N"], a["n_c"]
        self.n_cx, self.n_cy = a.get("n_cx", a["n_c"]), a.get("n_cy", a["n_c"])
        self.model = CoarseDeepONet(self.n_cx, p=a["p"], hidden=a["hidden"], layers=a["layers"],
                                    trunk_hidden=a["trunk_hidden"], trunk_layers=a["trunk_layers"],
                                    skip=a.get("skip", True), trunk=a.get("trunk", "fourier"),
                                    mlp=bool(a.get("mlp", 1)), n_cy=self.n_cy)
        self.model.load_state_dict(ckp["model"])
        self.model.eval()
        with torch.no_grad():
            self.basis_T = self.model.basis().T.contiguous()  # (p, M)
            # fold the linear maps that follow the last nonlinearity into
            # single matrices (deployment-time weight folding):
            #   y = x @ A_lin + h(x) @ A_mlp + c
            if self.model.branch_mlp is not None:
                last = self.model.branch_mlp[-1]
                self.A_mlp = (last.weight.T @ self.basis_T).contiguous()    # (hidden, M)
                self.c = (last.bias @ self.basis_T).contiguous()            # (M,)
                self.hidden_net = self.model.branch_mlp[:-1]
            else:
                self.A_mlp = None
            if self.model.branch_skip is not None:
                self.A_lin = (self.model.branch_skip.weight.T @ self.basis_T).contiguous()  # (M, M)
            else:
                self.A_lin = None
        self.branch = self.model.branch
        self.in_scale = float(ckp.get("input_scale", 1.0))
        self.inv_scale = 1.0 / ckp["target_scale"]
        self.transfer = BandTransfer(self.N, self.n_cx, self.n_cy)
        self.args = a

    def _net(self, xc):
        """xc: (B, n_c, n_c) float64 normalised coarse residual -> coarse du."""
        B = xc.shape[0]
        x = torch.from_numpy((xc.reshape(B, -1) * self.in_scale).astype(np.float32))
        with torch.no_grad():
            y = x @ self.A_lin if self.A_lin is not None else 0.0
            if self.A_mlp is not None:
                y = y + self.hidden_net(x) @ self.A_mlp + self.c
        return y.numpy().astype(np.float64).reshape(B, self.n_cx, self.n_cy)

    def _net_reference(self, xc):
        """Unfolded evaluation (for checking the folded weights)."""
        B = xc.shape[0]
        x = torch.from_numpy((xc.reshape(B, -1) * self.in_scale).astype(np.float32))
        with torch.no_grad():
            y = self.branch(x) @ self.basis_T
        return y.numpy().astype(np.float64).reshape(B, self.n_cx, self.n_cy)

    def correct(self, r):
        """r: (B, N, N) residual -> additive correction du (B, N, N)."""
        rc = self.transfer.restrict(r)
        rn = np.sqrt((rc ** 2).sum(axis=(-2, -1), keepdims=True))
        rn = np.maximum(rn, 1e-300)
        yc = self._net(rc / rn) * (rn * self.inv_scale)
        return self.transfer.prolong(yc)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _random_band_fields(pde, transfer, rng, n, mode):
    """Coarse (in-band) residual/error training pairs with diverse spectra.
    Returns rc, ec with shape (n, n_c, n_c), where ec = R A^{-1} P rc."""
    N = pde.N
    if mode == "grf":            # forcing-like inputs (the solution operator's data)
        r = GRF2D(N, rng=rng).sample(n)
    elif mode == "rough":        # residual-like inputs: A applied to a GRF error
        e = GRF2D(N, rng=rng).sample(n)
        r = pde.apply_A(e)
    elif mode == "white":        # flat in-band spectrum (worst case for the net)
        r = rng.standard_normal((n, N, N))
    elif mode == "tilted":       # random power-law tilt + random anisotropy
        k = np.fft.fftfreq(N) * N
        kx, ky = np.meshgrid(k, k[: N // 2 + 1], indexing="ij")
        gam = rng.uniform(-1.0, 3.0, size=(n, 1, 1))
        ax = np.exp(rng.uniform(-1.0, 1.0, size=(n, 1, 1)))
        k2 = (ax * kx[None]) ** 2 + (ky[None] / ax) ** 2 + 1.0
        z = rng.standard_normal((n, N, N // 2 + 1)) + 1j * rng.standard_normal((n, N, N // 2 + 1))
        r = np.fft.irfft2(z * k2 ** (-gam / 2), s=(N, N), norm="ortho")
    else:
        raise ValueError(mode)
    r = demean(r)
    rc = transfer.restrict(r)
    e = pde.solve_direct(transfer.prolong(rc))
    ec = transfer.restrict(e)
    return rc, ec


def make_training_pairs(pde, transfer, rng, n, mix=("grf", "rough", "white", "tilted"), chunk=1024):
    """Pairs are generated in chunks so that only coarse (n_c x n_c) fields are
    kept in memory (fine-grid fields are transient), which keeps large grids
    tractable."""
    parts_r, parts_e = [], []
    per = int(math.ceil(n / len(mix)))
    for m in mix:
        done = 0
        while done < per:
            k = min(chunk, per - done)
            rc, ec = _random_band_fields(pde, transfer, rng, k, m)
            parts_r.append(rc)
            parts_e.append(ec)
            done += k
    rc = np.concatenate(parts_r)[:n]
    ec = np.concatenate(parts_e)[:n]
    rn = np.sqrt((rc ** 2).sum(axis=(-2, -1), keepdims=True))
    return (rc / rn).reshape(n, -1).astype(np.float32), (ec / rn).reshape(n, -1).astype(np.float32)


def train(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.ckp_dir, exist_ok=True)
    tag = f"deeponet_{args.equation}_{args.N}"
    device = torch.device(args.device)
    pde = FastStencilPDE(args.N, equation=args.equation, b_vec=(args.b_vel, args.b_vel))
    cx = args.coarsen_x if args.coarsen_x else args.coarsen
    cy = args.coarsen_y if args.coarsen_y else args.coarsen
    n_cx, n_cy = args.N // cx, args.N // cy
    n_c = n_cx
    transfer = BandTransfer(args.N, n_cx, n_cy)
    M = n_cx * n_cy
    input_scale = math.sqrt(M)  # unit-norm inputs -> O(1) entries

    t0 = time.time()
    X, Y = make_training_pairs(pde, transfer, rng, args.n_train)
    Xv, Yv = make_training_pairs(pde, transfer, np.random.default_rng(args.seed + 1), args.n_val)
    init_ck = None
    if args.init_from:
        init_ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        target_scale = float(init_ck["target_scale"])
    else:
        target_scale = float(math.sqrt(M) / np.median(np.linalg.norm(Y, axis=1)))
    X *= input_scale
    Xv *= input_scale
    Y *= target_scale
    Yv *= target_scale
    extra = None
    if args.extra_pairs:
        d = np.load(args.extra_pairs)
        Xe = d["X"].astype(np.float32) * input_scale
        Ye = d["Y"].astype(np.float32) * target_scale
        extra = (Xe, Ye)
        print(f"  extra rollout pairs: {len(Xe)}")
    print(f"[{tag}] data: {len(X)} train / {len(Xv)} val pairs on {n_cx}x{n_cy} sensors "
          f"in {time.time()-t0:.1f}s; target_scale {target_scale:.3g}")

    model = CoarseDeepONet(n_cx, p=args.p, hidden=args.hidden, layers=args.layers,
                           trunk_hidden=args.trunk_hidden, trunk_layers=args.trunk_layers,
                           skip=bool(args.skip), trunk=args.trunk, mlp=bool(args.mlp), n_cy=n_cy).to(device)
    if init_ck is not None:
        model.load_state_dict(init_ck["model"])
        print(f"  initialised from {args.init_from}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params/1e6:.2f}M")
    if args.linear_fit == "ls" and model.branch_skip is not None and init_ck is None:
        # Fit the linear path of the branch net by ridge regression on the
        # training pairs (float64, CPU). y = x @ W^T @ B^T  =>  x @ W^T = y @ B
        # (B has orthonormal columns), i.e. an ordinary least-squares problem.
        t1 = time.time()
        B = model.basis().detach().cpu().double().numpy()      # (M, p)
        Xd = X.astype(np.float64)
        Td = Y.astype(np.float64) @ B                            # (n, p)
        G = Xd.T @ Xd
        G[np.diag_indices_from(G)] += args.ls_ridge * np.trace(G) / len(G)
        Wt = np.linalg.solve(G, Xd.T @ Td)                       # (M, p)
        with torch.no_grad():
            model.branch_skip.weight.copy_(torch.tensor(Wt.T, dtype=torch.float32))
        print(f"  linear path fitted by least squares in {time.time()-t1:.0f}s", flush=True)
        args.linear_steps = 0
        _ls_val = None
    # two-stage optimisation: the linear branch path first (it fits the linear
    # inverse essentially exactly), then the whole network at a reduced
    # learning rate so the nonlinear path can only refine.
    linear_params = [model.branch_skip.weight] if model.branch_skip is not None else []
    other_params = [p_ for n_, p_ in model.named_parameters() if not n_.startswith("branch_skip")]
    if not other_params and args.linear_fit == "ls":
        args.steps = 0  # nothing left to train by SGD
    stage1 = min(args.linear_steps, args.steps) if linear_params else 0
    stage2 = args.steps - stage1
    opt = torch.optim.AdamW([{"params": linear_params, "lr": args.lr},
                             {"params": other_params, "lr": args.lr * args.mlp_lr_factor}],
                            lr=args.lr, weight_decay=args.weight_decay)
    sched1 = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.lr * args.mlp_lr_factor],
                                                 total_steps=max(stage1, 1), pct_start=0.05,
                                                 anneal_strategy="cos", div_factor=10.0,
                                                 final_div_factor=100.0) if stage1 else None
    sched2 = None
    Xt, Yt = torch.tensor(X).to(device), torch.tensor(Y).to(device)
    if extra is not None:
        Xt = torch.cat([Xt, torch.tensor(extra[0]).to(device)])
        Yt = torch.cat([Yt, torch.tensor(extra[1]).to(device)])
    Xv_d, Yv_d = torch.tensor(Xv).to(device), torch.tensor(Yv).to(device)

    def rel_err(pred, tgt):
        return torch.linalg.norm(pred - tgt, dim=1) / torch.linalg.norm(tgt, dim=1)

    def validate():
        model.eval()
        errs = []
        with torch.no_grad():
            for i in range(0, len(Xv_d), 1024):
                errs.append(rel_err(model(Xv_d[i:i + 1024]), Yv_d[i:i + 1024]).cpu())
        e = torch.cat(errs)
        model.train()
        return e.median().item(), e.mean().item(), e.max().item()

    best = np.inf
    n = len(Xt)
    med, mean, mx = validate()
    print(f"  before SGD: val_rel med {med:.2e} mean {mean:.2e} max {mx:.2e}", flush=True)
    if args.steps == 0:
        best = med
        torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                    "args": {**vars(args), "n_c": n_c, "n_cx": n_cx, "n_cy": n_cy}, "target_scale": target_scale,
                    "input_scale": input_scale},
                   f"{args.ckp_dir}/{tag}{args.suffix}_best.pth")
    model.train()
    t0 = time.time()
    run_loss = 0.0
    for p_ in other_params:
        p_.requires_grad_(stage1 == 0)
    for step in range(args.steps):
        if step == stage1 and stage2 > 0:
            for p_ in other_params:
                p_.requires_grad_(True)
            if args.freeze_linear:
                for p_ in linear_params:
                    p_.requires_grad_(False)
            groups = [{"params": other_params, "lr": args.lr * args.mlp_lr_factor}]
            if not args.freeze_linear:
                groups.append({"params": linear_params, "lr": args.lr * args.mlp_lr_factor})
            opt = torch.optim.AdamW(groups, lr=args.lr * args.mlp_lr_factor,
                                    weight_decay=args.weight_decay)
            sched2 = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=args.lr * args.mlp_lr_factor, total_steps=stage2, pct_start=0.05,
                anneal_strategy="cos", div_factor=10.0, final_div_factor=100.0)
            print(f"  -> stage 2 (full network, lr x{args.mlp_lr_factor})", flush=True)
        idx = torch.randint(0, n, (args.batch_size,), device=device)
        xb, yb = Xt[idx], Yt[idx]
        if args.mixup:
            # the target map is linear, so random linear combinations of
            # training pairs are exact new pairs (fresh data every step; the
            # branch MLP cannot memorise the finite training set)
            idx2 = torch.randint(0, n, (args.batch_size,), device=device)
            a = torch.randn(args.batch_size, 1, device=device)
            b = torch.randn(args.batch_size, 1, device=device)
            xb = a * xb + b * Xt[idx2]
            yb = a * yb + b * Yt[idx2]
            nrm = torch.linalg.norm(xb, dim=1, keepdim=True).clamp_min(1e-12) / input_scale
            xb, yb = xb / nrm, yb / nrm
        opt.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = torch.mean(rel_err(pred, yb) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        (sched2 if step >= stage1 and sched2 is not None else sched1).step()
        run_loss = 0.98 * run_loss + 0.02 * loss.item() if step else loss.item()
        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            med, mean, mx = validate()
            mark = ""
            if step + 1 == stage1 and stage2 > 0:
                best = np.inf  # checkpoint selection restarts with the full network
            if med < best and (step >= stage1 or stage2 == 0):
                best = med
                torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                            "args": {**vars(args), "n_c": n_c, "n_cx": n_cx, "n_cy": n_cy}, "target_scale": target_scale,
                            "input_scale": input_scale},
                           f"{args.ckp_dir}/{tag}{args.suffix}_best.pth")
                mark = " *"
            print(f"  step {step+1:6d} train_relmse {run_loss:.3e} val_rel med {med:.2e} "
                  f"mean {mean:.2e} max {mx:.2e}  ({time.time()-t0:.0f}s){mark}", flush=True)
    with open(f"{args.ckp_dir}/{tag}{args.suffix}_meta.json", "w") as fh:
        json.dump({"best_val_median_rel": best, "target_scale": target_scale,
                   "input_scale": input_scale, "n_params": n_params, **vars(args), "n_c": n_c,
                   "n_cx": n_cx, "n_cy": n_cy},
                  fh, indent=1)
    print(f"[{tag}] done; best val median rel err {best:.3e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--equation", default="Poisson", choices=["Poisson", "ConvDiff", "AnisoDiff", "VarCoeff"])
    p.add_argument("--N", type=int, default=128)
    p.add_argument("--coarsen", type=int, default=4)
    p.add_argument("--coarsen_x", type=int, default=0, help="sensor-grid coarsening along x (0: --coarsen)")
    p.add_argument("--coarsen_y", type=int, default=0, help="sensor-grid coarsening along y (0: --coarsen)")
    p.add_argument("--p", type=int, default=1024)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--trunk_hidden", type=int, default=512)
    p.add_argument("--trunk_layers", type=int, default=2)
    p.add_argument("--skip", type=int, default=1)
    p.add_argument("--mlp", type=int, default=0,
                   help="add a GELU MLP path to the branch net (off: linear branch)")
    p.add_argument("--trunk", default="fourier", choices=["fourier", "mlp"])
    p.add_argument("--mixup", type=int, default=1)
    p.add_argument("--linear_steps", type=int, default=10000)
    p.add_argument("--linear_fit", default="ls", choices=["ls", "sgd"])
    p.add_argument("--freeze_linear", type=int, default=1,
                   help="stage 2 trains only the nonlinear branch path")
    p.add_argument("--ls_ridge", type=float, default=1e-8)
    p.add_argument("--mlp_lr_factor", type=float, default=0.05)
    p.add_argument("--n_train", type=int, default=64000)
    p.add_argument("--n_val", type=int, default=2000)
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--b_vel", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--ckp_dir", default="./checkpoints")
    p.add_argument("--suffix", default="")
    p.add_argument("--extra_pairs", default=None, help="npz with X,Y rollout residual pairs")
    p.add_argument("--init_from", default=None)
    train(p.parse_args())
