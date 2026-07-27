"""Scale-equivariant FNO corrector for Poisson (periodic), trained on FFT truth.

Why scale-equivariant: inside a hybrid loop the network sees residuals whose
amplitude decays over many orders of magnitude. Training on f/||f|| -> u/||f||
and applying  NO(r) = ||r|| * net(r/||r||) / s  keeps the corrector useful
late in the solve (same trick as the wallclock-rebuttal DeepONet pipeline).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from fast_pde import FastStencilPDE, GRF2D, demean, l2

# repo root for ml_solver.FNOforPDE
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from ml_solver import FNOforPDE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=31)
    p.add_argument("--n_train", type=int, default=8000)
    p.add_argument("--n_val", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--trunc_mode", type=int, default=12)
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--ckp_dir", type=str, default="./checkpoints")
    p.add_argument("--tag", type=str, default="fno_poisson")
    return p.parse_args()


def main():
    args = parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.ckp_dir, exist_ok=True)
    tag = f"{args.tag}_N{args.N}"

    print(f"[{tag}] generating data on device={device} ...", flush=True)
    t0 = time.time()
    pde = FastStencilPDE(args.N, equation="Poisson")
    grf = GRF2D(args.N, rng=np.random.default_rng(args.seed))
    f_all = grf.sample(args.n_train + args.n_val)
    u_all = demean(pde.solve_direct(f_all))

    fnorm = l2(f_all)[:, None, None]
    f_n = (f_all / np.maximum(fnorm, 1e-300)).astype(np.float32)
    u_n = (u_all / np.maximum(fnorm, 1e-300)).astype(np.float32)
    target_scale = float(1.0 / max(np.median(np.linalg.norm(u_n[: args.n_train].reshape(args.n_train, -1), axis=1)), 1e-12))
    u_n = u_n * target_scale
    print(f"  data in {time.time()-t0:.1f}s; target_scale={target_scale:.4f}", flush=True)

    f_tr = torch.from_numpy(f_n[: args.n_train])
    u_tr = torch.from_numpy(u_n[: args.n_train])
    f_va = torch.from_numpy(f_n[args.n_train:]).to(device)
    u_va = torch.from_numpy(u_n[args.n_train:]).to(device)

    model = FNOforPDE(
        trunc_mode=args.trunc_mode,
        dim=2,
        N=args.N,
        in_channels=1,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(args.epochs // 2, 1), gamma=0.5)

    def val_rel():
        model.eval()
        errs = []
        with torch.no_grad():
            for i in range(0, len(f_va), 256):
                xb = f_va[i:i + 256].unsqueeze(1)
                pred = model(xb).reshape(xb.shape[0], -1)
                tgt = u_va[i:i + 256].reshape(xb.shape[0], -1)
                errs.append((torch.linalg.norm(pred - tgt, dim=1) / torch.linalg.norm(tgt, dim=1)).cpu())
        e = torch.cat(errs)
        return e.median().item(), e.mean().item()

    best = np.inf
    n_batches = int(np.ceil(args.n_train / args.batch_size))
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(args.n_train)
        tot = 0.0
        for b in range(n_batches):
            idx = perm[b * args.batch_size:(b + 1) * args.batch_size]
            xb = f_tr[idx].unsqueeze(1).to(device)
            yb = u_tr[idx].to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = torch.mean((pred - yb) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        sched.step()
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            med, mean = val_rel()
            mark = ""
            if med < best:
                best = med
                torch.save(
                    {"model": model.state_dict(), "args": vars(args), "target_scale": target_scale},
                    f"{args.ckp_dir}/{tag}_best.pth",
                )
                mark = " *"
            print(
                f"  epoch {epoch:4d}  train_mse {tot/n_batches:.3e}  "
                f"val_relL2 med {med:.4f} mean {mean:.4f}{mark}",
                flush=True,
            )

    with open(f"{args.ckp_dir}/{tag}_meta.json", "w") as fh:
        json.dump({"best_val_median_relL2": best, "target_scale": target_scale, **vars(args)}, fh, indent=1)
    print(f"[{tag}] done. best val median rel L2 = {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
