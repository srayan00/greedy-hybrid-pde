"""Train a DeepONet solution-operator surrogate for the wall-clock benchmarks.

Differences from train_ml_solver.py (all needed to make wall-clock evaluation
feasible and the surrogate usable inside a solver loop):

  * Ground-truth solutions come from the FFT diagonalization of the *same*
    discrete operator (fast_pde.FastStencilPDE.solve_direct), instead of dense
    torch.linalg.lstsq, so data generation is O(N^2 log N) per sample and we can
    train at N = 63 / 127 quickly.
  * Scale-equivariant training: the map f -> u is linear, so we train on
    f/||f|| -> u/||f|| and at inference apply  NO(r) := ||r|| * net(r/||r||).
    Without this the network is only accurate at the amplitude scale of the
    training data, which is exactly why the DeepONet correction stops being
    useful once the residual has decayed a few orders of magnitude.
  * The architecture is the paper's DeepONet (ml_solver.DeepONet) unchanged.

Usage:
  python train_fast_deeponet.py --equation Poisson --N 63 --epochs 300
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fast_pde import GRF2D, demean, l2
from ml_solver import DeepONet
from pde_factory import ckp_tag, make_pde

parser = argparse.ArgumentParser()
parser.add_argument("--equation", type=str, default="Poisson",
                    choices=["Poisson", "ConvDiff", "AnisoDiff"])
parser.add_argument("--N", type=int, default=63)
parser.add_argument("--n_train", type=int, default=10000)
parser.add_argument("--n_val", type=int, default=1000)
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--branch_dim", type=int, default=128)
parser.add_argument("--hidden", type=int, default=256)
parser.add_argument("--layers", type=int, default=4)
parser.add_argument("--b_vel", type=float, default=20.0)
parser.add_argument("--eps_x", type=float, default=0.01)
parser.add_argument("--eps_y", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
parser.add_argument("--ckp_dir", type=str, default="./checkpoints")
parser.add_argument("--tag_suffix", type=str, default="",
                    help="Appended to checkpoint tag (e.g. L2H64) so arch ablations "
                         "do not overwrite the default model.")
args = parser.parse_args()

torch.manual_seed(args.seed)
os.makedirs(args.ckp_dir, exist_ok=True)
tag = ckp_tag(args.equation, args.N, eps_x=args.eps_x, eps_y=args.eps_y)
if args.tag_suffix:
    tag = f"{tag}_{args.tag_suffix}"
device = torch.device(args.device)

print(f"[{tag}] generating {args.n_train + args.n_val} samples on N={args.N} grid ...")
t0 = time.time()
pde = make_pde(args.equation, args.N, b_vel=args.b_vel, eps_x=args.eps_x, eps_y=args.eps_y)
grf = GRF2D(args.N, rng=np.random.default_rng(args.seed))
f_all = grf.sample(args.n_train + args.n_val)
u_all = demean(pde.solve_direct(f_all))

# scale-equivariant normalization: f/||f|| -> u/||f||, then a single global
# rescaling so targets are O(1) (stored in the checkpoint, undone at inference)
fnorm = l2(f_all)[:, None, None]
f_all = (f_all / fnorm).astype(np.float32).reshape(len(f_all), -1)
u_all = (u_all / fnorm).astype(np.float32).reshape(len(u_all), -1)
target_scale = float(1.0 / np.median(np.linalg.norm(u_all[: args.n_train], axis=1)))
u_all = u_all * target_scale
print(f"  data done in {time.time()-t0:.1f}s; target_scale = {target_scale:.3f}; "
      f"||u_scaled|| median = {np.median(np.linalg.norm(u_all, axis=1)):.4f}")

f_tr = torch.tensor(f_all[: args.n_train])
u_tr = torch.tensor(u_all[: args.n_train])
f_va = torch.tensor(f_all[args.n_train:]).to(device)
u_va = torch.tensor(u_all[args.n_train:]).to(device)

model = DeepONet(N=args.N, dim=2, device=device, in_channels=1, boundary="Periodic",
                 branch_dim=args.branch_dim, hidden_branch=args.hidden,
                 num_branch_layers=args.layers, hidden_trunk=args.hidden,
                 num_trunk_layers=args.layers).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(args.epochs // 2, 1), gamma=0.5)


def val_rel_err():
    model.eval()
    errs = []
    with torch.no_grad():
        for i in range(0, len(f_va), 512):
            pred = model(f_va[i:i + 512]).reshape(-1, args.N * args.N)
            tgt = u_va[i:i + 512]
            errs.append((torch.linalg.norm(pred - tgt, dim=1)
                         / torch.linalg.norm(tgt, dim=1)).cpu())
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
        xb = f_tr[idx].to(device, non_blocking=True)
        yb = u_tr[idx].to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        pred = model(xb).reshape(len(idx), -1)
        loss = torch.mean((pred - yb) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot += loss.item()
    sched.step()
    if epoch % 10 == 0 or epoch == args.epochs - 1:
        med, mean = val_rel_err()
        marker = ""
        if med < best:
            best = med
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "target_scale": target_scale},
                       f"{args.ckp_dir}/{tag}_best.pth")
            marker = " *"
        print(f"  epoch {epoch:4d}  train_mse {tot/n_batches:.3e}  "
              f"val_relL2 med {med:.4f} mean {mean:.4f}{marker}", flush=True)

med, mean = val_rel_err()
if med < best:
    best = med
    torch.save({"model": model.state_dict(), "args": vars(args),
                "target_scale": target_scale}, f"{args.ckp_dir}/{tag}_best.pth")
with open(f"{args.ckp_dir}/{tag}_meta.json", "w") as fh:
    json.dump({"best_val_median_relL2": best, "target_scale": target_scale, **vars(args)}, fh, indent=1)
print(f"[{tag}] done. best val median rel L2 = {best:.4f}")
