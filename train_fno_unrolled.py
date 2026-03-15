"""
Unrolled fine-tuning of the FNO through an iterative solver loop.

Instead of training the FNO on i.i.d. (residual -> correction) pairs,
we run T steps of the hybrid solver (SOR portfolio + FNO) with oracle
greedy selection, and backprop through the trajectory to update the FNO.

At each step the FNO sees the *actual* residual that arises from the
mixed SOR+FNO trajectory -- not a random residual.  The loss is the
FNO's prediction error at every step (even when the oracle chose SOR),
which teaches it to be accurate on the real in-loop distribution.
"""

import argparse
import json
import os
import time
import torch
import numpy as np
from ml_solver import FNOforPDE
from numerical_solver import SORSolver, GaussSeidelSolver, WeightedJacobiSolver
from pde import PoissonEquation2D, ConvectionDiffusion2D

parser = argparse.ArgumentParser()
parser.add_argument("--fno_model_name", type=str, required=True,
                    help="Pre-trained FNO checkpoint name (e.g. hier_fno)")
parser.add_argument("--from_scratch", action="store_true",
                    help="Initialize FNO from scratch instead of loading pre-trained weights")
parser.add_argument("--save_name", type=str, required=True,
                    help="Name for the fine-tuned FNO (e.g. hier_fno_unrolled)")
parser.add_argument("--equation", type=str, default="Poisson", choices=["Poisson", "ConvDiff"])
parser.add_argument("--solver_specs", type=str, default="sor_1.0,sor_1.3,sor_1.6")
parser.add_argument("--b_vel", type=float, default=20.0)
parser.add_argument("--ckp_dir", type=str, default="./checkpoints")
parser.add_argument("--data_dir", type=str, default="./data")
parser.add_argument("--data_name", type=str, default="hier_")
parser.add_argument("--n_train", type=int, default=2048,
                    help="Number of training samples to use from pre-generated data")
parser.add_argument("--n_val", type=int, default=256,
                    help="Number of validation samples to use")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--T_unroll", type=int, default=30,
                    help="Starting unrolled iterations per training step")
parser.add_argument("--T_unroll_max", type=int, default=200,
                    help="Max unroll length (grown during training)")
parser.add_argument("--T_grow_every", type=int, default=15,
                    help="Grow T_unroll every this many epochs")
parser.add_argument("--T_grow_by", type=int, default=20,
                    help="Amount to grow T_unroll by")
parser.add_argument("--loss_mode", type=str, default="fno_error",
                    choices=["fno_error", "trajectory_error"],
                    help="fno_error: FNO prediction error at each step; "
                         "trajectory_error: actual trajectory error")
parser.add_argument("--N", type=int, default=31)


class LightweightPDE:
    """A minimal PDE wrapper that shares a pre-built A matrix
    and only updates b per batch.  Avoids expensive build_matrix calls."""

    def __init__(self, A_shared, b, equation_name="Poisson"):
        self.equation = equation_name
        self.A_shared = A_shared         # (N^2, N^2) — built once
        self.b = b                        # (B, N^2) — updated per batch
        self._A_expanded = None
        self._A_expanded_bs = None

    @property
    def A(self):
        bs = self.b.shape[0]
        if self._A_expanded is None or self._A_expanded_bs != bs:
            self._A_expanded = self.A_shared.unsqueeze(0).expand(bs, -1, -1)
            self._A_expanded_bs = bs
        return self._A_expanded

    def compute_residual(self, u):
        return self.b - torch.bmm(self.A, u.unsqueeze(-1)).squeeze(-1)

    def set_batch(self, b_new):
        self.b = b_new
        self._A_expanded = None
        self._A_expanded_bs = None


def make_solver(spec, pde, device):
    parts = spec.split("_")
    name = parts[0]
    if name == "sor":
        omega = float(parts[1]) if len(parts) > 1 else 1.5
        return SORSolver(pde, omega=omega, device=device)
    elif name == "jacobi":
        w = float(parts[1]) if len(parts) > 1 else 1.0
        return WeightedJacobiSolver(pde, device=device, weight=w)
    elif name == "gs":
        return GaussSeidelSolver(pde, device=device)
    raise ValueError(f"Unknown solver: {spec}")


def load_pretrained_fno(ckp_dir, model_name, equation, device):
    args_path = os.path.join(ckp_dir, f"fno_{model_name}_{equation}_Periodic_2d_1c_args.json")
    if os.path.exists(args_path):
        with open(args_path) as f:
            fno_args = json.load(f)
    else:
        with open("args/fno_n31_args.json") as f:
            fno_args = json.load(f)

    model = FNOforPDE(
        trunc_mode=fno_args["trunc_mode"], dim=2, in_channels=1,
        hidden_size=fno_args["hidden_size"], num_layers=fno_args["num_layers"],
    ).to(device)

    ckp_path = os.path.join(ckp_dir, f"fno_{model_name}_{equation}_Periodic_2d_1c_best.pth")
    if not os.path.exists(ckp_path):
        ckp_path = ckp_path.replace("_best.pth", "_full.pth")
    return model, fno_args, ckp_path


def build_system_matrix(N, equation, device, b_vel=20.0):
    """Build the system matrix A once using a dummy single-sample PDE."""
    x = torch.linspace(0, 1, N + 1, device=device)[:-1]
    y = torch.linspace(0, 1, N + 1, device=device)[:-1]
    a_func = lambda xi, yi: 1.0
    dummy_f = torch.zeros(1, N * N, device=device)
    if equation == "Poisson":
        pde = PoissonEquation2D(a_func=a_func, f_func=dummy_f,
                                boundary="Periodic", x=x, y=y,
                                solve=False, device=device)
    else:
        pde = ConvectionDiffusion2D(a_func=a_func, f_func=dummy_f,
                                     b_vec=(b_vel, b_vel),
                                     boundary="Periodic", x=x, y=y,
                                     solve=False, device=device)
    A = pde.A
    if A.ndim == 3:
        A = A[0]
    return A


def load_data(data_dir, data_name, equation, n_samples, split="train"):
    """Load pre-generated data (input f, target u) from .pt files."""
    eq_name = equation
    if split == "train":
        base = 10000
    else:
        base = 2000
    fname = f"{data_name}{split}_data_{eq_name}_Periodic_2d_1c_{base}s.pt"
    path = os.path.join(data_dir, fname)
    print(f"Loading {path}...")
    data = torch.load(path, map_location="cpu", weights_only=False)
    inputs = data[0][:n_samples]   # (N_samples, C, 31, 31)
    targets = data[1][:n_samples]  # (N_samples, 31, 31)

    f_flat = inputs[:, -1].reshape(n_samples, -1)   # last channel = f
    u_flat = targets.reshape(n_samples, -1)
    return f_flat, u_flat


def unrolled_step(fno, classical_solvers, pde, u_prev, u_sol,
                  subtract_mean, loss_mode):
    """One unrolled iteration: all candidates, oracle pick, FNO loss."""
    bs = u_prev.shape[0]

    # Classical solver candidates (detached — no grad through classical)
    with torch.no_grad():
        classical_cands = []
        classical_errs = []
        for solver in classical_solvers:
            solver.equation = pde
            u_cand = solver.iteration(u_prev)
            if subtract_mean:
                u_cand = u_cand - torch.mean(u_cand, dim=-1, keepdim=True)
            classical_cands.append(u_cand)
            classical_errs.append(torch.linalg.norm(u_cand - u_sol, dim=-1))

    # FNO candidate (WITH gradient)
    residual = pde.compute_residual(u_prev)
    fno_correction = fno(residual[:, None, :]).reshape(bs, -1)
    u_fno = u_prev + fno_correction
    if subtract_mean:
        u_fno = u_fno - torch.mean(u_fno, dim=-1, keepdim=True)
    err_fno = torch.linalg.norm(u_fno - u_sol, dim=-1)

    # Oracle greedy: pick solver with smallest error per sample
    all_errs = torch.stack(classical_errs + [err_fno], dim=0)  # (K+1, B)
    best_idx = torch.argmin(all_errs, dim=0)                   # (B,)
    ml_idx = len(classical_solvers)

    # Build u_next from oracle selection.
    # Classical candidates are detached; FNO keeps gradient when selected.
    all_cands = torch.stack(classical_cands + [u_fno], dim=0)  # (K+1, B, N^2)
    u_next = all_cands[best_idx, torch.arange(bs)]

    # Loss
    if loss_mode == "fno_error":
        step_loss = torch.mean(err_fno ** 2)
    else:
        traj_err = torch.linalg.norm(u_next - u_sol, dim=-1)
        step_loss = torch.mean(traj_err ** 2)

    fno_frac = (best_idx == ml_idx).float().mean().item()
    return u_next, step_loss, fno_frac


def train_one_epoch(fno, optimizer, classical_solvers, pde,
                    train_f, train_u, T_unroll, batch_size,
                    subtract_mean, loss_mode, device):
    fno.train()
    n = train_f.shape[0]
    perm = torch.randperm(n, device=device)
    total_loss = 0.0
    total_fno_usage = 0.0
    n_batches = 0

    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]
        f_b = train_f[idx]
        u_b = train_u[idx]

        pde.set_batch(f_b)

        optimizer.zero_grad()
        u_prev = torch.zeros_like(u_b)
        accum_loss = torch.tensor(0.0, device=device)
        batch_fno = 0.0

        for t in range(T_unroll):
            u_prev_in = u_prev.detach() if t > 0 else u_prev
            u_next, sloss, fno_f = unrolled_step(
                fno, classical_solvers, pde, u_prev_in, u_b,
                subtract_mean, loss_mode)
            accum_loss = accum_loss + sloss
            batch_fno += fno_f
            u_prev = u_next

        accum_loss = accum_loss / T_unroll
        accum_loss.backward()
        torch.nn.utils.clip_grad_norm_(fno.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += accum_loss.item()
        total_fno_usage += batch_fno / T_unroll
        n_batches += 1

    return total_loss / max(n_batches, 1), total_fno_usage / max(n_batches, 1)


@torch.no_grad()
def validate(fno, classical_solvers, pde, val_f, val_u,
             T_unroll, subtract_mean, loss_mode, device):
    fno.eval()
    bs = val_f.shape[0]
    pde.set_batch(val_f)

    u_prev = torch.zeros_like(val_u)
    total_loss = 0.0
    total_fno = 0.0
    ml_idx = len(classical_solvers)

    for t in range(T_unroll):
        classical_cands = []
        classical_errs = []
        for solver in classical_solvers:
            solver.equation = pde
            u_cand = solver.iteration(u_prev)
            if subtract_mean:
                u_cand = u_cand - torch.mean(u_cand, dim=-1, keepdim=True)
            classical_cands.append(u_cand)
            classical_errs.append(torch.linalg.norm(u_cand - val_u, dim=-1))

        residual = pde.compute_residual(u_prev)
        fno_corr = fno(residual[:, None, :]).reshape(bs, -1)
        u_fno = u_prev + fno_corr
        if subtract_mean:
            u_fno = u_fno - torch.mean(u_fno, dim=-1, keepdim=True)
        err_fno = torch.linalg.norm(u_fno - val_u, dim=-1)

        all_errs = torch.stack(classical_errs + [err_fno], dim=0)
        best_idx = torch.argmin(all_errs, dim=0)
        all_cands = torch.stack(classical_cands + [u_fno], dim=0)
        u_prev = all_cands[best_idx, torch.arange(bs)]

        if loss_mode == "fno_error":
            total_loss += torch.mean(err_fno ** 2).item()
        else:
            traj_err = torch.linalg.norm(u_prev - val_u, dim=-1)
            total_loss += torch.mean(traj_err ** 2).item()
        total_fno += (best_idx == ml_idx).float().mean().item()

    return total_loss / T_unroll, total_fno / T_unroll


if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = args.N
    N2 = N * N

    print(f"{'='*60}")
    print(f"Unrolled FNO fine-tuning")
    print(f"  equation={args.equation}, solvers={args.solver_specs}")
    print(f"  T_unroll={args.T_unroll} -> {args.T_unroll_max}")
    print(f"  loss={args.loss_mode}, lr={args.lr}")
    print(f"  device={device}")
    print(f"{'='*60}\n")

    # 1. Load pre-trained FNO
    fno, fno_args, ckp_path = load_pretrained_fno(args.ckp_dir, args.fno_model_name,
                                                    args.equation, device)
    if not args.from_scratch:
        ckp = torch.load(ckp_path, map_location=device, weights_only=False)
        state = {k: v for k, v in ckp["model"].items() if not k.startswith("_")}
        fno.load_state_dict(state)
        print(f"Loaded pre-trained FNO from {ckp_path} (epoch {ckp.get('epoch', '?')})")
    else:
        print("Initializing FNO from scratch (no pre-trained weights)")
    n_params = sum(p.numel() for p in fno.parameters())
    print(f"FNO parameters: {n_params:,}")

    # 2. Build shared system matrix A (once)
    print("Building system matrix A...")
    t0 = time.time()
    A_shared = build_system_matrix(N, args.equation, device, args.b_vel)
    print(f"  A shape: {A_shared.shape}  ({time.time()-t0:.1f}s)")

    # 3. Load pre-generated data
    subtract_mean = args.equation in ("Poisson", "ConvDiff")
    train_f, train_u = load_data(args.data_dir, args.data_name,
                                  args.equation, args.n_train, "train")
    val_f, val_u = load_data(args.data_dir, args.data_name,
                              args.equation, args.n_val, "val")
    train_f = train_f.to(device)
    train_u = train_u.to(device)
    val_f = val_f.to(device)
    val_u = val_u.to(device)
    if subtract_mean:
        train_u = train_u - torch.mean(train_u, dim=-1, keepdim=True)
        val_u = val_u - torch.mean(val_u, dim=-1, keepdim=True)
    print(f"  train: {train_f.shape}, val: {val_f.shape}")

    # 4. Create lightweight PDE and classical solvers
    pde = LightweightPDE(A_shared, train_f[:1], args.equation)
    solver_specs = [s.strip() for s in args.solver_specs.split(",")]
    classical_solvers = [make_solver(s, pde, device) for s in solver_specs]
    print(f"  classical solvers: {solver_specs}")

    # 5. Optimizer and scheduler
    optimizer = torch.optim.Adam(fno.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    save_prefix = os.path.join(
        args.ckp_dir,
        f"fno_{args.save_name}_{args.equation}_Periodic_2d_1c")

    # Save FNO args JSON so router training can find it
    args_save_path = f"{save_prefix}_args.json"
    with open(args_save_path, "w") as fp:
        json.dump(fno_args, fp)
    print(f"Saved FNO args to {args_save_path}")

    best_val_loss = float("inf")
    T_unroll = args.T_unroll

    print(f"\n{'='*60}")
    print(f"Training for {args.epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(args.epochs):
        if epoch > 0 and epoch % args.T_grow_every == 0 and T_unroll < args.T_unroll_max:
            T_unroll = min(T_unroll + args.T_grow_by, args.T_unroll_max)
            print(f"  >> T_unroll -> {T_unroll}")

        t0 = time.time()
        train_loss, train_fno = train_one_epoch(
            fno, optimizer, classical_solvers, pde,
            train_f, train_u, T_unroll, args.batch_size,
            subtract_mean, args.loss_mode, device)

        val_loss, val_fno = validate(
            fno, classical_solvers, pde, val_f, val_u,
            min(T_unroll, 100), subtract_mean, args.loss_mode, device)

        scheduler.step()
        dt = time.time() - t0

        print(f"Ep {epoch:3d} | T={T_unroll:3d} | "
              f"train={train_loss:.3e} fno={train_fno:.1%} | "
              f"val={val_loss:.3e} fno={val_fno:.1%} | "
              f"lr={scheduler.get_last_lr()[0]:.1e} | {dt:.0f}s")

        ckp = {
            "model": fno.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "T_unroll": T_unroll,
            "args": vars(args),
        }
        torch.save(ckp, f"{save_prefix}_full.pth")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckp, f"{save_prefix}_best.pth")
            print(f"  ** best val_loss={val_loss:.3e}")

    print(f"\nDone! best_val_loss={best_val_loss:.3e}")
    print(f"Saved to {save_prefix}_best.pth / _full.pth")
