"""Scale-equivariant DeepONet corrector (from wallclock-rebuttal fork)."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from ml_solver import DeepONet


class DeepONetCorrector:
    """Loads a train_fast_deeponet.py checkpoint; caches the trunk basis
    (the grid is fixed, so the trunk MLP never needs re-evaluation) and applies
    the scale-equivariant correction: du = ||r|| * net(r/||r||) / s."""

    def __init__(self, ckp_path, device="cpu", threads=1):
        torch.set_num_threads(threads)
        ckp = torch.load(ckp_path, map_location="cpu", weights_only=False)
        a = ckp["args"]
        dev = torch.device(device)
        model = DeepONet(
            N=a["N"],
            dim=2,
            device=dev,
            in_channels=1,
            boundary="Periodic",
            branch_dim=a["branch_dim"],
            hidden_branch=a["hidden"],
            num_branch_layers=a["layers"],
            hidden_trunk=a["hidden"],
            num_trunk_layers=a["layers"],
        ).to(dev)
        model.load_state_dict(ckp["model"])
        model.eval()
        with torch.no_grad():
            trunk = model.trunk_net(model.coords)
        self.trunk_T = trunk.transpose(0, 1).contiguous().to(dev)
        self.branch = model.branch_net
        self.inv_scale = 1.0 / ckp["target_scale"]
        self.N = a["N"]
        self.device = dev

    def correct(self, r):
        """r: (B, N, N) float64 residual -> additive correction du (B, N, N)."""
        B = r.shape[0]
        rn = np.sqrt((r ** 2).sum(axis=(-2, -1), keepdims=True))
        rn = np.maximum(rn, 1e-300)
        x = torch.from_numpy((r / rn).reshape(B, -1).astype(np.float32)).to(self.device)
        with torch.no_grad():
            out = self.branch(x) @ self.trunk_T
        du = out.cpu().numpy().astype(np.float64).reshape(r.shape) * rn * self.inv_scale
        return du - du.mean(axis=(-2, -1), keepdims=True)
