"""Diagnose why the 2D Helmholtz DeepONet fails on some samples."""
import torch, json, numpy as np
from data_generation import GaussianRandomFieldHierarchical
from pde import HelmholtzEquation2D
from ml_solver import DeepONet

device = torch.device('cuda')
N = 31
n_test = 128

grf = GaussianRandomFieldHierarchical(N, 2, 0.01, 100.0, 0.1, 1000.0,
    [0.5,1.0,1.5,2.0,2.5,3.0,4.0], device, seed=9999)
f_raw = grf.generate(n_test, pushfoward=None)
grf_k2 = GaussianRandomFieldHierarchical(N, 2, 0.01, 100.0, 0.1, 1000.0,
    [0.5,1.0,1.5,2.0,2.5,3.0,4.0], device, seed=10000)
k2_raw = grf_k2.generate(n_test)

f_flat = f_raw.reshape(n_test, N*N)
k2_flat = k2_raw.reshape(n_test, N*N)

x = torch.linspace(0, 1, N+1, device=device)[:-1]
y = torch.linspace(0, 1, N+1, device=device)[:-1]
pde = HelmholtzEquation2D(a_func=lambda xi,yi: 1.0, f_func=f_flat, k2=k2_flat,
    boundary='Periodic', x=x, y=y, device=device)
u_sol = pde.u.clone().detach().float().reshape(n_test, N*N)

# Solution magnitude distribution
u_norms = torch.linalg.norm(u_sol, dim=-1).cpu().numpy()
u_max = torch.abs(u_sol).max(dim=-1).values.cpu().numpy()
k2_norms = torch.linalg.norm(k2_flat, dim=-1).cpu().numpy()
f_norms = torch.linalg.norm(f_flat, dim=-1).cpu().numpy()

print("=== Solution magnitude distribution ===")
print(f"  ||u||_2:  min={u_norms.min():.4e}  max={u_norms.max():.4e}  "
      f"ratio={u_norms.max()/u_norms.min():.1f}x")
print(f"  max|u|:   min={u_max.min():.4e}  max={u_max.max():.4e}  "
      f"ratio={u_max.max()/u_max.min():.1f}x")
print(f"  ||f||_2:  min={f_norms.min():.4e}  max={f_norms.max():.4e}")
print(f"  ||k2||_2: min={k2_norms.min():.4e}  max={k2_norms.max():.4e}")
print(f"\n  Percentiles of ||u||_2:")
for p in [1, 5, 25, 50, 75, 95, 99]:
    print(f"    p{p:02d} = {np.percentile(u_norms, p):.4e}")

# Load DeepONet and evaluate per-sample
with open('args/deeponet_args.json') as fj:
    dargs = json.load(fj)
don = DeepONet(N=N, dim=2, in_channels=2, device=device, boundary='Periodic',
    branch_dim=dargs['branch_dim'], hidden_branch=dargs['hidden_branch'],
    num_branch_layers=dargs['num_branch_layers'], hidden_trunk=dargs['hidden_trunk'],
    num_trunk_layers=dargs['num_trunk_layers']).to(device)
ckp = torch.load('checkpoints/deeponet_hier_v2_Helmholtz_Periodic_2d_1c_best.pth',
                  map_location=device)
don.load_state_dict(ckp['model']); don.eval()

inputs = torch.cat([k2_flat[:, None, :], f_flat[:, None, :]], dim=1)
with torch.no_grad():
    preds = don(inputs).reshape(n_test, -1)

abs_errors = torch.linalg.norm(preds - u_sol, dim=-1).cpu().numpy()
rel_errors = abs_errors / np.maximum(u_norms, 1e-15)

print(f"\n=== DeepONet prediction errors ===")
print(f"  Abs L2:  mean={abs_errors.mean():.4e}  std={abs_errors.std():.4e}")
print(f"  Rel L2:  mean={rel_errors.mean():.4e}  std={rel_errors.std():.4e}")

# Show correlation: do large-solution samples have lower relative error?
print(f"\n=== Per-sample: solution norm vs abs/rel error ===")
sort_idx = np.argsort(u_norms)
print(f"  {'||u||':>12s}  {'Abs err':>12s}  {'Rel err':>12s}  {'||pred||':>12s}")
pred_norms = torch.linalg.norm(preds, dim=-1).cpu().numpy()
for i in [0, 1, 2, 3, 4, n_test//4, n_test//2, 3*n_test//4, n_test-4, n_test-3, n_test-2, n_test-1]:
    idx = sort_idx[i]
    print(f"  {u_norms[idx]:12.4e}  {abs_errors[idx]:12.4e}  {rel_errors[idx]:12.4e}  {pred_norms[idx]:12.4e}")
