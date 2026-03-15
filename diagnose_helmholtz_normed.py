"""Check the normalized-f v3 Helmholtz DeepONet quality."""
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

# Normalize f to unit norm (matching training)
f_flat = f_raw.reshape(n_test, -1)
f_norms = torch.linalg.norm(f_flat, dim=-1, keepdim=True).clamp(min=1e-15)
f_raw = f_raw / f_norms.unsqueeze(-1)
f_flat = f_raw.reshape(n_test, N*N)

grf_k2 = GaussianRandomFieldHierarchical(N, 2, 0.01, 100.0, 0.1, 1000.0,
    [0.5,1.0,1.5,2.0,2.5,3.0,4.0], device, seed=10000)
k2_raw = grf_k2.generate(n_test)
k2_flat = k2_raw.reshape(n_test, N*N)

x = torch.linspace(0, 1, N+1, device=device)[:-1]
y = torch.linspace(0, 1, N+1, device=device)[:-1]
pde = HelmholtzEquation2D(a_func=lambda xi,yi: 1.0, f_func=f_flat, k2=k2_flat,
    boundary='Periodic', x=x, y=y, device=device)
u_sol = pde.u.clone().detach().float().reshape(n_test, N*N)

u_norms = torch.linalg.norm(u_sol, dim=-1).cpu().numpy()

print("=== Solution magnitude distribution (normalized f) ===")
print(f"  ||u||_2:  min={u_norms.min():.4e}  max={u_norms.max():.4e}  ratio={u_norms.max()/u_norms.min():.1f}x")
for p in [1, 25, 50, 75, 99]:
    print(f"    p{p:02d} = {np.percentile(u_norms, p):.4e}")

# Load model
args_path = 'checkpoints/deeponet_hier_v3_Helmholtz_Periodic_2d_1c_args.json'
with open(args_path) as fj:
    dargs = json.load(fj)
don = DeepONet(N=N, dim=2, in_channels=2, device=device, boundary='Periodic',
    branch_dim=dargs['branch_dim'], hidden_branch=dargs['hidden_branch'],
    num_branch_layers=dargs['num_branch_layers'], hidden_trunk=dargs['hidden_trunk'],
    num_trunk_layers=dargs['num_trunk_layers']).to(device)
ckp = torch.load('checkpoints/deeponet_hier_v3_Helmholtz_Periodic_2d_1c_best.pth',
                  map_location=device)
don.load_state_dict(ckp['model']); don.eval()
print(f"Loaded model from epoch {ckp['epoch']}")

inputs = torch.cat([k2_flat[:, None, :], f_flat[:, None, :]], dim=1)
with torch.no_grad():
    preds = don(inputs).reshape(n_test, -1)

pred_norms = torch.linalg.norm(preds, dim=-1).cpu().numpy()
abs_errors = torch.linalg.norm(preds - u_sol, dim=-1).cpu().numpy()
rel_errors = abs_errors / np.maximum(u_norms, 1e-15)

print(f"\n=== Prediction quality ===")
print(f"  ||pred||:  min={pred_norms.min():.4e}  max={pred_norms.max():.4e}  mean={pred_norms.mean():.4e}")
print(f"  ||u||:     min={u_norms.min():.4e}  max={u_norms.max():.4e}  mean={u_norms.mean():.4e}")
print(f"  Std of ||pred||: {pred_norms.std():.4e}")
print(f"  Correlation: {np.corrcoef(pred_norms, u_norms)[0,1]:.4f}")
print(f"  Mean abs L2:  {abs_errors.mean():.4e} ± {abs_errors.std():.4e}")
print(f"  Mean rel L2:  {rel_errors.mean():.4e} ± {rel_errors.std():.4e}")

# Per-quartile
quartiles = np.percentile(u_norms, [0, 25, 50, 75, 100])
print(f"\n=== Per-quartile relative L2 error ===")
for i in range(4):
    lo, hi = quartiles[i], quartiles[i+1]
    mask = (u_norms >= lo) & (u_norms <= hi + 1e-20)
    if mask.sum() > 0:
        print(f"  Q{i+1} (||u|| in [{lo:.2e}, {hi:.2e}]): rel_L2={rel_errors[mask].mean():.4f}  "
              f"abs_L2={abs_errors[mask].mean():.4e}  n={mask.sum()}")
