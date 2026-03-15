"""Check per-sample greedy trajectories to understand the convergence behavior."""
import torch, json, numpy as np
from data_generation import GaussianRandomFieldHierarchical
from pde import HelmholtzEquation2D
from ml_solver import DeepONet
from numerical_solver import WeightedJacobiSolver

device = torch.device('cuda')
N = 31; n_test = 128

grf = GaussianRandomFieldHierarchical(N, 2, 0.01, 100.0, 0.1, 1000.0,
    [0.5,1.0,1.5,2.0,2.5,3.0,4.0], device, seed=9999)
f_raw = grf.generate(n_test, pushfoward=None)
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
    boundary='Periodic', x=x, y=y, A=None, solve=True, device=device)
u_sol = pde.u.clone().detach().float().reshape(n_test, N*N)
A = pde.A; b = pde.b

# Load DeepONet
args_path = 'checkpoints/deeponet_hier_v3_Helmholtz_Periodic_2d_1c_args.json'
with open(args_path) as fj: dargs = json.load(fj)
don = DeepONet(N=N, dim=2, in_channels=2, device=device, boundary='Periodic',
    branch_dim=dargs['branch_dim'], hidden_branch=dargs['hidden_branch'],
    num_branch_layers=dargs['num_branch_layers'], hidden_trunk=dargs['hidden_trunk'],
    num_trunk_layers=dargs['num_trunk_layers']).to(device)
ckp = torch.load('checkpoints/deeponet_hier_v3_Helmholtz_Periodic_2d_1c_best.pth', map_location=device)
don.load_state_dict(ckp['model']); don.eval()

jacobi = WeightedJacobiSolver(equation=pde, device=device, weight=1.0)

u_jac = torch.zeros_like(u_sol)
u_greedy = torch.zeros_like(u_sol)
max_iters = 300

errs_jac = []
errs_greedy = []
don_used = np.zeros((n_test, max_iters), dtype=bool)

for t in range(max_iters):
    u_jac = jacobi.iteration(u_jac)
    errs_jac.append(torch.linalg.norm(u_jac - u_sol, dim=-1).cpu().numpy())

    u_jac_cand = jacobi.iteration(u_greedy)
    residual = b - torch.bmm(A, u_greedy.unsqueeze(-1)).squeeze(-1)
    res_norms = torch.linalg.norm(residual, dim=-1, keepdim=True).clamp(min=1e-15)
    res_normed = residual / res_norms
    inp = torch.cat([k2_flat[:, None, :], res_normed[:, None, :]], dim=1)
    with torch.no_grad():
        corr = don(inp).reshape(u_greedy.shape) * res_norms
    u_don_cand = u_greedy + corr

    err_j = torch.linalg.norm(u_jac_cand - u_sol, dim=-1)
    err_d = torch.linalg.norm(u_don_cand - u_sol, dim=-1)
    use_don = (err_d < err_j)
    don_used[:, t] = use_don.cpu().numpy()
    u_greedy = torch.where(use_don.unsqueeze(-1), u_don_cand, u_jac_cand)
    errs_greedy.append(torch.linalg.norm(u_greedy - u_sol, dim=-1).cpu().numpy())

errs_jac = np.array(errs_jac)
errs_greedy = np.array(errs_greedy)

# Compare per-sample: is greedy EVER worse than jacobi?
final_jac = errs_jac[-1]
final_greedy = errs_greedy[-1]
greedy_worse = final_greedy > final_jac
print(f"Samples where greedy final error > jacobi final error: {greedy_worse.sum()}/{n_test}")
print(f"  Mean ratio (greedy/jacobi): {(final_greedy/final_jac).mean():.3f}")
print(f"  Max ratio: {(final_greedy/final_jac).max():.3f}")

# For a few samples where greedy is worse, show when DeepONet was used
worst_ratio_idx = np.argsort(final_greedy / final_jac)[-5:]
print(f"\nWorst 5 samples (highest greedy/jacobi ratio):")
for idx in worst_ratio_idx:
    don_iters = np.where(don_used[idx])[0]
    last_don = don_iters[-1] if len(don_iters) > 0 else -1
    print(f"  Sample {idx}: jac_err={final_jac[idx]:.4e} greedy_err={final_greedy[idx]:.4e} "
          f"ratio={final_greedy[idx]/final_jac[idx]:.2f} "
          f"#don_calls={len(don_iters)} last_don_at={last_don}")

# Show error at step 25 vs step 300 for greedy
err_25_greedy = errs_greedy[24]
print(f"\nGreedy: mean error at step 25: {err_25_greedy.mean():.4e}")
print(f"Greedy: mean error at step 300: {final_greedy.mean():.4e}")
print(f"Jacobi: mean error at step 25: {errs_jac[24].mean():.4e}")
print(f"Jacobi: mean error at step 300: {final_jac.mean():.4e}")

# Check if there are samples where greedy error INCREASES after a DeepONet step
print(f"\n=== DeepONet steps that INCREASE error (should be 0 by oracle design) ===")
n_increase = 0
for t in range(1, max_iters):
    for i in range(n_test):
        if don_used[i, t] and errs_greedy[t, i] > errs_greedy[t-1, i]:
            n_increase += 1
print(f"Number of (sample, step) where DeepONet was chosen but error increased: {n_increase}")
