"""Find what makes certain Helmholtz samples work well for the greedy approach,
and test controlled k2 ranges to find a regime where greedy outperforms."""
import torch, json, numpy as np
from data_generation import GaussianRandomFieldHierarchical
from pde import HelmholtzEquation2D
from ml_solver import DeepONet
from numerical_solver import WeightedJacobiSolver

device = torch.device('cuda')
N = 31; n_test = 128; max_iters = 300

def run_comparison(f_flat, k2_flat, u_sol, pde, don, label):
    A = pde.A; b = pde.b
    jacobi = WeightedJacobiSolver(equation=pde, device=device, weight=1.0)
    
    u_jac = torch.zeros_like(u_sol)
    u_greedy = torch.zeros_like(u_sol)
    u_hints = torch.zeros_like(u_sol)
    tau = 24
    
    errs_jac, errs_greedy, errs_hints = [], [], []
    don_count = np.zeros(n_test)
    
    for t in range(max_iters):
        # Jacobi
        u_jac = jacobi.iteration(u_jac)
        errs_jac.append(torch.linalg.norm(u_jac - u_sol, dim=-1).cpu().numpy())
        
        # HINTS
        if (t + 1) % tau == 0:
            res_h = b - torch.bmm(A, u_hints.unsqueeze(-1)).squeeze(-1)
            rn_h = torch.linalg.norm(res_h, dim=-1, keepdim=True).clamp(min=1e-15)
            inp_h = torch.cat([k2_flat[:, None, :], (res_h/rn_h)[:, None, :]], dim=1)
            with torch.no_grad():
                c_h = don(inp_h).reshape(u_hints.shape) * rn_h
            u_hints = u_hints + c_h
        else:
            u_hints = jacobi.iteration(u_hints)
        errs_hints.append(torch.linalg.norm(u_hints - u_sol, dim=-1).cpu().numpy())
        
        # Greedy
        u_jac_c = jacobi.iteration(u_greedy)
        res_g = b - torch.bmm(A, u_greedy.unsqueeze(-1)).squeeze(-1)
        rn_g = torch.linalg.norm(res_g, dim=-1, keepdim=True).clamp(min=1e-15)
        inp_g = torch.cat([k2_flat[:, None, :], (res_g/rn_g)[:, None, :]], dim=1)
        with torch.no_grad():
            c_g = don(inp_g).reshape(u_greedy.shape) * rn_g
        u_don_c = u_greedy + c_g
        err_j = torch.linalg.norm(u_jac_c - u_sol, dim=-1)
        err_d = torch.linalg.norm(u_don_c - u_sol, dim=-1)
        use_don = (err_d < err_j)
        don_count += use_don.cpu().numpy()
        u_greedy = torch.where(use_don.unsqueeze(-1), u_don_c, u_jac_c)
        errs_greedy.append(torch.linalg.norm(u_greedy - u_sol, dim=-1).cpu().numpy())
    
    errs_jac = np.array(errs_jac); errs_greedy = np.array(errs_greedy); errs_hints = np.array(errs_hints)
    final_j = errs_jac[-1].mean(); final_g = errs_greedy[-1].mean(); final_h = errs_hints[-1].mean()
    auc_j = errs_jac.mean(axis=1).sum(); auc_g = errs_greedy.mean(axis=1).sum(); auc_h = errs_hints.mean(axis=1).sum()
    
    print(f"\n=== {label} ===")
    print(f"  {'Strategy':15s}  {'Final L2':>12s}  {'AUC':>12s}")
    print(f"  {'Jacobi':15s}  {final_j:12.4e}  {auc_j:12.4e}")
    print(f"  {'HINTS':15s}  {final_h:12.4e}  {auc_h:12.4e}")
    print(f"  {'True Greedy':15s}  {final_g:12.4e}  {auc_g:12.4e}")
    print(f"  DeepONet usage: mean {don_count.mean():.1f} calls, min {don_count.min():.0f}, max {don_count.max():.0f}")
    greedy_wins = (errs_greedy[-1] <= errs_jac[-1]).sum()
    print(f"  Per-sample greedy ≤ jacobi at final: {greedy_wins}/{n_test}")
    return final_j, final_g, final_h, auc_j, auc_g, auc_h

# Load current DeepONet
args_path = 'checkpoints/deeponet_hier_v3_Helmholtz_Periodic_2d_1c_args.json'
with open(args_path) as fj: dargs = json.load(fj)
don = DeepONet(N=N, dim=2, in_channels=2, device=device, boundary='Periodic',
    branch_dim=dargs['branch_dim'], hidden_branch=dargs['hidden_branch'],
    num_branch_layers=dargs['num_branch_layers'], hidden_trunk=dargs['hidden_trunk'],
    num_trunk_layers=dargs['num_trunk_layers']).to(device)
ckp = torch.load('checkpoints/deeponet_hier_v3_Helmholtz_Periodic_2d_1c_best.pth', map_location=device)
don.load_state_dict(ckp['model']); don.eval()

# ---- Test 1: Current setup (hierarchical GRF k2 with exp pushforward) ----
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
x = torch.linspace(0, 1, N+1, device=device)[:-1]; y = x.clone()
pde = HelmholtzEquation2D(a_func=lambda xi,yi: 1.0, f_func=f_flat, k2=k2_flat,
    boundary='Periodic', x=x, y=y, device=device)
u_sol = pde.u.clone().detach().float().reshape(n_test, N*N)

# Check DeepONet quality
inputs = torch.cat([k2_flat[:, None, :], f_flat[:, None, :]], dim=1)
with torch.no_grad(): preds = don(inputs).reshape(n_test, -1)
rel_errs = (torch.linalg.norm(preds - u_sol, dim=-1) / torch.linalg.norm(u_sol, dim=-1).clamp(min=1e-15)).cpu().numpy()
print(f"Current DeepONet: mean rel L2 = {rel_errs.mean():.4f}")

run_comparison(f_flat, k2_flat, u_sol, pde, don, "Hierarchical k2 (exp GRF, current)")

# ---- Test 2: Constant k2 ----
k2_const = 10.0 * torch.ones(n_test, N*N, device=device)
pde2 = HelmholtzEquation2D(a_func=lambda xi,yi: 1.0, f_func=f_flat, k2=k2_const,
    boundary='Periodic', x=x, y=y, device=device)
u_sol2 = pde2.u.clone().detach().float().reshape(n_test, N*N)
inputs2 = torch.cat([k2_const[:, None, :], f_flat[:, None, :]], dim=1)
with torch.no_grad(): preds2 = don(inputs2).reshape(n_test, -1)
rel_errs2 = (torch.linalg.norm(preds2 - u_sol2, dim=-1) / torch.linalg.norm(u_sol2, dim=-1).clamp(min=1e-15)).cpu().numpy()
print(f"\nConstant k2=10: DeepONet rel L2 = {rel_errs2.mean():.4f}")
run_comparison(f_flat, k2_const, u_sol2, pde2, don, "Constant k2 = 10")

# ---- Test 3: Mild k2 variation (narrow range via sigmoid) ----
grf_k2_mild = GaussianRandomFieldHierarchical(N, 2, 0.01, 100.0, 0.1, 1000.0,
    [0.5,1.0,1.5,2.0,2.5,3.0,4.0], device, seed=10000)
k2_base_raw = grf_k2_mild.generate(n_test, pushfoward=None)
k2_mild = 5.0 + 10.0 * torch.sigmoid(k2_base_raw.reshape(n_test, N*N))
print(f"\nMild k2: range [{k2_mild.min():.2f}, {k2_mild.max():.2f}]")
pde3 = HelmholtzEquation2D(a_func=lambda xi,yi: 1.0, f_func=f_flat, k2=k2_mild,
    boundary='Periodic', x=x, y=y, device=device)
u_sol3 = pde3.u.clone().detach().float().reshape(n_test, N*N)
inputs3 = torch.cat([k2_mild[:, None, :], f_flat[:, None, :]], dim=1)
with torch.no_grad(): preds3 = don(inputs3).reshape(n_test, -1)
rel_errs3 = (torch.linalg.norm(preds3 - u_sol3, dim=-1) / torch.linalg.norm(u_sol3, dim=-1).clamp(min=1e-15)).cpu().numpy()
print(f"Mild k2: DeepONet rel L2 = {rel_errs3.mean():.4f}")
run_comparison(f_flat, k2_mild, u_sol3, pde3, don, "Mild k2 (sigmoid, [5,15])")
