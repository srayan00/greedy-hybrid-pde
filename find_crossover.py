"""Find the iteration where Jacobi's error crosses below the greedy's error.
Also test whether a smaller number of max_iters makes greedy outperform everywhere."""
import torch, json, numpy as np
from data_generation import GaussianRandomFieldHierarchical
from pde import HelmholtzEquation2D
from ml_solver import DeepONet
from numerical_solver import WeightedJacobiSolver

device = torch.device('cuda')
N = 31; n_test = 256; max_iters = 300

def run_and_crossover(k2_flat, f_flat, u_sol, pde, don, label):
    A = pde.A; b = pde.b
    jacobi = WeightedJacobiSolver(equation=pde, device=device, weight=1.0)
    u_jac = torch.zeros_like(u_sol)
    u_greedy = torch.zeros_like(u_sol)
    u_hints = torch.zeros_like(u_sol)
    tau = 24
    errs_jac, errs_greedy, errs_hints = [], [], []
    
    for t in range(max_iters):
        u_jac = jacobi.iteration(u_jac)
        errs_jac.append(torch.linalg.norm(u_jac - u_sol, dim=-1).cpu().numpy())
        
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
        u_greedy = torch.where(use_don.unsqueeze(-1), u_don_c, u_jac_c)
        errs_greedy.append(torch.linalg.norm(u_greedy - u_sol, dim=-1).cpu().numpy())
    
    errs_jac = np.array(errs_jac)
    errs_greedy = np.array(errs_greedy)
    errs_hints = np.array(errs_hints)
    
    mean_jac = errs_jac.mean(axis=1)
    mean_greedy = errs_greedy.mean(axis=1)
    mean_hints = errs_hints.mean(axis=1)
    
    print(f"\n=== {label} ===")
    print(f"DeepONet usage: {(errs_greedy < errs_jac).mean()*100:.1f}%")
    
    crossover_jac = None
    for i in range(len(mean_jac)):
        if mean_jac[i] < mean_greedy[i]:
            crossover_jac = i + 1
            break
    
    crossover_hints = None
    for i in range(len(mean_hints)):
        if mean_hints[i] < mean_greedy[i]:
            crossover_hints = i + 1
            break
    
    print(f"Jacobi crosses below greedy at iteration: {crossover_jac if crossover_jac else 'never'}")
    print(f"HINTS crosses below greedy at iteration: {crossover_hints if crossover_hints else 'never'}")
    
    checkpoints = [25, 50, 75, 100, 150, 200, 300]
    print(f"\n{'Iters':>6s} | {'Jac Final':>12s} {'Greedy Final':>12s} {'HINTS Final':>12s} | {'Jac AUC':>12s} {'Greedy AUC':>12s} {'HINTS AUC':>12s} | {'G<J Final':>8s} {'G<H Final':>8s} {'G<J AUC':>8s} {'G<H AUC':>8s}")
    print("-" * 140)
    for cp in checkpoints:
        jf = mean_jac[cp-1]; gf = mean_greedy[cp-1]; hf = mean_hints[cp-1]
        ja = mean_jac[:cp].sum(); ga = mean_greedy[:cp].sum(); ha = mean_hints[:cp].sum()
        gj_f = "YES" if gf < jf else "no"
        gh_f = "YES" if gf < hf else "no"
        gj_a = "YES" if ga < ja else "no"
        gh_a = "YES" if ga < ha else "no"
        print(f"{cp:6d} | {jf:12.4e} {gf:12.4e} {hf:12.4e} | {ja:12.4e} {ga:12.4e} {ha:12.4e} | {gj_f:>8s} {gh_f:>8s} {gj_a:>8s} {gh_a:>8s}")

# Load DeepONet
args_path = 'checkpoints/deeponet_hier_v3_Helmholtz_Periodic_2d_1c_args.json'
with open(args_path) as fj: dargs = json.load(fj)
don = DeepONet(N=N, dim=2, in_channels=2, device=device, boundary='Periodic',
    branch_dim=dargs['branch_dim'], hidden_branch=dargs['hidden_branch'],
    num_branch_layers=dargs['num_branch_layers'], hidden_trunk=dargs['hidden_trunk'],
    num_trunk_layers=dargs['num_trunk_layers']).to(device)
ckp = torch.load('checkpoints/deeponet_hier_v3_Helmholtz_Periodic_2d_1c_best.pth', map_location=device)
don.load_state_dict(ckp['model']); don.eval()

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

run_and_crossover(k2_flat, f_flat, u_sol, pde, don, "Exp GRF k2 (n=256)")
