"""Quick script to compute final L2 error mean+std for all strategies."""
import torch, json, numpy as np
from data_generation import GaussianRandomFieldHierarchical
from pde import PoissonEquation2D
from ml_solver import DeepONet
from numerical_solver import WeightedJacobiSolver
from hybrid_solver import LSTMGreedyRouter
import models

device = torch.device('cuda')
N = 31; n_test = 128; max_iters = 300; tau = 24

grf = GaussianRandomFieldHierarchical(N, 2, 0.01, 100.0, 0.1, 1000.0,
    [0.5,1.0,1.5,2.0,2.5,3.0,4.0], device, seed=9999)
pushfwd = lambda x: x - torch.mean(x, dim=(-2,-1), keepdim=True)
f_raw = grf.generate(n_test, pushfoward=pushfwd)
f_flat = f_raw.reshape(n_test, N*N)
x = torch.linspace(0,1,N+1,device=device)[:-1]
y = torch.linspace(0,1,N+1,device=device)[:-1]
pde = PoissonEquation2D(a_func=lambda xi,yi:1.0, f_func=f_flat,
    boundary='Periodic', x=x, y=y, device=device)
u_sol = pde.u.clone().detach().float().reshape(n_test, N*N)
u_sol = u_sol - torch.mean(u_sol, dim=-1, keepdim=True)
A, b = pde.A, pde.b

with open('args/deeponet_args.json') as fj:
    dargs = json.load(fj)
don = DeepONet(N=N, dim=2, in_channels=1, device=device, boundary='Periodic',
    branch_dim=dargs['branch_dim'], hidden_branch=dargs['hidden_branch'],
    num_branch_layers=dargs['num_branch_layers'], hidden_trunk=dargs['hidden_trunk'],
    num_trunk_layers=dargs['num_trunk_layers']).to(device)
ckp = torch.load('checkpoints/deeponet_hier_v2_Poisson_Periodic_2d_1c_best.pth', map_location=device)
don.load_state_dict(ckp['model']); don.eval()

with open('args/lstm_args.json') as fj:
    largs = json.load(fj)
router = LSTMGreedyRouter(None, N*N*2, largs['hidden_dim'], largs['num_layers'], 2, largs['dropout']).to(device)
rckp = torch.load('checkpoints/lstmrouter_hier_v1_Poisson_Periodic_2d_1c_jacobi_best.pth',
                   map_location=device, weights_only=False)
router.load_state_dict(rckp['model']); router.eval()

u_j = torch.zeros_like(u_sol); u_h = torch.zeros_like(u_sol)
u_g = torch.zeros_like(u_sol); u_r = torch.zeros_like(u_sol)
hidden = None
jsolver = WeightedJacobiSolver(equation=pde, device=device, weight=1.0)

for t in range(max_iters):
    u_j = jsolver.iteration(u_j)
    u_j = u_j - torch.mean(u_j, dim=-1, keepdim=True)
    if (t+1) % tau == 0:
        res_h = b - torch.bmm(A, u_h.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            corr = don(res_h[:,None,:]).reshape(u_h.shape)
        u_h = u_h + corr
    else:
        u_h = jsolver.iteration(u_h)
    u_h = u_h - torch.mean(u_h, dim=-1, keepdim=True)
    u_g_jac = jsolver.iteration(u_g)
    u_g_jac = u_g_jac - torch.mean(u_g_jac, dim=-1, keepdim=True)
    res_g = b - torch.bmm(A, u_g.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        corr_g = don(res_g[:,None,:]).reshape(u_g.shape)
    u_g_don = u_g + corr_g
    u_g_don = u_g_don - torch.mean(u_g_don, dim=-1, keepdim=True)
    ej = torch.linalg.norm(u_g_jac - u_sol, dim=-1)
    ed = torch.linalg.norm(u_g_don - u_sol, dim=-1)
    u_g = torch.where((ed < ej).unsqueeze(-1), u_g_don, u_g_jac)
    res_r = b - torch.bmm(A, u_r.unsqueeze(-1)).squeeze(-1)
    rec = torch.cat((res_r[:,None,:], u_r.unsqueeze(1)), dim=1)
    with torch.no_grad():
        chosen, _, hidden = router.predict(rec.reshape(n_test,-1), hidden, with_scores=True)
    u_r_jac = jsolver.iteration(u_r)
    with torch.no_grad():
        corr_r = don(res_r[:,None,:]).reshape(u_r.shape)
    u_r_don = u_r + corr_r
    u_r_jac = u_r_jac - torch.mean(u_r_jac, dim=-1, keepdim=True)
    u_r_don = u_r_don - torch.mean(u_r_don, dim=-1, keepdim=True)
    u_r = torch.where((chosen==1).unsqueeze(-1), u_r_don, u_r_jac)

from scipy import stats as spstats

# Track per-sample AUC (sum of L2 error across iterations)
auc_j = torch.zeros(n_test, device=device)
auc_h = torch.zeros(n_test, device=device)
auc_g = torch.zeros(n_test, device=device)
auc_r = torch.zeros(n_test, device=device)

# Re-run to collect per-sample AUC
u_j2 = torch.zeros_like(u_sol); u_h2 = torch.zeros_like(u_sol)
u_g2 = torch.zeros_like(u_sol); u_r2 = torch.zeros_like(u_sol)
hidden2 = None
for t in range(max_iters):
    u_j2 = jsolver.iteration(u_j2)
    u_j2 = u_j2 - torch.mean(u_j2, dim=-1, keepdim=True)
    auc_j += torch.linalg.norm(u_j2 - u_sol, dim=-1)
    if (t+1) % tau == 0:
        res_h2 = b - torch.bmm(A, u_h2.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            corr2 = don(res_h2[:,None,:]).reshape(u_h2.shape)
        u_h2 = u_h2 + corr2
    else:
        u_h2 = jsolver.iteration(u_h2)
    u_h2 = u_h2 - torch.mean(u_h2, dim=-1, keepdim=True)
    auc_h += torch.linalg.norm(u_h2 - u_sol, dim=-1)
    u_g2_jac = jsolver.iteration(u_g2)
    u_g2_jac = u_g2_jac - torch.mean(u_g2_jac, dim=-1, keepdim=True)
    res_g2 = b - torch.bmm(A, u_g2.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        corr_g2 = don(res_g2[:,None,:]).reshape(u_g2.shape)
    u_g2_don = u_g2 + corr_g2
    u_g2_don = u_g2_don - torch.mean(u_g2_don, dim=-1, keepdim=True)
    ej2 = torch.linalg.norm(u_g2_jac - u_sol, dim=-1)
    ed2 = torch.linalg.norm(u_g2_don - u_sol, dim=-1)
    u_g2 = torch.where((ed2 < ej2).unsqueeze(-1), u_g2_don, u_g2_jac)
    auc_g += torch.minimum(ej2, ed2)
    res_r2 = b - torch.bmm(A, u_r2.unsqueeze(-1)).squeeze(-1)
    rec2 = torch.cat((res_r2[:,None,:], u_r2.unsqueeze(1)), dim=1)
    with torch.no_grad():
        chosen2, _, hidden2 = router.predict(rec2.reshape(n_test,-1), hidden2, with_scores=True)
    u_r2_jac = jsolver.iteration(u_r2)
    with torch.no_grad():
        corr_r2 = don(res_r2[:,None,:]).reshape(u_r2.shape)
    u_r2_don = u_r2 + corr_r2
    u_r2_jac = u_r2_jac - torch.mean(u_r2_jac, dim=-1, keepdim=True)
    u_r2_don = u_r2_don - torch.mean(u_r2_don, dim=-1, keepdim=True)
    u_r2 = torch.where((chosen2==1).unsqueeze(-1), u_r2_don, u_r2_jac)
    auc_r += torch.linalg.norm(u_r2 - u_sol, dim=-1)

err_j = torch.linalg.norm(u_j2 - u_sol, dim=-1).cpu().numpy()
err_h = torch.linalg.norm(u_h2 - u_sol, dim=-1).cpu().numpy()
err_g = torch.linalg.norm(u_g2 - u_sol, dim=-1).cpu().numpy()
err_r = torch.linalg.norm(u_r2 - u_sol, dim=-1).cpu().numpy()
auc_j = auc_j.cpu().numpy()
auc_h = auc_h.cpu().numpy()
auc_g = auc_g.cpu().numpy()
auc_r = auc_r.cpu().numpy()

_, p_l2_rj = spstats.ttest_rel(err_r, err_j, alternative='less')
_, p_l2_rh = spstats.ttest_rel(err_r, err_h, alternative='less')
_, p_auc_rj = spstats.ttest_rel(auc_r, auc_j, alternative='less')
_, p_auc_rh = spstats.ttest_rel(auc_r, auc_h, alternative='less')

def fmt_p(p):
    if p < 1e-16:
        return "< 1e-16"
    return f"{p:.2e}"

print("=== Final L2 Error (after 300 iters) ===")
print(f"Jacobi:      {err_j.mean():.4e} ({err_j.std():.4e})  p = {fmt_p(p_l2_rj)}")
print(f"HINTS:       {err_h.mean():.4e} ({err_h.std():.4e})  p = {fmt_p(p_l2_rh)}")
print(f"LSTM Router: {err_r.mean():.4e} ({err_r.std():.4e})")
print(f"Oracle:      {err_g.mean():.4e} ({err_g.std():.4e})")
print()
print("=== AUC (sum of per-sample L2 errors over 300 iters) ===")
print(f"Jacobi:      {auc_j.mean():.4e} ({auc_j.std():.4e})  p = {fmt_p(p_auc_rj)}")
print(f"HINTS:       {auc_h.mean():.4e} ({auc_h.std():.4e})  p = {fmt_p(p_auc_rh)}")
print(f"LSTM Router: {auc_r.mean():.4e} ({auc_r.std():.4e})")
print(f"Oracle:      {auc_g.mean():.4e} ({auc_g.std():.4e})")
