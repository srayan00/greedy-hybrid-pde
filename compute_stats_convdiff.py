"""Compute per-sample statistics and paired t-tests for ConvDiff results."""
import torch
import numpy as np
from scipy import stats
import json, sys

sys.path.insert(0, ".")
from verify_pipeline import (generate_test_data, load_deeponet,
                              load_lstm_router, run_iterative_comparison)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

equation = "ConvDiff"
dim = 2
N = 31
n_test = 128
max_iters = 300
b_vel = 20.0
reaction_c = 0.0

print(f"=== ConvDiff Stats (b_vel={b_vel}, reaction_c={reaction_c}) ===")

inputs, u_sol, f_raw, x, pde, k2_raw = generate_test_data(
    n_test, N, dim, device, equation=equation,
    grf_mode="hierarchical", b_vel=b_vel, reaction_c=reaction_c)

deeponet = load_deeponet("./checkpoints", "hier_cd2", N, dim, equation, device)
lstm_router = load_lstm_router(
    "./checkpoints", "router_cd", N, dim, equation, "jacobi", device)

result = run_iterative_comparison(
    deeponet, f_raw, u_sol, N, dim, device,
    equation=equation, lstm_router=lstm_router,
    b_vel=b_vel, reaction_c=reaction_c,
    max_iters=max_iters, tau=24)

(errors_jacobi, errors_hints, errors_greedy, greedy_choices,
 greedy_choices_per_sample, snapshots,
 errors_router, router_choices, router_choices_per_sample,
 ej_ps, eh_ps, eg_ps, er_ps) = result

# ej_ps etc. are (max_iters, n_test)
final_jac = ej_ps[-1]
final_hints = eh_ps[-1]
final_greedy = eg_ps[-1]
final_router = er_ps[-1]

auc_jac = ej_ps.sum(axis=0)
auc_hints = eh_ps.sum(axis=0)
auc_greedy = eg_ps.sum(axis=0)
auc_router = er_ps.sum(axis=0)

def fmt(arr):
    return f"{np.mean(arr):.4e} ({np.std(arr):.4e})"

def paired_t_one_sided(baseline, lstm):
    t_stat, p_two = stats.ttest_rel(baseline, lstm)
    p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
    return p_one

p_l2_jac = paired_t_one_sided(final_jac, final_router)
p_l2_hints = paired_t_one_sided(final_hints, final_router)
p_auc_jac = paired_t_one_sided(auc_jac, auc_router)
p_auc_hints = paired_t_one_sided(auc_hints, auc_router)

print()
print("Final L2 Error (mean (std))")
print("-" * 75)
print(f"{'Strategy':<20} {'Final L2 Error':<28} {'p(LSTM < row)':<20}")
print("-" * 75)
print(f"{'Jacobi Only':<20} {fmt(final_jac):<28} {p_l2_jac:<20.4e}")
print(f"{'HINTS':<20} {fmt(final_hints):<28} {p_l2_hints:<20.4e}")
print(f"{'LSTM Router':<20} {fmt(final_router):<28} {'--':<20}")
print(f"{'True Greedy':<20} {fmt(final_greedy):<28} {'--':<20}")
print("-" * 75)
print()
print("AUC (sum of errors, lower = better)")
print("-" * 75)
print(f"{'Strategy':<20} {'AUC':<28} {'p(LSTM < row)':<20}")
print("-" * 75)
print(f"{'Jacobi Only':<20} {fmt(auc_jac):<28} {p_auc_jac:<20.4e}")
print(f"{'HINTS':<20} {fmt(auc_hints):<28} {p_auc_hints:<20.4e}")
print(f"{'LSTM Router':<20} {fmt(auc_router):<28} {'--':<20}")
print(f"{'True Greedy':<20} {fmt(auc_greedy):<28} {'--':<20}")
print("-" * 75)
