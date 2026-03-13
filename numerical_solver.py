import torch
from pde import PDE


class NumericalSolver:
    def __init__(self, equation: PDE, device = None):
        self.equation = equation
        self.device = device if device else torch.device("cpu")

    def iteration(self, u_old):
        return u_old # Placeholder for actual iteration logic
    
    def solve(self, tol=1e-6, max_iter=1000, u_init=None):
        u_old = u_init if u_init is not None else torch.zeros_like(self.equation.b, device=self.device)
        for it in range(max_iter):
            u_new = self.iteration(u_old)
            if torch.norm(u_new - u_old, float('inf')) < tol:
                print(f'Converged in {it} iterations.')
                return u_new
            u_old = u_new
        print('Max iterations reached without convergence.')
        return u_old
    
    def batch_solve(self, tol=1e-10, max_iter=1000, u_init=None):
        u_old = u_init if u_init is not None else torch.zeros_like(self.equation.b, device=self.device)
        mask = torch.zeros(self.equation.b.shape[0], dtype=torch.bool, device=self.device)
        u_new = torch.zeros_like(u_old, device=self.device)
        for it in range(max_iter):
            u_new = self.iteration(u_old, ~mask)
            mask = torch.linalg.norm(u_new - u_old, float('inf'), dim=1) < tol
            print(f'Iteration {it}, converged samples: {mask.sum().item()}/{mask.shape[0]}')
            if mask.all():
                print(f'Converged in {it} iterations.')
                return u_new
            u_old = u_new
        print('Max iterations reached without convergence.')
        return u_old
    
    def batch_solve_stopping(self, stopping, max_iters = 1000, u_init = None):
        u_old = u_init if u_init is not None else torch.zeros_like(self.equation.b, device=self.device)
        mask = torch.zeros(self.equation.b.shape[0], dtype=torch.bool, device=self.device)
        u_new = torch.zeros_like(u_old, device=self.device)
        for it in range(max_iters):
            mask = stopping <= it
            u_new = self.iteration(u_old, ~mask)
            # mask = stopping(u_new, u_old)
            print(f'Iteration {it}, converged samples: {mask.sum().item()}/{mask.shape[0]}')
            if mask.all():
                print(f'Converged in {it} iterations.')
                return u_new
            u_old = u_new


class WeightedJacobiSolver(NumericalSolver):
    def __init__(self, equation: PDE, device = torch.device("cpu"), weight=1.0):
        super().__init__(equation, device)
        self.weight = weight
    
    def iteration(self, u_old, mask = None):
        if mask is None:
            mask = torch.ones(self.equation.b.shape[0], device=self.device, dtype=torch.bool)
        D = torch.diag_embed(torch.diagonal((self.equation.A), dim1=-2, dim2=-1))
        D_inv = torch.linalg.inv(D)
        is_batch = D_inv.ndim == 3
        if is_batch:
            u_new = u_old.clone()
            output = torch.bmm(self.equation.A[mask], u_old[mask].unsqueeze(-1)).squeeze(-1)
            u_new[mask] = u_old[mask] + self.weight * torch.bmm(D_inv[mask], (self.equation.b[mask] - output).unsqueeze(-1)).squeeze(-1)
        else:
            output = self.equation.A @ u_old
            u_new = u_old + self.weight * D_inv @ (self.equation.b - output)
        return u_new
    
    
class GaussSeidelSolver(NumericalSolver):
    def __init__(self, equation: PDE, device = torch.device("cpu")):
        super().__init__(equation, device)
    
    def iteration(self, u_old, mask = None):
        # L is lower triangular part of A
        if mask is None:
            mask = torch.ones(self.equation.b.shape[0], device=self.device, dtype=torch.bool)
        L = torch.tril(self.equation.A)
        L_inv = torch.linalg.inv(L)
        is_batch = L_inv.ndim == 3
        if is_batch:
            u_new = u_old.clone()
            output = torch.bmm(self.equation.A[mask], u_old[mask].unsqueeze(-1)).squeeze(-1)
            u_new[mask] = u_old[mask] + torch.bmm(L_inv[mask], (self.equation.b[mask] - output).unsqueeze(-1)).squeeze(-1)
        else:
            output = self.equation.A @ u_old
            u_new = u_old + L_inv @ (self.equation.b - output)
        return u_new

class MultigridSolver(NumericalSolver):
    def __init__(self, equation: PDE, levels=2, device = torch.device("cpu")):
        super().__init__(equation, device)
        raise NotImplementedError("Multigrid solver not implemented yet.")