import numpy
import torch
from ml_solver import MLSolver, DeepONet, FNOforPDE
from numerical_solver import NumericalSolver
from pde import PDE

class HINTSrouter(torch.nn.Module):
    def __init__(self, num_solvers: int, tau: int):
        super().__init__()
        if num_solvers != 2:
            raise ValueError("HINTSrouter can only be used with two solvers.")
        self.tau = tau
        self.num_solvers = num_solvers
    
    def forward(self, iteration):
        score = torch.zeros(iteration.shape[0], self.num_solvers)
        indices = torch.remainder(iteration, self.tau) == 0
        score[:, indices] = 1.0
        return score
    
    def predict(self, iteration, with_scores=True):
        scores = self.forward(iteration)
        chosen_solver = torch.argmax(scores, dim=1)
        if with_scores:
            return chosen_solver, scores
        else:
            return chosen_solver
        

class HybridSolver(torch.nn.Module):
    def __init__(self, suite_solver: List[NumericalSolver, MLSolver], router: torch.nn.Module, equation: PDE, tol: float, max_iters: int, threshold: float) -> None:
        super().__init__()
        if len(suite_solver) < 2:
            raise ValueError("suite_solver must contain at least two solvers.")
        if isinstance(router, HINTSrouter):
            if len(suite_solver) != 2:
                raise ValueError("HINTSrouter can only be used with two solvers in suite_solver.")
            if not (isinstance(suite_solver[0], NumericalSolver) and isinstance(suite_solver[1], MLSolver)):
                raise TypeError("When using HINTSrouter, the first solver must be a NumericalSolver and the second must be an MLSolver.")
        else:
            for i in range(len(suite_solver)):
                if not isinstance(suite_solver[i], (NumericalSolver, MLSolver)):
                    raise TypeError("Each solver in suite_solver must be an instance of NumericalSolver or MLSolver.")
        self.suite_solver = suite_solver
        self.router = router
        self.tol = tol
        self.max_iters = max_iters
        self.threshold = threshold
        self.equation = equation


    def iteration(self, a, f, iteration_num, u_prev = None):
        if u_prev is None:
            u_prev = torch.zeros_like(f)
        use_ml_solver = self.router(iteration_num)
        residual = self.equation.compute_residual(u_prev.numpy())
        if  isinstance(self.suite_solver[use_ml_solver], MLSolver):
            # Use the machine learning solver
            return self.suite_solver[use_ml_solver](a, f, u_prev, residual)
        else:
            # Use the numerical solver
            return self.suite_solver[use_ml_solver](a, f, u_prev, residual)