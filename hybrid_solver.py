import numpy
import torch
from ml_solver import MLSolver, DeepONet, FNOforPDE
from numerical_solver import NumericalSolver
from pde import PDE, PoissonEquation1D, PoissonEquation2D

class Router(torch.nn.Module):
    def __init__(self, num_solvers: int):
        super().__init__()
        self.num_solvers = num_solvers
        self.type = None
    def forward(self, iteration):
        raise NotImplementedError

class ConstantRouter(Router):
    def __init__(self, num_solvers: int, constant_index: int = 0):
        super().__init__(num_solvers)
        self.num_solvers = num_solvers
        self.type = "Constant"
        self.constant_index = constant_index

    def forward(self, iteration):
        scores = torch.zeros(iteration.shape[0], self.num_solvers)
        scores[:, self.constant_index] = 1.0
        return scores
    
    def predict(self, iteration, with_scores=True):
        scores = self.forward(iteration)
        chosen_solver = torch.argmax(scores, dim=1)
        if with_scores:
            return chosen_solver, scores
        else:
            return chosen_solver

class HybridSolver(torch.nn.Module):
    def __init__(self, N: int, dim: int, in_channels: int, boundary: str, equation: PDE, suite_solver: list[NumericalSolver, MLSolver], router: torch.nn.Module, tol: float, max_iters: int, threshold: float) -> None:
        super().__init__()
        if len(suite_solver) < 2:
            raise ValueError("suite_solver must contain at least two solvers.")
        if isinstance(router, HINTSRouter):
            if len(suite_solver) != 2:
                raise ValueError("HINTRouter can only be used with two solvers in suite_solver.")
            if not (isinstance(suite_solver[0], NumericalSolver) and isinstance(suite_solver[1], MLSolver)):
                raise TypeError("When using HINTSrouter, the first solver must be a NumericalSolver and the second must be an MLSolver.")
        else:
            for i in range(len(suite_solver)):
                if not isinstance(suite_solver[i], (NumericalSolver, MLSolver)):
                    raise TypeError("Each solver in suite_solver must be an instance of NumericalSolver or MLSolver.")
        if not isinstance(equation, PoissonEquation1D) and not isinstance(equation, PoissonEquation2D):
            raise ValueError("Unsupported equation type. Supported types are 'Poisson1D' and 'Poisson2D'.")
        self.N = N
        self.dim = dim
        self.in_channels = in_channels
        self.boundary = boundary
        self.xs = torch.linspace(0, 1, N + 1)[:-1] if boundary == "Periodic" else torch.linspace(0, 1, N)
        if self.dim > 1:
            self.ys = torch.linspace(0, 1, N + 1)[:-1] if boundary == "Periodic" else torch.linspace(0, 1, N)
        self.suite_solver = suite_solver
        self.router = router
        self.tol = tol
        self.max_iters = max_iters
        self.threshold = threshold
        self.equation = equation
    
    def forward(self, f, a = None, u0 = None, return_dict = False, training = False, teacher_forcing = 0.0, ground_truth = None):
        if training and ground_truth is None:
            raise ValueError("ground_truth must be provided during training.")
        if training and not return_dict:
            raise ValueError("return_dict must be True during training.")
        if u0 is None:
            u0 = torch.zeros_like(f)
        
        u_prev = u0
        predictions = ()
        routing_scores = () if return_dict else None
        complete_expert_predictions = () if return_dict and training else None
        bs = f.shape[0]
        equations = self.prepare_equations(f, a)
        for iteration_num in range(self.max_iters):
            print(f"Iteration {iteration_num+1}/{self.max_iters}")
            
            residual = torch.zeros_like(f)
            for b in range(bs):
                residual[b] = torch.tensor(equations[b].compute_residual(u_prev[b].detach().numpy()), dtype=torch.float32)
                # equations[b].b = residual[b].detach().numpy()
            print(f"u_prev shape: {u_prev}")
            print(f"residual shape: {residual}")
            inputs = self.prepare_inputs(torch.tensor(residual, dtype=torch.float32).unsqueeze(1), a)
            if self.router.type in ["HINTS", "Constant"]:
                print(torch.tensor([iteration_num]).repeat(bs).shape)
                use_ml_solver, scores = self.router.predict(torch.tensor([iteration_num]).repeat(bs), with_scores=True)
            else:
                raise NotImplementedError("Only HINTRouter is implemented in this version.")
            if training:
                all_expert_predictions = ()
                for i in range(len(self.suite_solver)):
                    if isinstance(self.suite_solver[i], MLSolver):
                        print(f"inputs shape for solver {i}: {inputs.shape}")
                        all_expert_predictions += (u_prev + self.suite_solver[i](inputs),)
                    else:
                        expert_predictions = torch.zeros_like(u_prev)
                        for b in range(bs):
                            new_solver = self.suite_solver[i].__class__(equations[b])
                            expert_predictions[b] = torch.Tensor(new_solver.iteration(u_prev[b].detach().numpy()))
                        all_expert_predictions += (expert_predictions,)
            else:
                predictionsz = torch.zeros_like(u_prev)
                for j in range(bs):
                    if use_ml_solver[j] == 0:
                        continue
                    if isinstance(self.suite_solver[use_ml_solver[j]], MLSolver):
                        a_func = a[j] if a else None
                        f_func = residual[j]
                        inputs_j = self.prepare_inputs(f_func.unsqueeze(0), a_func.unsqueeze(0) if a else None)
                        u_new_j = u_prev[j] + self.suite_solver[use_ml_solver[j]](inputs_j)
                        predictionsz[j] = u_new_j
                    else:
                        new_solver = self.suite_solver[use_ml_solver[j]].__class__(equations[j])
                        u_new_j = new_solver.iteration(u_prev[j].detach().numpy())
                        predictionsz[j] = torch.tensor(u_new_j, dtype=torch.float32).unsqueeze(0)
            if training:
                all_expert_predictions = torch.stack(all_expert_predictions, dim=0)
                error = torch.linalg.norm(all_expert_predictions - ground_truth, dim=2)
                print(f"Shape of error: {error.shape}")
                best_solver = torch.argmin(error, dim=0)
                print(f"Best solver indices: {best_solver}")
                print(f"router solver indices: {use_ml_solver}")
                teacher_forcing_mask = (torch.rand(bs) < teacher_forcing).long()
                chosen_solver = teacher_forcing_mask * best_solver + (1 - teacher_forcing_mask) * use_ml_solver
                print(f"Chosen solver indices: {chosen_solver}")
                predictionsz = all_expert_predictions[chosen_solver, torch.arange(bs)]
            u_prev = predictionsz   

            if return_dict:
                predictions += (predictionsz,)
                if training:
                    print(f"Shape of concatenated expert predictions: {all_expert_predictions.shape}")
                    complete_expert_predictions += (all_expert_predictions,)
                routing_scores += (scores,)
            
        if return_dict:
            output_dict = {
                "predictions": torch.stack(predictions, dim=0),
                "routing_scores": torch.stack(routing_scores, dim=0) if routing_scores else None,
                "complete_expert_predictions": torch.stack(complete_expert_predictions, dim=0) if complete_expert_predictions else None
            }
            return output_dict
        return predictions


                # if training:
                #     all_expert_predictions = torch.stack(all_expert_predictions, dim=0)
                #     error = torch.linalg.norm(all_expert_predictions - ground_truth, dim=1)


            
                
    def prepare_equations(self, f, a):
        equations = []
        bs = f.shape[0]
        for b in range(bs):
            if a:
                a_func = a[b].numpy()
            else:
                a_func = lambda x: 1.0
                if self.dim == 2:
                    a_func = lambda x, y: 1.0
            f_func = f[b].numpy()
            if self.dim == 1:
                equation = self.equation.__class__(a_func = a_func,
                                                   f_func = f_func,
                                                   boundary = self.boundary, 
                                                   x = self.xs.numpy(), 
                                                   A = None,
                                                   solve = False)
            else:
                equation = self.equation.__class__(a_func = a_func,
                                                   f_func = f_func,
                                                   boundary = self.boundary,
                                                   x = self.xs.numpy(),
                                                   y = self.ys.numpy(),
                                                   A = None,
                                                   solve = False)
            equations.append(equation)
        return equations                     

    def prepare_inputs(self, f, a):
        if a is None:
            return f
        return torch.cat((f, a), dim=1)