import numpy
import torch
from ml_solver import MLSolver, DeepONet, FNOforPDE
from numerical_solver_pytorch import NumericalSolver
from pde_pytorch import PDE, PoissonEquation1D, PoissonEquation2D
import models

class Router(torch.nn.Module):
    def __init__(self, num_solvers: int):
        super().__init__()
        self.num_solvers = num_solvers
        self.type = None
    def forward(self, iteration):
        raise NotImplementedError

class ConstantRouter(Router):
    def __init__(self, num_solvers: int, constant_index: int = 0, device = torch.device("cpu")):
        super().__init__(num_solvers)
        self.num_solvers = num_solvers
        self.type = "Constant"
        self.constant_index = constant_index
        self.device = device

    def forward(self, iteration):
        scores = torch.zeros(iteration.shape[0], self.num_solvers, device = self.device)
        scores[:, self.constant_index] = 1.0
        return scores
    
    def predict(self, iteration, with_scores=True):
        scores = self.forward(iteration)
        chosen_solver = torch.argmax(scores, dim=1)
        if with_scores:
            return chosen_solver, scores
        else:
            return chosen_solver

class HINTSRouter(Router):
    def __init__(self, num_solvers: int, tau: int, device = torch.device("cpu")):
        super().__init__(num_solvers)
        if num_solvers != 2:
            raise ValueError("HINTRouter can only be used with two solvers.")
        self.tau = tau
        self.num_solvers = num_solvers
        self.type = "HINTS"
        self.device = device
    
    def forward(self, iteration):
        score = torch.zeros(iteration.shape[0], self.num_solvers, device = self.device)
        indices = (torch.remainder(iteration + 1, self.tau) == 0) + 0
        score[torch.arange(iteration.shape[0]), indices] = 1.0
        return score
    
    def predict(self, iteration, with_scores=True):
        scores = self.forward(iteration)
        chosen_solver = torch.argmax(scores, dim=1)
        if with_scores:
            return chosen_solver, scores
        else:
            return chosen_solver

class LSTMGreedyRouter(Router):
    def __init__(self, encoder_dim, decoder_dim, hidden_dim, num_layers, num_solvers, dropout):
        super(LSTMGreedyRouter, self).__init__(num_solvers)
        self.type = "LSTMGreedy"
        self.lm = None
        self.hidden_dim = hidden_dim
        if encoder_dim is None:
            self.encoder_dim = 0
        else:
            if isinstance(encoder_dim, int):
                self.encoder_dim = encoder_dim
            elif isinstance(encoder_dim, tuple):
                self.encoder_dim = encoder_dim[1]
            self.lm = torch.nn.Linear(self.encoder_dim, self.hidden_dim)
        self.decoder_dim = decoder_dim
        self.model = models.LSTMModel(self.decoder_dim , self.hidden_dim, self.num_solvers, num_layers, dropout)
    
    def initHidden(self, encoder_hidden):
        if encoder_hidden is None:
            return torch.zeros(1, 1, self.hidden_dim)
        if len(encoder_hidden.shape) == 3:
            encoder_hidden = torch.mean(encoder_hidden, dim = 1)
        return self.lm(encoder_hidden)
    
    def forward(self, input, hidden):
        x, hidden = self.model(input, hidden)
        return x, hidden
    
    def predict(self, decoder_hidden, hidden, with_scores = False):
        final_score, hidden = self.forward(decoder_hidden, hidden)
        decision = torch.max(final_score, dim = 1).indices
        if with_scores:
            return (decision, final_score, hidden)
        else:
            return (decision, hidden)

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
                    print(f"invalid index{i}")
                    raise TypeError("Each solver in suite_solver must be an instance of NumericalSolver or MLSolver.")
        if equation.equation not in ["Helmholtz", "Poisson"]:
            raise ValueError("Unsupported equation type. Supported types are 'Poisson1D', 'Poisson2D', 'Helmholtz1D', 'Helmholtz2D.")
        self.N = N
        self.dim = dim
        self.in_channels = in_channels
        self.boundary = boundary
        self.xs = torch.linspace(0, 1, N + 1)[:-1] if boundary == "Periodic" else torch.linspace(0, 1, N)
        if self.dim > 1:
            self.ys = torch.linspace(0, 1, N + 1)[:-1] if boundary == "Periodic" else torch.linspace(0, 1, N)
        # Keep original list for routing logic, but register trainable ML solvers so their
        # parameters appear in model.named_parameters()/optimizer.
        self.suite_solver = suite_solver
        self.ml_solvers = torch.nn.ModuleList([s for s in suite_solver if isinstance(s, MLSolver)])
        self.router = router
        self.tol = tol
        self.max_iters = max_iters
        self.curr_iters = 0
        self.threshold = threshold
        self.equation = equation

    def reset(self):
        self.curr_iters = 0

    def forward(self, f, 
                a = None, k2 = None, u0 = None, return_dict = False, 
                training = False, teacher_forcing = 0.0, ground_truth = None, 
                hidden_state_for_recurrent = None, num_iters = None):
        if training and ground_truth is None:
            raise ValueError("ground_truth must be provided during training.")
        if training and not return_dict:
            raise ValueError("return_dict must be True during training.")
        if u0 is None:
            u0 = torch.zeros_like(f, device=f.device)
        if num_iters is None:
            end_iters = self.max_iters
        else:
            end_iters = min(num_iters + self.curr_iters, self.max_iters)
        start_iter = self.curr_iters
        u_prev = u0
        predictions = ()
        routing_scores = () if return_dict else None
        residuals = () if return_dict else None
        complete_expert_predictions = () if return_dict and training else None
        bs = f.shape[0]
        equations = self.prepare_equations(f, a, k2)

        for iteration_num in range(start_iter, end_iters):
            if iteration_num % 25 == 0:
                print(f"Iteration {iteration_num+1}/{self.max_iters}")
            residual = torch.zeros_like(f, device=f.device)
            for b in range(bs):
                # print(f"SHAPE of A {equations[b].A.shape}")
                # print(f"SHAPE of b {equations[b].b.shape}")
                # print(f"SHAPE of u_prev {u_prev[b].shape}")
                residual[b] = equations[b].compute_residual(u_prev[b])
            inputs = self.prepare_inputs(residual.unsqueeze(1), a, k2)
            if self.router.type in ["HINTS", "Constant"]:
                use_ml_solver, scores = self.router.predict(torch.tensor([iteration_num]).repeat(bs), with_scores=True)
            elif self.router.type == "LSTMGreedy":
                recurrent_inputs = torch.cat((inputs, u_prev.unsqueeze(1)), dim = 1)
                print(f"shape of recurrent inputs {recurrent_inputs.shape}")
                bs = recurrent_inputs.shape[0]
                use_ml_solver, scores, hidden_state_for_recurrent = self.router.predict(recurrent_inputs.reshape(bs, -1), hidden_state_for_recurrent, with_scores=True)
            else:
                raise NotImplementedError("Only HINTRouter is implemented in this version.")
            if training:
                all_expert_predictions = ()
                for i in range(len(self.suite_solver)):
                    if isinstance(self.suite_solver[i], MLSolver):
                        print(f"inputs shape for solver {i}: {inputs.shape}")
                        
                        all_expert_predictions += (u_prev + self.suite_solver[i](inputs),) if self.dim == 1 else (u_prev + self.suite_solver[i](inputs).reshape(bs, -1),)
                    else:
                        expert_predictions = torch.zeros_like(u_prev)
                        for b in range(bs):
                            # new_solver = self.suite_solver[i].__class__(equations[b])
                            # expert_predictions[b] = new_solver.iteration(u_prev[b])
                            self.suite_solver[i].equation = equations[b]
                            expert_predictions[b] = self.suite_solver[i].iteration(u_prev[b])
                        all_expert_predictions += (expert_predictions,)
            else:
                predictionsz = torch.zeros_like(u_prev)
                for j in range(bs):
                    if isinstance(self.suite_solver[use_ml_solver[j]], MLSolver):
                        a_func = a[j] if a else None
                        f_func = residual[j]
                        inputs_j = self.prepare_inputs(f_func.unsqueeze(0), a_func.unsqueeze(0) if a else None)
                        print(f"u_prevjshape of {u_prev[j].shape}")
                        u_new_j = u_prev[j] + self.suite_solver[use_ml_solver[j]](inputs_j) if self.dim == 1 else u_prev[j] + self.suite_solver[use_ml_solver[j]](inputs_j).flatten()
                        predictionsz[j] = u_new_j
                    else:
                        # new_solver = self.suite_solver[use_ml_solver[j]].__class__(equations[j])
                        # u_new_j = new_solver.iteration(u_prev[j])
                        self.suite_solver[use_ml_solver[j]].equation = equations[b]
                        u_new_j = self.suite_solver[use_ml_solver[j]].iteration(u_prev[b])
                        predictionsz[j] = u_new_j.unsqueeze(0)
            if training:
                all_expert_predictions = torch.stack(all_expert_predictions, dim=0)
                error = torch.linalg.norm(all_expert_predictions - ground_truth, dim=2)
                best_solver = torch.argmin(error, dim=0)
                teacher_forcing_mask = (torch.rand(bs, device = best_solver.device) < teacher_forcing).long()
                chosen_solver = teacher_forcing_mask * best_solver + (1 - teacher_forcing_mask) * use_ml_solver
                teacher_forced_prediction = all_expert_predictions[best_solver, torch.arange(bs)]
                next_predictions = all_expert_predictions[chosen_solver, torch.arange(bs)]
                predictionsz = all_expert_predictions[use_ml_solver, torch.arange(bs)]
            u_prev = predictionsz if not training else next_predictions   
            residual = torch.zeros_like(f, device=f.device)
            for b in range(bs):
                residual[b] = equations[b].compute_residual(u_prev[b])  
            self.curr_iters += 1
            if return_dict:
                predictions += (predictionsz,)
                if training:
                    complete_expert_predictions += (all_expert_predictions,)
                routing_scores += (scores,)
                residuals += (residual, )
            
        if return_dict:
            output_dict = {
                "predictions": torch.stack(predictions, dim=0),
                "routing_scores": torch.stack(routing_scores, dim=0) if routing_scores else None,
                "complete_expert_predictions": torch.stack(complete_expert_predictions, dim=0) if complete_expert_predictions else None,
                "hidden_state_for_recurrent": hidden_state_for_recurrent if self.router.type == "LSTMGreedy" else None,
                "residuals": torch.stack(residuals, dim = 0)
            }
            return output_dict
        return predictions
            
                
    def prepare_equations(self, f, a, k2):
        equations = []
        bs = f.shape[0]
        for b in range(bs):
            if a:
                a_func = a[b]
            else:
                a_func = lambda x: 1.0
                if self.dim == 2:
                    a_func = lambda x, y: 1.0
            f_func = f[b]
            k2_func = k2[b] if self.equation.equation == "Helmholtz" else None
            if self.dim == 1:
                if self.equation.equation == "Poisson":
                    equation = self.equation.__class__(a_func = a_func,
                                                    f_func = f_func,
                                                    boundary = self.boundary, 
                                                    x = self.xs,#.numpy(), 
                                                    A = None,
                                                    solve = False,
                                                    device = f.device)
                else:
                    equation = self.equation.__class__(a_func = a_func,
                                                f_func = f_func,
                                                k2 = k2_func,
                                                boundary = self.boundary, 
                                                x = self.xs,#.numpy(), 
                                                A = None,
                                                solve = False,
                                                device = f.device)
            else:
                if self.equation.equation == "Poisson":
                    equation = self.equation.__class__(a_func = a_func,
                                                    f_func = f_func,
                                                    boundary = self.boundary,
                                                    x = self.xs,
                                                    y = self.ys,
                                                    A = None,
                                                    solve = False,
                                                    device = f.device)
                else:
                    equation = self.equation.__class__(a_func = a_func,
                                                   f_func = f_func,
                                                   k2 = k2_func,
                                                   boundary = self.boundary,
                                                   x = self.xs,
                                                   y = self.ys,
                                                   A = None,
                                                   solve = False,
                                                   device = f.device)
            equations.append(equation)
        return equations                     

    def prepare_inputs(self, f, a, k2 = None):
        # print(k2.shape)
        print(f.shape)
        if self.equation.equation == "Poisson":
            if a is None:
                return f
            return torch.cat((f, a), dim=1)
        if a is None:
            return torch.cat((k2.unsqueeze(1), f), dim=1)
        return torch.cat((a, k2.unsqueeze(1), f), dim=1)
    
    def detach_hidden(self, hidden_state):
        for i in range(len(hidden_state)):
            for j in range(len(hidden_state[i])):
                hidden_state[i][j] = hidden_state[i][j].detach()
        return hidden_state


# class HybridSolver(torch.nn.Module):
#     def __init__(self, N: int, dim: int, in_channels: int, boundary: str, equation: PDE, suite_solver: list[NumericalSolver, MLSolver], router: torch.nn.Module, tol: float, max_iters: int, threshold: float) -> None:
#         super().__init__()
#         if len(suite_solver) < 2:
#             raise ValueError("suite_solver must contain at least two solvers.")
#         if isinstance(router, HINTSRouter):
#             if len(suite_solver) != 2:
#                 raise ValueError("HINTRouter can only be used with two solvers in suite_solver.")
#             if not (isinstance(suite_solver[0], NumericalSolver) and isinstance(suite_solver[1], MLSolver)):
#                 raise TypeError("When using HINTSrouter, the first solver must be a NumericalSolver and the second must be an MLSolver.")
#         else:
#             for i in range(len(suite_solver)):
#                 if not isinstance(suite_solver[i], (NumericalSolver, MLSolver)):
#                     print(f"invalid index{i}")
#                     raise TypeError("Each solver in suite_solver must be an instance of NumericalSolver or MLSolver.")
#         if not isinstance(equation, PoissonEquation1D) and not isinstance(equation, PoissonEquation2D):
#             raise ValueError("Unsupported equation type. Supported types are 'Poisson1D' and 'Poisson2D'.")
#         self.N = N
#         self.dim = dim
#         self.in_channels = in_channels
#         self.boundary = boundary
#         self.xs = torch.linspace(0, 1, N + 1)[:-1] if boundary == "Periodic" else torch.linspace(0, 1, N)
#         if self.dim > 1:
#             self.ys = torch.linspace(0, 1, N + 1)[:-1] if boundary == "Periodic" else torch.linspace(0, 1, N)
#         self.suite_solver = suite_solver
#         self.router = router
#         self.tol = tol
#         self.max_iters = max_iters
#         self.curr_iters = 0
#         self.threshold = threshold
#         self.equation = equation

#     def reset(self):
#         self.curr_iters = 0

#     def forward(self, f, 
#                 a = None, u0 = None, return_dict = False, 
#                 training = False, teacher_forcing = 0.0, ground_truth = None, 
#                 hidden_state_for_recurrent = None, num_iters = None):
#         if training and ground_truth is None:
#             raise ValueError("ground_truth must be provided during training.")
#         if training and not return_dict:
#             raise ValueError("return_dict must be True during training.")
#         if u0 is None:
#             u0 = torch.zeros_like(f, device=f.device)
#         if num_iters is None:
#             end_iters = self.max_iters
#         else:
#             end_iters = min(num_iters + self.curr_iters, self.max_iters)
#         start_iter = self.curr_iters
#         u_prev = u0
#         predictions = ()
#         routing_scores = () if return_dict else None
#         complete_expert_predictions = () if return_dict and training else None
#         bs = f.shape[0]
#         equations = self.prepare_equations(f, a)
#         # if self.router.type == "LSTMGreedy":
#         #     hidden_state_for_recurrent = None
#         # for iteration_num in range(self.max_iters):
#         for iteration_num in range(start_iter, end_iters):
#             if iteration_num % 25 == 0:
#                 print(f"Iteration {iteration_num+1}/{self.max_iters}")

#             residual = torch.zeros_like(f, device=f.device)
#             for b in range(bs):
#                 residual[b] = equations[b].compute_residual(u_prev[b])
#             inputs = self.prepare_inputs(residual.unsqueeze(1), a)
#             if self.router.type in ["HINTS", "Constant"]:
#                 use_ml_solver, scores = self.router.predict(torch.tensor([iteration_num]).repeat(bs), with_scores=True)
#             elif self.router.type == "LSTMGreedy":
#                 recurrent_inputs = torch.cat((inputs, u_prev.unsqueeze(1)), dim = 1)
#                 bs = recurrent_inputs.shape[0]
#                 use_ml_solver, scores, hidden_state_for_recurrent = self.router.predict(recurrent_inputs.reshape(bs, -1), hidden_state_for_recurrent, with_scores=True)
#             else:
#                 raise NotImplementedError("Only HINTRouter is implemented in this version.")
#             if training:
#                 all_expert_predictions = ()
#                 for i in range(len(self.suite_solver)):
#                     if isinstance(self.suite_solver[i], MLSolver):
#                         print(f"inputs shape for solver {i}: {inputs.shape}")
#                         all_expert_predictions += (u_prev + self.suite_solver[i](inputs),)
#                     else:
#                         expert_predictions = torch.zeros_like(u_prev)
#                         for b in range(bs):
#                             new_solver = self.suite_solver[i].__class__(equations[b])
#                             expert_predictions[b] = new_solver.iteration(u_prev[b])
#                         all_expert_predictions += (expert_predictions,)
#             else:
#                 predictionsz = torch.zeros_like(u_prev)
#                 for j in range(bs):
#                     if isinstance(self.suite_solver[use_ml_solver[j]], MLSolver):
#                         a_func = a[j] if a else None
#                         f_func = residual[j]
#                         inputs_j = self.prepare_inputs(f_func.unsqueeze(0), a_func.unsqueeze(0) if a else None)
#                         u_new_j = u_prev[j] + self.suite_solver[use_ml_solver[j]](inputs_j)
#                         predictionsz[j] = u_new_j
#                     else:
#                         new_solver = self.suite_solver[use_ml_solver[j]].__class__(equations[j])
#                         u_new_j = new_solver.iteration(u_prev[j])
#                         predictionsz[j] = u_new_j.unsqueeze(0)
#             if training:
#                 all_expert_predictions = torch.stack(all_expert_predictions, dim=0)
#                 error = torch.linalg.norm(all_expert_predictions - ground_truth, dim=2)
#                 best_solver = torch.argmin(error, dim=0)
#                 teacher_forcing_mask = (torch.rand(bs, device = best_solver.device) < teacher_forcing).long()
#                 chosen_solver = teacher_forcing_mask * best_solver + (1 - teacher_forcing_mask) * use_ml_solver
#                 teacher_forced_prediction = all_expert_predictions[best_solver, torch.arange(bs)]
#                 next_predictions = all_expert_predictions[chosen_solver, torch.arange(bs)]
#                 predictionsz = all_expert_predictions[use_ml_solver, torch.arange(bs)]
#             u_prev = predictionsz if not training else next_predictions   
#             self.curr_iters += 1
#             if return_dict:
#                 predictions += (predictionsz,)
#                 if training:
#                     complete_expert_predictions += (all_expert_predictions,)
#                 routing_scores += (scores,)
            
#         if return_dict:
#             output_dict = {
#                 "predictions": torch.stack(predictions, dim=0),
#                 "routing_scores": torch.stack(routing_scores, dim=0) if routing_scores else None,
#                 "complete_expert_predictions": torch.stack(complete_expert_predictions, dim=0) if complete_expert_predictions else None,
#                 "hidden_state_for_recurrent": hidden_state_for_recurrent if self.router.type == "LSTMGreedy" else None
#             }
#             return output_dict
#         return predictions
            
                
#     def prepare_equations(self, f, a):
#         equations = []
#         bs = f.shape[0]
#         for b in range(bs):
#             if a:
#                 a_func = a[b]
#             else:
#                 a_func = lambda x: 1.0
#                 if self.dim == 2:
#                     a_func = lambda x, y: 1.0
#             f_func = f[b]
#             if self.dim == 1:
#                 equation = self.equation.__class__(a_func = a_func,
#                                                    f_func = f_func,
#                                                    boundary = self.boundary, 
#                                                    x = self.xs,#.numpy(), 
#                                                    A = None,
#                                                    solve = False,
#                                                    device = f.device)
#             else:
#                 equation = self.equation.__class__(a_func = a_func,
#                                                    f_func = f_func,
#                                                    boundary = self.boundary,
#                                                    x = self.xs,
#                                                    y = self.ys,
#                                                    A = None,
#                                                    solve = False,
#                                                    device = f.device)
#             equations.append(equation)
#         return equations                     

#     def prepare_inputs(self, f, a):
#         if a is None:
#             return f
#         return torch.cat((f, a), dim=1)
    
#     def detach_hidden(self, hidden_state):
#         for i in range(len(hidden_state)):
#             for j in range(len(hidden_state[i])):
#                 hidden_state[i][j] = hidden_state[i][j].detach()
#         return hidden_state



# class HybridSolver(torch.nn.Module):
#     def __init__(self, N: int, dim: int, in_channels: int, boundary: str, equation: PDE, suite_solver: list[NumericalSolver, MLSolver], router: torch.nn.Module, tol: float, max_iters: int, threshold: float) -> None:
#         super().__init__()
#         if len(suite_solver) < 2:
#             raise ValueError("suite_solver must contain at least two solvers.")
#         if isinstance(router, HINTSRouter):
#             if len(suite_solver) != 2:
#                 raise ValueError("HINTRouter can only be used with two solvers in suite_solver.")
#             if not (isinstance(suite_solver[0], NumericalSolver) and isinstance(suite_solver[1], MLSolver)):
#                 raise TypeError("When using HINTSrouter, the first solver must be a NumericalSolver and the second must be an MLSolver.")
#         else:
#             for i in range(len(suite_solver)):
#                 if not isinstance(suite_solver[i], (NumericalSolver, MLSolver)):
#                     print(f"invalid index{i}")
#                     raise TypeError("Each solver in suite_solver must be an instance of NumericalSolver or MLSolver.")
#         if not isinstance(equation, PoissonEquation1D) and not isinstance(equation, PoissonEquation2D):
#             raise ValueError("Unsupported equation type. Supported types are 'Poisson1D' and 'Poisson2D'.")
#         self.N = N
#         self.dim = dim
#         self.in_channels = in_channels
#         self.boundary = boundary
#         self.xs = torch.linspace(0, 1, N + 1)[:-1] if boundary == "Periodic" else torch.linspace(0, 1, N)
#         if self.dim > 1:
#             self.ys = torch.linspace(0, 1, N + 1)[:-1] if boundary == "Periodic" else torch.linspace(0, 1, N)
#         self.suite_solver = suite_solver
#         self.router = router
#         self.tol = tol
#         self.max_iters = max_iters
#         self.curr_iters = 0
#         self.threshold = threshold
#         self.equation = equation

#     def reset(self):
#         self.curr_iters = 0

#     def forward(self, f, 
#                 a = None, u0 = None, return_dict = False, 
#                 training = False, teacher_forcing = 0.0, ground_truth = None, 
#                 hidden_state_for_recurrent = None, num_iters = None):
#         if training and ground_truth is None:
#             raise ValueError("ground_truth must be provided during training.")
#         if training and not return_dict:
#             raise ValueError("return_dict must be True during training.")
#         if u0 is None:
#             u0 = torch.zeros_like(f, device=f.device)
#         if num_iters is None:
#             end_iters = self.max_iters
#         else:
#             end_iters = min(num_iters + self.curr_iters, self.max_iters)
#         start_iter = self.curr_iters
#         u_prev = u0
#         predictions = ()
#         routing_scores = () if return_dict else None
#         complete_expert_predictions = () if return_dict and training else None
#         bs = f.shape[0]
#         equations = self.prepare_equations(f, a)
#         if self.router.type == "LSTMGreedy":
#             hidden_state_for_recurrent = None
#         # for iteration_num in range(self.max_iters):
#         for iteration_num in range(start_iter, end_iters):
#             if iteration_num % 25 == 0:
#                 print(f"Iteration {iteration_num+1}/{self.max_iters}")

#             residual = torch.zeros_like(f, device=f.device)
#             for b in range(bs):
#                 residual[b] = equations[b].compute_residual(u_prev[b])
#             inputs = self.prepare_inputs(residual.unsqueeze(1), a)
#             if self.router.type in ["HINTS", "Constant"]:
#                 use_ml_solver, scores = self.router.predict(torch.tensor([iteration_num]).repeat(bs), with_scores=True)
#             elif self.router.type == "LSTMGreedy":
#                 recurrent_inputs = torch.cat((inputs, u_prev.unsqueeze(1)), dim = 1)
#                 bs = recurrent_inputs.shape[0]
#                 use_ml_solver, scores, hidden_state_for_recurrent = self.router.predict(recurrent_inputs.reshape(bs, -1), hidden_state_for_recurrent, with_scores=True)
#             else:
#                 raise NotImplementedError("Only HINTRouter is implemented in this version.")
#             if training:
#                 all_expert_predictions = ()
#                 for i in range(len(self.suite_solver)):
#                     if isinstance(self.suite_solver[i], MLSolver):
#                         print(f"inputs shape for solver {i}: {inputs.shape}")
#                         all_expert_predictions += (u_prev + self.suite_solver[i](inputs),)
#                     else:
#                         expert_predictions = torch.zeros_like(u_prev)
#                         for b in range(bs):
#                             new_solver = self.suite_solver[i].__class__(equations[b])
#                             expert_predictions[b] = new_solver.iteration(u_prev[b])
#                         all_expert_predictions += (expert_predictions,)
#             else:
#                 predictionsz = torch.zeros_like(u_prev)
#                 for j in range(bs):
#                     if isinstance(self.suite_solver[use_ml_solver[j]], MLSolver):
#                         a_func = a[j] if a else None
#                         f_func = residual[j]
#                         inputs_j = self.prepare_inputs(f_func.unsqueeze(0), a_func.unsqueeze(0) if a else None)
#                         u_new_j = u_prev[j] + self.suite_solver[use_ml_solver[j]](inputs_j)
#                         predictionsz[j] = u_new_j
#                     else:
#                         new_solver = self.suite_solver[use_ml_solver[j]].__class__(equations[j])
#                         u_new_j = new_solver.iteration(u_prev[j])
#                         predictionsz[j] = u_new_j.unsqueeze(0)
#             if training:
#                 all_expert_predictions = torch.stack(all_expert_predictions, dim=0)
#                 error = torch.linalg.norm(all_expert_predictions - ground_truth, dim=2)
#                 best_solver = torch.argmin(error, dim=0)
#                 teacher_forcing_mask = (torch.rand(bs, device = best_solver.device) < teacher_forcing).long()
#                 chosen_solver = teacher_forcing_mask * best_solver + (1 - teacher_forcing_mask) * use_ml_solver
#                 predictionsz = all_expert_predictions[chosen_solver, torch.arange(bs)]
#             u_prev = predictionsz   
#             self.curr_iters += 1
#             if return_dict:
#                 predictions += (predictionsz,)
#                 if training:
#                     complete_expert_predictions += (all_expert_predictions,)
#                 routing_scores += (scores,)
            
#         if return_dict:
#             output_dict = {
#                 "predictions": torch.stack(predictions, dim=0),
#                 "routing_scores": torch.stack(routing_scores, dim=0) if routing_scores else None,
#                 "complete_expert_predictions": torch.stack(complete_expert_predictions, dim=0) if complete_expert_predictions else None
#             }
#             return output_dict
#         return predictions
            
                
#     def prepare_equations(self, f, a):
#         equations = []
#         bs = f.shape[0]
#         for b in range(bs):
#             if a:
#                 a_func = a[b]
#             else:
#                 a_func = lambda x: 1.0
#                 if self.dim == 2:
#                     a_func = lambda x, y: 1.0
#             f_func = f[b]
#             if self.dim == 1:
#                 equation = self.equation.__class__(a_func = a_func,
#                                                    f_func = f_func,
#                                                    boundary = self.boundary, 
#                                                    x = self.xs,#.numpy(), 
#                                                    A = None,
#                                                    solve = False,
#                                                    device = f.device)
#             else:
#                 equation = self.equation.__class__(a_func = a_func,
#                                                    f_func = f_func,
#                                                    boundary = self.boundary,
#                                                    x = self.xs,
#                                                    y = self.ys,
#                                                    A = None,
#                                                    solve = False,
#                                                    device = f.device)
#             equations.append(equation)
#         return equations                     

#     def prepare_inputs(self, f, a):
#         if a is None:
#             return f
#         return torch.cat((f, a), dim=1)
