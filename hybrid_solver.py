import numpy
import torch
from ml_solver import MLSolver, DeepONet, FNOforPDE
from numerical_solver import NumericalSolver
from pde import PDE, PoissonEquation1D, PoissonEquation2D
import models

class Router(torch.nn.Module):
    def __init__(self, num_solvers: int):
        super().__init__()
        self.num_solvers = num_solvers
        self.type = None
    def forward(self, iteration):
        raise NotImplementedError

class ConstantRouter(Router):
    """
    A router that always chooses the same solver
    """
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
    """
    A router that switches between solvers based on a periodic schedule
    """
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
    """
    A router that uses an LSTM to predict which solver to use 
    """
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
    """
    A hybrid solver that combines multiple solvers and uses a router to decide which solver to use at each iteration.
    """
    def __init__(self, N: int, dim: int, in_channels: int, boundary: str, equation: PDE, suite_solver: list[NumericalSolver, MLSolver], router: torch.nn.Module, tol: float, max_iters: int, threshold: float) -> None:
        """
        N: resolution of the grid
        dim: dimension of the PDE (1 or 2)
        in_channels: number of input channels for the ML solvers (e.g. 2 for a and f in Poisson, 3 for a, k2, and f in Helmholtz)
        boundary: type of boundary condition ("Periodic" or "Dirichlet")
        equation: PDE object containing the matrix A and vector b for the linear system Au = b. It can either be Poisson/Helmholtz
        suite_solver: list of solvers to choose from. It can contain any number of NumericalSolvers and MLSolvers but if the router is HINTRouter then it can only contain 2 solvers and the first one must be a NumericalSolver and the second one must be an MLSolver.
        router: any one of the above routers
        tol: tolerance for convergence, default 1e-6
        max_iters: maximum number of iterations, default 1000
        threshold: threshold for the router to decide which solver to use. 
        """
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
        if equation.equation not in ["Helmholtz", "Poisson", "ConvDiff", "Reaction"]:
            raise ValueError("Unsupported equation type. Supported: Poisson, Helmholtz, ConvDiff, Reaction.")
        self.N = N
        self.dim = dim
        self.in_channels = in_channels
        self.boundary = boundary
        self.xs = torch.linspace(0, 1, N + 1)[:-1] if boundary == "Periodic" else torch.linspace(0, 1, N)
        if self.dim > 1:
            self.ys = torch.linspace(0, 1, N + 1)[:-1] if boundary == "Periodic" else torch.linspace(0, 1, N)
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
        """
        f: right hand side of the PDE, shape (B, N) for 1D and (B, N, N) for 2D
        a: coefficient function for the PDE, shape (B, N) for 1D and (B, N, N) for 2D. It can be None if the PDE is constant coefficient.
        k2: coefficient function for the Helmholtz equation, shape (B, N) for 1D and (B, N, N) for 2D. It can be None if the PDE is Poisson.
        u0: initial guess for the solution, shape (B, N) for 1D and (B, N, N) for 2D. If None, it will be initialized to zero.
        return_dict: whether to return a dictionary containing the predictions, routing scores, and residuals at each iteration.
        training: whether the model is being trained. If True, the router will use teacher forcing to choose the solver based on the ground truth solution.
        teacher_forcing: the probability of using teacher forcing during training. It should be a value between 0 and 1. If teacher_forcing is 0, the router will always use its own predictions to choose the solver. If teacher_forcing is 1, the router will always use the ground truth solution to choose the solver.
        ground_truth: the ground truth solution, shape (B, N) for 1D and (B, N, N) for 2D. It is required if training is True.
        hidden_state_for_recurrent: the hidden state for the recurrent router, if applicable. It should be a tuple of (h_0, c_0) where h_0 and c_0 are the initial hidden and cell states for the LSTM. The shape of h_0 and c_0 should be (num_layers, B, hidden_dim).
        num_iters: the number of iterations to run the solver for. If None, it will run for self.max_iters iterations.
        """
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
        prev_step_errors = () if return_dict and training else None
        bs = f.shape[0]
        equations = self.prepare_equations(f, a, k2)
        
        for iteration_num in range(start_iter, end_iters):
            residual = equations.compute_residual(u_prev)
            # For Helmholtz, normalize residual to unit norm (DeepONet trained on unit-norm f).
            # Store the norms to scale ML solver output back after prediction.
            if self.equation.equation == "Helmholtz":
                res_norms = torch.linalg.norm(residual, dim=-1).clamp(min=1e-15)
                residual_for_input = residual / res_norms.unsqueeze(-1)
            else:
                res_norms = None
                residual_for_input = residual
            inputs = self.prepare_inputs(residual_for_input.unsqueeze(1), a, k2)
            if self.router.type in ["HINTS", "Constant"]:
                use_ml_solver, scores = self.router.predict(torch.tensor([iteration_num]).repeat(bs), with_scores=True)
            elif self.router.type == "LSTMGreedy":
                recurrent_inputs = torch.cat((inputs, u_prev.unsqueeze(1)), dim = 1)
                bs = recurrent_inputs.shape[0]
                use_ml_solver, scores, hidden_state_for_recurrent = self.router.predict(recurrent_inputs.reshape(bs, -1), hidden_state_for_recurrent, with_scores=True)
            else:
                raise NotImplementedError("Only HINTRouter is implemented in this version.")
            if training:
                all_expert_predictions = ()
                for i in range(len(self.suite_solver)):
                    if isinstance(self.suite_solver[i], MLSolver):
                        ml_out = self.suite_solver[i](inputs)
                        if self.dim != 1:
                            ml_out = ml_out.reshape(bs, -1)
                        if res_norms is not None:
                            ml_out = ml_out * res_norms.unsqueeze(-1)
                        all_expert_predictions += (u_prev + ml_out,)
                    else:
                        self.suite_solver[i].equation = equations
                        expert_predictions = self.suite_solver[i].iteration(u_prev)
                        all_expert_predictions += (expert_predictions,)
            else:
                predictionsz = torch.zeros_like(u_prev)
                for i in range(len(self.suite_solver)):
                    if isinstance(self.suite_solver[i], MLSolver):
                        mask_i = (use_ml_solver == i)
                        if mask_i.any():
                            ml_out = self.suite_solver[i](inputs[mask_i])
                            if self.dim != 1:
                                ml_out = ml_out.reshape(mask_i.sum(), -1)
                            if res_norms is not None:
                                ml_out = ml_out * res_norms[mask_i].unsqueeze(-1)
                            predictionsz[mask_i] = u_prev[mask_i] + ml_out
                    else:
                        self.suite_solver[i].equation = equations
                        u_new_i = self.suite_solver[i].iteration(u_prev, use_ml_solver == i)
                        predictionsz = u_new_i
            if training:
                all_expert_predictions = torch.stack(all_expert_predictions, dim=0)
                has_null_space = self.equation.equation == "Poisson" or (self.equation.equation == "ConvDiff" and getattr(self.equation, 'reaction', 0.0) == 0.0)
                if has_null_space and self.boundary == "Periodic":
                    all_expert_predictions = all_expert_predictions - torch.mean(all_expert_predictions, dim=2, keepdim=True)
                error = torch.linalg.norm(all_expert_predictions - ground_truth, dim=2)
                best_solver = torch.argmin(error, dim=0)
                best_error = torch.min(error, dim=0).values
                teacher_forcing_mask = (torch.rand(bs, device = best_solver.device) < teacher_forcing).long()
                chosen_solver = teacher_forcing_mask * best_solver + (1 - teacher_forcing_mask) * use_ml_solver
                teacher_forced_prediction = all_expert_predictions[best_solver, torch.arange(bs)]
                next_predictions = all_expert_predictions[chosen_solver, torch.arange(bs)]
                predictionsz = all_expert_predictions[use_ml_solver, torch.arange(bs)]
            u_prev = predictionsz if not training else next_predictions  
            residual = equations.compute_residual(u_prev)   
            self.curr_iters += 1
            if return_dict:
                predictions += (predictionsz,)
                if training:
                    complete_expert_predictions += (all_expert_predictions,)
                    prev_step_errors += (best_error,)
                routing_scores += (scores,)
                residuals += (residual, )
            
        if return_dict:
            output_dict = {
                "predictions": torch.stack(predictions, dim=0),
                "routing_scores": torch.stack(routing_scores, dim=0) if routing_scores else None,
                "complete_expert_predictions": torch.stack(complete_expert_predictions, dim=0) if complete_expert_predictions else None,
                "hidden_state_for_recurrent": hidden_state_for_recurrent if self.router.type == "LSTMGreedy" else None,
                "residuals": torch.stack(residuals, dim = 0),
                "prev_step_errors": torch.stack(prev_step_errors, dim = 0) if prev_step_errors else None
            }
            return output_dict
        return predictions
            
                
    def prepare_equations(self, f, a, k2):
        a_func = a if a is not None else lambda x: 1.0
        if self.dim == 2:
            a_func = a if a is not None else lambda x, y: 1.0
        f_func = f
        k2_func = k2 if self.equation.equation == "Helmholtz" else None

        cached_A = getattr(self, '_cached_A', None)
        use_cache = (a is None) and (k2 is None) and (cached_A is not None)

        A_arg = cached_A if use_cache else None

        if self.dim == 1:
            if self.equation.equation == "Poisson":
                equation = self.equation.__class__(a_func = a_func,
                                                f_func = f_func,
                                                boundary = self.boundary, 
                                                x = self.xs,
                                                A = A_arg,
                                                solve = False,
                                                device = f.device)
            else:
                equation = self.equation.__class__(a_func = a_func,
                                            f_func = f_func,
                                            k2 = k2_func,
                                            boundary = self.boundary, 
                                            x = self.xs,
                                            A = A_arg,
                                            solve = False,
                                            device = f.device)
        else:
            if self.equation.equation == "Poisson":
                equation = self.equation.__class__(a_func = a_func,
                                                f_func = f_func,
                                                boundary = self.boundary,
                                                x = self.xs,
                                                y = self.ys,
                                                A = A_arg,
                                                solve = False,
                                                device = f.device)
            elif self.equation.equation == "ConvDiff":
                equation = self.equation.__class__(a_func = a_func,
                                               f_func = f_func,
                                               b_vec = self.equation.b_vec,
                                               boundary = self.boundary,
                                               x = self.xs,
                                               y = self.ys,
                                               A = A_arg,
                                               solve = False,
                                               device = f.device,
                                               reaction = self.equation.reaction)
            elif self.equation.equation == "Reaction":
                equation = self.equation.__class__(a_func = a_func,
                                               f_func = f_func,
                                               reaction = self.equation.reaction,
                                               boundary = self.boundary,
                                               x = self.xs,
                                               y = self.ys,
                                               A = A_arg,
                                               solve = False,
                                               device = f.device)
            else:
                equation = self.equation.__class__(a_func = a_func,
                                               f_func = f_func,
                                               k2 = k2_func,
                                               boundary = self.boundary,
                                               x = self.xs,
                                               y = self.ys,
                                               A = A_arg,
                                               solve = False,
                                               device = f.device)

        if (a is None) and (k2 is None) and (cached_A is None):
            self._cached_A = equation.A

        return equation                    

    def prepare_inputs(self, f, a, k2 = None):
        if self.equation.equation in ("Poisson", "ConvDiff", "Reaction"):
            if a is None:
                return f
            return torch.cat((a.unsqueeze(1), f), dim=1)
        if a is None:
            return torch.cat((k2.unsqueeze(1), f), dim=1)
        return torch.cat((a.unsqueeze(1), k2.unsqueeze(1), f), dim=1)
    
    def detach_hidden(self, hidden_state):
        for i in range(len(hidden_state)):
            for j in range(len(hidden_state[i])):
                hidden_state[i][j] = hidden_state[i][j].detach()
        return hidden_state