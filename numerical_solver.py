import torch
from pde import PDE


class NumericalSolver:
    def __init__(self, equation: PDE, device = None):
        """
        equation: PDE object containing the matrix A and vector b for the linear system Au = b. It can either be Poisson/Helmholtz
        device: torch device to run the solver on. If None, it will use cuda if available, otherwise cpu.
        """
        self.equation = equation
        self.device = device if device else torch.device("cpu")

    def iteration(self, u_old):
        return u_old # Placeholder for actual iteration logic
    
    """
    This function was used before I vectorized my code but now I have implemented batch_solve which is more efficient. It can be ignored.
    """
    
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
    
    def batch_solve(self, tol=1e-10, max_iter=1000, u_init=None, mask = None):
        """
        tol: tolerance for convergence, default 1e-10
        max_iter: maximum number of iterations, default 1000
        u_init: initial guess for the solution, default None (zero vector)
        """
        u_old = u_init if u_init is not None else torch.zeros_like(self.equation.b, device=self.device)
        mask = torch.zeros(self.equation.b.shape[0], dtype=torch.bool, device=self.device) if mask is None else mask
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
    """
    This function was used when I wanted to train the DeepONet to be a residual corrector but it isn't used anymore. It can be ignored.
    """
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
        self._cached_D_inv = None
        self._cached_A_id = None
    
    def _get_D_inv(self, A):
        A_id = A.data_ptr()
        if self._cached_D_inv is None or A_id != self._cached_A_id:
            self._cached_A_id = A_id
            D = torch.diag_embed(torch.diagonal(A, dim1=-2, dim2=-1))
            self._cached_D_inv = torch.linalg.inv(D)
        return self._cached_D_inv
    
    def iteration(self, u_old, mask = None):
        """
        u_old: (B, N) or (B, N^2) tensor containing the current solution estimates for a batch of samples
        mask: (B,) boolean tensor indicating which samples have not yet converged. If None, it is assumed that all samples are not converged.
        """
        if mask is None:
            mask = torch.ones(self.equation.b.shape[0], device=self.device, dtype=torch.bool)
        D_inv = self._get_D_inv(self.equation.A)
        is_batch = self.equation.is_batch
        if is_batch and self.equation.in_channels > 1:
            u_new = u_old.clone()
            output = torch.bmm(self.equation.A[mask], u_old[mask].unsqueeze(-1)).squeeze(-1) # prediction: (B, N) or (B, N^2)
            u_new[mask] = u_old[mask] + self.weight * torch.bmm(D_inv[mask], (self.equation.b[mask] - output).unsqueeze(-1)).squeeze(-1)
        elif is_batch:
            u_new = u_old.clone()
            output = torch.matmul(self.equation.A, u_old[mask].unsqueeze(-1)).squeeze(-1) # prediction: (B, N) or (B, N^2)
            u_new[mask] = u_old[mask] + self.weight * torch.matmul(D_inv, (self.equation.b[mask] - output).unsqueeze(-1)).squeeze(-1)
        else:
            output = self.equation.A @ u_old
            u_new = u_old + self.weight * D_inv @ (self.equation.b - output)
        return u_new
    
    
class GaussSeidelSolver(NumericalSolver):
    def __init__(self, equation: PDE, device = torch.device("cpu")):
        super().__init__(equation, device)
        self._cached_L_inv = None
        self._cached_A_id = None
    
    def _get_L_inv(self, A):
        A_id = A.data_ptr()
        if self._cached_L_inv is None or A_id != self._cached_A_id:
            self._cached_A_id = A_id
            self._cached_L_inv = torch.linalg.inv(torch.tril(A))
        return self._cached_L_inv

    def iteration(self, u_old, mask = None):
        """
        u_old: (B, N) or (B, N^2) tensor containing the current solution estimates for a batch of samples
        mask: (B,) boolean tensor indicating which samples have not yet converged. If None, it is assumed that all samples are not converged.
        """
        if mask is None:
            mask = torch.ones(self.equation.b.shape[0], device=self.device, dtype=torch.bool)
        L_inv = self._get_L_inv(self.equation.A)
        is_batch = self.equation.is_batch
        if is_batch and self.equation.in_channels > 1:
            u_new = u_old.clone()
            output = torch.bmm(self.equation.A[mask], u_old[mask].unsqueeze(-1)).squeeze(-1)
            u_new[mask] = u_old[mask] + torch.bmm(L_inv[mask], (self.equation.b[mask] - output).unsqueeze(-1)).squeeze(-1)
        elif is_batch:
            u_new = u_old.clone()
            output = torch.matmul(self.equation.A, u_old[mask].unsqueeze(-1)).squeeze(-1)
            u_new[mask] = u_old[mask] + torch.matmul(L_inv, (self.equation.b[mask] - output).unsqueeze(-1)).squeeze(-1)
        else:
            output = self.equation.A @ u_old
            u_new = u_old + L_inv @ (self.equation.b - output)
        return u_new

class SuccessiveOverRelaxationSolver(NumericalSolver):
    def __init__(self, equation: PDE, device = torch.device("cpu"), omega = 1.0):
        super().__init__(equation, device)
        self.omega = omega
        self._cached_D_plus_omega_L_inv = None
        self._cached_A_id = None
    
    def _get_D_plus_omega_L_inv(self, A):
        A_id = A.data_ptr()
        if self._cached_D_plus_omega_L_inv is None or A_id != self._cached_A_id:
            self._cached_A_id = A_id
            D = torch.diag_embed(torch.diagonal(A, dim1=-2, dim2=-1))
            L = torch.tril(A, diagonal = -1)
            self._cached_D_plus_omega_L_inv = torch.linalg.inv(D + self.omega * L)
        return self._cached_D_plus_omega_L_inv
    
    def iteration(self, u_old, mask = None):
        """
        u_old: (B, N) or (B, N^2) tensor containing the current solution estimates for a batch of samples
        mask: (B,) boolean tensor indicating which samples have not yet converged. If None, it is assumed that all samples are not converged.
        """
        if mask is None:
            mask = torch.ones(self.equation.b.shape[0], device=self.device, dtype=torch.bool)
        precond = self.omega * self._get_D_plus_omega_L_inv(self.equation.A)
        is_batch = self.equation.is_batch
        if is_batch and self.equation.in_channels > 1:
            u_new = u_old.clone()
            output = torch.bmm(self.equation.A[mask], u_old[mask].unsqueeze(-1)).squeeze(-1)
            u_new[mask] = u_old[mask] + torch.bmm(precond[mask], (self.equation.b[mask] - output).unsqueeze(-1)).squeeze(-1)
        elif is_batch:
            u_new = u_old.clone()
            output = torch.matmul(self.equation.A, u_old[mask].unsqueeze(-1)).squeeze(-1)
            u_new[mask] = u_old[mask] + torch.matmul(precond, (self.equation.b[mask] - output).unsqueeze(-1)).squeeze(-1)
        else:
            output = self.equation.A @ u_old
            u_new = u_old + precond @ (self.equation.b - output)
        return u_new
    
class SymmetricSuccessiveOverRelaxationSolver(NumericalSolver):
    def __init__(self, equation: PDE, device = torch.device("cpu"), omega = 1.0):
        super().__init__(equation, device)
        self.omega = omega
        self.preconditioner = None
        self._cached_A_id = None
    
    def _get_preconditioner(self, A):
        A_id = A.data_ptr()
        if self.preconditioner is None or A_id != self._cached_A_id:
            self._cached_A_id = A_id
            D = torch.diag_embed(torch.diagonal(A, dim1=-2, dim2=-1))
            U = torch.triu(A, diagonal = 1)
            L = torch.tril(A, diagonal = -1)
            first_term = D/self.omega + L
            second_term = self.omega/(2 - self.omega) + torch.linalg.inv(D)
            third_term = D/self.omega + U
            if self.equation.is_batch and self.equation.in_channels > 1:
                self.preconditioner = torch.bmm(torch.bmm(first_term, second_term), third_term)
            else:
                self.preconditioner = torch.matmul(first_term, second_term) @ third_term
            self.preconditioner = torch.linalg.inv(self.preconditioner)
        return self.preconditioner
    
    def iteration(self, u_old, mask = None):
        """
        u_old: (B, N) or (B, N^2) tensor containing the current solution estimates for a batch of samples
        mask: (B,) boolean tensor indicating which samples have not yet converged. If None, it is assumed that all samples are not converged.
        """
        if mask is None:
            mask = torch.ones(self.equation.b.shape[0], device=self.device, dtype=torch.bool)
        is_batch = self.equation.is_batch
        precond = self._get_preconditioner(self.equation.A)
        if is_batch and self.equation.in_channels > 1:
            u_new = u_old.clone()
            output = torch.bmm(self.equation.A[mask], u_old[mask].unsqueeze(-1)).squeeze(-1)
            u_new[mask] = u_old[mask] + torch.bmm(precond[mask], (self.equation.b[mask] - output).unsqueeze(-1)).squeeze(-1)
        elif is_batch:
            u_new = u_old.clone()
            output = torch.matmul(self.equation.A, u_old[mask].unsqueeze(-1)).squeeze(-1)
            u_new[mask] = u_old[mask] + torch.matmul(precond, (self.equation.b[mask] - output).unsqueeze(-1)).squeeze(-1)
        else:
            output = self.equation.A @ u_old
            u_new = u_old + precond @ (self.equation.b - output)
        return u_new

class RichardsonSolver(NumericalSolver):
    def __init__(self, equation: PDE, device = torch.device("cpu"), omega = 1.0):
        super().__init__(equation, device)
        self.omega = omega
    
    def iteration(self, u_old, mask = None):
        """
        u_old: (B, N) or (B, N^2) tensor containing the current solution estimates for a batch of samples
        mask: (B,) boolean tensor indicating which samples have not yet converged. If None, it is assumed that all samples are not converged.
        """
        if mask is None:
            mask = torch.ones(self.equation.b.shape[0], device=self.device, dtype=torch.bool)
        is_batch = self.equation.is_batch
        if is_batch and self.equation.in_channels > 1:
            u_new = u_old.clone()
            output = torch.bmm(self.equation.A[mask], u_old[mask].unsqueeze(-1)).squeeze(-1)
            u_new[mask] = u_old[mask] + self.omega * (self.equation.b[mask] - output)
        elif is_batch:
            u_new = u_old.clone()
            output = torch.matmul(self.equation.A, u_old[mask].unsqueeze(-1)).squeeze(-1)
            u_new[mask] = u_old[mask] + self.omega * (self.equation.b[mask] - output)
        else:
            output = self.equation.A @ u_old
            u_new = u_old + self.omega * (self.equation.b - output)
        return u_new

# class MultigridSolver(NumericalSolver):
#     def __init__(self, equation: PDE, levels=2, device = torch.device("cpu")):
#         super().__init__(equation, device)
#         raise NotImplementedError("Multigrid solver not implemented yet.")

class MultigridSolver(NumericalSolver):
    def __init__(self, equation: PDE, levels=2, device = torch.device("cpu")):
        super().__init__(equation, device)
        if levels > 2:
            raise NotImplementedError("MultigridSolver currently supports up to 2 levels only.")
        self.levels = levels
        self.equations = [self.equation]
        self.restrictor, self.interpolator = self.build_restrictor_interpolator()
        self.restrictors = []
        self.interpolators = []
        if self.levels > 2:
            for i in range(levels - 1):
                restrictor, interpolator = self.build_restrictor_interpolator()
                self.restrictors.append(restrictor)
                self.interpolators.append(interpolator)
                if self.equation.dimension == 1:
                    new_equation = self.equation.__class__(self.equation.a_func, self.equation.f_func, self.equation.boundary,
                                                        self.equation.x[::2], A=self.build_coefficient_matrix(self.equation, restrictor, interpolator))
                else:
                    new_equation = self.equation.__class__(self.equation.a_func, self.equation.f_func, self.equation.boundary,
                                                        self.equation.x[::2], self.equation.y[::2], A=self.build_coefficient_matrix(self.equation, restrictor, interpolator))
                self.equations.append(new_equation)

    def index(self, i, j, len_y=None):
        if len_y is None:
            len_y = len(self.equation.y)
        return i * len_y + j if self.equation.dimension == 2 else i

    def build_restrictor_interpolator_dirichlet(self, curr_equation=None):
        if curr_equation is None:
            curr_equation = self.equation
        # device = curr_equation.A.device if hasattr(curr_equation.A, 'device') else 'cpu'
        if curr_equation.dimension == 1:
            n_h = len(curr_equation.x) - 2
            n_2h = n_h // 2
            R = torch.zeros((n_2h, n_h), device=self.device)
            I = torch.zeros((n_h, n_2h), device=self.device)
            for i in range(n_2h):
                j = 2 * i
                R[i, j] = 0.25
                R[i, j + 1] = 0.5
                R[i, j + 2] = 0.25
                I[j, i] = 0.5
                I[j + 1, i] = 1.0
                I[j + 2, i] = 0.5
        else:
            n_hx, n_hy = len(curr_equation.x) - 2, len(curr_equation.y) - 2 # Exclude boundary points
            n_h = n_hx * n_hy
            n_2hx, n_2hy = n_hx // 2, n_hy // 2
            n_2h = n_2hx * n_2hy
            R = torch.zeros((n_2h, n_h), device=self.device)
            I = torch.zeros((n_h, n_2h), device=self.device)
            for i in range(n_2hx):
                for j in range(n_2hy):
                    idx_2h = self.index(i, j, n_2hy)
                    R[idx_2h, self.index(2 * i + 1, 2 * j + 1, n_hy)] = 0.25
                    R[idx_2h, self.index(2 * i + 2, 2 * j + 1, n_hy)] = 0.125
                    R[idx_2h, self.index(2 * i, 2 * j + 1, n_hy)] = 0.125
                    R[idx_2h, self.index(2 * i + 1, 2 * j + 2, n_hy)] = 0.125
                    R[idx_2h, self.index(2 * i + 1, 2 * j, n_hy)] = 0.125
                    R[idx_2h, self.index(2 * i + 2, 2 * j, n_hy)] = 0.0625
                    R[idx_2h, self.index(2 * i, 2 * j + 2, n_hy)] = 0.0625
                    R[idx_2h, self.index(2 * i + 2, 2 * j + 2, n_hy)] = 0.0625
                    R[idx_2h, self.index(2 * i, 2 * j, n_hy)] = 0.0625
                    I[self.index(2 * i + 1, 2 * j + 1, n_hy), idx_2h] = 1
                    I[self.index(2 * i + 2, 2 * j + 1, n_hy), idx_2h] = 0.5
                    I[self.index(2 * i, 2 * j + 1, n_hy), idx_2h] = 0.5
                    I[self.index(2 * i + 1, 2 * j + 2, n_hy), idx_2h] = 0.5
                    I[self.index(2 * i + 1, 2 * j, n_hy), idx_2h] = 0.5
                    I[self.index(2 * i + 2, 2 * j, n_hy), idx_2h] = 0.25
                    I[self.index(2 * i, 2 * j + 2, n_hy), idx_2h] = 0.25
                    I[self.index(2 * i + 2, 2 * j + 2, n_hy), idx_2h] = 0.25
                    I[self.index(2 * i, 2 * j, n_hy), idx_2h] = 0.25
        return R, I
    
    def build_restrictor_interpolator_periodic(self, curr_equation=None):
        # Apply a simple full weighting restriction for 1d
        if curr_equation is None:
            curr_equation = self.equation
        # Apply a simple full weighting restriction for 1d
        # device = curr_equation.A.device if hasattr(curr_equation.A, 'device') else 'cpu'
        if curr_equation.dimension == 1:
            n_h = len(curr_equation.x)
            n_2h = (n_h + 1) // 2 # include border points
            R = torch.zeros((n_2h, n_h), device=self.device)
            I = torch.zeros((n_h, n_2h), device=self.device)
            for i in range(n_2h):
                j = 2 * i
                R[i, (j - 1 + n_h) % n_h] = 0.25
                R[i, j] = 0.5
                R[i, (j + 1) % n_h] = 0.25
                I[(j - 1 + n_h) % n_h, i] = 0.5
                I[j, i] = 1.0
                I[(j + 1 + n_h) % n_h, i] = 0.5
        else:
            n_hx, n_hy = len(curr_equation.x), len(curr_equation.y)
            n_h = n_hx * n_hy
            n_2hx, n_2hy = (n_hx + 1) // 2, (n_hy + 1) // 2
            n_2h = n_2hx * n_2hy
            R = torch.zeros((n_2h, n_h), device=self.device)
            I = torch.zeros((n_h, n_2h), device=self.device)
            for i in range(n_2hx):
                for j in range(n_2hy):
                    idx_2h = self.index(i, j, n_2hy)
                    R[idx_2h, self.index(2 * i, 2 * j, n_hy)] = 0.25

                    R[idx_2h, self.index((2 * i - 1 + n_hx) % n_hx, 2 * j, n_hy)] = 0.125
                    R[idx_2h, self.index((2 * i + 1) % n_hx, 2 * j, n_hy)] = 0.125
                    R[idx_2h, self.index(2 * i, (2 * j - 1 + n_hy) % n_hy, n_hy)] = 0.125
                    R[idx_2h, self.index(2 * i, (2 * j + 1) % n_hy, n_hy)] = 0.125

                    R[idx_2h, self.index((2 * i - 1 + n_hx) % n_hx, (2 * j - 1 + n_hy) % n_hy, n_hy)] = 0.0625
                    R[idx_2h, self.index((2 * i + 1) % n_hx, (2 * j - 1 + n_hy) % n_hy, n_hy)] = 0.0625
                    R[idx_2h, self.index((2 * i - 1 + n_hx) % n_hx, (2 * j + 1) % n_hy, n_hy)] = 0.0625
                    R[idx_2h, self.index((2 * i + 1) % n_hx, (2 * j + 1) % n_hy, n_hy)] = 0.0625

                    I[self.index(2 * i, 2 * j, n_hy), idx_2h] = 1

                    I[self.index((2 * i - 1 + n_hx) % n_hx, 2 * j, n_hy), idx_2h] = 0.5
                    I[self.index((2 * i + 1) % n_hx, 2 * j, n_hy), idx_2h] = 0.5
                    I[self.index(2 * i, (2 * j - 1 + n_hy) % n_hy, n_hy), idx_2h] = 0.5
                    I[self.index(2 * i, (2 * j + 1) % n_hy, n_hy), idx_2h] = 0.5

                    I[self.index((2 * i - 1 + n_hx) % n_hx, (2 * j - 1 + n_hy) % n_hy, n_hy), idx_2h] = 0.25
                    I[self.index((2 * i + 1) % n_hx, (2 * j - 1 + n_hy) % n_hy, n_hy), idx_2h] = 0.25
                    I[self.index((2 * i - 1 + n_hx) % n_hx, (2 * j + 1) % n_hy, n_hy), idx_2h] = 0.25
                    I[self.index((2 * i + 1) % n_hx, (2 * j + 1) % n_hy, n_hy), idx_2h] = 0.25
        return R, I

    def build_restrictor_interpolator(self, curr_equation=None):
        if self.equation.boundary == 'Dirichlet':
            restrictor, interpolator = self.build_restrictor_interpolator_dirichlet(curr_equation)
        elif self.equation.boundary == 'Periodic':
            restrictor, interpolator = self.build_restrictor_interpolator_periodic(curr_equation)
        else:
            raise ValueError("Boundary condition must be either 'Dirichlet' or 'Periodic'")
        curr_equation = curr_equation if curr_equation is not None else self.equation
        is_batch = curr_equation.is_batch
        if is_batch and curr_equation.in_channels > 1:
            restrictor = restrictor.unsqueeze(0).repeat(curr_equation.batch_size, 1, 1)
            interpolator = interpolator.unsqueeze(0).repeat(curr_equation.batch_size, 1, 1)
        return restrictor, interpolator
    
    def build_coefficient_matrix_periodic(self, curr_equation=None, restrictor=None, interpolator=None, mask = None):
        if mask is None:
            mask = torch.ones(curr_equation.b.shape[0], dtype=torch.bool, device=self.device)
        if curr_equation is None:
            curr_equation = self.equation
        if restrictor is None or interpolator is None:
            restrictor, interpolator = self.build_restrictor_interpolator(curr_equation)
        new_A = curr_equation.A[mask] if curr_equation.is_batch and curr_equation.in_channels > 1 else curr_equation.A
        return restrictor @ new_A @ interpolator
    
    def build_coefficient_matrix_dirichlet(self, curr_equation=None, restrictor=None, interpolator=None, mask = None):
        if mask is None:
            mask = torch.ones(curr_equation.b.shape[0], dtype=torch.bool, device=self.device)
        if curr_equation.dimension == 1:
            new_A = curr_equation.A[1:-1, 1:-1].clone()  # Exclude Dirichlet boundaries
            new_A = restrictor @ new_A @ interpolator
            # include dirichlet boundaries
            new_A = torch.nn.functional.pad(new_A, (1, 1, 1, 1), mode='constant', value=0)
            new_A[0, 0] = 1.0
            new_A[-1, -1] = 1.0
        else:
            new_A = curr_equation.A.reshape((len(curr_equation.x), len(curr_equation.y), len(curr_equation.x), len(curr_equation.y)))[1:-1, 1:-1, 1:-1, 1:-1].clone()  # Exclude Dirichlet boundaries
            new_len_x = len(curr_equation.x) - 2
            new_len_y = len(curr_equation.y) - 2
            new_A = new_A.reshape((new_len_x * new_len_y, new_len_x * new_len_y))
            new_A = restrictor @ new_A @ interpolator
            new_A = new_A.reshape((new_len_x // 2, new_len_y // 2, new_len_x // 2, new_len_y // 2))
            new_A = torch.nn.functional.pad(new_A, (1, 1, 1, 1, 1, 1, 1, 1), mode='constant', value=0)
            for i in range((new_len_x // 2) + 2):
                new_A[i, 0, i, 0] = 1.0
                new_A[i, -1, i, -1] = 1.0
            for j in range((new_len_y // 2) + 2):
                new_A[0, j, 0, j] = 1.0
                new_A[-1, j, -1, j] = 1.0
            len_x, len_y = new_A.shape[0], new_A.shape[1]
            new_A = new_A.reshape((len_x * len_y, len_x * len_y))
        return new_A
    
    def build_coefficient_matrix(self, curr_equation=None, restrictor=None, interpolator=None, mask = None):
        if curr_equation is None:
            curr_equation = self.equation
        if mask is None:
            mask = torch.ones(curr_equation.b.shape[0], dtype=torch.bool, device=self.device)
        if self.equation.boundary == 'Dirichlet':
            return self.build_coefficient_matrix_dirichlet(curr_equation, restrictor, interpolator, mask)
        elif self.equation.boundary == 'Periodic':
            return self.build_coefficient_matrix_periodic(curr_equation, restrictor, interpolator, mask)
        else:
            raise ValueError("Boundary condition must be either 'Dirichlet' or 'Periodic'")

    def restrict(self, u, curr_equation=None, restrictor=None):
        if restrictor is None:
            restrictor = self.restrictor
        if curr_equation is None:
            curr_equation = self.equation
        if curr_equation.boundary == 'Dirichlet':
            if curr_equation.dimension == 1:
                # batch implementation
                ans = (restrictor @ u[:,1:-1].unsqueeze(-1)).squeeze(-1)
                return torch.cat((torch.tensor([0], device=ans.device), ans, torch.tensor([0], device=self.device)))
            else:
                len_x = len(curr_equation.x)
                len_y = len(curr_equation.y)
                ans = restrictor @ u.reshape((len_x, len_y))[:, 1:-1, 1:-1].reshape(-1, len_x * len_y)
                new_len_x = (len_x - 2)// 2
                new_len_y = (len_y - 2)// 2
                return torch.nn.functional.pad(ans.reshape((-1, new_len_x, new_len_y)), (1, 1, 1, 1), mode='constant', value=0).reshape(-1, new_len_x * new_len_y)  # Add Dirichlet boundary conditions
        return torch.bmm(restrictor, u.unsqueeze(-1)).squeeze(-1)#restrictor @ u
    
    def prolong(self, u_coarse, curr_equation=None, interpolator=None):
        if interpolator is None:
            interpolator = self.interpolator
        if curr_equation is None:
            curr_equation = self.equation
        if curr_equation.boundary == 'Dirichlet':
            if curr_equation.dimension == 1:
                ans = interpolator @ u_coarse[1:-1]
                return torch.cat((torch.tensor([0], device=ans.device), ans, torch.tensor([0], device=self.device)))  # Add Dirichlet boundary conditions
            else:
                len_x = (len(curr_equation.x) - 2)
                len_y = (len(curr_equation.y) - 2)
                ans = interpolator @ u_coarse.reshape((len_x // 2 + 2, len_y // 2 + 2))[1:-1, 1:-1].flatten()
                return torch.nn.functional.pad(ans.reshape((len_x, len_y)), (1, 1, 1, 1), mode='constant', value=0).flatten()  # Add Dirichlet boundary conditions
        return torch.bmm(interpolator, u_coarse.unsqueeze(-1)).squeeze(-1) #interpolator @ u_coarse
    
    def valid_level(self, n):
        return n > 3


    def iteration(self, u_old, mask = None, curr_equation=None, level = 1):
        if mask is None:
            mask = torch.ones(u_old.shape[0], dtype=torch.bool, device=self.device)
        if curr_equation is None:
            curr_equation = self.equation
        # else:
        restrictor, interpolator = self.build_restrictor_interpolator(curr_equation)
        # Pre-smoothing
        u = WeightedJacobiSolver(curr_equation, self.device, weight=0.8).batch_solve(u_init=u_old, max_iter=3, mask=~mask)
        # Compute residual
        r = curr_equation.compute_residual(u, mask)
        # r = curr_equation.b - curr_equation.A @ u

        # Restrict residual to coarser grid
        r_coarse = self.restrict(r, curr_equation, restrictor[mask])

        u_init = torch.zeros_like(r_coarse)

        # Solver on coarser grid
        new_A = self.build_coefficient_matrix(curr_equation, restrictor[mask], interpolator[mask], mask)
        if curr_equation.equation == "Poisson":
            if curr_equation.dimension == 1:
                equation_coarse =  curr_equation.__class__(a_func=None, f_func=r_coarse, boundary=curr_equation.boundary, x=curr_equation.x[::2], A=new_A, solve = False)
            else:
                equation_coarse =  curr_equation.__class__(None, r_coarse, curr_equation.boundary, curr_equation.x[::2], curr_equation.y[::2], A=new_A, solve = False)
        else:
            if curr_equation.dimension == 1:
                equation_coarse =  curr_equation.__class__(a_func=None, f_func=r_coarse, k2= None, boundary=curr_equation.boundary, x=curr_equation.x[::2], A=new_A, solve = False)
            else:
                equation_coarse =  curr_equation.__class__(None, r_coarse, None, curr_equation.boundary, curr_equation.x[::2], curr_equation.y[::2], A=new_A, solve = False)
        solver_coarse = WeightedJacobiSolver(equation_coarse, self.device, weight=0.8)
        u_coarse = solver_coarse.batch_solve(u_init=u_init, max_iter=3)

        # Prolongate solution to fine grid
        u_prolonged = self.prolong(u_coarse, curr_equation, interpolator[mask])

        # Update solution
        u[mask] += u_prolonged

        # Post-smoothing
        u = WeightedJacobiSolver(curr_equation, self.device, weight=0.8).batch_solve(u_init=u, max_iter=3, mask=~mask)
        return u

