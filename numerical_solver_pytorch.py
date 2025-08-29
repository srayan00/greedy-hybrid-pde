import torch
from pde_pytorch import PDE


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

    
class WeightedJacobiSolver(NumericalSolver):
    def __init__(self, equation: PDE, device = torch.device("cpu"), weight=1.0):
        super().__init__(equation, device)
        self.weight = weight
    
    def iteration(self, u_old):
        # D is diagonal of A
        D = torch.diag(torch.diag(self.equation.A))
        D_inv = torch.linalg.inv(D)
        u_new = u_old + self.weight * D_inv @ (self.equation.b - self.equation.A @ u_old)
        return u_new

class GaussSeidelSolver(NumericalSolver):
    def __init__(self, equation: PDE, device = torch.device("cpu")):
        super().__init__(equation, device)
    
    def iteration(self, u_old):
        # L is lower triangular part of A
        L = torch.tril(self.equation.A)
        L_inv = torch.linalg.inv(L)
        u_new = u_old.to(self.device) + L_inv @ (self.equation.b - self.equation.A @ u_old)
        return u_new

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
        device = curr_equation.A.device if hasattr(curr_equation.A, 'device') else 'cpu'
        if curr_equation.dimension == 1:
            n_h = len(curr_equation.x) - 2
            n_2h = n_h // 2
            R = torch.zeros((n_2h, n_h), device=self.device)
            I = torch.zeros((n_h, n_2h), device=device)
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
        device = curr_equation.A.device if hasattr(curr_equation.A, 'device') else 'cpu'
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
        return restrictor, interpolator
    
    def build_coefficient_matrix_periodic(self, curr_equation=None, restrictor=None, interpolator=None):
        if curr_equation is None:
            curr_equation = self.equation
        if restrictor is None or interpolator is None:
            restrictor, interpolator = self.build_restrictor_interpolator(curr_equation)
        new_A = curr_equation.A
        return restrictor @ new_A @ interpolator
    
    def build_coefficient_matrix_dirichlet(self, curr_equation=None, restrictor=None, interpolator=None):
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
    
    def build_coefficient_matrix(self, curr_equation=None, restrictor=None, interpolator=None):
        if curr_equation is None:
            curr_equation = self.equation
        if self.equation.boundary == 'Dirichlet':
            return self.build_coefficient_matrix_dirichlet(curr_equation, restrictor, interpolator)
        elif self.equation.boundary == 'Periodic':
            return self.build_coefficient_matrix_periodic(curr_equation, restrictor, interpolator)
        else:
            raise ValueError("Boundary condition must be either 'Dirichlet' or 'Periodic'")

    def restrict(self, u, curr_equation=None, restrictor=None):
        if restrictor is None:
            restrictor = self.restrictor
        if curr_equation is None:
            curr_equation = self.equation
        if curr_equation.boundary == 'Dirichlet':
            if curr_equation.dimension == 1:
                ans = restrictor @ u[1:-1]
                return torch.cat((torch.tensor([0], device=ans.device), ans, torch.tensor([0], device=self.device)))
            else:
                len_x = len(curr_equation.x)
                len_y = len(curr_equation.y)
                ans = restrictor @ u.reshape((len_x, len_y))[1:-1, 1:-1].flatten()
                new_len_x = (len_x - 2)// 2
                new_len_y = (len_y - 2)// 2
                return torch.nn.functional.pad(ans.reshape((new_len_x, new_len_y)), (1, 1, 1, 1), mode='constant', value=0).flatten()  # Add Dirichlet boundary conditions

        return restrictor @ u
    
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
        return interpolator @ u_coarse
    
    def valid_level(self, n):
        return n > 3


    def iteration(self, u_old, curr_equation=None, level = 1):
        if curr_equation is None:
            curr_equation = self.equation
        # else:
        restrictor, interpolator = self.build_restrictor_interpolator(curr_equation)
        # Pre-smoothing
        print("Pre-smoothing")
        u = WeightedJacobiSolver(curr_equation, weight=0.8).solve(u_init=u_old, max_iter=3)
        # Compute residual
        print("Computing residual")
        r = curr_equation.b - curr_equation.A @ u

        # Restrict residual to coarser grid
        print("Restricting residual")
        r_coarse = self.restrict(r, curr_equation, restrictor)

        u_init = torch.zeros_like(r_coarse)

        # Solver on coarser grid
        print("Solving on coarser grid")
        new_A = self.build_coefficient_matrix(curr_equation, restrictor, interpolator)

        if curr_equation.dimension == 1:
            equation_coarse =  curr_equation.__class__(a_func=None, f_func=r_coarse, boundary=curr_equation.boundary, x=curr_equation.x[::2], A=new_A)
        else:
            equation_coarse =  curr_equation.__class__(None, r_coarse, curr_equation.boundary, curr_equation.x[::2], curr_equation.y[::2], A=new_A)

        # equation_coarse = self.equation.__class__(None, r_coarse, self.equation.boundary, self.equation.x[::2], self.equation.y[::2] if self.equation.dimension == 2 else None, A=new_A)
        # equation_coarse = PoissonEquation2D(None, r_coarse, self.equation.boundary, self.equation.x[::2], self.equation.y[::2], A=new_A)
        solver_coarse = WeightedJacobiSolver(equation_coarse, weight=0.8)
        u_coarse = solver_coarse.solve(u_init=u_init, max_iter=3)

        # Prolongate solution to fine grid
        print("Prolongating solution to fine grid")
        u_prolonged = self.prolong(u_coarse, curr_equation, interpolator)

        # Update solution
        print("Updating solution")
        u += u_prolonged

        # Post-smoothing
        print("Post-smoothing")
        u = WeightedJacobiSolver(curr_equation, weight=0.8).solve(u_init=u, max_iter=3)
        return u

