import numpy as np
import torch



class PDE:
    def __init__(self, a_func, f_func, boundary, x, y=None, A = None):
        """
        a_func: diffusion coefficient function a(x) or a(x, y). Can be a function, or a tensor aligned with the grid.
        f_func: source term function f(x) or f(x, y). Can be a function, or a tensor aligned with the grid.
        boundary: "Dirichlet" or "Periodic"
        x: 1D tensor of grid points in x direction
        y: 1D tensor of grid points in y direction (optional, for 2D problems)
        A: Pre-built system matrix (optional). If None, it will be built based on a_func and the grid.
        """
        self.x = x
        self.y = y
        self.a_func = a_func
        self.f_func = f_func
        self.boundary = boundary 
        self.dimension = 1 if y is None else 2
        self.A = A 
        self.equation = None
        self.b = None
        self.u = None
        self.is_batch = False
        self.is_coefficient = isinstance(a_func, torch.Tensor) 

    def build_matrix(self):
        """
        This function builds the A matrix in Au = b based on the PDE specification.
        """
        if self.boundary == 'Dirichlet':
            return self.build_matrix_dirichlet()
        elif self.boundary == 'Periodic':
            return self.build_matrix_periodic()
        else:
            raise ValueError("Boundary condition must be either 'Dirichlet' or 'Periodic'")
        
    def build_matrix_periodic_batch(self):
        return NotImplementedError
    
    def build_matrix_periodic(self):
        return NotImplementedError
    
    def build_matrix_dirichlet_batch(self):
        return NotImplementedError
    
    def build_matrix_dirichlet(self):
        return NotImplementedError
    
    def build_rhs(self):
        """
        This function builds the b vector in Au = b based on the PDE specification.
        """
        return NotImplementedError
    
    def solve(self):
        """
        This function solves the linear system Au = b to get the solution u using torch.linalg.lstsq.
        """
        return NotImplementedError
    
    def compute_residual(self, u_approx, mask = None):
        """
        This function computes the residual for the given approximate solution.
        """
        if mask is None:
            mask = torch.ones(self.b.shape[0], dtype=torch.bool, device=self.device)
        if self.A is None or self.b is None:
            raise ValueError("Matrix A and vector b must be built before computing residual.")
        if self.is_batch:
            output = torch.bmm(self.A[mask], u_approx[mask].unsqueeze(-1)).squeeze(-1)
        else:
            output = self.A @ u_approx
        if self.is_batch:
            residual = self.b[mask] - output
        else:
            residual = self.b - output
        # if self.equation == "Poisson" and not self.is_coefficient and self.boundary == "Periodic":
        #     # For Poisson with constant coefficient and periodic BCs, the solution is only determined up to a constant. Remove mean to get meaningful residual.
        #     residual = residual - residual.mean(dim=-1, keepdim=True)
        return residual


class PoissonEquation1D(PDE):
    def __init__(self, a_func, f_func, boundary, x, A=None, solve=True, device='cpu'):
        super().__init__(a_func, f_func, boundary, x, None, A)
        
        # Check if this is batch mode (batch_size > 1)
        self.is_batch = False
        self.batch_size = 1
        
        if isinstance(a_func, torch.Tensor):
            if a_func.ndim == 2:  # (batch_size, grid_size)
                self.batch_size = a_func.shape[0]
                self.is_batch = True
            elif a_func.ndim > 2:
                raise ValueError("a_func tensor should be 1D or 2D (batch) for PoissonEquation1D")
        
        if isinstance(f_func, torch.Tensor):
            if f_func.ndim == 2:  # (batch_size, grid_size)
                if not self.is_batch:
                    self.batch_size = f_func.shape[0]
                    self.is_batch = True
                elif f_func.shape[0] != self.batch_size:
                    raise ValueError("Batch sizes of a_func and f_func must match")
            elif f_func.ndim > 2:
                raise ValueError("f_func tensor should be 1D or 2D (batch) for PoissonEquation1D")
        
        self.device = device
        self.equation = "Poisson"
        self.A = self.build_matrix() if A is None else A
        self.b = self.build_rhs() # if not isinstance(self.f_func, torch.Tensor) else self.f_func
        self.u = self.solve() if solve else None
    
    def _f_at(self, i):
        """
        Helper function to get f at grid point i, handling both tensor and callable cases, and batch vs non-batch.
        """
        if isinstance(self.f_func, torch.Tensor):
            return self.f_func[i] if not self.is_batch else self.f_func[:, i]  # (batch_size,) or scalar
        else:
            return self.f_func(self.x[i]) if not self.is_batch else torch.tensor([self.f_func(self.x[i])], device=self.device).expand(self.batch_size)  # scalar or (batch_size,)
        
    def _a_at(self, i):
        """
        Helper function to get a at grid point i, handling both tensor and callable cases, and batch vs non-batch.
        """
        if isinstance(self.a_func, torch.Tensor):
            return self.a_func[i] if not self.is_batch else self.a_func[:, i]  # (batch_size,) or scalar
        else:
            return self.a_func(self.x[i]) if not self.is_batch else torch.tensor([self.a_func(self.x[i])], device=self.device).expand(self.batch_size)  # scalar or (batch_size,)
        
    def _set_matrix_entry(self, A, i, j, value):
        """
        Helper function to set entry (i, j) of matrix A, handling both batch and non-batch cases.
        """
        if self.is_batch:
            A[:, i, j] = value
        else:
            A[i, j] = value

    def _set_vector_entry(self, b, i, value):
        """
        Helper function to set entry i of vector b, handling both batch and non-batch cases.
        """
        if self.is_batch:
            b[:, i] = value
        else:
            b[i] = value

    def build_matrix_dirichlet(self):
        """
        Builds the system matrix A for the 1D Poisson equation with Dirichlet boundary conditions using finite difference discretization.
        """
        n = len(self.x)
        h = self.x[1] - self.x[0]
        A = torch.zeros((n, n), device=self.device) if not self.is_batch else torch.zeros((self.batch_size, n, n), device=self.device)
        
        for i in range(1, n - 1):
            a_iminusone = self._a_at(i - 1)
            a_iplusone = self._a_at(i + 1)
            a_i = self._a_at(i)
            a_iminushalf = (a_iminusone + a_i) / 2
            a_iplushalf = (a_i + a_iplusone) / 2

            self._set_matrix_entry(A, i, i - 1, -a_iminushalf / h**2)
            self._set_matrix_entry(A, i, i, (a_iminushalf + a_iplushalf) / h**2)
            self._set_matrix_entry(A, i, i + 1, -a_iplushalf / h**2)
            

        self._set_matrix_entry(A, 0, 0, 1)
        self._set_matrix_entry(A, -1, -1, 1)
            
        return A
    

    def build_matrix_periodic(self):
        """
        Builds the system matrix A for the 1D Poisson equation with Periodic boundary conditions using finite difference discretization.
        """
        n = len(self.x)
        h = self.x[1] - self.x[0]
        A = torch.zeros((n, n), device=self.device) if not self.is_batch else torch.zeros((self.batch_size, n, n), device=self.device)
        for i in range(n):
            a_iminusone = self._a_at((i - 1 + n) % n)
            a_iplusone = self._a_at((i + 1) % n)
            a_i = self._a_at(i)
            a_iminushalf = (a_iminusone + a_i) / 2
            a_iplushalf = (a_i + a_iplusone) / 2
            self._set_matrix_entry(A, i, (i - 1 + n) % n, -a_iminushalf / h**2)
            self._set_matrix_entry(A, i, i, (a_iminushalf + a_iplushalf) / h**2)
            self._set_matrix_entry(A, i, (i + 1) % n, -a_iplushalf / h**2)
                
        return A

    def build_rhs(self):
        """
        Builds the b for the linear system Au = b based on the source term f and the grid, handling both batch and non-batch cases.
        """
        n = len(self.x)
        b = torch.zeros((self.batch_size, n), device=self.device) if self.is_batch else torch.zeros(n, device=self.device)
        for i in range(n):
            self._set_vector_entry(b, i, self._f_at(i))
        if self.boundary == 'Dirichlet':
            if self.is_batch:
                b[:, 0] = 0
                b[:, -1] = 0
            else:
                b[0] = 0
                b[-1] = 0
        return b

    def solve(self):
        u, *_ = torch.linalg.lstsq(self.A, self.b.unsqueeze(-1))  # (batch_size, n, 1) or (n, 1)
        return u.squeeze(-1)  


class PoissonEquation2D(PDE):
    def __init__(self, a_func, f_func, boundary, x, y, A=None, solve=True, device='cpu'):
        super().__init__(a_func, f_func, boundary, x, y, A)
        self.is_batch = False
        self.batch_size = 1
        
        if isinstance(a_func, torch.Tensor):
            if a_func.ndim == 2:  # (batch_size, grid_size)
                self.batch_size = a_func.shape[0]
                self.is_batch = True
            elif a_func.ndim > 2:
                raise ValueError("a_func tensor should be 1D or 2D (batch) for PoissonEquation2D")
        
        if isinstance(f_func, torch.Tensor):
            if f_func.ndim == 2:  # (batch_size, grid_size)
                if not self.is_batch:
                    self.batch_size = f_func.shape[0]
                    self.is_batch = True
                elif f_func.shape[0] != self.batch_size:
                    raise ValueError("Batch sizes of a_func and f_func must match")
            elif f_func.ndim > 2:
                raise ValueError("f_func tensor should be 1D or 2D (batch) for PoissonEquation2D")
        
        self.device = device
        self.equation = "Poisson"
        self.A = self.build_matrix() if A is None else A
        self.b = self.build_rhs() if not isinstance(self.f_func, torch.Tensor) else self.f_func
        self.u = self.solve() if solve else None

    def index(self, i, j):
        return i * len(self.y) + j
    
    def _a_at(self, i, j):
        if isinstance(self.a_func, torch.Tensor):
            return self.a_func[self.index(i, j)] if not self.is_batch else self.a_func[:, self.index(i, j)]  # (batch_size,) or scalar
        else:
            return self.a_func(self.x[i], self.y[j]) if not self.is_batch else torch.tensor([self.a_func(self.x[i], self.y[j])], device=self.device).expand(self.batch_size)  # scalar or (batch_size,)
    
    def _f_at(self, i, j):
        if isinstance(self.f_func, torch.Tensor):
            return self.f_func[self.index(i, j)] if not self.is_batch else self.f_func[:, self.index(i, j)]  # (batch_size,) or scalar
        else:
            return self.f_func(self.x[i], self.y[j]) if not self.is_batch else torch.tensor([self.f_func(self.x[i], self.y[j])], device=self.device).expand(self.batch_size)  # scalar or (batch_size,)
    
    def _set_matrix_entry(self, A, i, j, new_i, new_j, value):
        idx = self.index(i, j)
        new_idx = self.index(new_i, new_j)
        if self.is_batch:
            A[:, idx, new_idx] = value
        else:
            A[idx, new_idx] = value
    
    def _set_vector_entry(self, b, i, j, value):
        idx = self.index(i, j)
        if self.is_batch:
            b[:, idx] = value
        else:
            b[idx] = value
    
    def build_matrix_dirichlet(self):
        n = len(self.x)
        m = len(self.y)
        A = torch.zeros((self.batch_size, n * m, n * m), device=self.device) if self.is_batch else torch.zeros((n * m, n * m), device=self.device)
        h_x = self.x[1] - self.x[0]
        h_y = self.y[1] - self.y[0]
        for i in range(1, n - 1):
            for j in range(1, m - 1):
                idx = self.index(i, j)
                a_ij = self._a_at(i, j)
                a_iplusone = self._a_at(i + 1, j)
                a_iminusone = self._a_at(i - 1, j)
                a_jplusone = self._a_at(i, j + 1)
                a_jminusone = self._a_at(i, j - 1)
                a_iminushalf = (a_iminusone + a_ij) / 2
                a_iplushalf = (a_ij + a_iplusone) / 2
                a_jminushalf = (a_jminusone + a_ij) / 2
                a_jplushalf = (a_ij + a_jplusone) / 2
                self._set_matrix_entry(A, i, j, i - 1, j, -a_iminushalf / h_x**2)
                self._set_matrix_entry(A, i, j, i + 1, j, -a_iplushalf / h_x**2)
                self._set_matrix_entry(A, i, j, i, j - 1, -a_jminushalf / h_y**2)
                self._set_matrix_entry(A, i, j, i, j + 1, -a_jplushalf / h_y**2)
                self._set_matrix_entry(A, i, j, i, j, (a_iminushalf + a_iplushalf) / h_x**2 + (a_jminushalf + a_jplushalf) / h_y**2)
        for i in range(n):
            for j in [0, m - 1]:
                idx = self.index(i, j)
                if self.is_batch:
                    A[:, idx, :] = 0
                    A[:, idx, idx] = 1
                else:
                    A[idx, :] = 0
                    A[idx, idx] = 1
        for j in range(m):
            for i in [0, n - 1]:
                idx = self.index(i, j)
                if self.is_batch:
                    A[:, idx, :] = 0
                    A[:, idx, idx] = 1
                else:
                    A[idx, :] = 0
                    A[idx, idx] = 1
        return A

  
    
    def build_matrix_periodic(self):
        n = len(self.x)
        m = len(self.y)
        A = torch.zeros((n * m, n * m), device=self.device) if not self.is_batch else torch.zeros((self.batch_size, n * m, n * m), device=self.device)
        h_x = self.x[1] - self.x[0]
        h_y = self.y[1] - self.y[0]
        for i in range(n):
            for j in range(m):
                a_ij = self._a_at(i, j)
                a_iplusone = self._a_at((i + 1) % n, j)
                a_iminusone = self._a_at((i - 1 + n) % n, j)
                a_jplusone = self._a_at(i, (j + 1) % m)
                a_jminusone = self._a_at(i, (j - 1 + m) % m)
                a_iminushalf = (a_iminusone + a_ij) / 2
                a_iplushalf = (a_ij + a_iplusone) / 2
                a_jminushalf = (a_jminusone + a_ij) / 2
                a_jplushalf = (a_ij + a_jplusone) / 2
                self._set_matrix_entry(A, i, j, (i - 1 + n) % n, j, -a_iminushalf / h_x**2)
                self._set_matrix_entry(A, i, j, (i + 1) % n, j, -a_iplushalf / h_x**2)
                self._set_matrix_entry(A, i, j, i, (j - 1 + m) % m, -a_jminushalf / h_y**2)
                self._set_matrix_entry(A, i, j, i, (j + 1) % m, -a_jplushalf / h_y**2)
                self._set_matrix_entry(A, i, j, i, j, (a_iminushalf + a_iplushalf) / h_x**2 + (a_jminushalf + a_jplushalf) / h_y**2)
        return A

    def build_rhs(self):
        n = len(self.x)
        m = len(self.y)
        b = torch.zeros((n * m), device=self.device) if not self.is_batch else torch.zeros((self.batch_size, n * m), device=self.device)
        for i in range(n):
            for j in range(m):
                self._set_vector_entry(b, i, j, self._f_at(i, j))
        if self.boundary == 'Dirichlet':
            for i in range(n):
                if self.is_batch:
                    b[:, self.index(i, 0)] = 0
                    b[:, self.index(i, m - 1)] = 0
                else:
                    b[self.index(i, 0)] = 0
                    b[self.index(i, m - 1)] = 0
            for j in range(m):
                if self.is_batch:
                    b[:, self.index(0, j)] = 0
                    b[:, self.index(n - 1, j)] = 0
                else:
                    b[self.index(0, j)] = 0
                    b[self.index(n - 1, j)] = 0
        return b

    def solve(self):
        # Use torch.linalg.lstsq for least squares solution
        # batches of 64 solving at a time
        if self.is_batch:
            u = torch.zeros((self.batch_size, len(self.x) * len(self.y), 1), device=self.device)
            for i in range(0, self.batch_size, 64):
                u[i:i+64, :, :], *_ = torch.linalg.lstsq(self.A[i:i+64, :, :], self.b[i:i+64, :].unsqueeze(-1))
            return u.squeeze(-1).reshape((self.batch_size, len(self.x), len(self.y)))
        else:
            u, *_ = torch.linalg.lstsq(self.A, self.b.unsqueeze(-1))
            return u.squeeze(-1).reshape((len(self.x), len(self.y)))



class HelmholtzEquation1D(PDE):
    """
    Discretizes and solves:  -(a(x) u')' + k2 * u = f  on a 1D uniform grid.

    Parameters
    ----------
    a_func : callable or 1D torch.Tensor
        Diffusion coefficient a(x). If tensor, must be length n and aligned with x.
    f_func : callable or 1D torch.Tensor
        RHS f(x). If tensor, must be length n and aligned with x.
    k2 : float | callable | 1D torch.Tensor
        Helmholtz parameter k^2 (can vary with x if tensor/callable).
    boundary : {'Dirichlet','Periodic'}
        Boundary condition type.
    x : 1D torch.Tensor
        Grid points (assumed uniform).
    A : torch.Tensor or None
        Pre-built system matrix (optional).
    solve : bool
        If True, immediately solves Au=b.
    device : {'cpu','cuda'}
        Device for tensors.
    """

    def __init__(self, a_func, f_func, k2, boundary, x, A=None, solve=True, device='cpu'):
        self.k2 = k2
        super().__init__(a_func, f_func, boundary, x, None, A)
        self.is_batch = False
        self.batch_size = 1
        
        if isinstance(a_func, torch.Tensor):
            if a_func.ndim == 2:  # (batch_size, grid_size)
                self.batch_size = a_func.shape[0]
                self.is_batch = True
            elif a_func.ndim > 2:
                raise ValueError("a_func tensor should be 1D or 2D (batch) for HelmholtzEquation1D")
        
        if isinstance(f_func, torch.Tensor):
            if f_func.ndim == 2:  # (batch_size, grid_size)
                if not self.is_batch:
                    self.batch_size = f_func.shape[0]
                    self.is_batch = True
                elif f_func.shape[0] != self.batch_size:
                    raise ValueError("Batch sizes of a_func and f_func must match")
            elif f_func.ndim > 2:
                raise ValueError("f_func tensor should be 1D or 2D (batch) for HelmholtzEquation1D")
        
        if isinstance(k2, torch.Tensor):
            if k2.ndim == 2:  # (batch_size, grid_size)
                if not self.is_batch:
                    self.batch_size = k2.shape[0]
                    self.is_batch = True
                elif k2.shape[0] != self.batch_size:
                    raise ValueError("Batch size of k2 must match a_func and f_func")
            elif k2.ndim > 2:
                raise ValueError("k2 tensor should be 1D or 2D (batch) for HelmholtzEquation1D")
        
        self.device = device
        self.equation = "Helmholtz"
        self.A = self.build_matrix() if A is None else A # .to(device)
        self.b = self.build_rhs() if not isinstance(self.f_func, torch.Tensor) else self.f_func #.to(device)
        self.u = self.solve() if solve else None

    # ----- helpers -----
    def _a_at(self, i):
        if isinstance(self.a_func, torch.Tensor):
            return self.a_func[i] if not self.is_batch else self.a_func[:, i]  # (batch_size,) or scalar
        else:
            return self.a_func(self.x[i]) if not self.is_batch else torch.tensor([self.a_func(self.x[i])], device=self.device).expand(self.batch_size)  # scalar or (batch_size,)

    def _k2_at(self, i):
        if isinstance(self.k2, torch.Tensor):
            return self.k2[i] if not self.is_batch else self.k2[:, i]  # (batch_size,) or scalar
        elif callable(self.k2):
            return self.k2(self.x[i]) if not self.is_batch else torch.tensor([self.k2(self.x[i])], device=self.device).expand(self.batch_size)  # scalar or (batch_size,)
        else:
            return torch.as_tensor(self.k2, device=self.device) if not self.is_batch else torch.tensor([self.k2], device=self.device).expand(self.batch_size)
    
    def _f_at(self, i):
        if isinstance(self.f_func, torch.Tensor):
            return self.f_func[i] if not self.is_batch else self.f_func[:, i]  # (batch_size,) or scalar
        else:
            return self.f_func(self.x[i]) if not self.is_batch else torch.tensor([self.f_func(self.x[i])], device=self.device).expand(self.batch_size)  # scalar or (batch_size,)
        
    def _set_matrix_entry(self, A, i, j, value):
        if self.is_batch:
            A[:, i, j] = value
        else:
            A[i, j] = value
    
    def _set_vector_entry(self, b, i, value):
        if self.is_batch:
            b[:, i] = value
        else:
            b[i] = value

    # ----- matrices -----
    def build_matrix_dirichlet(self):
        n = len(self.x)
        h = self.x[1] - self.x[0]
        A = torch.zeros((self.batch_size, n, n), device=self.device) if self.is_batch else torch.zeros((n, n), device=self.device)

        for i in range(1, n - 1):
            a_im1 = self._a_at(i - 1)
            a_i   = self._a_at(i)
            a_ip1 = self._a_at(i + 1)
            a_imh = (a_im1 + a_i) / 2.0
            a_iph = (a_i + a_ip1) / 2.0

            self._set_matrix_entry(A, i, i - 1, -a_imh / h**2)
            self._set_matrix_entry(A, i, i, (a_imh + a_iph) / h**2 - self._k2_at(i))
            self._set_matrix_entry(A, i, i + 1, -a_iph / h**2)

        # Dirichlet BC rows (u=0 at boundaries)
        if self.is_batch:
            A[:, 0, 0] = 1.0
            A[:, -1, -1] = 1.0
        else:
            A[0, 0] = 1.0
            A[-1, -1] = 1.0
        return A
    
    def build_matrix_periodic(self):
        n = len(self.x)
        A = torch.zeros((self.batch_size, n, n), device=self.device) if self.is_batch else torch.zeros((n, n), device=self.device)
        h = self.x[1] - self.x[0]
        for i in range(n):
            im1 = (i - 1 + n) % n
            ip1 = (i + 1) % n

            a_im1 = self._a_at(im1)
            a_i   = self._a_at(i)
            a_ip1 = self._a_at(ip1)
            a_imh = (a_im1 + a_i) / 2.0
            a_iph = (a_i + a_ip1) / 2.0

            self._set_matrix_entry(A, i, im1, -a_imh / h**2)
            self._set_matrix_entry(A, i, i, (a_imh + a_iph) / h**2 - self._k2_at(i))
            self._set_matrix_entry(A, i, ip1, -a_iph / h**2)
        return A

    # ----- RHS -----
    def build_rhs(self):
        n = len(self.x)
        b = torch.zeros((self.batch_size, n), device=self.device) if self.is_batch else torch.zeros(n, device=self.device)
        for i in range(n):
            self._set_vector_entry(b, i, self._f_at(i))
        if self.boundary == 'Dirichlet':
            # enforce u=0 at boundaries
            b = b.clone()
            if self.is_batch:
                b[:, 0] = 0.0
                b[:, -1] = 0.0
            else:
                b[0] = 0.0 
                b[-1] = 0.0
        return b

    def solve(self):
        u, *_ = torch.linalg.lstsq(self.A, self.b.unsqueeze(-1))
        return u.squeeze(-1)


class HelmholtzEquation2D(PDE):
    def __init__(self, a_func, f_func, k2, boundary, x, y, A=None, solve=True, device='cpu'):
        super().__init__(a_func, f_func, boundary, x, y, A)
        self.k2 = k2
        self.is_batch = False
        self.batch_size = 1
        
        # Check batch dimensions for 2D case (flattened grid)
        expected_size = len(x) * len(y)
        
        if isinstance(a_func, torch.Tensor):
            if a_func.ndim == 2:  # (batch_size, grid_size_flattened)
                if a_func.shape[1] == expected_size:
                    self.batch_size = a_func.shape[0]
                    self.is_batch = True
                else:
                    raise ValueError(f"a_func tensor should have shape (batch_size, {expected_size}) for HelmholtzEquation2D")
            elif a_func.ndim > 2 or (a_func.ndim == 1 and len(a_func) != expected_size):
                raise ValueError(f"a_func tensor should be 1D with length {expected_size} or 2D (batch) for HelmholtzEquation2D")
        
        if isinstance(f_func, torch.Tensor):
            if f_func.ndim == 2:  # (batch_size, grid_size_flattened)
                if not self.is_batch:
                    if f_func.shape[1] == expected_size:
                        self.batch_size = f_func.shape[0]
                        self.is_batch = True
                    else:
                        raise ValueError(f"f_func tensor should have shape (batch_size, {expected_size}) for HelmholtzEquation2D")
                elif f_func.shape[0] != self.batch_size or f_func.shape[1] != expected_size:
                    raise ValueError(f"f_func tensor should have shape ({self.batch_size}, {expected_size}) to match batch configuration")
            elif f_func.ndim > 2 or (f_func.ndim == 1 and len(f_func) != expected_size):
                raise ValueError(f"f_func tensor should be 1D with length {expected_size} or 2D (batch) for HelmholtzEquation2D")
        
        if isinstance(k2, torch.Tensor):
            if k2.ndim == 2:  # (batch_size, grid_size_flattened)
                if not self.is_batch:
                    if k2.shape[1] == expected_size:
                        self.batch_size = k2.shape[0]
                        self.is_batch = True
                    else:
                        raise ValueError(f"k2 tensor should have shape (batch_size, {expected_size}) for HelmholtzEquation2D")
                elif k2.shape[0] != self.batch_size or k2.shape[1] != expected_size:
                    raise ValueError(f"k2 tensor should have shape ({self.batch_size}, {expected_size}) to match batch configuration")
            elif k2.ndim > 2 or (k2.ndim == 1 and len(k2) != expected_size):
                raise ValueError(f"k2 tensor should be 1D with length {expected_size} or 2D (batch) for HelmholtzEquation2D")

        self.device = device
        self.equation = "Helmholtz"

        self.A = self.build_matrix() if A is None else A# .to(device)
        self.b = self.build_rhs() if not isinstance(self.f_func, torch.Tensor) else self.f_func# .to(device)
        self.u = self.solve() if solve else None

    def index(self, i, j):
        return i * len(self.y) + j

    def _a_at(self, i, j):
        if isinstance(self.a_func, torch.Tensor):
            return self.a_func[self.index(i, j)] if not self.is_batch else self.a_func[:, self.index(i, j)]  # (batch_size,) or scalar
        else:
            return self.a_func(self.x[i], self.y[j]) if not self.is_batch else torch.tensor([self.a_func(self.x[i], self.y[j])], device=self.device).expand(self.batch_size)  # scalar or (batch_size,)

    def _k2_at(self, i, j):
        if isinstance(self.k2, torch.Tensor):
            return self.k2[self.index(i, j)] if not self.is_batch else self.k2[:, self.index(i, j)]  # (batch_size,) or scalar
        elif callable(self.k2):
            return self.k2(self.x[i], self.y[j]) if not self.is_batch else torch.tensor([self.k2(self.x[i], self.y[j])], device=self.device).expand(self.batch_size)  # scalar or (batch_size,)
        else:  # scalar
            return torch.as_tensor(self.k2, device=self.device) if not self.is_batch else torch.tensor([self.k2], device=self.device).expand(self.batch_size)
    
    def _f_at(self, i, j):
        if isinstance(self.f_func, torch.Tensor):
            return self.f_func[self.index(i, j)] if not self.is_batch else self.f_func[:, self.index(i, j)]  # (batch_size,) or scalar
        else:
            return self.f_func(self.x[i], self.y[j]) if not self.is_batch else torch.tensor([self.f_func(self.x[i], self.y[j])], device=self.device).expand(self.batch_size)  # scalar or (batch_size,)
    
    def _set_matrix_entry(self, A, i, j, new_i, new_j, value):
        idx = self.index(i, j)
        new_idx = self.index(new_i, new_j)
        if self.is_batch:
            A[:, idx, new_idx] = value
        else:
            A[idx, new_idx] = value
    
    def _set_vector_entry(self, b, i, j, value):
        idx = self.index(i, j)
        if self.is_batch:
            b[:, idx] = value
        else:
            b[idx] = value

    def build_matrix_dirichlet(self):
        n, m = len(self.x), len(self.y)
        A = torch.zeros((self.batch_size, n * m, n * m), device=self.device) if self.is_batch else torch.zeros((n * m, n * m), device=self.device)
        h_x = self.x[1] - self.x[0]
        h_y = self.y[1] - self.y[0]

        for i in range(1, n - 1):
            for j in range(1, m - 1):
                idx = self.index(i, j)
                a_ij = self._a_at(i, j)
                a_ip1 = self._a_at(i + 1, j)
                a_im1 = self._a_at(i - 1, j)
                a_jp1 = self._a_at(i, j + 1)
                a_jm1 = self._a_at(i, j - 1)

                a_imh = (a_im1 + a_ij) / 2
                a_iph = (a_ip1 + a_ij) / 2
                a_jmh = (a_jm1 + a_ij) / 2
                a_jph = (a_jp1 + a_ij) / 2

                self._set_matrix_entry(A, i, j, i - 1, j, -a_imh / h_x**2)
                self._set_matrix_entry(A, i, j, i + 1, j, -a_iph / h_x**2)
                self._set_matrix_entry(A, i, j, i, j - 1, -a_jmh / h_y**2)
                self._set_matrix_entry(A, i, j, i, j + 1, -a_jph / h_y**2)
                self._set_matrix_entry(A, i, j, i, j, (a_imh + a_iph) / h_x**2 + (a_jmh + a_jph) / h_y**2 - self._k2_at(i, j))

        # Dirichlet boundary rows
        for i in range(n):
            for j in [0, m - 1]:
                idx = self.index(i, j)
                if self.is_batch:
                    A[:, idx, :] = 0
                    A[:, idx, idx] = 1
                else:
                    A[idx, :] = 0
                    A[idx, idx] = 1
        for j in range(m):
            for i in [0, n - 1]:
                idx = self.index(i, j)
                if self.is_batch:
                    A[:, idx, :] = 0
                    A[:, idx, idx] = 1
                else:
                    A[idx, :] = 0
                    A[idx, idx] = 1
        return A

    def build_matrix_periodic(self):
        n, m = len(self.x), len(self.y)
        A = torch.zeros((self.batch_size, n * m, n * m), device=self.device) if self.is_batch else torch.zeros((n * m, n * m), device=self.device)
        h_x = self.x[1] - self.x[0]
        h_y = self.y[1] - self.y[0]

        for i in range(n):
            for j in range(m):
                idx = self.index(i, j)
                a_ij = self._a_at(i, j)
                a_ip1 = self._a_at((i + 1) % n, j)
                a_im1 = self._a_at((i - 1 + n) % n, j)
                a_jp1 = self._a_at(i, (j + 1) % m)
                a_jm1 = self._a_at(i, (j - 1 + m) % m)

                a_imh = (a_im1 + a_ij) / 2
                a_iph = (a_ip1 + a_ij) / 2
                a_jmh = (a_jm1 + a_ij) / 2
                a_jph = (a_jp1 + a_ij) / 2

                self._set_matrix_entry(A, i, j, (i - 1 + n) % n, j, -a_imh / h_x**2)
                self._set_matrix_entry(A, i, j, (i + 1) % n, j, -a_iph / h_x**2)
                self._set_matrix_entry(A, i, j, i, (j - 1 + m) % m, -a_jmh / h_y**2)
                self._set_matrix_entry(A, i, j, i, (j + 1) % m, -a_jph / h_y**2)
                self._set_matrix_entry(A, i, j, i, j, (a_imh + a_iph) / h_x**2 + (a_jmh + a_jph) / h_y**2 - self._k2_at(i, j))
        return A

    def build_rhs(self):
        n, m = len(self.x), len(self.y)
        b = torch.zeros((self.batch_size, n * m), device=self.device) if self.is_batch else torch.zeros(n * m, device=self.device)
        for i in range(n):
            for j in range(m):
                self._set_vector_entry(b, i, j, self._f_at(i, j))

        if self.boundary == 'Dirichlet':
            for i in range(n):
                self._set_vector_entry(b, i, 0, 0)
                self._set_vector_entry(b, i, m - 1, 0)
            for j in range(m):
                self._set_vector_entry(b, 0, j, 0)
                self._set_vector_entry(b, n - 1, j, 0)
        return b

    def solve(self):
        # u, *_ = torch.linalg.lstsq(self.A, self.b.unsqueeze(-1))
        # if self.is_batch:
        #     return u.squeeze(-1).reshape((self.batch_size, len(self.x), len(self.y)))
        # else:
        #     return u.squeeze(-1).reshape((len(self.x), len(self.y)))
        
        if self.is_batch:
            u = torch.zeros((self.batch_size, len(self.x) * len(self.y), 1), device=self.device)
            for i in range(0, self.batch_size, 64):
                u[i:i+64, :, :], *_ = torch.linalg.lstsq(self.A[i:i+64, :, :], self.b[i:i+64, :].unsqueeze(-1))
            return u.squeeze(-1).reshape((self.batch_size, len(self.x), len(self.y)))
        else:
            u, *_ = torch.linalg.lstsq(self.A, self.b.unsqueeze(-1))
            return u.squeeze(-1).reshape((len(self.x), len(self.y)))
  

class ConvectionDiffusion2D(PDE):
    """
    Discretizes and solves:  -∇·(a∇u) + b·∇u + c*u = f  on a 2D uniform grid
    using central differences for both diffusion and advection terms.

    Parameters
    ----------
    a_func : callable or tensor
        Diffusion coefficient (scalar, callable, or tensor).
    f_func : callable or tensor
        RHS forcing, shape (batch, N*N) for batch mode.
    b_vec : tuple (b1, b2)
        Constant advection velocity vector.
    reaction : float
        Constant reaction coefficient c (default 0.0 = pure convection-diffusion).
        When c > 0 the operator is positive-definite (no null space).
    boundary : {'Dirichlet', 'Periodic'}
    x, y : 1D tensors of grid points.
    """

    def __init__(self, a_func, f_func, b_vec, boundary, x, y, A=None, solve=True, device='cpu', reaction=0.0):
        super().__init__(a_func, f_func, boundary, x, y, A)
        self.b_vec = b_vec
        self.reaction = reaction
        self.is_batch = False
        self.batch_size = 1

        if isinstance(a_func, torch.Tensor):
            if a_func.ndim == 2:
                self.batch_size = a_func.shape[0]
                self.is_batch = True

        if isinstance(f_func, torch.Tensor):
            if f_func.ndim == 2:
                if not self.is_batch:
                    self.batch_size = f_func.shape[0]
                    self.is_batch = True
                elif f_func.shape[0] != self.batch_size:
                    raise ValueError("Batch sizes of a_func and f_func must match")

        self.device = device
        self.equation = "ConvDiff"
        self.A = self.build_matrix() if A is None else A
        self.b = self.build_rhs() if not isinstance(self.f_func, torch.Tensor) else self.f_func
        self.u = self.solve() if solve else None

    def index(self, i, j):
        return i * len(self.y) + j

    def _a_at(self, i, j):
        if isinstance(self.a_func, torch.Tensor):
            return self.a_func[self.index(i, j)] if not self.is_batch else self.a_func[:, self.index(i, j)]
        else:
            return self.a_func(self.x[i], self.y[j]) if not self.is_batch else torch.tensor([self.a_func(self.x[i], self.y[j])], device=self.device).expand(self.batch_size)

    def _f_at(self, i, j):
        if isinstance(self.f_func, torch.Tensor):
            return self.f_func[self.index(i, j)] if not self.is_batch else self.f_func[:, self.index(i, j)]
        else:
            return self.f_func(self.x[i], self.y[j]) if not self.is_batch else torch.tensor([self.f_func(self.x[i], self.y[j])], device=self.device).expand(self.batch_size)

    def _set_matrix_entry(self, A, i, j, new_i, new_j, value):
        idx = self.index(i, j)
        new_idx = self.index(new_i, new_j)
        if self.is_batch:
            A[:, idx, new_idx] = value
        else:
            A[idx, new_idx] = value

    def _set_vector_entry(self, b, i, j, value):
        idx = self.index(i, j)
        if self.is_batch:
            b[:, idx] = value
        else:
            b[idx] = value

    def build_matrix_periodic(self):
        n, m = len(self.x), len(self.y)
        A = torch.zeros((self.batch_size, n * m, n * m), device=self.device) if self.is_batch else torch.zeros((n * m, n * m), device=self.device)
        h_x = self.x[1] - self.x[0]
        h_y = self.y[1] - self.y[0]
        b1, b2 = self.b_vec

        for i in range(n):
            for j in range(m):
                a_ij = self._a_at(i, j)
                a_ip1 = self._a_at((i + 1) % n, j)
                a_im1 = self._a_at((i - 1 + n) % n, j)
                a_jp1 = self._a_at(i, (j + 1) % m)
                a_jm1 = self._a_at(i, (j - 1 + m) % m)

                a_imh = (a_im1 + a_ij) / 2
                a_iph = (a_ip1 + a_ij) / 2
                a_jmh = (a_jm1 + a_ij) / 2
                a_jph = (a_jp1 + a_ij) / 2

                # Diffusion + advection (central differences) + reaction
                self._set_matrix_entry(A, i, j, (i - 1 + n) % n, j, -a_imh / h_x**2 - b1 / (2 * h_x))
                self._set_matrix_entry(A, i, j, (i + 1) % n, j,     -a_iph / h_x**2 + b1 / (2 * h_x))
                self._set_matrix_entry(A, i, j, i, (j - 1 + m) % m, -a_jmh / h_y**2 - b2 / (2 * h_y))
                self._set_matrix_entry(A, i, j, i, (j + 1) % m,     -a_jph / h_y**2 + b2 / (2 * h_y))
                self._set_matrix_entry(A, i, j, i, j, (a_imh + a_iph) / h_x**2 + (a_jmh + a_jph) / h_y**2 + self.reaction)
        return A

    def build_matrix_dirichlet(self):
        n, m = len(self.x), len(self.y)
        A = torch.zeros((self.batch_size, n * m, n * m), device=self.device) if self.is_batch else torch.zeros((n * m, n * m), device=self.device)
        h_x = self.x[1] - self.x[0]
        h_y = self.y[1] - self.y[0]
        b1, b2 = self.b_vec

        for i in range(1, n - 1):
            for j in range(1, m - 1):
                a_ij = self._a_at(i, j)
                a_ip1 = self._a_at(i + 1, j)
                a_im1 = self._a_at(i - 1, j)
                a_jp1 = self._a_at(i, j + 1)
                a_jm1 = self._a_at(i, j - 1)

                a_imh = (a_im1 + a_ij) / 2
                a_iph = (a_ip1 + a_ij) / 2
                a_jmh = (a_jm1 + a_ij) / 2
                a_jph = (a_jp1 + a_ij) / 2

                self._set_matrix_entry(A, i, j, i - 1, j, -a_imh / h_x**2 - b1 / (2 * h_x))
                self._set_matrix_entry(A, i, j, i + 1, j, -a_iph / h_x**2 + b1 / (2 * h_x))
                self._set_matrix_entry(A, i, j, i, j - 1, -a_jmh / h_y**2 - b2 / (2 * h_y))
                self._set_matrix_entry(A, i, j, i, j + 1, -a_jph / h_y**2 + b2 / (2 * h_y))
                self._set_matrix_entry(A, i, j, i, j, (a_imh + a_iph) / h_x**2 + (a_jmh + a_jph) / h_y**2 + self.reaction)

        for i in range(n):
            for j in [0, m - 1]:
                idx = self.index(i, j)
                if self.is_batch:
                    A[:, idx, :] = 0; A[:, idx, idx] = 1
                else:
                    A[idx, :] = 0; A[idx, idx] = 1
        for j in range(m):
            for i in [0, n - 1]:
                idx = self.index(i, j)
                if self.is_batch:
                    A[:, idx, :] = 0; A[:, idx, idx] = 1
                else:
                    A[idx, :] = 0; A[idx, idx] = 1
        return A

    def build_rhs(self):
        n, m = len(self.x), len(self.y)
        b = torch.zeros((self.batch_size, n * m), device=self.device) if self.is_batch else torch.zeros(n * m, device=self.device)
        for i in range(n):
            for j in range(m):
                self._set_vector_entry(b, i, j, self._f_at(i, j))
        if self.boundary == 'Dirichlet':
            for i in range(n):
                self._set_vector_entry(b, i, 0, 0)
                self._set_vector_entry(b, i, m - 1, 0)
            for j in range(m):
                self._set_vector_entry(b, 0, j, 0)
                self._set_vector_entry(b, n - 1, j, 0)
        return b

    def solve(self):
        u, *_ = torch.linalg.lstsq(self.A, self.b.unsqueeze(-1))
        if self.is_batch:
            return u.squeeze(-1).reshape((self.batch_size, len(self.x), len(self.y)))
        else:
            return u.squeeze(-1).reshape((len(self.x), len(self.y)))

class ConvectionDiffusion2D(PDE):
    """
    Discretizes and solves:  -∇·(a∇u) + b·∇u + c*u = f  on a 2D uniform grid
    using central differences for both diffusion and advection terms.

    Parameters
    ----------
    a_func : callable or tensor
        Diffusion coefficient (scalar, callable, or tensor).
    f_func : callable or tensor
        RHS forcing, shape (batch, N*N) for batch mode.
    b_vec : tuple (b1, b2)
        Constant advection velocity vector.
    reaction : float
        Constant reaction coefficient c (default 0.0 = pure convection-diffusion).
        When c > 0 the operator is positive-definite (no null space).
    boundary : {'Dirichlet', 'Periodic'}
    x, y : 1D tensors of grid points.
    """

    def __init__(self, a_func, f_func, b_vec, boundary, x, y, A=None, solve=True, device='cpu', reaction=0.0):
        super().__init__(a_func, f_func, boundary, x, y, A)
        self.b_vec = b_vec
        self.reaction = reaction
        self.is_batch = False
        self.batch_size = 1

        if isinstance(a_func, torch.Tensor):
            if a_func.ndim == 2:
                self.batch_size = a_func.shape[0]
                self.is_batch = True

        if isinstance(f_func, torch.Tensor):
            if f_func.ndim == 2:
                if not self.is_batch:
                    self.batch_size = f_func.shape[0]
                    self.is_batch = True
                elif f_func.shape[0] != self.batch_size:
                    raise ValueError("Batch sizes of a_func and f_func must match")

        self.device = device
        self.equation = "ConvDiff"
        self.A = self.build_matrix() if A is None else A
        self.b = self.build_rhs() if not isinstance(self.f_func, torch.Tensor) else self.f_func
        self.u = self.solve() if solve else None

    def index(self, i, j):
        return i * len(self.y) + j

    def _a_at(self, i, j):
        if isinstance(self.a_func, torch.Tensor):
            return self.a_func[self.index(i, j)] if not self.is_batch else self.a_func[:, self.index(i, j)]
        else:
            return self.a_func(self.x[i], self.y[j]) if not self.is_batch else torch.tensor([self.a_func(self.x[i], self.y[j])], device=self.device).expand(self.batch_size)

    def _f_at(self, i, j):
        if isinstance(self.f_func, torch.Tensor):
            return self.f_func[self.index(i, j)] if not self.is_batch else self.f_func[:, self.index(i, j)]
        else:
            return self.f_func(self.x[i], self.y[j]) if not self.is_batch else torch.tensor([self.f_func(self.x[i], self.y[j])], device=self.device).expand(self.batch_size)

    def _set_matrix_entry(self, A, i, j, new_i, new_j, value):
        idx = self.index(i, j)
        new_idx = self.index(new_i, new_j)
        if self.is_batch:
            A[:, idx, new_idx] = value
        else:
            A[idx, new_idx] = value

    def _set_vector_entry(self, b, i, j, value):
        idx = self.index(i, j)
        if self.is_batch:
            b[:, idx] = value
        else:
            b[idx] = value

    def build_matrix_periodic(self):
        n, m = len(self.x), len(self.y)
        A = torch.zeros((self.batch_size, n * m, n * m), device=self.device) if self.is_batch else torch.zeros((n * m, n * m), device=self.device)
        h_x = self.x[1] - self.x[0]
        h_y = self.y[1] - self.y[0]
        b1, b2 = self.b_vec

        for i in range(n):
            for j in range(m):
                a_ij = self._a_at(i, j)
                a_ip1 = self._a_at((i + 1) % n, j)
                a_im1 = self._a_at((i - 1 + n) % n, j)
                a_jp1 = self._a_at(i, (j + 1) % m)
                a_jm1 = self._a_at(i, (j - 1 + m) % m)

                a_imh = (a_im1 + a_ij) / 2
                a_iph = (a_ip1 + a_ij) / 2
                a_jmh = (a_jm1 + a_ij) / 2
                a_jph = (a_jp1 + a_ij) / 2

                # Diffusion + advection (central differences) + reaction
                self._set_matrix_entry(A, i, j, (i - 1 + n) % n, j, -a_imh / h_x**2 - b1 / (2 * h_x))
                self._set_matrix_entry(A, i, j, (i + 1) % n, j,     -a_iph / h_x**2 + b1 / (2 * h_x))
                self._set_matrix_entry(A, i, j, i, (j - 1 + m) % m, -a_jmh / h_y**2 - b2 / (2 * h_y))
                self._set_matrix_entry(A, i, j, i, (j + 1) % m,     -a_jph / h_y**2 + b2 / (2 * h_y))
                self._set_matrix_entry(A, i, j, i, j, (a_imh + a_iph) / h_x**2 + (a_jmh + a_jph) / h_y**2 + self.reaction)
        return A

    def build_matrix_dirichlet(self):
        n, m = len(self.x), len(self.y)
        A = torch.zeros((self.batch_size, n * m, n * m), device=self.device) if self.is_batch else torch.zeros((n * m, n * m), device=self.device)
        h_x = self.x[1] - self.x[0]
        h_y = self.y[1] - self.y[0]
        b1, b2 = self.b_vec

        for i in range(1, n - 1):
            for j in range(1, m - 1):
                a_ij = self._a_at(i, j)
                a_ip1 = self._a_at(i + 1, j)
                a_im1 = self._a_at(i - 1, j)
                a_jp1 = self._a_at(i, j + 1)
                a_jm1 = self._a_at(i, j - 1)

                a_imh = (a_im1 + a_ij) / 2
                a_iph = (a_ip1 + a_ij) / 2
                a_jmh = (a_jm1 + a_ij) / 2
                a_jph = (a_jp1 + a_ij) / 2

                self._set_matrix_entry(A, i, j, i - 1, j, -a_imh / h_x**2 - b1 / (2 * h_x))
                self._set_matrix_entry(A, i, j, i + 1, j, -a_iph / h_x**2 + b1 / (2 * h_x))
                self._set_matrix_entry(A, i, j, i, j - 1, -a_jmh / h_y**2 - b2 / (2 * h_y))
                self._set_matrix_entry(A, i, j, i, j + 1, -a_jph / h_y**2 + b2 / (2 * h_y))
                self._set_matrix_entry(A, i, j, i, j, (a_imh + a_iph) / h_x**2 + (a_jmh + a_jph) / h_y**2 + self.reaction)

        for i in range(n):
            for j in [0, m - 1]:
                idx = self.index(i, j)
                if self.is_batch:
                    A[:, idx, :] = 0; A[:, idx, idx] = 1
                else:
                    A[idx, :] = 0; A[idx, idx] = 1
        for j in range(m):
            for i in [0, n - 1]:
                idx = self.index(i, j)
                if self.is_batch:
                    A[:, idx, :] = 0; A[:, idx, idx] = 1
                else:
                    A[idx, :] = 0; A[idx, idx] = 1
        return A

    def build_rhs(self):
        n, m = len(self.x), len(self.y)
        b = torch.zeros((self.batch_size, n * m), device=self.device) if self.is_batch else torch.zeros(n * m, device=self.device)
        for i in range(n):
            for j in range(m):
                self._set_vector_entry(b, i, j, self._f_at(i, j))
        if self.boundary == 'Dirichlet':
            for i in range(n):
                self._set_vector_entry(b, i, 0, 0)
                self._set_vector_entry(b, i, m - 1, 0)
            for j in range(m):
                self._set_vector_entry(b, 0, j, 0)
                self._set_vector_entry(b, n - 1, j, 0)
        return b

    def solve(self):
        if self.is_batch:
            u = torch.zeros((self.batch_size, len(self.x) * len(self.y), 1), device=self.device)
            for i in range(0, self.batch_size, 64):
                u[i:i+64, :, :], *_ = torch.linalg.lstsq(self.A[i:i+64, :, :], self.b[i:i+64, :].unsqueeze(-1))
            return u.squeeze(-1).reshape((self.batch_size, len(self.x), len(self.y)))
        else:
            u, *_ = torch.linalg.lstsq(self.A, self.b.unsqueeze(-1))
            return u.squeeze(-1).reshape((len(self.x), len(self.y)))
        # u, *_ = torch.linalg.lstsq(self.A, self.b.unsqueeze(-1))
        # if self.is_batch:
        #     return u.squeeze(-1).reshape((self.batch_size, len(self.x), len(self.y)))
        # else:
        #     return u.squeeze(-1).reshape((len(self.x), len(self.y)))

