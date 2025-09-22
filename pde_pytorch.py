import numpy as np
import torch




class PDE:
    def __init__(self, a_func, f_func, boundary, x, y=None, A = None):
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
    
    def build_matrix(self):
        if self.boundary == 'Dirichlet':
            return self.build_matrix_dirichlet()
        elif self.boundary == 'Periodic':
            return self.build_matrix_periodic()
        else:
            raise ValueError("Boundary condition must be either 'Dirichlet' or 'Periodic'")
    
    def build_matrix_periodic(self):
        return NotImplementedError
    
    def build_matrix_dirichlet(self):
        return NotImplementedError
    
    def build_rhs(self):
        return NotImplementedError
    
    def solve(self):
        return NotImplementedError
    
    def compute_residual(self, u_approx):
        if self.A is None or self.b is None:
            raise ValueError("Matrix A and vector b must be built before computing residual.")
        residual = self.b - self.A @ u_approx
        return residual


class PoissonEquation1D(PDE):
    def __init__(self, a_func, f_func, boundary, x, A=None, solve=True, device='cpu'):
        super().__init__(a_func, f_func, boundary, x, A)
        if isinstance(a_func, torch.Tensor) and len(a_func.shape) > 1:
            raise ValueError("a_func tensor should be 1D for PoissonEquation1D")
        if isinstance(f_func, torch.Tensor) and len(f_func.shape) > 1:
            raise ValueError("f_func tensor should be 1D for PoissonEquation1D")
        self.device = device
        self.equation = "Poisson"
        self.A = self.build_matrix() if A is None else A
        self.b = self.build_rhs() if not isinstance(self.f_func, torch.Tensor) else self.f_func
        self.u = self.solve() if solve else None

    def build_matrix_dirichlet(self):
        n = len(self.x)
        A = torch.zeros((n, n), device=self.device)
        h = self.x[1] - self.x[0]
        for i in range(1, n - 1):
            a_iminusone = self.a_func[i - 1] if isinstance(self.a_func, torch.Tensor) else self.a_func(self.x[i - 1])
            a_iplusone = self.a_func[i + 1] if isinstance(self.a_func, torch.Tensor) else self.a_func(self.x[i + 1])
            a_i = self.a_func[i] if isinstance(self.a_func, torch.Tensor) else self.a_func(self.x[i])
            a_iminushalf = (a_iminusone + a_i) / 2
            a_iplushalf = (a_i + a_iplusone) / 2
            A[i, i - 1] = -a_iminushalf / h**2
            A[i, i] = (a_iminushalf + a_iplushalf) / h**2
            A[i, i + 1] = -a_iplushalf / h**2
        A[0, 0] = 1
        A[-1, -1] = 1
        return A

    def build_matrix_periodic(self):
        n = len(self.x)
        A = torch.zeros((n, n), device=self.device)
        h = self.x[1] - self.x[0]
        for i in range(n):
            a_iminusone = self.a_func[(i - 1 + n) % n] if isinstance(self.a_func, torch.Tensor) else self.a_func(self.x[(i - 1 + n) % n])
            a_iplusone = self.a_func[(i + 1) % n] if isinstance(self.a_func, torch.Tensor) else self.a_func(self.x[(i + 1) % n])
            a_i = self.a_func[i] if isinstance(self.a_func, torch.Tensor) else self.a_func(self.x[i])
            a_iminushalf = (a_iminusone + a_i) / 2
            a_iplushalf = (a_i + a_iplusone) / 2
            A[i, (i - 1 + n) % n] = -a_iminushalf / h**2
            A[i, i] = (a_iminushalf + a_iplushalf) / h**2
            A[i, (i + 1) % n] = -a_iplushalf / h**2
        return A

    def build_rhs(self):
        n = len(self.x)
        b = torch.zeros(n, device=self.device)
        h = self.x[1] - self.x[0]
        for i in range(n):
            b[i] = self.f_func(self.x[i])
        if self.boundary == 'Dirichlet':
            b[0] = 0
            b[-1] = 0
        return b

    def solve(self):
        # Use torch.linalg.lstsq for least squares solution
        u, *_ = torch.linalg.lstsq(self.A, self.b.unsqueeze(1))
        return u.squeeze(1)


class PoissonEquation2D(PDE):
    def __init__(self, a_func, f_func, boundary, x, y, A=None, solve=True, device='cpu'):
        super().__init__(a_func, f_func, boundary, x, y, A)
        if isinstance(a_func, torch.Tensor) and len(a_func.shape) > 1:
            raise ValueError("a_func tensor should be 1D for PoissonEquation2D")
        if isinstance(f_func, torch.Tensor) and len(f_func.shape) > 1:
            raise ValueError("f_func tensor should be 1D for PoissonEquation2D")
        self.device = device
        self.equation = "Poisson"
        self.A = self.build_matrix() if A is None else A
        self.b = self.build_rhs() if not isinstance(self.f_func, torch.Tensor) else self.f_func
        self.u = self.solve() if solve else None

    def index(self, i, j):
        return i * len(self.y) + j

    def build_matrix_dirichlet(self):
        n = len(self.x)
        m = len(self.y)
        A = torch.zeros((n * m, n * m), device=self.device)
        h_x = self.x[1] - self.x[0]
        h_y = self.y[1] - self.y[0]
        for i in range(1, n - 1):
            for j in range(1, m - 1):
                idx = self.index(i, j)
                a_ij = self.a_func[idx] if isinstance(self.a_func, torch.Tensor) else self.a_func(self.x[i], self.y[j])
                a_iplusone = self.a_func[self.index(i + 1, j)] if isinstance(self.a_func, torch.Tensor) else self.a_func(self.x[i + 1], self.y[j])
                a_iminusone = self.a_func[self.index(i - 1, j)] if isinstance(self.a_func, torch.Tensor) else self.a_func(self.x[i - 1], self.y[j])
                a_jplusone = self.a_func[self.index(i, j + 1)] if isinstance(self.a_func, torch.Tensor) else self.a_func(self.x[i], self.y[j + 1])
                a_jminusone = self.a_func[self.index(i, j - 1)] if isinstance(self.a_func, torch.Tensor) else self.a_func(self.x[i], self.y[j - 1])
                a_iminushalf = (a_iminusone + a_ij) / 2
                a_iplushalf = (a_ij + a_iplusone) / 2
                a_jminushalf = (a_jminusone + a_ij) / 2
                a_jplushalf = (a_ij + a_jplusone) / 2
                A[idx, self.index(i - 1, j)] = -a_iminushalf / h_x**2
                A[idx, self.index(i + 1, j)] = -a_iplushalf / h_x**2
                A[idx, self.index(i, j - 1)] = -a_jminushalf / h_y**2
                A[idx, self.index(i, j + 1)] = -a_jplushalf / h_y**2
                A[idx, idx] = (a_iminushalf + a_iplushalf) / h_x**2 + (a_jminushalf + a_jplushalf) / h_y**2

        for i in range(n):
            for j in [0, m - 1]:
                idx = self.index(i, j)
                A[idx, :] = 0
                A[idx, idx] = 1
        for j in range(m):
            for i in [0, n - 1]:
                idx = self.index(i, j)
                A[idx, :] = 0
                A[idx, idx] = 1
        return A

    def build_matrix_periodic(self):
        n = len(self.x)
        m = len(self.y)
        A = torch.zeros((n * m, n * m), device=self.device)
        h_x = self.x[1] - self.x[0]
        h_y = self.y[1] - self.y[0]
        for i in range(n):
            for j in range(m):
                idx = self.index(i, j)
                a_ij = self.a_func[idx] if isinstance(self.a_func, torch.Tensor) else self.a_func(self.x[i], self.y[j])
                a_iplusone = self.a_func[self.index((i + 1) % n, j)] if isinstance(self.a_func, torch.Tensor) else self.a_func(self.x[(i + 1) % n], self.y[j])
                a_iminusone = self.a_func[self.index((i - 1 + n) % n, j)] if isinstance(self.a_func, torch.Tensor) else self.a_func(self.x[(i - 1 + n) % n], self.y[j])
                a_jplusone = self.a_func[self.index(i, (j + 1) % m)] if isinstance(self.a_func, torch.Tensor) else self.a_func(self.x[i], self.y[(j + 1) % m])
                a_jminusone = self.a_func[self.index(i, (j - 1 + m) % m)] if isinstance(self.a_func, torch.Tensor) else self.a_func(self.x[i], self.y[(j - 1 + m) % m])
                a_iminushalf = (a_iminusone + a_ij) / 2
                a_iplushalf = (a_ij + a_iplusone) / 2
                a_jminushalf = (a_jminusone + a_ij) / 2
                a_jplushalf = (a_ij + a_jplusone) / 2
                A[idx, self.index((i - 1 + n) % n, j)] = -a_iminushalf / h_x**2
                A[idx, self.index((i + 1) % n, j)] = -a_iplushalf / h_x**2
                A[idx, self.index(i, (j - 1 + m) % m)] = -a_jminushalf / h_y**2
                A[idx, self.index(i, (j + 1) % m)] = -a_jplushalf / h_y**2
                A[idx, idx] = (a_iminushalf + a_iplushalf) / h_x**2 + (a_jminushalf + a_jplushalf) / h_y**2
        return A

    def build_rhs(self):
        n = len(self.x)
        m = len(self.y)
        b = torch.zeros(n * m, device=self.device)
        for i in range(n):
            for j in range(m):
                idx = self.index(i, j)
                b[idx] = self.f_func(self.x[i], self.y[j])
        if self.boundary == 'Dirichlet':
            for i in range(n):
                b[self.index(i, 0)] = 0
                b[self.index(i, m - 1)] = 0
            for j in range(m):
                b[self.index(0, j)] = 0
                b[self.index(n - 1, j)] = 0
        return b

    def solve(self):
        # Use torch.linalg.lstsq for least squares solution
        u, *_ = torch.linalg.lstsq(self.A, self.b.unsqueeze(1))
        return u.squeeze(1).reshape((len(self.x), len(self.y)))



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
        super().__init__(a_func, f_func, boundary, x, A)
        if isinstance(a_func, torch.Tensor) and a_func.ndim > 1:
            raise ValueError("a_func tensor should be 1D for HelmholtzEquation1D")
        if isinstance(f_func, torch.Tensor) and f_func.ndim > 1:
            raise ValueError("f_func tensor should be 1D for HelmholtzEquation1D")
        if isinstance(k2, torch.Tensor) and k2.ndim > 1:
            raise ValueError("k2 tensor should be 1D for HelmholtzEquation1D")
        self.device = device
        self.equation = "Helmholtz"
        self.A = self.build_matrix() if A is None else A # .to(device)
        self.b = self.build_rhs() if not isinstance(self.f_func, torch.Tensor) else self.f_func #.to(device)
        self.u = self.solve() if solve else None

    # ----- helpers -----
    def _a_at(self, i):
        if isinstance(self.a_func, torch.Tensor):
            return self.a_func[i]
        else:
            return self.a_func(self.x[i])

    def _k2_at(self, i):
        if isinstance(self.k2, torch.Tensor):
            return self.k2[i]
        elif callable(self.k2):
            return self.k2(self.x[i])
        else:
            return torch.as_tensor(self.k2, device=self.device)

    # ----- matrices -----
    def build_matrix_dirichlet(self):
        n = len(self.x)
        A = torch.zeros((n, n), device=self.device)
        h = self.x[1] - self.x[0]

        # interior stencil for  -(a u')' + k2*u
        for i in range(1, n - 1):
            a_im1 = self._a_at(i - 1)
            a_i   = self._a_at(i)
            a_ip1 = self._a_at(i + 1)
            a_imh = (a_im1 + a_i) / 2.0
            a_iph = (a_i + a_ip1) / 2.0

            A[i, i - 1] = -a_imh / h**2
            A[i, i]     = (a_imh + a_iph) / h**2 + self._k2_at(i)
            A[i, i + 1] = -a_iph / h**2

        # Dirichlet BC rows (u=0 at boundaries)
        A[0, 0]   = 1.0
        A[-1, -1] = 1.0
        return A

    def build_matrix_periodic(self):
        n = len(self.x)
        A = torch.zeros((n, n), device=self.device)
        h = self.x[1] - self.x[0]

        for i in range(n):
            im1 = (i - 1 + n) % n
            ip1 = (i + 1) % n

            a_im1 = self._a_at(im1)
            a_i   = self._a_at(i)
            a_ip1 = self._a_at(ip1)
            a_imh = (a_im1 + a_i) / 2.0
            a_iph = (a_i + a_ip1) / 2.0

            A[i, im1] = -a_imh / h**2
            A[i, i]   = (a_imh + a_iph) / h**2 + self._k2_at(i)
            A[i, ip1] = -a_iph / h**2

        return A

    # ----- RHS -----
    def build_rhs(self):
        n = len(self.x)
        b = torch.zeros(n, device=self.device)

        if isinstance(self.f_func, torch.Tensor):
            b = self.f_func.to(self.device)
        else:
            for i in range(n):
                b[i] = self.f_func(self.x[i])

        if self.boundary == 'Dirichlet':
            # enforce u=0 at boundaries
            b = b.clone()
            b[0] = 0.0
            b[-1] = 0.0

        return b

    # ----- solve -----
    def solve(self):
        # Use least-squares for robustness (works for both Dirichlet & Periodic)
        u, *_ = torch.linalg.lstsq(self.A, self.b.unsqueeze(1))
        return u.squeeze(1)


import torch

class HelmholtzEquation2D(PDE):
    def __init__(self, a_func, f_func, k2, boundary, x, y, A=None, solve=True, device='cpu'):
        super().__init__(a_func, f_func, boundary, x, y, A)
        if isinstance(a_func, torch.Tensor) and a_func.ndim > 1:
            raise ValueError("a_func tensor should be 1D for HelmholtzEquation2D")
        if isinstance(f_func, torch.Tensor) and f_func.ndim > 1:
            raise ValueError("f_func tensor should be 1D for HelmholtzEquation2D")
        if isinstance(k2, torch.Tensor) and k2.ndim > 1:
            raise ValueError("k2 tensor should be 1D for HelmholtzEquation2D")

        self.device = device
        self.equation = "Helmholtz"
        self.k2 = k2

        self.A = self.build_matrix() if A is None else A# .to(device)
        self.b = self.build_rhs() if not isinstance(self.f_func, torch.Tensor) else self.f_func# .to(device)
        self.u = self.solve() if solve else None

    def index(self, i, j):
        return i * len(self.y) + j

    def _a_at(self, i, j):
        if isinstance(self.a_func, torch.Tensor):
            return self.a_func[self.index(i, j)]
        else:
            return self.a_func(self.x[i], self.y[j])

    def _k2_at(self, i, j):
        if isinstance(self.k2, torch.Tensor):
            return self.k2[self.index(i, j)]
        elif callable(self.k2):
            return self.k2(self.x[i], self.y[j])
        else:  # scalar
            return torch.as_tensor(self.k2, device=self.device)

    def build_matrix_dirichlet(self):
        n, m = len(self.x), len(self.y)
        A = torch.zeros((n * m, n * m), device=self.device)
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

                A[idx, self.index(i - 1, j)] = -a_imh / h_x**2
                A[idx, self.index(i + 1, j)] = -a_iph / h_x**2
                A[idx, self.index(i, j - 1)] = -a_jmh / h_y**2
                A[idx, self.index(i, j + 1)] = -a_jph / h_y**2
                A[idx, idx] = (a_imh + a_iph) / h_x**2 + (a_jmh + a_jph) / h_y**2 + self._k2_at(i, j)

        # Dirichlet boundary rows
        for i in range(n):
            for j in [0, m - 1]:
                idx = self.index(i, j)
                A[idx, :] = 0
                A[idx, idx] = 1
        for j in range(m):
            for i in [0, n - 1]:
                idx = self.index(i, j)
                A[idx, :] = 0
                A[idx, idx] = 1
        return A

    def build_matrix_periodic(self):
        n, m = len(self.x), len(self.y)
        A = torch.zeros((n * m, n * m), device=self.device)
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

                A[idx, self.index((i - 1 + n) % n, j)] = -a_imh / h_x**2
                A[idx, self.index((i + 1) % n, j)] = -a_iph / h_x**2
                A[idx, self.index(i, (j - 1 + m) % m)] = -a_jmh / h_y**2
                A[idx, self.index(i, (j + 1) % m)] = -a_jph / h_y**2
                A[idx, idx] = (a_imh + a_iph) / h_x**2 + (a_jmh + a_jph) / h_y**2 + self._k2_at(i, j)
        return A

    def build_rhs(self):
        n, m = len(self.x), len(self.y)
        b = torch.zeros(n * m, device=self.device)

        if isinstance(self.f_func, torch.Tensor):
            b = self.f_func.to(self.device)
        else:
            for i in range(n):
                for j in range(m):
                    b[self.index(i, j)] = self.f_func(self.x[i], self.y[j])

        if self.boundary == 'Dirichlet':
            for i in range(n):
                b[self.index(i, 0)] = 0
                b[self.index(i, m - 1)] = 0
            for j in range(m):
                b[self.index(0, j)] = 0
                b[self.index(n - 1, j)] = 0
        return b

    def solve(self):
        u, *_ = torch.linalg.lstsq(self.A, self.b.unsqueeze(1))
        return u.squeeze(1).reshape((len(self.x), len(self.y)))
  