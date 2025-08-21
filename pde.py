import numpy as np
from scipy.interpolate import interp1d, interp2d
import matplotlib.pyplot as plt




class PDE:
    def __init__(self, a_func, f_func, boundary, x, y=None, A = None):
        self.x = x
        self.y = y
        self.a_func = a_func
        self.f_func = f_func
        self.boundary = boundary 
        self.dimension = 1 if y is None else 2
        self.A = A 
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
    def __init__(self, a_func, f_func, boundary, x, A = None):
        super().__init__(a_func, f_func, boundary, x, A)
        self.A = self.build_matrix() if A is None else A
        self.b = self.build_rhs() if not isinstance(self.f_func, np.ndarray) else self.f_func
        self.u = self.solve()
    
    
    def build_matrix_dirichlet(self):
        n = len(self.x)
        A = np.zeros((n, n))
        h = self.x[1] - self.x[0]
        for i in range(1, n - 1):
            a_iminusone = self.a_func[i - 1] if isinstance(self.a_func, np.ndarray) else self.a_func(self.x[i - 1])
            a_iplusone = self.a_func[i + 1] if isinstance(self.a_func, np.ndarray) else self.a_func(self.x[i + 1])
            a_i = self.a_func[i] if isinstance(self.a_func, np.ndarray) else self.a_func(self.x[i])
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
        A = np.zeros((n, n))
        h = self.x[1] - self.x[0]
        
        for i in range(n): 
            a_iminusone = self.a_func[(i - 1 + n) % n] if isinstance(self.a_func, np.ndarray) else self.a_func(self.x[(i - 1 + n) % n])
            a_iplusone = self.a_func[(i + 1) % n] if isinstance(self.a_func, np.ndarray) else self.a_func(self.x[(i + 1) % n])
            a_i = self.a_func[i] if isinstance(self.a_func, np.ndarray) else self.a_func(self.x[i])
            a_iminushalf = (a_iminusone + a_i) / 2
            a_iplushalf = (a_i + a_iplusone) / 2
            A[i, (i - 1 + n) % n] = -a_iminushalf / h**2
            A[i, i] = (a_iminushalf + a_iplushalf) / h**2
            A[i, (i + 1) % n] = -a_iplushalf / h**2
        
        return A
    
    def build_rhs(self):
        n = len(self.x)
        b = np.zeros(n)
        h = self.x[1] - self.x[0]
        for i in range(n):
            b[i] = self.f_func(self.x[i])
        if self.boundary == 'Dirichlet':
            b[0] = 0
            b[-1] = 0
        return b
    
    def solve(self):
        u = np.linalg.solve(self.A, self.b)
        return u


class PoissonEquation2D(PDE):
    def __init__(self, a_func, f_func, boundary, x, y, A=None):
        super().__init__(a_func, f_func, boundary,x, y, A)
        self.A = self.build_matrix() if A is None else A
        self.b = self.build_rhs() if not isinstance(self.f_func, np.ndarray) else self.f_func
        self.u = self.solve()

    def index(self, i, j):
        return i * len(self.y) + j

    def build_matrix_dirichlet(self):
        n = len(self.x)
        m = len(self.y)
        A = np.zeros((n * m, n * m))
        h_x = self.x[1] - self.x[0]
        h_y = self.y[1] - self.y[0]
        for i in range(1, n - 1):
            for j in range(1, m - 1):
                idx = self.index(i, j)
                a_ij = self.a_func[idx] if isinstance(self.a_func, np.ndarray) else self.a_func(self.x[i], self.y[j])
                a_iplusone = self.a_func[self.index(i + 1, j)] if isinstance(self.a_func, np.ndarray) else self.a_func(self.x[i + 1], self.y[j])
                a_iminusone = self.a_func[self.index(i - 1, j)] if isinstance(self.a_func, np.ndarray) else self.a_func(self.x[i - 1], self.y[j])
                a_jplusone = self.a_func[self.index(i, j + 1)] if isinstance(self.a_func, np.ndarray) else self.a_func(self.x[i], self.y[j + 1])
                a_jminusone = self.a_func[self.index(i, j - 1)] if isinstance(self.a_func, np.ndarray) else self.a_func(self.x[i], self.y[j - 1])
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
            for j in [0, m- 1]:
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
        A = np.zeros((n * m, n * m))
        h_x = self.x[1] - self.x[0]
        h_y = self.y[1] - self.y[0]
        for i in range(n):
            for j in range(m):
                idx = self.index(i, j)
                a_ij = self.a_func[idx] if isinstance(self.a_func, np.ndarray) else self.a_func(self.x[i], self.y[j])
                a_iplusone = self.a_func[self.index((i + 1) % n, j)] if isinstance(self.a_func, np.ndarray) else self.a_func(self.x[(i + 1) % n], self.y[j])
                a_iminusone = self.a_func[self.index((i - 1 + n) % n, j)] if isinstance(self.a_func, np.ndarray) else self.a_func(self.x[(i - 1 + n) % n], self.y[j])
                a_jplusone = self.a_func[self.index(i, (j + 1) % m)] if isinstance(self.a_func, np.ndarray) else self.a_func(self.x[i], self.y[(j + 1) % m])
                a_jminusone = self.a_func[self.index(i, (j - 1 + m) % m)] if isinstance(self.a_func, np.ndarray) else self.a_func(self.x[i], self.y[(j - 1 + m) % m])
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
        b = np.zeros(n * m)
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
        u = np.linalg.solve(self.A, self.b)
        return u.reshape((len(self.x), len(self.y)))
