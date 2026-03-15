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
    
    def batch_solve(self, tol=1e-10, max_iter=1000, u_init=None):
        """
        tol: tolerance for convergence, default 1e-10
        max_iter: maximum number of iterations, default 1000
        u_init: initial guess for the solution, default None (zero vector)
        """
        u_old = u_init if u_init is not None else torch.zeros_like(self.equation.b, device=self.device)
        mask = torch.zeros(self.equation.b.shape[0], dtype=torch.bool, device=self.device)
        u_new = torch.zeros_like(u_old, device=self.device)
        for it in range(max_iter):
            u_new = self.iteration(u_old, ~mask)
            mask = torch.linalg.norm(u_new - u_old, float('inf'), dim=1) < tol
            if it % 100 == 0:
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
            if it % 100 == 0:
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
        """
        u_old: (B, N) or (B, N^2) tensor containing the current solution estimates for a batch of samples
        mask: (B,) boolean tensor indicating which samples have not yet converged. If None, it is assumed that all samples are not converged.
        """
        if mask is None:
            mask = torch.ones(self.equation.b.shape[0], device=self.device, dtype=torch.bool)
        d_inv = 1.0 / torch.diagonal(self.equation.A, dim1=-2, dim2=-1)
        is_batch = self.equation.A.ndim == 3
        if is_batch:
            u_new = u_old.clone()
            output = torch.bmm(self.equation.A[mask], u_old[mask].unsqueeze(-1)).squeeze(-1)
            u_new[mask] = u_old[mask] + self.weight * d_inv[mask] * (self.equation.b[mask] - output)
        else:
            output = self.equation.A @ u_old
            u_new = u_old + self.weight * d_inv * (self.equation.b - output)
        return u_new
    
    
class GaussSeidelSolver(NumericalSolver):
    def __init__(self, equation: PDE, device = torch.device("cpu")):
        super().__init__(equation, device)
        self._cached_L_inv = None
        self._cached_A_id = None

    def _get_L_inv(self, A):
        """Compute and cache L_inv; reuse when A hasn't changed."""
        a_id = A.data_ptr()
        if self._cached_L_inv is not None and self._cached_A_id == a_id:
            return self._cached_L_inv
        is_batch = A.ndim == 3
        A_ref = A[0] if is_batch else A
        L = torch.tril(A_ref)
        self._cached_L_inv = torch.linalg.inv(L)
        self._cached_A_id = a_id
        return self._cached_L_inv

    def iteration(self, u_old, mask = None):
        """
        u_old: (B, N) or (B, N^2) tensor containing the current solution estimates for a batch of samples
        mask: (B,) boolean tensor indicating which samples have not yet converged. If None, it is assumed that all samples are not converged.
        """
        if mask is None:
            mask = torch.ones(self.equation.b.shape[0], device=self.device, dtype=torch.bool)
        A = self.equation.A
        L_inv = self._get_L_inv(A)
        is_batch = A.ndim == 3
        if is_batch:
            u_new = u_old.clone()
            output = torch.bmm(A[mask], u_old[mask].unsqueeze(-1)).squeeze(-1)
            residual = self.equation.b[mask] - output
            u_new[mask] = u_old[mask] + (residual @ L_inv.T)
        else:
            output = A @ u_old
            u_new = u_old + L_inv @ (self.equation.b - output)
        return u_new


class SORSolver(NumericalSolver):
    """Successive Over-Relaxation: GS with relaxation parameter omega.

    omega=1 recovers Gauss-Seidel; omega>1 is over-relaxation; omega<1 is
    under-relaxation.  The iteration is  u_new = (D + omega*L)^{-1}
    [omega*b + ((1-omega)*D - omega*U) * u_old]  which simplifies to
    u_new = u_old + omega * (D+L)^{-1} * (b - A*u_old)  when rewritten as
    a relaxed GS step.
    """

    def __init__(self, equation: PDE, omega=1.5, device=torch.device("cpu")):
        super().__init__(equation, device)
        self.omega = omega
        self._cached_DL_inv = None
        self._cached_A_id = None

    def _get_DL_inv(self, A):
        a_id = A.data_ptr()
        if self._cached_DL_inv is not None and self._cached_A_id == a_id:
            return self._cached_DL_inv
        A_ref = A[0] if A.ndim == 3 else A
        DL = torch.tril(A_ref)
        self._cached_DL_inv = torch.linalg.inv(DL)
        self._cached_A_id = a_id
        return self._cached_DL_inv

    def iteration(self, u_old, mask=None):
        if mask is None:
            mask = torch.ones(self.equation.b.shape[0], device=self.device, dtype=torch.bool)
        A = self.equation.A
        DL_inv = self._get_DL_inv(A)
        is_batch = A.ndim == 3
        if is_batch:
            u_new = u_old.clone()
            output = torch.bmm(A[mask], u_old[mask].unsqueeze(-1)).squeeze(-1)
            residual = self.equation.b[mask] - output
            gs_step = residual @ DL_inv.T
            u_new[mask] = u_old[mask] + self.omega * gs_step
        else:
            output = A @ u_old
            gs_step = DL_inv @ (self.equation.b - output)
            u_new = u_old + self.omega * gs_step
        return u_new


class MultigridSolver(NumericalSolver):
    """Geometric multigrid V-cycle for 2D periodic grids with constant coefficient.

    Handles arbitrary grid sizes (not limited to powers of 2) by using
    interpolation-based restriction/prolongation between non-matching grids.
    Coarsening uses Nc = (N + 1) // 2 at each level.

    Assumes all batch samples share the same operator matrix A (constant
    coefficient).  Hierarchy is built once from A[0] and reused.
    """

    def __init__(self, equation: PDE, levels=3, device=torch.device("cpu"),
                 pre_smooth=2, post_smooth=2, omega=2.0 / 3.0):
        super().__init__(equation, device)
        self.levels = levels
        self.pre_smooth = pre_smooth
        self.post_smooth = post_smooth
        self.omega = omega

        A = equation.A
        A_shared = A[0] if A.ndim == 3 else A
        N_sq = A_shared.shape[0]
        N = int(round(N_sq ** 0.5))
        assert N * N == N_sq, f"Matrix size {N_sq} is not a perfect square"

        self.hierarchy = []
        A_cur = A_shared
        N_cur = N

        for _ in range(levels):
            Nc = (N_cur + 1) // 2
            if Nc < 3:
                break
            d_inv = 1.0 / torch.diagonal(A_cur)
            R = self._build_restriction_2d(N_cur, Nc, device)
            P = self._build_prolongation_2d(N_cur, Nc, device)
            A_coarse = R @ A_cur @ P
            self.hierarchy.append((A_cur, d_inv, R, P))
            A_cur = A_coarse
            N_cur = Nc

        self.n_levels = len(self.hierarchy)
        self._A_coarsest_pinv = torch.linalg.pinv(A_cur)

    # ------------------------------------------------------------------
    #  1-D operators via interpolation (handles any N, periodic)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_prolongation_1d(N_fine, N_coarse, device):
        """Linear interpolation from Nc-point coarse grid to N-point fine grid (periodic)."""
        h_c = 1.0 / N_coarse
        h_f = 1.0 / N_fine
        P = torch.zeros(N_fine, N_coarse, device=device)
        for i in range(N_fine):
            xf = i * h_f
            idx_left = int(xf / h_c)
            t = (xf - idx_left * h_c) / h_c
            P[i, idx_left % N_coarse] += 1.0 - t
            P[i, (idx_left + 1) % N_coarse] += t
        return P

    @staticmethod
    def _build_restriction_1d(N_fine, N_coarse, device):
        """Adjoint-of-prolongation restriction (transpose, row-normalised)."""
        P = MultigridSolver._build_prolongation_1d(N_fine, N_coarse, device)
        R = P.T.clone()
        row_sums = R.sum(dim=1, keepdim=True).clamp(min=1e-12)
        R = R / row_sums
        return R

    # ------------------------------------------------------------------
    #  2-D operators via Kronecker product of 1-D operators
    # ------------------------------------------------------------------
    @staticmethod
    def _build_restriction_2d(N_fine, N_coarse, device):
        R1 = MultigridSolver._build_restriction_1d(N_fine, N_coarse, device)
        return torch.kron(R1, R1)

    @staticmethod
    def _build_prolongation_2d(N_fine, N_coarse, device):
        P1 = MultigridSolver._build_prolongation_1d(N_fine, N_coarse, device)
        return torch.kron(P1, P1)

    # ------------------------------------------------------------------
    #  V-cycle
    # ------------------------------------------------------------------
    def _smooth(self, u, b, A, d_inv, n_iter):
        for _ in range(n_iter):
            r = b - u @ A.T
            u = u + self.omega * d_inv.unsqueeze(0) * r
        return u

    def _v_cycle(self, u, b, level):
        if level == self.n_levels:
            return b @ self._A_coarsest_pinv.T

        A, d_inv, R, P = self.hierarchy[level]
        u = self._smooth(u, b, A, d_inv, self.pre_smooth)
        r = b - u @ A.T
        r_c = r @ R.T
        e_c = self._v_cycle(torch.zeros_like(r_c), r_c, level + 1)
        u = u + e_c @ P.T
        u = self._smooth(u, b, A, d_inv, self.post_smooth)
        return u

    def iteration(self, u_old, mask=None):
        if mask is None:
            mask = torch.ones(self.equation.b.shape[0], device=self.device,
                              dtype=torch.bool)
        u_new = u_old.clone()
        u_new[mask] = self._v_cycle(u_old[mask], self.equation.b[mask], level=0)
        return u_new