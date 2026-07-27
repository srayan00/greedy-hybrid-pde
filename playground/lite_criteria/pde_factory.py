"""Build fast PDE objects for Poisson / ConvDiff / AnisoDiff."""

from __future__ import annotations


def make_pde(equation: str, N: int, *, b_vel: float = 20.0, eps_x: float = 0.01, eps_y: float = 1.0):
    if equation == "AnisoDiff":
        from fast_aniso_pde import FastAnisoPDE

        return FastAnisoPDE(N, eps_x=eps_x, eps_y=eps_y)
    from fast_pde import FastStencilPDE

    return FastStencilPDE(N, equation=equation, b_vec=(b_vel, b_vel))


def ckp_tag(equation: str, N: int, *, eps_x: float = 0.01, eps_y: float = 1.0) -> str:
    if equation == "AnisoDiff":
        return f"fast_deeponet_AnisoDiff_ex{eps_x:g}_ey{eps_y:g}_{N}"
    return f"fast_deeponet_{equation}_{N}"
