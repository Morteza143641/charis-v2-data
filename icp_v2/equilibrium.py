from __future__ import annotations
import numpy as np
from scipy.optimize import root, root_scalar
from .model import ModelParams, c_inf, flow_q, resistance_a, rhs


def _steady_residual_full(z, A, p: ModelParams):
    P, C = float(z[0]), float(z[1])
    q = flow_q(A, P, C, p)
    pressure_balance = (p.R_pv / p.R_f) * q - (P - p.P_vs) / p.R_o
    autoreg_balance = C - c_inf(q, p)
    return np.array([pressure_balance, autoreg_balance])


def solve_equilibrium(A: float = 100.0, p: ModelParams = ModelParams(), guess=None):
    if guess is None:
        guess = np.array([max(p.P_vs + 3.0, 9.5), p.C_an], dtype=float)
    sol = root(lambda z: _steady_residual_full(z, A, p), guess, options={"xtol": 1e-12})
    if not sol.success:
        sol = root(lambda z: _steady_residual_full(z, A, p),
                   np.array([min(0.45*A, 45.0), max(p.C_an, 0.25)]), options={"xtol": 1e-12})
    if not sol.success:
        raise RuntimeError(f"Equilibrium solve failed: {sol.message}")
    P, C = map(float, sol.x)
    q = flow_q(A, P, C, p); R_a = resistance_a(A, P, C, p)
    const = lambda t: float(A); zero = lambda t: 0.0
    d = rhs(0.0, np.array([P, C]), const, zero, p=p, autoregulation=True)
    return {"P": P, "C": C, "q": q, "R_a": R_a, "P_dot": float(d[0]), "C_dot": float(d[1]),
            "residual": _steady_residual_full([P, C], A, p)}


def solve_equilibrium_fixed_c(A: float = 100.0, p: ModelParams = ModelParams(), C: float | None = None):
    C = p.C_an if C is None else float(C)
    def f(P):
        q = flow_q(A, float(P), C, p)
        return (p.R_pv / p.R_f) * q - (float(P) - p.P_vs) / p.R_o
    lo = p.P_vs + 1e-8; hi = A - max(p.min_gap * 10.0, 1e-3)
    sol = root_scalar(f, bracket=(lo, hi), xtol=1e-12)
    P = float(sol.root); q = flow_q(A, P, C, p)
    return {"P": P, "C": C, "q": q, "R_a": resistance_a(A, P, C, p)}
