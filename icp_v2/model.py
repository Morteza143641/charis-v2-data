from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable, Tuple, Optional
import numpy as np
from scipy.integrate import solve_ivp


class ModelDomainError(RuntimeError):
    """Raised when the mechanistic model leaves its declared physiological domain."""


@dataclass(frozen=True)
class ModelParams:
    # Ursino-Lodi population parameters used in the frozen V2 specification.
    R_o: float = 526.3       # mmHg*s/mL
    R_f: float = 2.38e3      # mmHg*s/mL
    R_pv: float = 1.24       # mmHg*s/mL
    C_an: float = 0.15       # mL/mmHg
    k_E: float = 0.11        # 1/mL
    k_R: float = 4.91e4      # model-consistent units
    q_n: float = 12.5        # mL/s
    tau: float = 20.0        # s
    G: float = 1.5           # frozen published/model parameter
    P_vs: float = 6.0        # mmHg
    min_gap: float = 1e-4

    def to_dict(self):
        return asdict(self)


def c_inf(q: float, p: ModelParams = ModelParams()) -> float:
    x = (q - p.q_n) / p.q_n
    Delta = 0.75 if x <= 0.0 else 0.075
    k_sigma = Delta / 4.0
    z = np.clip(p.G * x / k_sigma, -700.0, 700.0)
    ez = np.exp(z)
    return ((p.C_an + Delta / 2.0) + (p.C_an - Delta / 2.0) * ez) / (1.0 + ez)


def resistance_a(A: float, P: float, C: float, p: ModelParams = ModelParams()) -> float:
    gap = A - P
    if not np.isfinite([A, P, C]).all():
        raise ModelDomainError("Non-finite state/input.")
    if P <= 0.0:
        raise ModelDomainError(f"P must remain positive; got {P}.")
    if C <= 0.0:
        raise ModelDomainError(f"C must remain positive; got {C}.")
    if gap <= p.min_gap:
        raise ModelDomainError(f"A-P approached model singularity: A-P={gap:.6g} mmHg.")
    R_a = p.k_R * p.C_an**2 / (C**2 * gap**2)
    if R_a <= 0.0 or not np.isfinite(R_a):
        raise ModelDomainError(f"Invalid R_a={R_a}.")
    return float(R_a)


def flow_q(A: float, P: float, C: float, p: ModelParams = ModelParams()) -> float:
    R_a = resistance_a(A, P, C, p)
    q = (A - P) / (R_a + p.R_pv)
    if q <= 0.0 or not np.isfinite(q):
        raise ModelDomainError(f"q must remain positive; got {q}.")
    return float(q)


def rhs(t: float, y: np.ndarray, A_fn: Callable[[float], float], dA_fn: Callable[[float], float],
        p: ModelParams = ModelParams(), autoregulation: bool = True,
        infusion_fn: Optional[Callable[[float], float]] = None) -> np.ndarray:
    P, C = float(y[0]), float(y[1])
    A = float(A_fn(t)); dA = float(dA_fn(t))
    q = flow_q(A, P, C, p)
    C_dot = (c_inf(q, p) - C) / p.tau if autoregulation else 0.0
    I_i = 0.0 if infusion_fn is None else float(infusion_fn(t))
    denom = 1.0 + p.k_E * P * C
    if denom <= 0.0 or not np.isfinite(denom):
        raise ModelDomainError(f"Invalid pressure-compliance denominator={denom}.")
    P_dot = p.k_E * P / denom * (C * dA + C_dot * (A - P) + (p.R_pv / p.R_f) * q - (P - p.P_vs) / p.R_o + I_i)
    if not np.isfinite([P_dot, C_dot]).all():
        raise ModelDomainError("Non-finite derivative.")
    return np.array([P_dot, C_dot], dtype=float)


def simulate(t_span: Tuple[float, float], y0: Tuple[float, float], A_fn: Callable[[float], float],
             dA_fn: Callable[[float], float], p: ModelParams = ModelParams(),
             autoregulation: bool = True, method: str = "DOP853", rtol: float = 1e-8,
             atol: float = 1e-10, max_step: float = np.inf, t_eval=None):
    fun = lambda t, y: rhs(t, y, A_fn, dA_fn, p=p, autoregulation=autoregulation)
    sol = solve_ivp(fun, t_span, np.asarray(y0, dtype=float), method=method, rtol=rtol, atol=atol,
                    max_step=max_step, t_eval=t_eval)
    if not sol.success:
        raise RuntimeError(sol.message)
    for ti, Pi, Ci in zip(sol.t, sol.y[0], sol.y[1]):
        flow_q(float(A_fn(float(ti))), float(Pi), float(Ci), p)
    return sol
