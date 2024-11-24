from typing import Callable, List, Optional, Tuple, TypedDict

import numpy as np
from scipy.integrate import solve_ivp


class IvpSolution(TypedDict):
    t: np.ndarray
    y: np.ndarray
    sol: Optional[Callable[[float], np.ndarray]]
    t_events: Optional[List[np.ndarray]]
    y_events: Optional[List[np.ndarray]]
    nfev: int
    njev: int
    nlu: int
    status: int
    message: str
    success: bool


def r9y_sys_nonrec(
    _: float, y: List[float], lam: Tuple[float, float, float]
) -> List[float]:
    p1, p2, p3 = y[:3]
    lam1, lam2, lam3 = lam

    return [
        -(lam1 + lam2 + lam3) * p1,
        lam1 * p1 - (lam2 + lam3) * p2,
        lam2 * p1 - (lam1 + lam3) * p3,
        lam3 * p1,
        lam3 * p2,
        lam2 * p2 + lam3 * p3,
        lam1 * p3,
    ]


def solve_r9y_sys_nonrec(
    t_span: Tuple[float, float],
    y0: List[float],
    lam: Tuple[float, float, float],
    t_eval: np.ndarray,
) -> IvpSolution:
    return solve_ivp(
        r9y_sys_nonrec,
        t_span,
        y0=y0,
        args=(lam,),
        t_eval=t_eval,
    )
