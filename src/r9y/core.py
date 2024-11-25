from enum import Enum
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


def r9y_sys_nonrec(_: float, y: List[float], lam: List[float]) -> List[float]:
    l1, l2, l3 = lam

    return [
        -(l1 + l2 + l3) * y[0],
        l1 * y[0] - (l2 + l3) * y[1],
        l2 * y[0] - (l1 + l3) * y[2],
        l3 * y[0],
        l3 * y[1],
        l2 * y[1] + l3 * y[2],
        l1 * y[2],
    ]


def solve_r9y_sys_nonrec(
    t_span: Tuple[float, float],
    y0: List[float],
    lam: List[float],
    t_eval: np.ndarray,
) -> IvpSolution:
    return solve_ivp(
        r9y_sys_nonrec,
        t_span,
        y0=y0,
        args=(lam,),
        t_eval=t_eval,
    )


def r9y_sys_rec(
    _: float,
    y: List[float],
    lam_h: List[float],
    mu_h: List[float],
    lam_s: List[float],
    mu_s: List[float],
) -> List[float]:
    l1h, l2h, l3h = lam_h
    m1h, m2h, m3h = mu_h
    (l1s,) = lam_s
    (m1s,) = mu_s

    return [
        -(l1s + l1h + l2h + l3h) * y[0] + m1h * y[2] + m2h * y[3] + m3h * y[4],
        -(m1s + l2h + l3h) * y[1] + l1s * y[0] + m2h * y[6] + m3h * y[7],
        -(m1h + l2h + l3h) * y[2] + l1h * y[0] + m2h * y[8] + m3h * y[9],
        -(m2h + l1s + l1h + l3h) * y[3] + l2h * y[0] + m1h * y[8] + m3h * y[10],
        -(m3h) * y[4] + l3h * y[0] + m1h * y[9] + m2h * y[10],
        -(l1h + l1s + l2h + l3h) * y[5]
        + m1s * y[1]
        + m1h * y[11]
        + m2h * y[13]
        + m3h * y[14],
        -(m2h + m1s) * y[6] + l2h * y[1] + l1s * y[3],
        -(m3h + m1s) * y[7] + l3h * y[1],
        -(m2h + m1h) * y[8] + l2h * y[2] + l1h * y[3],
        -(m3h + m1h) * y[9] + l3h * y[2],
        -(m3h + m2h) * y[10] + l3h * y[3],
        -(m1h + l2h + l3h) * y[11] + l1h * y[5] + m2h * y[15] + m3h * y[16],
        -(m1s + l2h + l3h) * y[12] + l1s * y[5] + m2h * y[18] + m3h * y[19],
        -(m2h + l1s + l1h + l3h) * y[13]
        + l2h * y[5]
        + m1s * y[6]
        + m1h * y[15]
        + m3h * y[20],
        -(m3h) * y[14] + l3h * y[5] + m1s * y[7] + m2h * y[20] + m1h * y[16],
        -(m2h + m1h) * y[15] + l2h * y[11] + l1h * y[13],
        -(m3h + m1h) * y[16] + l3h * y[11],
        -(l1s + l1h + l2h + l3h) * y[17]
        + m1s * y[12]
        + m1h * y[21]
        + m2h * y[23]
        + m3h * y[24],
        -(m2h + m1s) * y[18] + l2h * y[12] + l1s * y[13],
        -(m3h + m1s) * y[19] + l3h * y[12],
        -(m3h + m2h) * y[20] + l3h * y[13],
        -(m1h + l2h + l3h) * y[21] + l1h * y[17] + m2h * y[25] + m3h * y[26],
        -(l2h + l3h) * y[22] + l1s * y[17] + m2h * y[27] + m3h * y[28],
        -(m2h + l3h + l1s + l1h) * y[23]
        + l2h * y[17]
        + m3h * y[29]
        + m1s * y[18]
        + m1h * y[25],
        -(m3h) * y[24] + l3h * y[17] + m1s * y[19] + m2h * y[29] + m1h * y[26],
        -(m2h + m1h) * y[25] + l2h * y[21] + l1h * y[23],
        -(m3h + m1h) * y[26] + l3h * y[21],
        -(m2h) * y[27] + l2h * y[22] + l1s * y[23],
        -(m3h) * y[28] + l3h * y[22],
        -(m3h + m2h) * y[29] + l3h * y[23],
    ]


def solve_r9y_sys_rec(
    t_span: Tuple[float, float],
    y0: List[float],
    lam_h: List[float],
    mu_h: List[float],
    lam_s: List[float],
    mu_s: List[float],
    t_eval: np.ndarray,
) -> IvpSolution:
    return solve_ivp(
        r9y_sys_rec,
        t_span,
        y0=y0,
        args=(lam_h, mu_h, lam_s, mu_s),
        t_eval=t_eval,
    )


class SystemType(Enum):
    NONRECOVERABLE = "Nonrecoverable"
    RECOVERABLE = "Recoverable"
