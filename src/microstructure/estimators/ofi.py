"""Order-flow imbalance (Cont, Kukanov & Stoikov 2014) and its regression.

OFI counts liquidity-consuming and liquidity-adding events at the best
quotes: bid improvements/size-adds are buying pressure (+), ask
improvements/size-adds are selling pressure (-). Price change over a
window is theorized (and empirically found) linear in the window's
summed OFI with slope ~ 1/depth.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def ofi_events(
    bid_p: np.ndarray, bid_q: np.ndarray, ask_p: np.ndarray, ask_q: np.ndarray
) -> np.ndarray:
    if not (bid_p.shape == bid_q.shape == ask_p.shape == ask_q.shape):
        raise ValueError("all four L1 arrays must have identical shape")
    e = np.zeros(bid_p.size - 1)
    b_now, b_prev = bid_p[1:], bid_p[:-1]
    a_now, a_prev = ask_p[1:], ask_p[:-1]
    e += np.where(b_now >= b_prev, bid_q[1:], 0.0)
    e -= np.where(b_now <= b_prev, bid_q[:-1], 0.0)
    e -= np.where(a_now <= a_prev, ask_q[1:], 0.0)
    e += np.where(a_now >= a_prev, ask_q[:-1], 0.0)
    return e


@dataclass(frozen=True)
class OLSFit:
    slope: float
    stderr: float
    r2: float


def ols_through_origin(x: np.ndarray, y: np.ndarray) -> OLSFit:
    sxx = float(x @ x)
    if sxx == 0.0:
        raise ValueError("x has zero variance")
    slope = float(x @ y) / sxx
    resid = y - slope * x
    dof = max(x.size - 1, 1)
    stderr = float(np.sqrt((resid @ resid) / dof / sxx))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else 0.0
    return OLSFit(slope=slope, stderr=stderr, r2=r2)
