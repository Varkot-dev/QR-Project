"""Average price response to a signed event (Bouchaud et al. 2004).

R(l) = E[ sign_t * (m_{t+l} - m_t) ], with m_t the mid strictly BEFORE
event t. For uncorrelated signs, R(l) recovers the impact kernel itself;
for real (long-memory) signs it mixes kernel and flow memory — that
distinction is the point of the analysis comparing both.
"""
from __future__ import annotations

import numpy as np


def response_function(signs: np.ndarray, mids: np.ndarray, max_lag: int) -> np.ndarray:
    if signs.shape != mids.shape:
        raise ValueError(f"signs {signs.shape} and mids {mids.shape} must match")
    n = signs.size
    if max_lag >= n:
        raise ValueError(f"max_lag {max_lag} must be < series length {n}")
    s = signs.astype(np.float64)
    m = mids.astype(np.float64)
    out = np.zeros(max_lag + 1)
    for lag in range(1, max_lag + 1):
        out[lag] = np.mean(s[:-lag] * (m[lag:] - m[:-lag]))
    return out
