"""Adversarial reproduction of Q4 headline regressions from the results parquet.

Own OLS via closed-form normal equations (not np.polyfit, no project code).
"""
import numpy as np
import polars as pl

df = pl.read_parquet("results/q4_cross_section.parquet")
x = np.log10(df["n_events"].to_numpy().astype(np.float64))


def ols(x: np.ndarray, y: np.ndarray) -> dict:
    xm, ym = x.mean(), y.mean()
    dx, dy = x - xm, y - ym
    slope = (dx @ dy) / (dx @ dx)
    intercept = ym - slope * xm
    resid = y - (slope * x + intercept)
    ss_res = resid @ resid
    ss_tot = dy @ dy
    return {"slope": float(slope), "intercept": float(intercept),
            "r2": float(1.0 - ss_res / ss_tot), "n": int(x.size)}

for name, col in [("gamma", "gamma"), ("p_flip", "p_flip")]:
    r = ols(x, df[col].to_numpy().astype(np.float64))
    print(name, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
