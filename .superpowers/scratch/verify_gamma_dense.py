"""Dense-lag direct dot-product ACF (lags 10..500) + polyfit, per month.

Fully independent of the repo's FFT sign_acf; replicates the claimed fit
window so fit-weighting is held constant. Also reports the 6-point fit.
"""
import json
from pathlib import Path

import numpy as np

from microstructure.signals.load import load_events

ROOT = Path("/Users/varshithkotagiri/Projects/QR project/data")
SPOT_LAGS = [10, 20, 50, 100, 200, 500]
CLAIMED = {"2023-06": 0.2858, "2023-07": 0.2055}

results = {}
for period in ["2023-06", "2023-07"]:
    df = load_events(ROOT, "ETHUSDT", [period])
    s = df["sign"].to_numpy().astype(np.float64)
    n = len(s)
    del df
    mu = s.mean()
    x = s - mu
    del s
    var = np.dot(x, x) / n
    lags = np.arange(10, 501)
    acf = np.empty(lags.size)
    for i, lag in enumerate(lags):
        acf[i] = (np.dot(x[:-lag], x[lag:]) / (n - lag)) / var
    del x
    lx, ly = np.log(lags.astype(float)), np.log(acf)
    slope, _ = np.polyfit(lx, ly, 1)
    gamma_dense = -slope
    # 6-point log-spaced fit
    idx = [np.where(lags == l)[0][0] for l in SPOT_LAGS]
    s6, _ = np.polyfit(lx[idx], ly[idx], 1)
    results[period] = {
        "n_events": int(n),
        "acf_spot": {str(l): float(acf[i]) for l, i in zip(SPOT_LAGS, idx)},
        "gamma_dense_10_500": float(gamma_dense),
        "gamma_6pt": float(-s6),
        "claimed": CLAIMED[period],
        "diff_dense": float(abs(gamma_dense - CLAIMED[period])),
        "dense_within_0.03": bool(abs(gamma_dense - CLAIMED[period]) <= 0.03),
    }

print(json.dumps(results, indent=2))
Path("/Users/varshithkotagiri/Projects/QR project/.superpowers/scratch/verify_gamma_dense.json").write_text(
    json.dumps(results, indent=2)
)
