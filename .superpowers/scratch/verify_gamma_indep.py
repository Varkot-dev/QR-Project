"""Independent spot-check of ETHUSDT monthly gammas.

Direct numpy dot-product ACF at selected lags + hand polyfit log-log slope.
Does NOT use microstructure.estimators.acf.
"""
import json
from pathlib import Path

import numpy as np

from microstructure.signals.load import load_events

ROOT = Path("/Users/varshithkotagiri/Projects/QR project/data")
LAGS = [10, 20, 50, 100, 200, 500]
CLAIMED = {"2023-06": 0.2858, "2023-07": 0.2055}

results = {}
for period in ["2023-06", "2023-07"]:
    df = load_events(ROOT, "ETHUSDT", [period])
    s = df["sign"].to_numpy().astype(np.float64)
    n = len(s)
    del df
    mu = s.mean()
    var = s.var()  # population variance
    acf_vals = {}
    for lag in LAGS:
        # direct dot product, unbiased-in-count cross moment
        c = np.dot(s[:-lag] - mu, s[lag:] - mu) / (n - lag)
        acf_vals[lag] = c / var
    xs = np.log(np.array(LAGS, dtype=float))
    ys = np.log(np.array([acf_vals[l] for l in LAGS]))
    slope, intercept = np.polyfit(xs, ys, 1)
    gamma = -slope
    results[period] = {
        "n_events": int(n),
        "mean_sign": float(mu),
        "acf": {str(l): float(acf_vals[l]) for l in LAGS},
        "gamma_indep": float(gamma),
        "claimed": CLAIMED[period],
        "abs_diff": float(abs(gamma - CLAIMED[period])),
        "within_0.03": bool(abs(gamma - CLAIMED[period]) <= 0.03),
    }
    del s

print(json.dumps(results, indent=2))
out = Path("/Users/varshithkotagiri/Projects/QR project/.superpowers/scratch/verify_gamma_indep.json")
out.write_text(json.dumps(results, indent=2))
