"""Adversarial re-test: lag-1/lag-2 sign autocorrelation restricted to
cross-millisecond pairs only. Naive numpy on ts/sign arrays; does NOT
use sign_acf and does NOT reuse the shuffle/netting approach.

BTCUSDT 2023-06.
"""
from pathlib import Path

import numpy as np

from microstructure.signals.load import load_events

ROOT = Path("/Users/varshithkotagiri/Projects/QR project/data")


def pair_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Plain Pearson correlation of paired samples (naive)."""
    mx, my = x.mean(), y.mean()
    sx, sy = x.std(), y.std()
    return float(((x - mx) * (y - my)).mean() / (sx * sy))


def main() -> None:
    ev = load_events(ROOT, "BTCUSDT", ["2023-06"])
    ts = ev["ts"].to_numpy().astype("datetime64[ms]").astype(np.int64)
    sign = ev["sign"].to_numpy().astype(np.float64)
    del ev
    n = len(sign)
    print(f"n_events={n}")

    out = {}
    for lag in (1, 2):
        x = sign[:-lag]
        y = sign[lag:]
        # all pairs (reference, should match sign_acf approximately)
        r_all = pair_corr(x, y)
        # cross-ms only: the two events of the pair are in different milliseconds
        mask = ts[lag:] != ts[:-lag]
        n_pairs = len(x)
        n_cross = int(mask.sum())
        r_cross = pair_corr(x[mask], y[mask])
        # same-ms only (for lag-1 mainly)
        n_same = n_pairs - n_cross
        r_same = pair_corr(x[~mask], y[~mask]) if n_same > 1 else float("nan")
        out[lag] = dict(
            r_all=r_all,
            r_cross_ms=r_cross,
            r_same_ms=r_same,
            n_pairs=n_pairs,
            n_cross_ms_pairs=n_cross,
            n_same_ms_pairs=n_same,
        )
        print(f"lag={lag}: all={r_all:.6f} cross_ms={r_cross:.6f} "
              f"same_ms={r_same:.6f} n_cross={n_cross} n_same={n_same}")

    # stricter variant for lag-1: both events of the pair are in
    # milliseconds that contain only ONE event each (no tie-break involved
    # anywhere near the pair)
    same_prev = np.concatenate(([False], ts[1:] == ts[:-1]))  # shares ms with prior
    same_next = np.concatenate((ts[:-1] == ts[1:], [False]))  # shares ms with next
    solo = ~(same_prev | same_next)  # event is alone in its ms
    m1 = solo[:-1] & solo[1:]
    r1_solo = pair_corr(sign[:-1][m1], sign[1:][m1])
    m2 = solo[:-2] & solo[2:]
    r2_solo = pair_corr(sign[:-2][m2], sign[2:][m2])
    print(f"solo-ms-only pairs: lag1={r1_solo:.6f} (n={int(m1.sum())}) "
          f"lag2={r2_solo:.6f} (n={int(m2.sum())})")

    import json
    print(json.dumps({"per_lag": out,
                      "solo": {"lag1": r1_solo, "n1": int(m1.sum()),
                               "lag2": r2_solo, "n2": int(m2.sum())}}))


if __name__ == "__main__":
    main()
