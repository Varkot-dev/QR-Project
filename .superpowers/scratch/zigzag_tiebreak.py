"""Test whether BTCUSDT short-lag sign ACF zigzag is a tie-break artifact.

A: baseline (deterministic sort ts,sign: sells before buys within same ms)
B: randomized tie-break within same-ts groups (rng(0))
C: same-ts groups netted to one event, sign = sign(sum sign*qty)
"""
from pathlib import Path

import numpy as np
import polars as pl

from microstructure.estimators.acf import sign_acf
from microstructure.signals.load import load_events

ROOT = Path("/Users/varshithkotagiri/Projects/QR project/data")
MAX_LAG = 30
EVEN = [2, 4, 6, 8, 10]
ODD = [1, 3, 5, 7, 9]


def zigzag(acf: np.ndarray) -> float:
    return float(acf[EVEN].mean() - acf[ODD].mean())


def report(label: str, acf: np.ndarray) -> None:
    print(f"--- {label} ---")
    print("acf[1..10] =", np.array2string(acf[1:11], precision=6, separator=", "))
    print(f"zigzag amplitude = {zigzag(acf):.6f}")


events = load_events(ROOT, "BTCUSDT", ["2023-06"])
n_events = events.height
ts = events["ts"].cast(pl.Int64).to_numpy()
signs = events["sign"].to_numpy().astype(np.int8)
qty = events["qty"].to_numpy()
del events

print(f"n_events = {n_events}")

# --- same-ts diagnostics ---
same_ts = ts[1:] == ts[:-1]
frac_same_ts = float(same_ts.mean())
opp = signs[1:][same_ts] != signs[:-1][same_ts]
frac_opp_among_same = float(opp.mean())
print(f"fraction of consecutive pairs sharing same ts = {frac_same_ts:.6f}")
print(f"among same-ts adjacent pairs, fraction opposite-signed = {frac_opp_among_same:.6f}")

# group structure: after (ts, side) aggregation each ts has at most 2 rows
starts = np.flatnonzero(np.concatenate(([True], ts[1:] != ts[:-1])))
sizes = np.diff(np.concatenate((starts, [n_events])))
print(f"max same-ts group size = {sizes.max()} (should be 2)")
n_pairs = int((sizes == 2).sum())
print(f"n same-ts pairs (size-2 groups) = {n_pairs}")

# --- A: baseline ---
acf_a = sign_acf(signs, MAX_LAG)
report("A baseline (deterministic tie-break)", acf_a)

# --- B: randomized tie-break ---
rng = np.random.default_rng(0)
signs_b = signs.copy()
pair_starts = starts[sizes == 2]
flip = rng.random(n_pairs) < 0.5
swap_idx = pair_starts[flip]
tmp = signs_b[swap_idx].copy()
signs_b[swap_idx] = signs_b[swap_idx + 1]
signs_b[swap_idx + 1] = tmp
print(f"pairs swapped = {int(flip.sum())} / {n_pairs}")
acf_b = sign_acf(signs_b, MAX_LAG)
report("B randomized tie-break", acf_b)
del signs_b

# --- C: netted same-ts groups ---
group_id = np.cumsum(np.concatenate(([0], (ts[1:] != ts[:-1]).astype(np.int64))))
net = np.zeros(len(starts))
np.add.at(net, group_id, signs.astype(np.float64) * qty)
signs_c = np.sign(net)
nz = signs_c != 0
n_zero_net = int((~nz).sum())
signs_c = signs_c[nz].astype(np.int8)
print(f"netted series: n = {signs_c.size}, zero-net groups dropped = {n_zero_net}")
acf_c = sign_acf(signs_c, MAX_LAG)
report("C netted same-ts groups", acf_c)
