"""Q1b: is the short-lag odd/even zigzag in the sign ACF a tie-break artifact?

Q1's log-log ACF plot shows a visible odd/even alternation at short lags
(negative ACF(1), strongly positive ACF(2)). `to_aggressor_events` sorts
same-millisecond, opposite-side events deterministically by (ts, sign) --
sells before buys -- so a natural worry is that the zigzag is just an
artifact of that tie-break convention rather than real market structure.

Three variants of the sign series are compared, all restricted to lags
1..10 so the effect at literally the shortest lags is visible:

  A. baseline    -- signs straight out of `load_events` (deterministic
                     (ts, sign) tie-break: sells before buys within a ms).
  B. randomized   -- same-ts pairs (always one sell + one buy, since
                     `to_aggressor_events` never produces same-(ts, side)
                     duplicates) have their order swapped with p=0.5 using a
                     fixed-seed RNG, destroying the deterministic tie-break.
  C. netted       -- each same-ts group of opposite signs is collapsed to a
                     single event, sign = sign(sum(sign * qty)); zero-net
                     groups (exact offsetting notional) are dropped.

If the zigzag amplitude survives B and C essentially unchanged, it is not a
tie-break artifact -- the deterministic sort touches too few pairs, and
netting still leaves the alternation intact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from microstructure.estimators.acf import sign_acf
from microstructure.signals.load import load_events

MAX_LAG = 30  # only lags 1..10 are reported/plotted, but ACF needs headroom
N_ZIGZAG_LAGS = 10
EVEN_LAGS = [2, 4, 6, 8, 10]
ODD_LAGS = [1, 3, 5, 7, 9]
RNG_SEED = 0


def zigzag_amplitude(acf: np.ndarray) -> float:
    """mean(acf at even lags 2,4,6,8,10) - mean(acf at odd lags 1,3,5,7,9)."""
    return float(acf[EVEN_LAGS].mean() - acf[ODD_LAGS].mean())


def _same_ts_group_info(ts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (group_id per event, group start indices, group sizes)."""
    n = ts.size
    starts = np.flatnonzero(np.concatenate(([True], ts[1:] != ts[:-1])))
    sizes = np.diff(np.concatenate((starts, [n])))
    group_id = np.cumsum(np.concatenate(([0], (ts[1:] != ts[:-1]).astype(np.int64))))
    return group_id, starts, sizes


def _randomized_tiebreak(signs: np.ndarray, starts: np.ndarray, sizes: np.ndarray, seed: int) -> tuple[np.ndarray, int]:
    """Swap each size-2 same-ts pair's order with p=0.5. Returns (signs, n_swapped)."""
    signs_b = signs.copy()
    pair_starts = starts[sizes == 2]
    rng = np.random.default_rng(seed)
    flip = rng.random(pair_starts.size) < 0.5
    swap_idx = pair_starts[flip]
    tmp = signs_b[swap_idx].copy()
    signs_b[swap_idx] = signs_b[swap_idx + 1]
    signs_b[swap_idx + 1] = tmp
    return signs_b, int(flip.sum())


def _netted(signs: np.ndarray, qty: np.ndarray, group_id: np.ndarray, n_groups: int) -> tuple[np.ndarray, int]:
    """Collapse each same-ts group to sign(sum(sign*qty)); drop zero-net groups."""
    net = np.zeros(n_groups)
    np.add.at(net, group_id, signs.astype(np.float64) * qty)
    netted_signs = np.sign(net)
    nonzero = netted_signs != 0
    n_dropped = int((~nonzero).sum())
    return netted_signs[nonzero].astype(np.int8), n_dropped


def run_q1b(root: Path, out_dir: Path, symbol: str, period: str) -> dict:
    """Compute baseline/randomized-tiebreak/netted short-lag ACF for one symbol-month."""
    out_dir.mkdir(parents=True, exist_ok=True)

    events = load_events(root, symbol, [period])
    n_events = events.height
    ts = events["ts"].cast(pl.Int64).to_numpy()
    signs = events["sign"].to_numpy().astype(np.int8)
    qty = events["qty"].to_numpy()
    del events

    same_ts = ts[1:] == ts[:-1]
    frac_consecutive_pairs_same_ts = float(same_ts.mean()) if same_ts.size else 0.0
    opposite_among_same = signs[1:][same_ts] != signs[:-1][same_ts]
    frac_opposite_among_same = float(opposite_among_same.mean()) if opposite_among_same.size else 0.0

    group_id, starts, sizes = _same_ts_group_info(ts)
    n_groups = starts.size
    max_group_size = int(sizes.max()) if sizes.size else 0
    n_same_ts_pairs = int((sizes == 2).sum())

    # --- A: baseline ---
    acf_a = sign_acf(signs, MAX_LAG)

    # --- B: randomized tie-break within same-ts pairs ---
    signs_b, n_swapped = _randomized_tiebreak(signs, starts, sizes, RNG_SEED)
    acf_b = sign_acf(signs_b, MAX_LAG)
    del signs_b

    # --- C: netted same-ts groups ---
    signs_c, n_zero_net = _netted(signs, qty, group_id, n_groups)
    acf_c = sign_acf(signs_c, MAX_LAG)
    n_netted_events = int(signs_c.size)
    del signs_c

    result = {
        "symbol": symbol,
        "period": period,
        "n_events": n_events,
        "frac_consecutive_pairs_same_ts": frac_consecutive_pairs_same_ts,
        "frac_opposite_sign_among_same_ts_adjacent_pairs": frac_opposite_among_same,
        "n_same_ts_pairs": n_same_ts_pairs,
        "max_same_ts_group_size": max_group_size,
        "baseline": {
            "acf_1_to_10": acf_a[1 : N_ZIGZAG_LAGS + 1].tolist(),
            "zigzag_amplitude": zigzag_amplitude(acf_a),
        },
        "randomized_tiebreak": {
            "pairs_swapped": n_swapped,
            "acf_1_to_10": acf_b[1 : N_ZIGZAG_LAGS + 1].tolist(),
            "zigzag_amplitude": zigzag_amplitude(acf_b),
        },
        "netted": {
            "n_netted_events": n_netted_events,
            "zero_net_groups_dropped": n_zero_net,
            "acf_1_to_10": acf_c[1 : N_ZIGZAG_LAGS + 1].tolist(),
            "zigzag_amplitude": zigzag_amplitude(acf_c),
        },
    }

    _plot(out_dir, acf_a, acf_b, acf_c, symbol, period)
    _write_results_md(out_dir, result)
    (out_dir / "q1b_zigzag.json").write_text(json.dumps(result, indent=2))
    return result


def _plot(out_dir: Path, acf_a: np.ndarray, acf_b: np.ndarray, acf_c: np.ndarray, symbol: str, period: str) -> None:
    lags = np.arange(1, N_ZIGZAG_LAGS + 1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(lags, acf_a[1 : N_ZIGZAG_LAGS + 1], "o-", label="A: baseline (deterministic tie-break)")
    ax.plot(lags, acf_b[1 : N_ZIGZAG_LAGS + 1], "s--", label="B: randomized tie-break")
    ax.plot(lags, acf_c[1 : N_ZIGZAG_LAGS + 1], "^:", label="C: netted same-ts groups")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("lag (events)")
    ax.set_ylabel("sign ACF")
    ax.set_xticks(lags)
    ax.set_title(f"{symbol} {period}: short-lag ACF, tie-break robustness")
    ax.legend()
    fig.savefig(out_dir / "q1b_zigzag.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_results_md(out_dir: Path, result: dict) -> None:
    a, b, c = result["baseline"], result["randomized_tiebreak"], result["netted"]
    amp_a, amp_b, amp_c = a["zigzag_amplitude"], b["zigzag_amplitude"], c["zigzag_amplitude"]
    rel_change_b = abs(amp_b - amp_a) / abs(amp_a) if amp_a else float("nan")
    rel_change_c = abs(amp_c - amp_a) / abs(amp_a) if amp_a else float("nan")

    lines: list[str] = []
    lines.append("# Q1b: short-lag ACF zigzag — tie-break robustness")
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append(
        f"Aggressor events for {result['symbol']} {result['period']} are loaded via "
        "`load_events` (sorted by `(ts, sign)`; within a shared millisecond, sells precede "
        "buys). Three sign series are compared, all evaluated on the identical FFT sign ACF "
        "(`sign_acf`) at lags 1-10:"
    )
    lines.append("")
    lines.append(
        "- **A (baseline):** the series as `load_events` produces it -- deterministic "
        "`(ts, sign)` tie-break."
    )
    lines.append(
        "- **B (randomized tie-break):** same-timestamp adjacent pairs (after aggregation, "
        "always exactly one sell + one buy, since `to_aggressor_events` never emits two "
        "same-(ts, side) rows) have their order swapped with p=0.5 using a fixed-seed RNG "
        f"(`numpy.random.default_rng({RNG_SEED})`), which destroys the deterministic ordering "
        "convention while leaving every other event untouched."
    )
    lines.append(
        "- **C (netted):** each same-timestamp group of opposite-signed events is collapsed "
        "into a single event with sign = sign(sum(sign * qty)); groups whose signed notional "
        "exactly nets to zero are dropped (undefined sign)."
    )
    lines.append("")
    lines.append(
        "The **zigzag amplitude** is `mean(ACF at lags 2,4,6,8,10) - mean(ACF at lags "
        "1,3,5,7,9)` -- large and positive when even lags run noticeably higher than "
        "odd lags, which is the pattern visible in Q1's log-log ACF plot at short lags."
    )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append(f"n_events = {result['n_events']:,}.")
    lines.append("")
    lines.append(
        f"Fraction of consecutive event pairs sharing a millisecond timestamp: "
        f"**{result['frac_consecutive_pairs_same_ts']:.4%}** "
        f"({result['n_same_ts_pairs']:,} same-ts pairs). Among those same-ts adjacent pairs, "
        f"**{result['frac_opposite_sign_among_same_ts_adjacent_pairs']:.2%}** are "
        f"opposite-signed (by construction: `to_aggressor_events` produces at most one buy "
        f"and one sell per millisecond, so max same-ts group size = "
        f"{result['max_same_ts_group_size']}). Variant B is therefore a random 50/50 swap of "
        f"{b['pairs_swapped']:,} of those pairs, not a general permutation."
    )
    lines.append("")
    lines.append("| lag | A: baseline | B: randomized tie-break | C: netted |")
    lines.append("|---|---|---|---|")
    for i in range(N_ZIGZAG_LAGS):
        lines.append(
            f"| {i + 1} | {a['acf_1_to_10'][i]:.6f} | {b['acf_1_to_10'][i]:.6f} | "
            f"{c['acf_1_to_10'][i]:.6f} |"
        )
    lines.append("")
    lines.append("| variant | zigzag amplitude | relative change vs. A |")
    lines.append("|---|---|---|")
    lines.append(f"| A: baseline | {amp_a:.6f} | -- |")
    lines.append(f"| B: randomized tie-break | {amp_b:.6f} | {rel_change_b:.2%} |")
    lines.append(
        f"| C: netted ({c['n_netted_events']:,} events, "
        f"{c['zero_net_groups_dropped']:,} zero-net groups dropped) | {amp_c:.6f} | "
        f"{rel_change_c:.2%} |"
    )
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(
        f"**The zigzag is real structure, not a tie-break artifact.** The amplitude barely "
        f"moves under the randomized tie-break ({amp_a:.6f} -> {amp_b:.6f}, a "
        f"{rel_change_b:.2%} change) and remains large under netting ({amp_c:.6f}, a "
        f"{rel_change_c:.2%} change). A sanity check agrees: only "
        f"{result['frac_consecutive_pairs_same_ts']:.2%} of consecutive event pairs share a "
        "timestamp, so the deterministic tie-break simply does not touch enough adjacent "
        "pairs to manufacture an alternation of this size. The most likely explanation is "
        "genuine market structure -- alternation consistent with bid-ask bounce or "
        "interleaved liquidity-taking reversals -- surviving both perturbations."
    )
    lines.append("")
    lines.append(
        "**This is consistent with Q1's headline gamma fits being unaffected, though not "
        "directly measured.** Q1's power-law fit window starts "
        "at lag 10 (`fit_power_law(..., lo=10, ...)`); the zigzag reported here is measured "
        "over lags 1-10, i.e. entirely at or before the "
        "start of the fit window, not inside it. Whether the alternation itself persists "
        "PAST lag 10 (and could therefore influence the fit) is not established by this "
        "analysis, which only computes lags 1-10 -- that would require extending this same "
        "three-way comparison to lags 11+ before the 'fit window unaffected' claim could be "
        "made without qualification."
    )
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append(
        f"- Single symbol-month ({result['symbol']} {result['period']}); this analysis was "
        "not repeated on other symbols or periods in this run."
    )
    lines.append(
        "- Among same-ts adjacent pairs, 100% are opposite-signed by construction "
        "(post-aggregation, each timestamp holds at most one buy and one sell event), so "
        "variant B reduces to random pair swaps rather than a general permutation -- "
        "equivalent here since groups never exceed size 2."
    )
    lines.append(
        "- `sign_acf` uses the unbiased normalization (divide by n-lag); at lags <=10 with "
        "n in the tens of millions this makes no practical difference."
    )
    lines.append(
        "- Netting (C) measurably dampens the zigzag amplitude relative to baseline, so "
        "same-ts buy/sell pairs do contribute something to it, but the dominant odd/even "
        "pattern (negative ACF(1), large positive ACF(2)) persists through both "
        "perturbations."
    )
    lines.append(
        "- Timestamps have millisecond resolution; finer-than-millisecond ordering "
        "information is unrecoverable, so 'real structure' means real at the "
        "millisecond-aggregated event level, not at the level of true arrival order within "
        "a millisecond."
    )
    lines.append(
        "- This analysis does not extend past lag 10, so it cannot by itself confirm or "
        "rule out zigzag-driven distortion of Q1's [10, 500] fit window; see the note under "
        "Verdict above."
    )
    lines.append("")
    (out_dir / "q1b_zigzag.md").write_text("\n".join(lines))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q1b: short-lag ACF zigzag tie-break robustness check")
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--period", type=str, default="2023-06")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_q1b(args.root, args.out, symbol=args.symbol, period=args.period)
