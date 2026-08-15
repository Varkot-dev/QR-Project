"""Q1: Is Binance aggressor order flow long-memory?

Method: aggressor-event sign series per symbol -> FFT ACF to max_lag ->
log-log power-law fit over lags [10, max_lag//2] -> compare exponent to
the equity/futures literature range (gamma ~ 0.3-0.7, Bouchaud 2004).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from microstructure.estimators.acf import fit_power_law, sign_acf
from microstructure.signals.load import load_events

LIT_RANGE = (0.3, 0.7)  # equity/futures sign-ACF exponent range, research/01+03


def run_q1(root: Path, out_dir: Path, symbols: list[str], periods: list[str],
           max_lag: int = 1000) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    fig, ax = plt.subplots(figsize=(7, 5))
    for sym in symbols:
        ev = load_events(root, sym, periods)
        signs = ev["sign"].to_numpy()
        acf = sign_acf(signs, max_lag)
        fit = fit_power_law(acf, lo=10, hi=max_lag // 2)
        results[sym] = {"gamma": fit.exponent, "stderr": fit.stderr,
                        "n_events": int(signs.size), "acf": acf.tolist()}
        lags = np.arange(1, max_lag + 1)
        ax.loglog(lags, np.clip(acf[1:], 1e-6, None), label=f"{sym} (γ̂={fit.exponent:.3f})")
    ax.set_xlabel("lag (events)"); ax.set_ylabel("sign ACF"); ax.legend()
    ax.set_title(f"Order-flow sign memory, periods {periods[0]}..{periods[-1]}")
    fig.savefig(out_dir / "q1_acf_loglog.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    _write_results_md(out_dir, results, periods)
    (out_dir / "q1_results.json").write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk != "acf"} for k, v in results.items()}, indent=2))
    return results


def _write_results_md(out_dir: Path, results: dict, periods: list[str]) -> None:
    lo, hi = LIT_RANGE
    lines: list[str] = []
    lines.append("# Q1: Order-flow memory — results")
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append(
        "For each symbol, Binance aggTrades prints for the given periods are collapsed "
        "into aggressor-level events (all same-millisecond, same-side prints merged into "
        "one taker decision — see `to_aggressor_events`), producing a ±1 sign series where "
        "+1 is a buyer-initiated (taker-buy) event and -1 is a seller-initiated event. The "
        "sample autocorrelation function (ACF) of this sign series is computed via FFT "
        "(Wiener-Khinchin) out to `max_lag` events. A power law of the form "
        "ACF(lag) ~ lag^(-γ) is then fit by ordinary least squares on log(ACF) vs log(lag) "
        "over the window [10, max_lag // 2], skipping any non-positive ACF values in that "
        "window. The fitted exponent γ̂ is the analysis's estimate of the order-flow "
        "long-memory decay rate."
    )
    lines.append("")
    lines.append(f"Periods analyzed: {', '.join(periods)}.")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| symbol | n_events | γ̂ | OLS stderr |")
    lines.append("|---|---|---|---|")
    for sym, r in results.items():
        lines.append(f"| {sym} | {r['n_events']:,} | {r['gamma']:.4f} | {r['stderr']:.4f} |")
    lines.append("")
    lines.append("## Benchmark vs. literature")
    lines.append("")
    lines.append(
        f"Literature range (equities/futures sign-ACF exponent, Bouchaud et al. 2004): "
        f"γ ≈ {lo:.1f}–{hi:.1f}, with persistence horizons of thousands of trades."
    )
    lines.append("")
    lines.append("| symbol | γ̂ | literature range | in range? |")
    lines.append("|---|---|---|---|")
    for sym, r in results.items():
        gamma = r["gamma"]
        in_range = lo <= gamma <= hi
        lines.append(f"| {sym} | {gamma:.4f} | {lo:.1f}–{hi:.1f} | {'yes' if in_range else 'no'} |")
    lines.append("")
    lines.append(
        "Falling inside or outside this range is a documented empirical finding either way, "
        "not a pass/fail criterion for the analysis."
    )
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- The OLS standard error on γ̂ is computed from the log-log regression residuals "
        "under an i.i.d.-error assumption; because the ACF values at nearby lags are "
        "themselves autocorrelated, this stderr **understates** the true uncertainty in γ̂."
    )
    lines.append(
        "- Millisecond-timestamp ties are merged by aggressor aggregation before computing "
        "signs (multiple same-ms, same-side prints from one sweep become a single event), "
        "so the event count is smaller than the raw aggTrades row count and lag-1 structure "
        "reflects aggressor decisions, not raw prints."
    )
    lines.append(
        "- The sample covers 2 months of a specific market regime for each symbol; the "
        "estimated γ̂ may not generalize to other periods, volatility regimes, or symbols."
    )
    lines.append("")
    (out_dir / "q1_results.md").write_text("\n".join(lines))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q1: order-flow sign memory analysis")
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT")
    parser.add_argument("--periods", type=str, default="2023-06,2023-07")
    parser.add_argument("--max-lag", type=int, default=1000)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    symbols = args.symbols.split(",")
    periods = args.periods.split(",")
    run_q1(args.root, args.out, symbols=symbols, periods=periods, max_lag=args.max_lag)
