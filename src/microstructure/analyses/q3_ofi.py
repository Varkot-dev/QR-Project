"""Q3: Is Binance mid-price change linear in order-flow imbalance (OFI)?

Method: daily bookTicker parquet -> per-update OFI (Cont, Kukanov & Stoikov
2014 formula) attached to the timestamp of the later update -> bucket into
fixed-length time bars -> per bar, regress delta_mid on summed OFI through
the origin. A secondary check regresses log|slope| on log(mean depth)
across five depth quintiles; Cont's linear-in-inverse-depth theory predicts
that regression's slope to be approximately -1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from microstructure.estimators.ofi import ofi_events, ols_through_origin
from microstructure.signals.load import load_book_ticker

CONT_EQUITIES_R2_RANGE = (0.65, 0.70)
N_DEPTH_QUANTILES = 5
CONT_DEPTH_SCALING_EXPONENT = -1.0


def _bucket_bars(bt: pl.DataFrame, window: str) -> pl.DataFrame:
    """Attach per-update OFI to the later update's ts, then bucket into bars.

    ofi_events returns n-1 values (one per consecutive update pair); each
    value is attached to the ts of the second (later) update in the pair,
    so the first row of the sorted frame carries no OFI contribution.
    """
    bt = bt.sort("ts")
    bid_p = bt["bid_price"].to_numpy()
    bid_q = bt["bid_qty"].to_numpy()
    ask_p = bt["ask_price"].to_numpy()
    ask_q = bt["ask_qty"].to_numpy()
    ofi = ofi_events(bid_p, bid_q, ask_p, ask_q)

    depth = ((bt["bid_qty"] + bt["ask_qty"]) / 2).to_numpy()

    events = pl.DataFrame(
        {
            "ts": bt["ts"][1:],
            "ofi": ofi,
            "mid": bt["mid"][1:],
            "depth": depth[1:],
        }
    ).sort("ts")

    bars = (
        events.group_by_dynamic("ts", every=window)
        .agg(
            pl.len().alias("n_updates"),
            pl.col("ofi").sum().alias("ofi_sum"),
            pl.col("mid").first().alias("mid_first"),
            pl.col("mid").last().alias("mid_last"),
            pl.col("depth").mean().alias("mean_depth"),
        )
        .filter(pl.col("n_updates") >= 2)
        .with_columns((pl.col("mid_last") - pl.col("mid_first")).alias("delta_mid"))
    )
    return bars


def _depth_scaling_check(bars: pl.DataFrame) -> dict:
    """Fit through-origin slope per depth quintile, then log|slope| vs log(depth).

    Quintiles with a non-positive fitted slope are excluded from the
    log-log fit (log of a non-positive number is undefined) and noted.
    """
    bars = bars.with_columns(
        pl.col("mean_depth").qcut(N_DEPTH_QUANTILES, labels=[str(i) for i in range(N_DEPTH_QUANTILES)])
        .alias("depth_q")
    )
    quintile_rows = []
    excluded: list[str] = []
    for q in sorted(bars["depth_q"].unique().to_list(), key=int):
        sub = bars.filter(pl.col("depth_q") == q)
        if sub.height < 2:
            excluded.append(q)
            continue
        x = sub["ofi_sum"].to_numpy()
        y = sub["delta_mid"].to_numpy()
        try:
            fit = ols_through_origin(x, y)
        except ValueError:
            excluded.append(q)
            continue
        mean_depth = float(sub["mean_depth"].mean())
        quintile_rows.append(
            {"quintile": q, "slope": fit.slope, "mean_depth": mean_depth, "n_bars": sub.height}
        )
        if fit.slope <= 0:
            excluded.append(q)

    usable = [r for r in quintile_rows if r["slope"] > 0]
    result: dict = {"quintiles": quintile_rows, "excluded_quintiles": excluded}
    if len(usable) < 2:
        result["scaling_exponent"] = None
        result["note"] = "fewer than 2 usable quintiles (positive slope required for log fit)"
        return result

    log_depth = np.log(np.array([r["mean_depth"] for r in usable]))
    log_abs_slope = np.log(np.abs(np.array([r["slope"] for r in usable])))
    # simple OLS (not through origin) of log|slope| vs log(depth)
    A = np.vstack([log_depth, np.ones_like(log_depth)]).T
    coef, *_ = np.linalg.lstsq(A, log_abs_slope, rcond=None)
    result["scaling_exponent"] = float(coef[0])
    result["intercept"] = float(coef[1])
    return result


def run_q3(root: Path, out_dir: Path, symbol: str, periods: list[str], window: str = "10s") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    bt = load_book_ticker(root, symbol, periods)
    bars = _bucket_bars(bt, window)

    x = bars["ofi_sum"].to_numpy()
    y = bars["delta_mid"].to_numpy()
    fit = ols_through_origin(x, y)
    depth_scaling = _depth_scaling_check(bars)

    result = {
        "slope": fit.slope,
        "stderr": fit.stderr,
        "r2": fit.r2,
        "n_windows": bars.height,
        "depth_scaling_check": depth_scaling,
    }

    _plot_scatter(out_dir, x, y, fit.slope, symbol, window)
    _write_results_md(out_dir, result, symbol, periods, window)
    (out_dir / "q3_results.json").write_text(json.dumps(result, indent=2))
    return result


def _plot_scatter(out_dir: Path, x: np.ndarray, y: np.ndarray, slope: float, symbol: str, window: str) -> None:
    n_bins = 60
    order = np.argsort(x)
    x_sorted, y_sorted = x[order], y[order]
    edges = np.quantile(x_sorted, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    bin_idx = np.clip(np.digitize(x_sorted, edges[1:-1]), 0, len(edges) - 2)
    bin_x, bin_y = [], []
    for b in range(len(edges) - 1):
        mask = bin_idx == b
        if mask.any():
            bin_x.append(x_sorted[mask].mean())
            bin_y.append(y_sorted[mask].mean())

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(bin_x, bin_y, s=18, alpha=0.7, label="binned mean Δmid")
    x_line = np.array([x.min(), x.max()])
    ax.plot(x_line, slope * x_line, color="red", label=f"through-origin fit (β̂={slope:.5f})")
    ax.set_xlabel(f"summed OFI per {window} bar")
    ax.set_ylabel("Δmid per bar")
    ax.set_title(f"{symbol}: OFI-linearity (Q3)")
    ax.legend()
    fig.savefig(out_dir / "q3_ofi_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_results_md(out_dir: Path, result: dict, symbol: str, periods: list[str], window: str) -> None:
    lo, hi = CONT_EQUITIES_R2_RANGE
    ds = result["depth_scaling_check"]
    lines: list[str] = []
    lines.append("# Q3: OFI linearity — results")
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append(
        f"For {symbol}, daily bookTicker L1 snapshots for the given periods are loaded and "
        "sorted by timestamp. Per-update order-flow imbalance (OFI) is computed with the "
        "Cont, Kukanov & Stoikov (2014) formula (`ofi_events`): each consecutive pair of L1 "
        "updates contributes a signed quantity reflecting bid/ask price improvements and "
        "same-price size changes. Each OFI value is attached to the timestamp of the later "
        f"update in its pair. Updates are then bucketed into fixed `{window}` bars via "
        "`pl.group_by_dynamic` on ts. Within each bar: OFI values are summed, delta_mid is "
        "computed as the last mid minus the first mid observed in the bar, and mean depth "
        "is the bar-average of (bid_qty + ask_qty)/2. Bars with fewer than 2 updates are "
        "dropped (no meaningful delta_mid). The resulting (summed OFI, delta_mid) pairs are "
        "regressed through the origin (`ols_through_origin`): delta_mid = beta * OFI_sum."
    )
    lines.append("")
    lines.append(
        "**Depth-scaling check**: bars are split into 5 depth quintiles by mean depth; a "
        "through-origin slope is fit per quintile, then log|slope| is regressed (ordinary "
        "least squares, not through origin) on log(mean depth) across quintiles. Cont's "
        "theory (slope ~ 1/depth) predicts this log-log regression's slope to be "
        "approximately -1. Quintiles whose fitted slope is zero or negative are excluded "
        "from the log-log fit (undefined under the log) and noted below."
    )
    lines.append("")
    lines.append(f"Periods analyzed: {', '.join(periods)}.")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| slope (β̂) | {result['slope']:.6f} |")
    lines.append(f"| stderr | {result['stderr']:.6f} |")
    lines.append(f"| R² | {result['r2']:.4f} |")
    lines.append(f"| n_windows | {result['n_windows']:,} |")
    lines.append("")
    lines.append("### Depth-scaling check")
    lines.append("")
    lines.append("| quintile | mean depth | slope | n_bars |")
    lines.append("|---|---|---|---|")
    for r in ds["quintiles"]:
        lines.append(f"| {r['quintile']} | {r['mean_depth']:.4f} | {r['slope']:.6f} | {r['n_bars']:,} |")
    lines.append("")
    if ds["excluded_quintiles"]:
        lines.append(
            f"Excluded from log-log fit (non-positive slope): {', '.join(ds['excluded_quintiles'])}."
        )
        lines.append("")
    if ds.get("scaling_exponent") is not None:
        lines.append(
            f"log|slope| vs log(mean depth) regression exponent: **{ds['scaling_exponent']:.4f}** "
            f"(Cont theory predicts ≈ {CONT_DEPTH_SCALING_EXPONENT:.0f})."
        )
    else:
        lines.append(f"Depth-scaling exponent not estimable: {ds.get('note', 'insufficient usable quintiles')}.")
    lines.append("")
    lines.append("## Benchmark vs. literature")
    lines.append("")
    lines.append(
        f"Cont, Kukanov & Stoikov (2014) report R² ≈ {lo:.0%}–{hi:.0%} for OFI-vs-price-change "
        "linear regressions on equities. Silantyev (2019) studied BitMEX order flow and found "
        "trade-flow imbalance (net signed trade volume) a stronger price-change predictor than "
        "book-based OFI in that venue."
    )
    lines.append("")
    lines.append("| our R² | Cont equities R² | benchmark comparison |")
    lines.append("|---|---|---|")
    r2 = result["r2"]
    cmp = "within" if lo <= r2 <= hi else ("above" if r2 > hi else "below")
    lines.append(f"| {r2:.4f} | {lo:.2f}–{hi:.2f} | {cmp} the Cont equities range |")
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- OFI is computed from L1 (best bid/ask) bookTicker snapshots only; no order-book "
        "depth beyond the top level is observed, so OFI understates true order-flow pressure "
        "from deeper levels."
    )
    lines.append(
        "- `ols_through_origin`'s stderr assumes i.i.d. residuals; bar-level delta_mid and OFI "
        "sums are likely autocorrelated across adjacent bars, so this stderr understates true "
        "uncertainty."
    )
    lines.append(
        "- The depth-scaling check uses only 5 quintiles over a fixed 14-day sample, giving a "
        "log-log regression with very few points; its exponent estimate carries wide "
        "(unreported) uncertainty and should be treated as suggestive, not confirmatory."
    )
    lines.append(
        "- The sample covers a single symbol over 14 days of a specific market regime; results "
        "may not generalize to other symbols, venues, or volatility regimes."
    )
    lines.append("")
    (out_dir / "q3_results.md").write_text("\n".join(lines))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q3: OFI linearity analysis")
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--symbol", type=str, default="ETHUSDT")
    parser.add_argument("--start-day", type=str, default="2023-06-01")
    parser.add_argument("--end-day", type=str, default="2023-06-14")
    parser.add_argument("--window", type=str, default="10s")
    return parser.parse_args(argv)


def _daily_periods(start_day: str, end_day: str) -> list[str]:
    from datetime import date, timedelta

    y, m, d = (int(v) for v in start_day.split("-"))
    start = date(y, m, d)
    y, m, d = (int(v) for v in end_day.split("-"))
    end = date(y, m, d)
    periods = []
    cur = start
    while cur <= end:
        periods.append(cur.isoformat())
        cur += timedelta(days=1)
    return periods


if __name__ == "__main__":
    args = _parse_args()
    periods = _daily_periods(args.start_day, args.end_day)
    run_q3(args.root, args.out, symbol=args.symbol, periods=periods, window=args.window)
