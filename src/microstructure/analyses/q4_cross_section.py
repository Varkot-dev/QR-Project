"""Q4: trades-side cross-section — does order-flow memory scale with activity?

Method: for each symbol in the universe, load one month of aggTrades,
collapse to aggressor events (`load_events`), and compute five per-symbol
statistics on the ±1 sign series: n_events (activity), gamma-hat (sign-ACF
power-law exponent, lags [10, max_lag//2], same fit window as Q1), lag-1
ACF, p_flip (P(sign_t+1 != sign_t)), zigzag amplitude (Q1b's definition:
mean ACF at even lags 2,4,6,8,10 minus mean ACF at odd lags 1,3,5,7,9), and
total traded quantity. Symbols with fewer than `min_events` events are
skipped (not failed); any other per-symbol error (missing parquet, bad
schema, etc.) is caught and logged as a failure. Neither skips nor
failures abort the run. Frames are processed and released one symbol at a
time to bound memory across the ~207-symbol universe.

Two cross-sectional OLS regressions are then fit on the successful set:
gamma-hat on log10(n_events), and p_flip on log10(n_events). Plain
`np.polyfit` is used (not `ols_through_origin`, since these regressions
are not through-origin — there is no reason to expect gamma or p_flip to
vanish at zero activity, so an intercept term is required); this is
documented in the md output. The regression stderr/R^2 the same OLS
machinery as `estimators.acf.fit_power_law` produces is heteroskedastic
across symbols: per-symbol gamma stderrs already understate uncertainty
(same i.i.d.-residual caveat as Q1), and that understatement is not
uniform across symbols with different n_events, so the cross-sectional
regression's own OLS assumptions (homoskedastic residuals) are violated
by construction. This is stated plainly in the md output rather than
papered over with a false confidence interval.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from microstructure.data.catalog import parquet_path
from microstructure.estimators.acf import fit_power_law, sign_acf
from microstructure.signals.load import load_events

EVEN_LAGS = [2, 4, 6, 8, 10]  # Q1b zigzag_amplitude definition
ODD_LAGS = [1, 3, 5, 7, 9]
FIT_LO = 10


def _zigzag_amplitude(acf: np.ndarray) -> float:
    return float(acf[EVEN_LAGS].mean() - acf[ODD_LAGS].mean())


def _symbol_stats(root: Path, symbol: str, period: str, max_lag: int) -> dict:
    """Compute all per-symbol stats. Raises on any failure; caller catches."""
    ev = load_events(root, symbol, [period])
    signs = ev["sign"].to_numpy()
    qty = ev["qty"].to_numpy()
    n_events = int(signs.size)
    total_qty = float(qty.sum())
    del ev, qty

    acf = sign_acf(signs, max_lag)
    fit = fit_power_law(acf, lo=FIT_LO, hi=max_lag // 2)
    acf1 = float(acf[1])
    p_flip = float(np.mean(signs[1:] != signs[:-1]))
    zigzag = _zigzag_amplitude(acf)
    del signs, acf

    return {
        "symbol": symbol,
        "n_events": n_events,
        "gamma": fit.exponent,
        "stderr": fit.stderr,
        "acf1": acf1,
        "p_flip": p_flip,
        "zigzag_amplitude": zigzag,
        "total_qty": total_qty,
    }


def _ols_with_intercept(x: np.ndarray, y: np.ndarray) -> dict:
    """OLS slope/intercept/R^2 via np.polyfit (not through-origin)."""
    (slope, intercept), cov = np.polyfit(x, y, 1, cov=True)
    yhat = slope * x + intercept
    resid = y - yhat
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "stderr": float(np.sqrt(cov[0, 0])),
        "r2": r2,
        "n": int(x.size),
    }


def _cross_sectional_regressions(records: list[dict]) -> dict:
    if len(records) < 3:
        return {
            "gamma_vs_activity": None,
            "p_flip_vs_activity": None,
            "note": "fewer than 3 successful symbols; regressions not fit",
        }
    log_n = np.array([np.log10(r["n_events"]) for r in records])
    if np.ptp(log_n) == 0.0:
        return {
            "gamma_vs_activity": None,
            "p_flip_vs_activity": None,
            "note": "log10(n_events) has zero variance across successful symbols; "
                    "regressions not fit",
        }
    gamma = np.array([r["gamma"] for r in records])
    p_flip = np.array([r["p_flip"] for r in records])
    return {
        "gamma_vs_activity": _ols_with_intercept(log_n, gamma),
        "p_flip_vs_activity": _ols_with_intercept(log_n, p_flip),
    }


def run_q4(
    root: Path,
    out_dir: Path,
    symbols: list[str],
    period: str = "2023-06",
    min_events: int = 1_000_000,
    max_lag: int = 1000,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    skips: list[dict] = []
    failures: list[dict] = []

    for symbol in symbols:
        p = parquet_path(root, symbol, "aggTrades", period)
        if not p.exists():
            failures.append({"symbol": symbol, "reason": f"parquet not found: {p}"})
            continue
        try:
            # Peek n_events cheaply is not available without loading; load
            # once and check min_events before computing expensive stats
            # would still require the full sign series for n_events itself,
            # so we compute after load but check min_events before the ACF
            # fit (which is the expensive step) to save work on tiny frames.
            ev = load_events(root, symbol, [period])
            n_events = ev.height
            if n_events < min_events:
                skips.append({"symbol": symbol, "reason": "below min_events", "n_events": n_events})
                del ev
                continue
            del ev
            stats = _symbol_stats(root, symbol, period, max_lag)
            records.append(stats)
        except Exception as e:  # noqa: BLE001 - per-symbol robustness is the point
            failures.append({"symbol": symbol, "reason": f"{type(e).__name__}: {e}"})

    regressions = _cross_sectional_regressions(records)

    result = {
        "period": period,
        "min_events": min_events,
        "max_lag": max_lag,
        "n_symbols_requested": len(symbols),
        "n_symbols_successful": len(records),
        "n_symbols_skipped": len(skips),
        "n_symbols_failed": len(failures),
        "symbols": records,
        "skips": skips,
        "failures": failures,
        "regressions": regressions,
    }

    _plot_gamma_vs_activity(out_dir, records)
    _plot_flip_vs_activity(out_dir, records)
    _write_results_md(out_dir, result)
    (out_dir / "q4_cross_section.json").write_text(json.dumps(result, indent=2))
    return result


def _plot_gamma_vs_activity(out_dir: Path, records: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    if records:
        log_n = np.array([np.log10(r["n_events"]) for r in records])
        gamma = np.array([r["gamma"] for r in records])
        ax.scatter(log_n, gamma, s=24, alpha=0.75)
        # Errorbars deliberately omitted: per-symbol OLS stderr on gamma-hat
        # understates true uncertainty (i.i.d.-residual assumption violated
        # by autocorrelated ACF values, same caveat as Q1) and is not
        # comparable across symbols with very different n_events, so
        # plotting it would imply a precision the estimate does not have.
        if len(records) >= 3 and np.ptp(log_n) > 0.0:
            slope, intercept = np.polyfit(log_n, gamma, 1)
            x_line = np.array([log_n.min(), log_n.max()])
            ax.plot(x_line, slope * x_line + intercept, color="red",
                     label=f"OLS fit (slope={slope:.4f})")
            ax.legend()
    ax.set_xlabel("log10(n_events)")
    ax.set_ylabel(r"$\hat{\gamma}$ (sign-ACF power-law exponent)")
    ax.set_title("Q4: order-flow memory exponent vs. activity")
    fig.savefig(out_dir / "q4_gamma_vs_activity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_flip_vs_activity(out_dir: Path, records: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    if records:
        log_n = np.array([np.log10(r["n_events"]) for r in records])
        p_flip = np.array([r["p_flip"] for r in records])
        ax.scatter(log_n, p_flip, s=24, alpha=0.75)
        if len(records) >= 3 and np.ptp(log_n) > 0.0:
            slope, intercept = np.polyfit(log_n, p_flip, 1)
            x_line = np.array([log_n.min(), log_n.max()])
            ax.plot(x_line, slope * x_line + intercept, color="red",
                     label=f"OLS fit (slope={slope:.4f})")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, label="no persistence")
    ax.set_xlabel("log10(n_events)")
    ax.set_ylabel(r"$p_{flip} = P(sign_{t+1} \neq sign_t)$")
    ax.set_title("Q4: sign-flip probability vs. activity")
    ax.legend()
    fig.savefig(out_dir / "q4_flip_vs_activity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fmt_reg(reg: dict | None, note: str | None = None) -> str:
    if reg is None:
        return f"not estimable ({note or 'insufficient data'})"
    return (
        f"slope = **{reg['slope']:.4f}** (stderr {reg['stderr']:.4f}), "
        f"intercept = {reg['intercept']:.4f}, R² = {reg['r2']:.4f}, n = {reg['n']}"
    )


def _direction_word(slope: float) -> str:
    if slope > 0:
        return "increases"
    if slope < 0:
        return "decreases"
    return "shows no relationship (slope ≈ 0) with"


def _write_results_md(out_dir: Path, result: dict) -> None:
    records = result["symbols"]
    period = result["period"]
    min_events = result["min_events"]
    regs = result["regressions"]

    lines: list[str] = []
    lines.append("# Q4: trades-side cross-section — results")
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append(
        f"For each symbol in the requested universe, one month ({period}) of aggTrades "
        "is loaded and collapsed into aggressor-level events (`load_events`), producing a "
        "±1 sign series per symbol. Symbols are processed one at a time and their frames "
        "released (`del`) before moving to the next symbol, bounding peak memory across "
        "the full universe. Symbols with fewer than "
        f"`min_events` = {min_events:,} events are **skipped** (logged, reason "
        "\"below min_events\") rather than analyzed; any other per-symbol failure "
        "(missing parquet, malformed data, or any other exception) is caught and logged "
        "into `failures` with the symbol and the exception, and never aborts the run for "
        "the remaining symbols."
    )
    lines.append("")
    lines.append("Per successful symbol, five statistics are computed on the sign series:")
    lines.append("")
    lines.append(
        "- **n_events**: activity, the number of aggressor events in the period."
    )
    lines.append(
        "- **γ̂ + OLS stderr**: sign-ACF power-law exponent, fit the same way as Q1 "
        "(`fit_power_law(sign_acf(signs, max_lag), lo=10, hi=max_lag//2)`)."
    )
    lines.append("- **lag-1 ACF**: `sign_acf(signs, max_lag)[1]`.")
    lines.append(
        "- **p_flip**: `P(sign_{t+1} != sign_t)`, the empirical fraction of consecutive "
        "sign flips — 0.5 is the no-persistence benchmark (independent coin flips)."
    )
    lines.append(
        "- **zigzag amplitude**: Phase-1.5's definition (Q1b) — mean ACF at even lags "
        "2,4,6,8,10 minus mean ACF at odd lags 1,3,5,7,9."
    )
    lines.append("- **total_qty**: sum of aggressor-event quantity over the period.")
    lines.append("")
    lines.append(
        "**Cross-sectional regressions**: on the successful set, ordinary least squares "
        "(`np.polyfit`, degree 1, with intercept — not through-origin, since there is no "
        "reason to expect γ̂ or p_flip to vanish at zero activity) is used to regress (a) "
        "γ̂ on log10(n_events) and (b) p_flip on log10(n_events)."
    )
    lines.append("")
    lines.append(
        "**Heteroskedasticity caveat**: each symbol's own γ̂ stderr comes from "
        "`fit_power_law`'s i.i.d.-residual OLS assumption applied to autocorrelated ACF "
        "values, which already understates that symbol's true uncertainty (documented in "
        "Q1). That understatement is *not* uniform across symbols — it scales with each "
        "symbol's own n_events and ACF shape — so per-symbol γ̂ noise is heteroskedastic "
        "across the cross-section. The cross-sectional regressions above therefore also "
        "violate the homoskedastic-residual assumption behind their own OLS stderr; the "
        "reported regression stderr/R² should be read as descriptive, not as a valid "
        "confidence interval on the true relationship."
    )
    lines.append("")
    lines.append("## Run summary")
    lines.append("")
    lines.append(
        f"Requested: {result['n_symbols_requested']}. Successful: "
        f"{result['n_symbols_successful']}. Skipped (below min_events): "
        f"{result['n_symbols_skipped']}. Failed: {result['n_symbols_failed']}."
    )
    lines.append("")

    if records:
        sorted_records = sorted(records, key=lambda r: r["n_events"], reverse=True)
        top10 = sorted_records[:10]
        bottom10 = sorted_records[-10:] if len(sorted_records) > 10 else []

        lines.append("## Top-10 by activity (n_events)")
        lines.append("")
        lines.append("| symbol | n_events | γ̂ | stderr | acf1 | p_flip | zigzag | total_qty |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in top10:
            lines.append(
                f"| {r['symbol']} | {r['n_events']:,} | {r['gamma']:.4f} | {r['stderr']:.4f} | "
                f"{r['acf1']:.4f} | {r['p_flip']:.4f} | {r['zigzag_amplitude']:.4f} | "
                f"{r['total_qty']:,.2f} |"
            )
        lines.append("")

        if bottom10:
            lines.append("## Bottom-10 by activity (n_events)")
            lines.append("")
            lines.append("| symbol | n_events | γ̂ | stderr | acf1 | p_flip | zigzag | total_qty |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for r in bottom10:
                lines.append(
                    f"| {r['symbol']} | {r['n_events']:,} | {r['gamma']:.4f} | {r['stderr']:.4f} | "
                    f"{r['acf1']:.4f} | {r['p_flip']:.4f} | {r['zigzag_amplitude']:.4f} | "
                    f"{r['total_qty']:,.2f} |"
                )
            lines.append("")
    else:
        lines.append("No symbols produced usable results — no table to show.")
        lines.append("")

    lines.append("## Cross-sectional regressions")
    lines.append("")
    reg_note = regs.get("note")
    lines.append(f"**γ̂ on log10(n_events)**: {_fmt_reg(regs.get('gamma_vs_activity'), reg_note)}")
    lines.append("")
    lines.append(f"**p_flip on log10(n_events)**: {_fmt_reg(regs.get('p_flip_vs_activity'), reg_note)}")
    lines.append("")

    lines.append("## Findings")
    lines.append("")
    gamma_reg = regs.get("gamma_vs_activity")
    flip_reg = regs.get("p_flip_vs_activity")
    if gamma_reg is not None:
        lines.append(
            f"The fitted order-flow memory exponent γ̂ **{_direction_word(gamma_reg['slope'])}** "
            f"with log-activity across the {gamma_reg['n']}-symbol successful set "
            f"(slope {gamma_reg['slope']:.4f}, R² {gamma_reg['r2']:.4f}), i.e. more actively "
            "traded symbols in this sample tend to show "
            f"{'stronger' if gamma_reg['slope'] > 0 else 'weaker' if gamma_reg['slope'] < 0 else 'no different'} "
            "long-memory decay than less actively traded ones."
        )
    else:
        lines.append(
            f"γ̂-vs-activity relationship not estimable: {reg_note or 'insufficient data'} "
            "in this run."
        )
    lines.append("")
    if flip_reg is not None:
        lines.append(
            f"The sign-flip probability p_flip **{_direction_word(flip_reg['slope'])}** with "
            f"log-activity (slope {flip_reg['slope']:.4f}, R² {flip_reg['r2']:.4f}); since "
            "p_flip = 0.5 corresponds to no persistence, this indicates that persistence "
            f"{'strengthens' if flip_reg['slope'] < 0 else 'weakens' if flip_reg['slope'] > 0 else 'is unrelated to activity level'} "
            "as activity increases (a slope below zero means p_flip falls toward more "
            "persistent behavior at higher activity)."
        )
    else:
        lines.append(
            f"p_flip-vs-activity relationship not estimable: {reg_note or 'insufficient data'} "
            "in this run."
        )
    lines.append("")

    if result["failures"]:
        lines.append("## Failures")
        lines.append("")
        lines.append("| symbol | reason |")
        lines.append("|---|---|")
        for f in result["failures"]:
            lines.append(f"| {f['symbol']} | {f['reason']} |")
        lines.append("")

    if result["skips"]:
        lines.append("## Skipped (below min_events)")
        lines.append("")
        lines.append("| symbol | n_events | reason |")
        lines.append("|---|---|---|")
        for s in result["skips"]:
            lines.append(f"| {s['symbol']} | {s.get('n_events', '?'):,} | {s['reason']} |")
        lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append(
        f"- Single month ({period}); this is one specific market regime, and per Phase 1.5 "
        "diagnostics, order-flow memory statistics are regime-dependent — these results "
        "may not generalize to other months or volatility regimes."
    )
    lines.append(
        "- Each symbol's γ̂ OLS stderr understates true uncertainty (autocorrelated ACF "
        "values violate the i.i.d.-residual assumption, same caveat as Q1), and this "
        "understatement is heteroskedastic across the cross-section (see Methodology); "
        "the cross-sectional regression stderr/R² inherit this problem and should be read "
        "as descriptive summaries, not as valid confidence intervals."
    )
    lines.append(
        "- `q4_gamma_vs_activity.png` deliberately omits per-symbol error bars on γ̂: "
        "plotting the OLS stderr would imply a precision the estimate does not have, for "
        "the same heteroskedasticity/understatement reason given above."
    )
    lines.append(
        "- Symbols are only included in the regressions if they clear `min_events`; the "
        "cross-section is therefore a survivorship-filtered subset of the requested "
        "universe, not the full universe."
    )
    lines.append("")
    (out_dir / "q4_cross_section.md").write_text("\n".join(lines))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q4: trades-side cross-section analysis")
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--symbols-file", type=Path, required=True,
                         help="one symbol per line")
    parser.add_argument("--period", type=str, default="2023-06")
    parser.add_argument("--min-events", type=int, default=1_000_000)
    parser.add_argument("--max-lag", type=int, default=1000)
    return parser.parse_args(argv)


def _read_symbols_file(path: Path) -> list[str]:
    lines = path.read_text().splitlines()
    return [s.strip() for s in lines if s.strip()]


if __name__ == "__main__":
    args = _parse_args()
    symbols = _read_symbols_file(args.symbols_file)
    run_q4(
        args.root, args.out, symbols=symbols, period=args.period,
        min_events=args.min_events, max_lag=args.max_lag,
    )
