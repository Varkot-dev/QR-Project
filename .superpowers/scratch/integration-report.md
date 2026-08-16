# Phase-1.5 diagnostics — integration report

Branch: `phase15-diagnostics` (off `main`), not pushed.

## Commits

1. `5ff0436` — feat: add Q0 aggregation-effect diagnostic
   Adds `src/microstructure/analyses/q0_aggregation_effect.py` +
   `tests/analyses/test_q0.py`, and runs it for real on BTCUSDT/ETHUSDT x
   2023-06/2023-07, writing `results/q0_aggregation_effect.{md,json}`.

2. `027c5ec` — feat: add Q1b zigzag tie-break robustness diagnostic
   Adds `src/microstructure/analyses/q1b_zigzag.py` +
   `tests/analyses/test_q1b.py`, and runs it for real on BTCUSDT 2023-06,
   writing `results/q1b_zigzag.{md,json,png}`.

3. `e5e0ada` — docs: link Phase-1.5 diagnostics from LEARNING.md and README
   Updates `LEARNING.md` §1 to point at `results/q0_aggregation_effect.md`
   as the committed artifact behind the raw-vs-agg table, adds a new
   subsection after §2 covering the zigzag verdict and the ETH weekly-gamma
   regime-robustness result, and updates `README.md`'s results section with
   a Phase 1.5 subsection (one line + link per artifact), repo map, and test
   count.

## Suite / lint status

- `uv run pytest -m "not network" -q` → **73 passed, 1 deselected** (67
  pre-existing + 6 new: 2 in `test_q0.py`, 4 in `test_q1b.py`).
- `uv run ruff check src/ tests/` → **All checks passed.**
- Both new analysis files are under 400 lines (208 and 315 lines).
- Real runs against the actual parquet data reproduced the verified
  finding's numbers to the same precision reported in the finding (Q0: all
  four symbol-month cells match `aggEffect`'s numbers to <1e-4; Q1b: BTC
  2023-06 amplitudes 0.138174 / 0.137872 / 0.128335 match `zigzag`'s A/B/C
  numbers exactly, including `pairs_swapped = 131,032`).

## Findings I had to soften, and why

1. **Q0's punchline ("raw gammas land inside the equity range").** The task
   instructions phrased the punchline as raw-print gamma landing inside
   0.3–0.7. The actual `aggEffect` finding's own verdict text says this
   holds strictly in only 1 of 4 cells (ETH 2023-07), with ETH 2023-06 at
   the boundary (0.708) and both BTC months overshooting past 0.7 (0.787,
   0.963). I did not write the stronger "lands inside the range" claim as
   a blanket statement — the md's punchline says raw gamma is inflated
   "landing INSIDE the equity range in half the cells... and OVERSHOOTING
   past it in the other half," and the literature-range table shows the
   per-cell yes/no explicitly (2 of 4 raw cells are "no" — BTC's two
   months). This matches the verified finding rather than the more
   convenient headline version.

2. **Q1b's "headline gamma fits are unaffected" claim.** The task asked me
   to check whether the zigzag amplitude persists beyond lag 10 before
   asserting the fit window (lo=10) is unaffected. The verified finding
   only reports ACF at lags 1–10, and my own re-computation is likewise
   restricted to lags 1–10 (matching the finding's own script scope), so I
   could not actually verify whether the alternation extends into lag
   10+. I qualified this in both the results md and LEARNING.md: I state
   that the zigzag is measured entirely at or before the fit window's
   start (which is true and defensible), but I explicitly flag that
   whether the alternation reaches *past* lag 10 is not established by
   this analysis, rather than asserting "unaffected" outright.

3. **ETH regime-robustness / SEC-suit causal link.** The `ethRobust`
   finding's own notes flag the SEC-filing timing as directionally
   consistent but not causally established (volatility/volume not
   controlled). LEARNING.md's new subsection preserves that hedge
   verbatim in spirit — "suggestive, not established causally" — rather
   than stating the regime shift is caused by the SEC suits.

No finding was reported as more confident than its verifier verdict
warranted; where a verdict was itself qualified (e.g. "regime-driven, not
robust" for ethRobust, "real structure" but with caveats for zigzag), those
qualifications are carried into the committed artifacts and docs.

## Files touched (absolute paths)

- `/Users/varshithkotagiri/Projects/QR project/src/microstructure/analyses/q0_aggregation_effect.py`
- `/Users/varshithkotagiri/Projects/QR project/tests/analyses/test_q0.py`
- `/Users/varshithkotagiri/Projects/QR project/results/q0_aggregation_effect.md`
- `/Users/varshithkotagiri/Projects/QR project/results/q0_aggregation_effect.json`
- `/Users/varshithkotagiri/Projects/QR project/src/microstructure/analyses/q1b_zigzag.py`
- `/Users/varshithkotagiri/Projects/QR project/tests/analyses/test_q1b.py`
- `/Users/varshithkotagiri/Projects/QR project/results/q1b_zigzag.md`
- `/Users/varshithkotagiri/Projects/QR project/results/q1b_zigzag.json`
- `/Users/varshithkotagiri/Projects/QR project/results/q1b_zigzag.png`
- `/Users/varshithkotagiri/Projects/QR project/LEARNING.md`
- `/Users/varshithkotagiri/Projects/QR project/README.md`

## Not done / left as-is

- The ETH weekly-gamma sub-period breakdown quoted in LEARNING.md is *not*
  a new committed analysis script with its own results artifact — it is
  the pre-existing `.superpowers/scratch/gamma_subperiods.py` output,
  quoted from the finding. LEARNING.md's prose says this explicitly rather
  than implying a new CLI analysis exists for it. This was outside the
  four numbered deliverables in the task.
- Did not push the branch (per instructions).
