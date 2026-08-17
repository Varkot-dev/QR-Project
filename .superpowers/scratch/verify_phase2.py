"""Trace Phase-2 numeric claims to artifacts and recompute derived claims."""
import json
import math
from pathlib import Path

import polars as pl

ROOT = Path("/Users/varshithkotagiri/Projects/QR project")
q4 = json.loads((ROOT / "results/q4_cross_section.json").read_text())
q5 = json.loads((ROOT / "results/q5_kernel_panel.json").read_text())
q4b = json.loads((ROOT / "results/q4b_tick_confound.json").read_text())

out = {}

# ---- Q4 basic counts ----
out["q4_counts"] = dict(
    requested=q4["n_symbols_requested"],
    successful=q4["n_symbols_successful"],
    skipped=q4["n_symbols_skipped"],
    failed=q4["n_symbols_failed"],
)

syms = q4["symbols"]
n_events = [s["n_events"] for s in syms]
gammas = [s["gamma"] for s in syms]
pflips = [s["p_flip"] for s in syms]
acf1 = [s["acf1"] for s in syms]
mn, mx = min(n_events), max(n_events)
out["q4_activity"] = dict(min=mn, max=mx, decades=math.log10(mx / mn))

# regressions
g = q4["regressions"]["gamma_vs_activity"]
p = q4["regressions"]["p_flip_vs_activity"]
out["gamma_reg"] = dict(slope=g["slope"], stderr=g["stderr"], r2=g["r2"])
out["pflip_reg"] = dict(slope=p["slope"], stderr=p["stderr"], r2=p["r2"])

# fitted-line movement of gamma across observed range
lmn, lmx = math.log10(mn), math.log10(mx)
move = g["slope"] * (lmx - lmn)
import statistics
sd_gamma = statistics.stdev(gammas)
out["gamma_move"] = dict(
    fitted_move=move,
    cross_sd=sd_gamma,
    ratio=abs(move) / sd_gamma,
    r2_pct=g["r2"] * 100,
    ratio_sq_pct=(move / sd_gamma) ** 2 * 100,
)

# gamma distribution
gs = sorted(gammas)
med = statistics.median(gammas)
q1_ = gs[len(gs) // 4]
# use numpy-style quartiles via statistics.quantiles
quarts = statistics.quantiles(gammas, n=4)
in_range = sum(1 for x in gammas if 0.3 <= x <= 0.7)
gmin = min(syms, key=lambda s: s["gamma"])
gmax = max(syms, key=lambda s: s["gamma"])
out["gamma_dist"] = dict(
    median=med,
    min=(gmin["symbol"], gmin["gamma"]),
    max=(gmax["symbol"], gmax["gamma"]),
    quartiles=quarts,
    n_in_03_07=in_range,
)

# fitted p_flip endpoints
out["pflip_fit_ends"] = dict(
    low=p["intercept"] + p["slope"] * lmn,
    high=p["intercept"] + p["slope"] * lmx,
)

# anti-persistent counts from JSON
ap = [s for s in syms if s["p_flip"] > 0.5]
ap_acf = [s for s in syms if s["acf1"] < 0]
out["antipersistent_json"] = dict(
    n_pflip_gt_half=len(ap),
    n_acf1_neg=len(ap_acf),
    sets_identical=set(s["symbol"] for s in ap) == set(s["symbol"] for s in ap_acf),
)
by_act = sorted(syms, key=lambda s: s["n_events"], reverse=True)
top20 = by_act[:20]
bot20 = by_act[-20:]
out["top_bottom_json"] = dict(
    top20_ap=sum(1 for s in top20 if s["p_flip"] > 0.5),
    bot20_ap=sum(1 for s in bot20 if s["p_flip"] > 0.5),
)

# ---- recompute from parquet ----
df = pl.read_parquet(ROOT / "results/q4_cross_section.parquet")
out["parquet_cols"] = df.columns
out["parquet_rows"] = df.height
d = df.sort("n_events", descending=True)
top = d.head(20)
bot = d.tail(20)
out["top_bottom_parquet"] = dict(
    top20_ap_pflip=int((top["p_flip"] > 0.5).sum()),
    bot20_ap_pflip=int((bot["p_flip"] > 0.5).sum()),
    top20_ap_acf1=int((top["acf1"] < 0).sum()),
    bot20_ap_acf1=int((bot["acf1"] < 0).sum()),
    total_ap=int((df["p_flip"] > 0.5).sum()),
)

# recompute regressions from parquet as sanity check
import numpy as np
x = np.log10(df["n_events"].to_numpy().astype(float))
ygam = df["gamma"].to_numpy()
ypf = df["p_flip"].to_numpy()
def ols(x, y):
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    beta, res, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    sigma2 = ss_res / (n - 2)
    cov = sigma2 * np.linalg.inv(X.T @ X)
    return dict(slope=float(beta[1]), intercept=float(beta[0]),
                stderr=float(np.sqrt(cov[1, 1])), r2=r2)
out["gamma_reg_recomputed"] = ols(x, ygam)
out["pflip_reg_recomputed"] = ols(x, ypf)
out["gamma_sd_parquet"] = float(ygam.std(ddof=1))
out["gamma_move_parquet"] = out["gamma_reg_recomputed"]["slope"] * (x.max() - x.min())

# ---- Q5 ----
recs = q5["records"]
out["q5_counts"] = dict(
    requested=q5["n_symbols_requested"],
    successful=q5["n_symbols_successful"],
    consistent=q5["n_consistent"],
    violated=q5["n_violated"],
)
viol = [r for r in recs if r["verdict"] != "consistent"]
out["q5_violators"] = {r["symbol"]: dict(delta=r["balance_delta"], verdict=r["verdict"],
                                          block_sd=r["beta_block_sd"]) for r in viol}
neg = [r for r in recs if r["balance_delta"] < 0]
out["q5_deltas"] = dict(
    n_negative=len(neg),
    violations_all_negative=all(r["balance_delta"] < 0 for r in viol),
    violation_delta_range=(min(r["balance_delta"] for r in viol),
                           max(r["balance_delta"] for r in viol)),
)
# floor binds when block_sd < 0.04
binds = [r["symbol"] for r in recs if r["beta_block_sd"] < 0.04]
out["q5_floor_binds"] = dict(n=len(binds), symbols=binds,
                              block_sds={r["symbol"]: r["beta_block_sd"] for r in recs})
# verify verdict rule reproduces stored verdicts
mismatch = []
for r in recs:
    tol = 2 * max(r["beta_block_sd"], 0.04)
    v = "consistent" if abs(r["balance_delta"]) <= tol else "violated"
    if v != r["verdict"]:
        mismatch.append((r["symbol"], v, r["verdict"]))
out["q5_verdict_rule_mismatches"] = mismatch

print(json.dumps(out, indent=2, default=str))
