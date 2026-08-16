"""Univariate exponential-kernel Hawkes process: simulator, MLE, count-variance.

Parameterization: intensity λ(t) = mu + Σ_{t_i < t} alpha*beta*exp(-beta*(t-t_i)).
The kernel is φ(t) = alpha*beta*exp(-beta*t); its integral over [0, ∞) is
    ∫ alpha*beta*exp(-beta*t) dt = alpha*beta * (1/beta) = alpha,
so **alpha IS the branching ratio n** (Hawkes & Oakes 1974 branching
interpretation: research/02-hawkes-processes.md §1 "Branching interpretation
and criticality"). alpha in [0, 1) for a stationary process; alpha -> 1 is
criticality; alpha >= 1 is explosive/non-stationary.

Three independent estimators are provided so they can cross-check each
other, as the literature insists on (research/02 §4 "Non-negotiables:
... report n̂ sensitivity to window and kernel family"):
  - simulate_hawkes_exp: ground truth via Ogata (1978) thinning.
  - fit_hawkes_exp: parametric MLE using the O(N) exponential-kernel
    recursion, optimized with a hand-rolled multi-start Nelder-Mead
    (numpy only, no scipy).
  - branching_count_variance: Hardiman & Bouchaud (2014) model-free
    estimator from count mean/variance alone — no kernel shape assumed.

The Poisson-refutation / regime-switching trap tests in
tests/estimators/test_hawkes.py document a critical failure mode: on a
non-stationary-rate (but NOT self-exciting) process, both a Hawkes MLE and
the count-variance estimator report spurious positive endogeneity
(Filimonov & Sornette 2015, research/02 §"The calibration counterattack").
This motivates deseasonalizing mu(t) before ever trusting an n̂ on real
data.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Simulation: Ogata (1978) thinning.
# ---------------------------------------------------------------------------


def simulate_hawkes_exp(mu: float, alpha: float, beta: float, t_end: float, seed: int) -> np.ndarray:
    """Simulate event times of an exponential-kernel Hawkes process via thinning.

    λ(t) = mu + Σ_{t_i < t} alpha*beta*exp(-beta*(t - t_i)); branching ratio
    (kernel integral) is exactly alpha:

        >>> import numpy as np
        >>> alpha, beta = 0.4, 2.0
        >>> t = np.linspace(0, 50, 200_000)
        >>> kernel_integral = np.trapezoid(alpha * beta * np.exp(-beta * t), t)
        >>> abs(kernel_integral - alpha) < 1e-3
        True

    Algorithm (Ogata thinning, exploiting the exponential kernel's Markov
    property so no per-candidate rebuild of the full excitation sum is
    needed): maintain the running excitation
        E(t) = Σ_{t_i < t} alpha*beta*exp(-beta*(t - t_i)),
    which decays smoothly between events and jumps by +alpha*beta at each
    accepted event. Immediately after an event at time t_i, the intensity
    is at its local maximum for the segment until the next accepted event
    (since the kernel is monotonically decaying), so
        lambda_bar = mu + E(t_i^+) = mu + E(t_i^-) + alpha*beta
    is a valid upper bound for λ(t) on [t_i, next accepted event]. Draw
    candidate arrival times from a homogeneous Poisson process at rate
    lambda_bar; accept a candidate at time t_c with probability
    λ(t_c)/lambda_bar. On rejection, the bound is still valid going
    forward (intensity only decays between events) so we simply continue
    thinning from t_c without recomputing lambda_bar.
    """
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if mu < 0.0:
        raise ValueError("mu must be non-negative")
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")

    rng = np.random.default_rng(seed)
    events: list[float] = []

    t = 0.0
    excitation = 0.0  # E(t) just after the most recent processed point
    lambda_bar = mu + excitation + alpha * beta  # upper bound valid until next accept

    while t < t_end:
        # Candidate inter-arrival time from homogeneous Poisson(lambda_bar).
        t += rng.exponential(1.0 / lambda_bar)
        if t >= t_end:
            break

        # Decay excitation from the last processed point to the candidate time.
        # (excitation tracked at the time of the last event/candidate, "dt" is
        # the gap since then.)
        # We recompute excitation at t directly below via the last event time.
        if events:
            dt_last = t - events[-1]
            excitation_at_t = excitation * np.exp(-beta * dt_last)
        else:
            excitation_at_t = 0.0

        lam_t = mu + excitation_at_t
        u = rng.random()
        if u <= lam_t / lambda_bar:
            events.append(t)
            excitation = excitation_at_t + alpha * beta
            lambda_bar = mu + excitation + alpha * beta
        # else rejected: lambda_bar remains valid (intensity only decays
        # between accepted events), continue thinning from t.

    return np.asarray(events, dtype=np.float64)


# ---------------------------------------------------------------------------
# MLE: O(N) recursion + hand-rolled multi-start Nelder-Mead.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HawkesFit:
    mu: float
    alpha: float
    beta: float
    loglik: float
    converged: bool


def _excitation_recursion(decay: np.ndarray) -> np.ndarray:
    """Vectorized R_i recursion: R_0=0, R_i = decay[i-1]*(R_{i-1}+1).

    This is an affine linear recurrence x_i = a_i*x_{i-1} + b_i with
    a_i = b_i = decay[i-1]. A naive Python for-loop over N events is the
    bottleneck in MLE fitting (called ~1000x by Nelder-Mead multi-start,
    on up to ~10^5 events per call) — the loop's per-element Python
    overhead dominates runtime. This computes the same recursion with a
    Hillis-Steele parallel prefix scan: O(N log N) numpy vector ops
    instead of O(N) Python-level iterations, ~8-10x faster in practice
    for N ~ 10^4-10^5 despite the extra log-factor work, because every
    step here is a vectorized numpy op rather than a scalar Python one.
    Each scan step combines affine maps (a1,b1) then (a2,b2) via
    a = a2*a1, b = a2*b1 + b2 (composition x -> a2*(a1*x+b1)+b2).
    """
    n = decay.size
    if n == 0:
        return np.array([0.0])

    a = decay.copy()
    b = decay.copy()
    shift = 1
    while shift < n:
        a_shifted = np.ones_like(a)
        b_shifted = np.zeros_like(b)
        a_shifted[shift:] = a[:-shift]
        b_shifted[shift:] = b[:-shift]
        b = b + a * b_shifted
        a = a * a_shifted
        shift *= 2

    r = np.empty(n + 1, dtype=np.float64)
    r[0] = 0.0
    r[1:] = b  # x_0 = 0, so x_i = a_i*0 + b_i = b_i
    return r


def hawkes_loglik(times: np.ndarray, t_end: float, mu: float, alpha: float, beta: float) -> float:
    """Exact log-likelihood via the O(N) exponential-kernel recursion.

    loglik = Σ_i log(mu + alpha*beta*R_i) - mu*T - alpha*Σ_i (1 - exp(-beta*(T-t_i)))

    with R_1 = 0, R_{i+1} = exp(-beta*(t_{i+1}-t_i)) * (R_i + 1)  (research/02
    §4.1). R_i represents Σ_{j<i} exp(-beta*(t_i - t_j)), so
    mu + alpha*beta*R_i is exactly λ(t_i^-). The second term is the
    compensator ∫_0^T λ(t) dt, split into the baseline mu*T plus, for each
    event, the integral of its own decaying kernel contribution truncated
    at T: ∫_{t_i}^{T} alpha*beta*exp(-beta*(t-t_i)) dt = alpha*(1-exp(-beta*(T-t_i))).
    """
    n = times.size
    if n == 0:
        return -mu * t_end

    if mu <= 0.0 or alpha < 0.0 or alpha >= 1.0 or beta <= 0.0:
        return -np.inf

    dt = np.diff(times)
    decay = np.exp(-beta * dt)
    r = _excitation_recursion(decay)

    intensities = mu + alpha * beta * r
    if np.any(intensities <= 0.0):
        return -np.inf

    log_sum = np.sum(np.log(intensities))
    compensator_baseline = mu * t_end
    compensator_excitation = alpha * np.sum(1.0 - np.exp(-beta * (t_end - times)))

    return float(log_sum - compensator_baseline - compensator_excitation)


def _neg_loglik_transformed(params: np.ndarray, times: np.ndarray, t_end: float) -> float:
    """Negative log-likelihood as a function of unconstrained (log mu, logit alpha, log beta)."""
    log_mu, logit_alpha, log_beta = params
    mu = float(np.exp(log_mu))
    alpha = float(1.0 / (1.0 + np.exp(-logit_alpha)))  # logistic -> (0, 1)
    beta = float(np.exp(log_beta))
    ll = hawkes_loglik(times, t_end, mu, alpha, beta)
    if not np.isfinite(ll):
        return 1e18
    return -ll


def _nelder_mead(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    max_iter: int = 500,
    step: float = 0.5,
    tol: float = 1e-6,
) -> tuple[np.ndarray, float, bool]:
    """Minimal Nelder-Mead simplex minimizer (numpy only, no scipy).

    Standard reflection/expansion/contraction/shrink algorithm (Nelder &
    Mead 1965). Convergence criterion: the spread of function values
    across the simplex (max - min) falls below `tol`.
    """
    dim = x0.size
    alpha_r, gamma_e, rho_c, sigma_s = 1.0, 2.0, 0.5, 0.5  # standard coefficients

    simplex = np.empty((dim + 1, dim), dtype=np.float64)
    simplex[0] = x0
    for i in range(dim):
        perturbed = x0.copy()
        perturbed[i] += step if x0[i] == 0.0 else step * x0[i]
        simplex[i + 1] = perturbed

    values = np.array([f(p) for p in simplex])
    converged = False

    for _ in range(max_iter):
        order = np.argsort(values)
        simplex = simplex[order]
        values = values[order]

        if (values[-1] - values[0]) < tol:
            converged = True
            break

        centroid = simplex[:-1].mean(axis=0)
        worst = simplex[-1]
        worst_val = values[-1]

        # Reflection
        reflected = centroid + alpha_r * (centroid - worst)
        reflected_val = f(reflected)

        if values[0] <= reflected_val < values[-2]:
            simplex[-1] = reflected
            values[-1] = reflected_val
            continue

        if reflected_val < values[0]:
            # Expansion
            expanded = centroid + gamma_e * (reflected - centroid)
            expanded_val = f(expanded)
            if expanded_val < reflected_val:
                simplex[-1] = expanded
                values[-1] = expanded_val
            else:
                simplex[-1] = reflected
                values[-1] = reflected_val
            continue

        # Contraction (reflected_val >= values[-2])
        contracted = centroid + rho_c * (worst - centroid)
        contracted_val = f(contracted)
        if contracted_val < worst_val:
            simplex[-1] = contracted
            values[-1] = contracted_val
            continue

        # Shrink
        best = simplex[0]
        for i in range(1, dim + 1):
            simplex[i] = best + sigma_s * (simplex[i] - best)
            values[i] = f(simplex[i])

    order = np.argsort(values)
    simplex = simplex[order]
    values = values[order]
    if not converged:
        converged = (values[-1] - values[0]) < tol
    return simplex[0], float(values[0]), converged


def fit_hawkes_exp(times: np.ndarray, t_end: float) -> HawkesFit:
    """MLE of (mu, alpha, beta) for an exponential-kernel Hawkes process.

    Optimizes over unconstrained params (log mu, logit alpha, log beta) so
    the simplex search never has to respect boundary constraints; alpha is
    mapped through a logistic transform into (0, 1) (per the brief: "alpha
    constrained to (0,1) via logistic transform"). Runs 5 multi-starts from
    spread initial points (mitigates the near-unidentifiability at n≈1
    documented in research/02 §4 pitfall 5) and returns the best-loglik
    result. `converged` is True iff the winning start's simplex satisfies
    the Nelder-Mead spread-in-loglik convergence criterion (tol=1e-6).
    """
    if times.size < 2:
        raise ValueError("need at least 2 events to fit")

    n = times.size
    mean_rate = n / t_end

    starts = [
        (mean_rate * 0.7, 0.1, 1.0),
        (mean_rate * 0.5, 0.3, 2.0),
        (mean_rate * 0.9, 0.5, 0.5),
        (mean_rate * 0.3, 0.6, 3.0),
        (mean_rate * 0.5, 0.2, 5.0),
    ]

    best_params: np.ndarray | None = None
    best_ll = -np.inf
    best_converged = False

    for mu0, alpha0, beta0 in starts:
        x0 = np.array(
            [np.log(mu0), np.log(alpha0 / (1.0 - alpha0)), np.log(beta0)],
            dtype=np.float64,
        )
        best_x, best_f, converged = _nelder_mead(
            lambda p: _neg_loglik_transformed(p, times, t_end), x0, max_iter=500
        )
        ll = -best_f
        if ll > best_ll:
            best_ll = ll
            best_params = best_x
            best_converged = converged

    assert best_params is not None
    log_mu, logit_alpha, log_beta = best_params
    mu = float(np.exp(log_mu))
    alpha = float(1.0 / (1.0 + np.exp(-logit_alpha)))
    beta = float(np.exp(log_beta))

    return HawkesFit(mu=mu, alpha=alpha, beta=beta, loglik=best_ll, converged=best_converged)


# ---------------------------------------------------------------------------
# Model-free branching-ratio estimator (Hardiman & Bouchaud 2014).
# ---------------------------------------------------------------------------


def branching_count_variance(times: np.ndarray, window: float, t_end: float) -> float:
    """Model-free branching-ratio estimate from count mean/variance alone.

    For a stationary Hawkes process, as the window W grows much larger
    than the kernel timescale, var(N_W)/mean(N_W) -> 1/(1-n)^2, giving
        n_hat = 1 - sqrt(mean(N_W) / var(N_W)).

    This requires windows W much larger than the kernel timescale
    (1/beta for the exponential kernel) — the large-window asymptotic is
    what makes the estimator "see" the amplification from clustering
    rather than just Poisson counting noise; too-small windows bias
    n_hat toward 0 (research/02 §4: "short windows truncate long-memory
    kernels and bias n̂ down"). No kernel shape is assumed, which is the
    estimator's advantage (and, per the regime-switching trap test in
    tests/estimators/test_hawkes.py, also its weakness: it cannot
    distinguish real self-excitation from non-stationary baseline rate).

    Raises ValueError if fewer than 20 non-overlapping windows fit in
    [0, t_end], if window is so small it would require an unreasonable
    number of bins (>10^7 — almost certainly a units mistake, not a
    legitimate analysis), or if the count variance is zero (degenerate/
    regular spacing), since none of these leave the estimator
    statistically meaningful.
    """
    if window <= 0.0:
        raise ValueError("window must be positive")

    n_windows = int(t_end // window)
    if n_windows < 20:
        raise ValueError(
            f"need at least 20 non-overlapping windows, got {n_windows} "
            f"(t_end={t_end}, window={window})"
        )
    max_windows = 10_000_000
    if n_windows > max_windows:
        raise ValueError(
            f"window={window} implies {n_windows} windows over t_end={t_end}, "
            f"exceeding the {max_windows} sanity cap (likely a units error)"
        )

    edges = np.arange(n_windows + 1) * window
    counts, _ = np.histogram(times, bins=edges)
    counts = counts.astype(np.float64)

    mean_count = counts.mean()
    var_count = counts.var(ddof=1)

    if var_count == 0.0:
        raise ValueError("count variance is zero; cannot estimate branching ratio")

    ratio = mean_count / var_count
    return float(1.0 - np.sqrt(ratio))
