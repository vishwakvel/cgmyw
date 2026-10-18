"""The P-measure fit: Anderson-Darling-weighted tail-probability matching
(PFIT-01/02/03), per ADR-0005 -- corrected 2026-08-07 from the originally
planned characteristic-function MLE.

Method (ADR-0005): compute empirical exceedance probabilities from the
observed daily log returns at a grid of thresholds; compute the model's tail
probabilities by normalizing the CGMYW Lévy density into a proper
probability density -- Lévy densities do NOT integrate to 1 on their own,
and skipping or botching this step is Dr. Madan's own flagged pitfall -- and
integrating past each threshold; fit by least-squares matching, weighted by
`1/(p(1-p))` so the extreme tails dominate the loss over the center (the
whole point of a tail-risk model).

Note on the normalization step in *this* implementation: psi_cgmyw is a
correctly compensated Lévy-Khintchine exponent (psi(0) == 0 exactly, per
01-01's MODEL-01 gate), so exp(psi(u))'s Fourier inversion is already very
close to a normalized PDF -- the residual gap normalize_to_pdf corrects is
FFT grid truncation/aliasing error (RESEARCH.md Pitfall 7, empirically up to
~1e-4 for in-bounds parameters on this grid), not literally "the Lévy
density has infinite mass" (that describes the Lévy *measure* ν(dx), a
different object from what gets FFT-inverted here). The correction step is
still required and cheap insurance regardless of the magnitude: an
optimizer excursion near a bound or a badly-conditioned draw could widen
that gap, and normalize_to_pdf is the one place that gets checked.

Module constants
-----------------
N_DENSITY, DX_DENSITY: the inverse-FFT grid. n=2**14, dx=2.5e-4 give a
    log-return half-width of n*dx/2 = 2.048 (about +/-205% in one day, far
    past any real observation, so truncation cannot bite) at a resolution of
    0.025% per grid point. The induced Fourier spacing is
    du = 2*pi/(n*dx) ~= 1.534 with u_max ~= 1.26e4. Checked against a 16x
    larger, 10x finer grid: tail probabilities agree to ~1e-5 across the
    bounds, including the published Y ~ 0.003 fits
    (test_tail_probabilities_converged_on_default_grid).
DENSITY_FLOOR: 1e-300, the clip applied before any log/divide inside
    normalize_to_pdf, so FFT roundoff producing a tiny negative value cannot
    propagate into the tail-probability integral as a NaN.
MIN_RETURNS: 60. Eight free parameters fitted to fewer than 60 daily
    observations is not a defensible estimate (D-04/PFIT-02's fail-fast
    principle, applied to insufficient data just as it applies to
    non-convergence).
MULTISTART_SEED: fixes the multi-start L-BFGS-B draws (D-07/T-02-06) so a
    reported parameter estimate is reproducible -- an unseeded multi-start
    makes reproducibility, a stated property of a defensible fit, impossible.
THRESHOLD_GRID: a module-level default threshold grid computed once (at
    import time) from a fixed synthetic representative return sample, so
    ad_weighted_objective has a sane default for standalone use/testing.
    D-11 requires the *real* threshold grid to be built from empirical
    quantiles of each window's own return sample, not a fixed hardcoded
    set -- fit_p_measure does this itself via compute_threshold_grid(returns)
    and passes the window-specific grid explicitly, rather than relying on
    this default. Thresholds close to the median get a large p(1-p)
    denominator and thus small AD weight, which is intentional -- the whole
    point is de-emphasizing the center.
P_BOUNDS: L-BFGS-B bounds per RESEARCH.md assumption A4, applied
    independently per tail. Each Y* interval is deliberately a single
    continuous span across the pole at 1.0 rather than two split intervals
    -- landing exactly on 1.0 has probability zero under a continuous
    optimizer, and ad_weighted_objective's +inf guard handles the near-pole
    region. Do not "fix" this later by splitting the interval.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import gamma

from .model import psi_cgmyw

N_DENSITY: int = 2**14
DX_DENSITY: float = 2.5e-4
DENSITY_FLOOR: float = 1e-300
MIN_RETURNS: int = 60
MULTISTART_SEED: int = 20260805

P_BOUNDS: dict[str, tuple[float, float]] = {
    "Cn": (1e-4, 50.0),
    "Cp": (1e-4, 50.0),
    # G/M up to 300 and Y down to 1e-3: Madan & Wang's published fits on the
    # same ten tickers (Table 1) have M up to ~132 and Y down to ~2e-4, which
    # the original (G,M <= 100, Y >= 0.02) box could not reach.
    "G": (1e-3, 300.0),
    "M": (1e-3, 300.0),
    "Yn": (1e-3, 1.98),
    "Yp": (1e-3, 1.98),
    "Wn": (1e-3, 50.0),
    "Wp": (1e-3, 50.0),
}
# Fixed parameter order (Cn, Cp, G, M, Yn, Yp, Wn, Wp) -- pinned in exactly
# one place since psi_cgmyw, ad_weighted_objective, and fit_p_measure all
# depend on the same ordering.
PARAM_ORDER: tuple[str, ...] = tuple(P_BOUNDS.keys())

# Three structured multi-start seeds (D-07), each a plausible-magnitude
# in-bounds guess well clear of every documented pole, chosen to give the
# optimizer a genuine shot at three qualitatively different regions of the
# 8-dimensional space rather than relying on log-uniform luck to find them:
#
# _CENTRAL_START: a "classical CGMY-ish" guess, moderate G/M, moderate Y,
#   small W (Wn+Yn = 0.42+0.6 = 1.02, not the 1.0 pole).
# _TAIL_HEAVY_START: near-zero Y with large G/M/W -- the corner an
#   independently published CGMYW fit on the same ten tickers actually lands
#   in (Madan & Wang, "Asset Pricing, Measure Changes, and a New Lévy Model",
#   2026, Table 1; AAPL there: Yn=Yp~0.003, G~52, M~98, Wn~22.3, Wp~4.8, on a
#   different, longer, fixed window than this project's rolling 252-day one).
#   Our own AAPL fits on 252-day windows land in different regimes window
#   to window (Y ~ 1.67 symmetric on 2023-01-13; Yn ~ 0.15 / Yp ~ 0.86 on
#   2025-08-29; see ADR-0008) -- a large gap
#   for the *same* general model and estimation family. This start exists so
#   that gap is a genuine finding about the window/objective, not an
#   unexplored corner of the multi-start.
# _HIGH_Y_START: Y > 1 (infinite-variation regime), on the other side of the
#   Gamma(-Y) pole at Y=1 from both of the above -- log-uniform draws land on
#   either side of that pole by chance, but nothing previously guaranteed the
#   optimizer actually explored the Y>1 side structurally.
_CENTRAL_START: dict[str, float] = dict(
    Cn=1.0, Cp=1.0, G=12.0, M=15.0, Yn=0.6, Yp=0.6, Wn=0.42, Wp=0.42
)
_TAIL_HEAVY_START: dict[str, float] = dict(
    Cn=0.5, Cp=0.5, G=50.0, M=90.0, Yn=0.05, Yp=0.05, Wn=15.0, Wp=5.0
)
_HIGH_Y_START: dict[str, float] = dict(
    Cn=1.0, Cp=1.0, G=15.0, M=15.0, Yn=1.5, Yp=1.5, Wn=1.0, Wp=1.0
)
# Ordered so start 0 is always the original central seed (keeps
# fit_p_measure's start-index diagnostics stable for n_starts>=1); starts 1
# and 2 are the two additional structured regimes when n_starts>=3.
_FIXED_STARTS: tuple[dict[str, float], ...] = (
    _CENTRAL_START,
    _TAIL_HEAVY_START,
    _HIGH_Y_START,
)

# Default quantiles for compute_threshold_grid (D-11). Widened from the
# original 8 (1/2/5/10/90/95/98/99th percentiles) to 14, adding resolution
# between the 1st/99th (kept as the outermost points -- more extreme than
# that starts running out of data at a 252-day window, ~2-3 points at the
# 1st percentile already) and the median. Motivation: 8 free parameters
# matched against exactly 8 threshold conditions is exactly determined, the
# textbook recipe for a non-unique fit -- confirmed empirically, not just in
# the abstract, by the gap against the independently published AAPL fit
# noted above (same general method, very different parameter values, both
# presumably tracking their own tail data reasonably). More conditions than
# parameters (over-identification) is the standard fix; it does not
# guarantee a unique optimum but gives the loss less freedom to be
# indifferent between very different parameter vectors.
DEFAULT_THRESHOLD_QUANTILES: tuple[float, ...] = (
    0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15,
    0.85, 0.90, 0.925, 0.95, 0.97, 0.98, 0.99,
)


def _fft_invert_characteristic_function(phi, n: int = N_DENSITY, dx: float = DX_DENSITY):
    """Centred inverse-FFT: given a characteristic function phi(u), return
    (x, density) approximating density(x) = (1/2pi) * integral phi(u)
    exp(-i*u*x) du on a grid.

    du = 2*pi/(n*dx); u = (arange(n) - n//2) * du; x = (arange(n) - n//2)*dx.
    The ifftshift/fftshift pair is what makes this the correct centred DFT
    pair for that integral convention; a plain np.fft.fft without it
    introduces an index-dependent phase error that looks like a smeared or
    shifted density.

    This is the single FFT-inversion code path shared by levy_density_grid
    (a CGMYW-specific caller) and, independently, by the test suite's own
    Gaussian characteristic-function closure -- gating the FFT machinery
    against a closed-form density (scipy.stats.norm.pdf) before it is
    trusted on a Lévy density that does not integrate to 1 on its own.
    """
    du = 2 * np.pi / (n * dx)  # Fourier-grid spacing implied by the space-grid spacing
    u = (np.arange(n) - n // 2) * du  # frequency grid, centred at u=0
    x = (np.arange(n) - n // 2) * dx  # space grid (log-return values), centred at x=0
    phi_u = phi(u)  # evaluate the characteristic function at every frequency-grid point
    # ifftshift moves the zero-frequency term back to index 0 (FFT's convention)
    # before the transform, fftshift moves the zero-space term back to the
    # centre afterward -- this pair is what makes the discrete sum equal the
    # continuous inversion integral on a grid centred at 0 instead of introducing
    # a spurious index-dependent phase. np.real drops FFT roundoff's tiny
    # imaginary part (the true answer is real since phi is a valid char. fn.).
    density = (du / (2 * np.pi)) * np.real(
        np.fft.fftshift(np.fft.fft(np.fft.ifftshift(phi_u)))
    )
    return x, density


def levy_density_grid(Cn, Cp, G, M, Yn, Yp, Wn, Wp, n: int = N_DENSITY, dx: float = DX_DENSITY):
    """Raw (NOT yet a probability density) CGMYW Lévy density on a grid,
    via the centred inverse-FFT of exp(psi_cgmyw(u, ...)) at dt=1.

    Unlike an MLE approach's exp(dt*psi(u)), this is the raw Lévy-density
    inversion at dt=1 -- the tail-matching method's own normalization step
    (normalize_to_pdf), not a dt scaling, is what turns this into a
    comparable probability object.

    Returns (x, density) WITHOUT normalizing or flooring -- that is
    normalize_to_pdf's job, kept separate so the test suite can prove the
    raw output genuinely is not a PDF before trusting the correction.
    """

    def phi(u):
        # exp(psi(u)) is the characteristic function E[e^{iuX}] at dt=1 --
        # psi_cgmyw returns the LOG of it (the Levy exponent), so exponentiate.
        return np.exp(psi_cgmyw(u, Cn, Cp, G, M, Yn, Yp, Wn, Wp))

    return _fft_invert_characteristic_function(phi, n=n, dx=dx)


def normalize_to_pdf(x, levy_density):
    """Normalize a raw Lévy density into a proper probability density.

    Lévy densities do not integrate to 1 by construction (ADR-0005; flagged
    directly by Dr. Madan as easy to get wrong) -- skipping this step, or
    normalizing incorrectly, silently invalidates every downstream tail
    probability. Clips to DENSITY_FLOOR before integrating (FFT roundoff can
    produce tiny negative values), integrates via trapezoid, and divides by
    that integral.

    Returns (x, pdf, normalization_constant) where pdf integrates to 1
    within tolerance and normalization_constant (the raw integral divided
    out) is positive and finite. Raises ValueError if the raw density is
    degenerate (non-finite or non-positive integral) -- callers in the hot
    optimizer path (ad_weighted_objective) catch this and return +inf.
    """
    x = np.asarray(x, dtype=float)
    # Floor before integrating: FFT roundoff can leave a handful of grid points
    # very slightly negative (they should be exactly 0), and a negative value
    # flowing into a later log/divide would produce a silent NaN.
    clipped = np.clip(np.asarray(levy_density, dtype=float), DENSITY_FLOOR, None)
    integral = float(np.trapezoid(clipped, x))  # total probability mass on the grid
    if not np.isfinite(integral) or integral <= 0.0:
        # A degenerate density (e.g. from an optimizer probe near a pole) --
        # let the caller (ad_weighted_objective) turn this into +inf.
        raise ValueError(
            f"non-normalizable density: raw integral={integral!r} is not "
            "strictly positive and finite"
        )
    pdf = clipped / integral  # rescale so the density now integrates to exactly 1
    return x, pdf, integral


def model_tail_probabilities(pdf_x, pdf, thresholds):
    """Model tail probabilities P(X>t) for t>=0, P(X<t) for t<0 (same
    convention as empirical_tail_probabilities), from a normalized PDF.

    Built from the cumulative-trapezoid CDF, interpolated at each exact
    threshold. Integrating a mask that starts at the first grid point past t
    instead drops the sliver [t, next grid point] -- up to dx * pdf(t), about
    0.005 at dx = 2.5e-4 near the center of a daily-return density, which is
    as large as the fit's own RMSE.
    """
    pdf_x = np.asarray(pdf_x, dtype=float)
    pdf = np.asarray(pdf, dtype=float)
    thresholds = np.atleast_1d(np.asarray(thresholds, dtype=float))
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(pdf_x))])
    at_t = np.interp(thresholds, pdf_x, cdf)  # P(X < t), 0 / total mass outside the grid
    return np.where(thresholds >= 0, cdf[-1] - at_t, at_t)


def empirical_tail_probabilities(returns, thresholds):
    """Empirical tail probabilities computed directly from the sample, same
    convention as model_tail_probabilities: P(return>t) for t>=0,
    P(return<t) for t<0. Pure numpy, no special functions."""
    returns = np.asarray(returns, dtype=float)
    thresholds = np.atleast_1d(np.asarray(thresholds, dtype=float))
    probs = np.empty_like(thresholds)
    for i, t in enumerate(thresholds):
        if t >= 0:
            probs[i] = np.mean(returns > t)
        else:
            probs[i] = np.mean(returns < t)
    return probs


def ad_weight(p_empirical):
    """Anderson-Darling weight 1/(p(1-p)), the mechanism that overweights
    the extreme tails relative to the center (ADR-0005). p is clamped into
    [1e-6, 1-1e-6] before the weight: an observed 0-or-1 empirical tail
    probability at a threshold is a small-sample artifact, not infinite
    information. Exposed standalone so its tail-vs-center behavior can be
    unit tested in isolation, not just inferred from a fit result."""
    p = np.clip(np.asarray(p_empirical, dtype=float), 1e-6, 1 - 1e-6)
    return 1.0 / (p * (1 - p))


def compute_threshold_grid(returns, quantiles=DEFAULT_THRESHOLD_QUANTILES):
    """Build a threshold grid from empirical quantiles of a return sample
    (D-11) -- keeps the grid meaningful across different windows/stocks
    without per-stock tuning, rather than a fixed hardcoded set."""
    returns = np.asarray(returns, dtype=float)
    return np.quantile(returns, quantiles)


# A fixed synthetic representative sample (not real data -- pfit.py has no
# import-time dependency on data.py or any file on disk) gives
# THRESHOLD_GRID a deterministic default value for ad_weighted_objective's
# signature and standalone testing.
_REPRESENTATIVE_SAMPLE = np.random.default_rng(MULTISTART_SEED).normal(0.0, 0.02, 5000)
THRESHOLD_GRID: np.ndarray = compute_threshold_grid(_REPRESENTATIVE_SAMPLE)


def ad_weighted_objective(theta, returns, thresholds=THRESHOLD_GRID):
    """Anderson-Darling-weighted tail-probability-matching objective
    (ADR-0005). theta = (Cn, Cp, G, M, Yn, Yp, Wn, Wp).

    Returns +inf (never NaN) whenever: an exception is raised while
    evaluating the density (ValueError/FloatingPointError/ZeroDivisionError
    -- e.g. from normalize_to_pdf's own degeneracy check), the raw or
    normalized density contains a non-finite value, or the resulting loss
    itself is non-finite. This is the mechanism that lets L-BFGS-B step
    cleanly away from the Yn/Yp in {0,1} and integer-Wn+Yn/Wp+Yp poles
    (psi_cgmyw deliberately does not trap these -- see model.py) and from
    non-normalizable densities. Returning NaN instead would corrupt the
    entire optimizer trajectory silently.
    """
    try:
        # theta's order must match PARAM_ORDER/P_BOUNDS exactly -- this unpack
        # is the one place that ordering actually gets consumed.
        Cn, Cp, G, M, Yn, Yp, Wn, Wp = theta
        # Step 1: this candidate parameter vector -> a raw (not-yet-normalized) density.
        x, raw_density = levy_density_grid(Cn, Cp, G, M, Yn, Yp, Wn, Wp)
        if not np.all(np.isfinite(raw_density)):
            # psi_cgmyw hit a pole (e.g. Yn==1, or Wn+Yn an integer) -- reject
            # this point rather than let a NaN reach the optimizer.
            return np.inf
        # Step 2: normalize into a proper PDF (raises ValueError on a
        # degenerate density, caught below and converted to +inf).
        pdf_x, pdf, norm_const = normalize_to_pdf(x, raw_density)
        if not np.isfinite(norm_const) or norm_const <= 0.0:
            return np.inf
        if not np.all(np.isfinite(pdf)):
            return np.inf

        # Step 3: model vs. empirical tail probabilities at the same thresholds.
        p_model = model_tail_probabilities(pdf_x, pdf, thresholds)
        p_empirical = empirical_tail_probabilities(returns, thresholds)
        # Step 4: AD-weighted squared error -- weights push the loss toward
        # matching the extreme thresholds tightly, at the cost of caring less
        # about thresholds near the median.
        weights = ad_weight(p_empirical)
        loss = float(np.sum(weights * (p_model - p_empirical) ** 2))
        if not np.isfinite(loss):
            return np.inf
        return loss
    except (ValueError, FloatingPointError, ZeroDivisionError):
        # Catches normalize_to_pdf's explicit raise plus any stray numeric
        # error from an extreme/degenerate theta -- always +inf, never let an
        # exception propagate up into scipy.optimize.minimize.
        return np.inf


def trailing_window_returns(dates, closes, end_date, window: int = 252):
    """The trailing P-fit window: min(252, available trading days) ending at
    end_date (D-04, ADR-0003). An expanding window until 252 days
    accumulate, never synthetic or extrapolated history -- the returned
    window is always a contiguous slice of the input dates/closes, never
    padded or interpolated.

    AAPL's 667-day history means every plausible date in this phase gets a
    full 252-day window -- the fallback below exists for Phase 2's earlier-
    history dates, and is exercised in this module's tests via a synthetic
    date/close series rather than by the real AAPL run.

    Locates the largest index with dates[idx] <= end_date (the most recent
    date at or before end_date, if end_date itself is absent). Raises
    ValueError if end_date precedes every date in the series, or if the
    resulting window yields fewer than MIN_RETURNS returns -- eight free
    parameters fitted to too little data is not a defensible estimate, and
    the fail-fast principle applies to insufficient data exactly as it does
    to non-convergence (D-08).
    """
    dates = np.asarray(dates, dtype=float)
    closes = np.asarray(closes, dtype=float)

    # searchsorted(..., side="right") - 1 finds the rightmost index whose date
    # is <= end_date -- i.e. "the most recent trading day at or before the
    # requested date," which also correctly handles end_date not being an
    # exact match (e.g. a weekend/holiday).
    idx = int(np.searchsorted(dates, end_date, side="right")) - 1
    if idx < 0:
        raise ValueError(
            f"end_date {end_date!r} is earlier than the series start "
            f"{dates[0]!r}; no trailing window exists"
        )

    available = idx + 1  # how many trading days exist at or before end_date
    take = min(window, available)  # D-04: min(252, available) -- the expanding-window rule
    # Slice backward from idx: [idx-take+1, idx] inclusive, i.e. the `take`
    # most recent days ending exactly at idx. This is always a contiguous
    # block of real rows -- never padded, never interpolated.
    window_dates = dates[idx - take + 1 : idx + 1]
    window_closes = closes[idx - take + 1 : idx + 1]
    assert len(window_dates) == take, (
        "trailing_window_returns must return a contiguous, unpadded slice "
        "(D-04's 'never synthetic history' clause)"
    )

    returns = np.diff(np.log(window_closes))  # `take` closes -> `take - 1` log returns
    if len(returns) < MIN_RETURNS:
        raise ValueError(
            f"trailing window ending {end_date!r} yields only {len(returns)} "
            f"return(s), fewer than MIN_RETURNS={MIN_RETURNS}; refusing to "
            "fit eight parameters to too little data"
        )
    return window_dates, returns


@dataclass(frozen=True)
class PFitResult:
    """The fitted P-measure parameters plus fit-quality diagnostics.

    as_tuple() returns (Cn, Cp, G, M, Yn, Yp, Wn, Wp) for direct hand-off to
    plan 01-03's report and, eventually, Phase 1.5's Q-calibration.
    """

    Cn: float
    Cp: float
    G: float
    M: float
    Yn: float
    Yp: float
    Wn: float
    Wp: float
    ad_objective: float
    n_returns: int
    window_start_date: int | None
    window_end_date: int | None
    starts_attempted: int
    starts_succeeded: int

    def as_tuple(self) -> tuple[float, float, float, float, float, float, float, float]:
        return (self.Cn, self.Cp, self.G, self.M, self.Yn, self.Yp, self.Wn, self.Wp)


def _variance_matched(start: dict[str, float], variance: float) -> tuple[float, ...]:
    """Rescale a seed's Cn, Cp so its variance per unit time is `variance`.

    The fixed seeds fix the *shape* (G, M, Y, W); their C values are only
    right for one return scale, and a seed off by 10x in variance sits on a
    plateau where every tail probability saturates and L-BFGS-B stops at
    once. Variance uses the CGMY moment C*Gamma(2-Y)*G^(Y-2) per side with
    the tilt folded in as G+W ((1+x)^-W ~ e^-Wx where the variance lives) --
    an approximation, good enough for a starting point.
    """
    s = dict(start)
    per_unit_c = gamma(2 - s["Yn"]) * (s["G"] + s["Wn"]) ** (s["Yn"] - 2) + gamma(2 - s["Yp"]) * (
        s["M"] + s["Wp"]
    ) ** (s["Yp"] - 2)
    s["Cn"] = s["Cp"] = float(np.clip(variance / per_unit_c, *P_BOUNDS["Cn"]))
    return tuple(s[name] for name in PARAM_ORDER)


def _log_ad_objective(z, returns, thresholds):
    """ad_weighted_objective in log-parameter space. Parameters span 1e-3..1e2
    (C ~ 1e-2, G ~ 1e2), so a level-space finite-difference gradient is
    badly scaled: on AAPL the level-space fit returned its seed G=50, M=90,
    W=15 essentially unchanged. In log space every coordinate is relative."""
    return ad_weighted_objective(np.exp(z), returns, thresholds)


def _log_uniform_draw(rng, bounds: tuple[float, float]) -> float:
    """Log-uniform draw inside (lo, hi) -- spreads probes across the several
    orders of magnitude the C/G/M/W parameters span far better than
    uniform-in-level."""
    lo, hi = bounds
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def fit_p_measure(returns, n_starts: int = 8, seed: int = MULTISTART_SEED, window_dates=None):
    """Multi-start bounded L-BFGS-B P-fit via Anderson-Darling-weighted
    tail-probability matching (ADR-0005, D-07/D-08/D-09).

    The threshold grid is recomputed from this window's own returns
    (compute_threshold_grid, D-11), not the module-level THRESHOLD_GRID
    default, and passed explicitly to ad_weighted_objective for every start.

    Optimization runs on log(theta) (_log_ad_objective) and each fixed
    seed's Cn, Cp are rescaled to the sample variance (_variance_matched).

    Starts 0-2 are the fixed, structured seeds in _FIXED_STARTS -- a
    classical-CGMY-ish guess, a near-zero-Y/large-G,M,W guess (the region an
    independently published fit of this same model landed in for the same
    tickers), and a Y>1 guess on the other side of the Gamma(-Y) pole -- each
    well clear of every documented pole. Remaining starts (n_starts-3, or all
    of them if n_starts<3, in which case only the first n_starts fixed seeds
    are used) are log-uniform draws inside P_BOUNDS from a seeded
    np.random.default_rng(seed) -- seeding is load-bearing, since an
    unseeded multi-start makes a reported parameter estimate irreproducible,
    and reproducibility is a stated property of a defensible fit.

    `returns` is used as-is, not de-meaned: the model has no separate drift
    parameter, so whatever mean the fitted (Cn,Cp,G,M,Yn,Yp,Wn,Wp) imply is
    an output of the fit, not something stripped out beforehand. This is
    deliberate, not an oversight -- the source note frames the P-fit as
    carrying a drift that is "then risk-neutralized" in the Q-step, which
    only makes sense if that drift survives the P-fit.

    A start is accepted only when the optimizer reports success, the final
    objective is finite, and every parameter component is inside P_BOUNDS.
    Among accepted starts, the lowest objective wins (D-07). If no start is
    accepted, raises RuntimeError enumerating all n_starts with each one's
    rejection reason (D-08) -- no least-bad fallback, no bound-widening, no
    retry (D-09).

    Also raises ValueError up front if `returns` contains any non-finite
    (NaN/inf) value: comparisons against NaN silently evaluate False
    throughout empirical_tail_probabilities/model_tail_probabilities, which
    would let a malformed input sample produce a plausible-looking (but
    meaningless) finite loss instead of failing loudly -- fail-fast applies
    to malformed input data exactly as it does to non-convergence (D-08).

    Emits a RuntimeWarning naming any parameter within 1e-6 of its P_BOUNDS
    edge -- a bound-pinned "convergence" is a bounds artifact, not a
    converged optimum, and surfacing it here is what lets plan 01-03's demo
    run notice it.
    """
    returns = np.asarray(returns, dtype=float)
    if not np.all(np.isfinite(returns)):
        raise ValueError(
            "returns array contains non-finite (NaN/inf) values; refusing "
            "to fit the P-measure against malformed data"
        )

    thresholds = compute_threshold_grid(returns)  # D-11: this window's own empirical quantiles
    bounds = [P_BOUNDS[name] for name in PARAM_ORDER]  # bounds list in the fixed param order
    log_bounds = [(np.log(lo), np.log(hi)) for lo, hi in bounds]  # the optimizer works on log(theta)

    # Build all n_starts initial guesses up front: the first min(n_starts, 3)
    # are the fixed, structured seeds in _FIXED_STARTS (in order, so start 0
    # is always _CENTRAL_START regardless of n_starts); any remaining starts
    # are seeded log-uniform draws -- same rng instance across all of them,
    # so re-running with the same seed reproduces the exact same starting
    # points every time.
    rng = np.random.default_rng(seed)
    starts: list[tuple[float, ...]] = [
        _variance_matched(fixed, float(np.var(returns))) for fixed in _FIXED_STARTS[:n_starts]
    ]
    n_random = max(0, n_starts - len(starts))
    for _ in range(n_random):
        starts.append(tuple(_log_uniform_draw(rng, P_BOUNDS[name]) for name in PARAM_ORDER))

    records = []
    accepted = []
    for i, x0 in enumerate(starts):
        # Run L-BFGS-B from this one starting point -- bounds enforce the
        # theoretical/practical parameter ranges directly during optimization,
        # not just as a post-hoc filter.
        z0 = np.log(np.array(x0, dtype=float))
        start_fun = float(_log_ad_objective(z0, returns, thresholds))
        result = minimize(
            _log_ad_objective,
            x0=z0,
            args=(returns, thresholds),
            method="L-BFGS-B",
            bounds=log_bounds,
        )
        fun = float(result.fun)
        # A start only counts as "accepted" if all three hold: it converged,
        # the final objective/params are finite (not the +inf a pole would
        # produce), and every param actually sits inside its bound (L-BFGS-B
        # can occasionally return something just outside due to
        # floating-point boundary handling). "Converged" includes L-BFGS-B's
        # ABNORMAL line-search stop when it improved on its start: that stop
        # is finite-difference gradient noise at the bottom of a basin (on
        # AAPL it was the best start), whereas a stop that never left its
        # start is a plateau and stays rejected.
        in_bounds = all(lo <= v <= hi for v, (lo, hi) in zip(result.x, log_bounds))
        finite = np.isfinite(fun) and bool(np.all(np.isfinite(result.x)))
        # Back to level space; clip absorbs exp(log(bound)) round-off.
        theta = np.clip(np.exp(result.x), *np.array(bounds).T) if finite else result.x
        converged = bool(result.success) or (
            "ABNORMAL" in str(result.message) and np.isfinite(fun) and fun < start_fun
        )
        accepted_start = converged and finite and in_bounds

        record = {
            "start_index": i,
            "success": bool(result.success),
            "message": str(result.message),
            "fun": fun if np.isfinite(fun) else float("inf"),
            "x": tuple(float(v) for v in theta),
            "in_bounds": in_bounds,
            "finite": finite,
            "accepted": accepted_start,
        }
        records.append(record)  # kept regardless of acceptance, for the failure-diagnostics message
        if accepted_start:
            accepted.append(record)

    if not accepted:
        lines = [
            f"P-fit failed: none of {n_starts} L-BFGS-B starts converged to "
            "a finite, in-bounds result. Per-start diagnostics:"
        ]
        for r in records:
            reasons = []
            if not r["success"]:
                reasons.append(f"non-convergent ({r['message']})")
            if not r["finite"]:
                reasons.append("non-finite objective/parameters")
            if not r["in_bounds"]:
                reasons.append("out-of-bounds parameters")
            lines.append(f"  start {r['start_index']}: " + ("; ".join(reasons) or "unknown rejection"))
        raise RuntimeError("\n".join(lines))

    # Among every accepted start, take the one with the lowest AD-weighted
    # objective (D-07) -- with 8 different starting points, the multi-start
    # protects against L-BFGS-B settling into a bad local optimum from any
    # single starting guess.
    best = min(accepted, key=lambda r: r["fun"])
    params = dict(zip(PARAM_ORDER, best["x"]))

    # A fitted value sitting right at a P_BOUNDS edge didn't actually
    # converge to an interior optimum -- it just ran into the wall. That's
    # not necessarily wrong, but it's a signal worth surfacing rather than
    # reporting silently as an ordinary converged fit.
    for name, val in params.items():
        lo, hi = P_BOUNDS[name]
        if abs(val - lo) < 1e-6 or abs(val - hi) < 1e-6:
            warnings.warn(
                f"P-fit parameter {name}={val!r} sits within 1e-6 of its "
                f"P_BOUNDS edge {(lo, hi)!r} -- this is a bounds artifact, "
                "not a converged optimum.",
                RuntimeWarning,
                stacklevel=2,
            )

    window_start_date = int(window_dates[0]) if window_dates is not None else None
    window_end_date = int(window_dates[-1]) if window_dates is not None else None

    return PFitResult(
        **params,
        ad_objective=best["fun"],
        n_returns=len(returns),
        window_start_date=window_start_date,
        window_end_date=window_end_date,
        starts_attempted=n_starts,
        starts_succeeded=len(accepted),
    )
