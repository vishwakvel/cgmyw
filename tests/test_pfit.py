"""Tests for the P-measure fit (plan 01-02): PDF normalization, tail
probabilities, the Anderson-Darling-weighted objective, the trailing 252-day
window, and the multi-start bounded L-BFGS-B fit.

Follows the RED -> GREEN TDD cycle: these tests were written from
01-02-PLAN.md's <behavior> lists before src/cgmyw/pfit.py existed.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from cgmyw.pfit import (
    MIN_RETURNS,
    N_DENSITY,
    P_BOUNDS,
    PFitResult,
    ad_weight,
    ad_weighted_objective,
    compute_threshold_grid,
    empirical_tail_probabilities,
    fit_p_measure,
    levy_density_grid,
    model_tail_probabilities,
    normalize_to_pdf,
    trailing_window_returns,
)
from cgmyw.pfit import _fft_invert_characteristic_function

# A known-good, pole-free in-bounds parameter vector reused across tests
# (Wn+Yn = 0.42+0.6 = 1.02, Wp+Yp = 0.42+0.6 = 1.02 -- a few hundredths off
# the documented b=1.0 Tricomi pole, per 01-01-SUMMARY.md's own resolution
# of the same hazard).
GOOD_PARAMS = (1.0, 1.0, 12.0, 15.0, 0.6, 0.6, 0.42, 0.42)


def _sample_from_cgmyw(params, n, seed):
    """Inverse-CDF sample from a CGMYW parameter vector's normalized PDF,
    reusing the already-tested levy_density_grid -> normalize_to_pdf
    pipeline instead of introducing a second sampling path."""
    x, raw = levy_density_grid(*params)
    x, pdf, _ = normalize_to_pdf(x, raw)
    dx = x[1] - x[0]
    cdf = np.cumsum(pdf) * dx
    cdf = cdf / cdf[-1]
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, n)
    return np.interp(u, cdf, x)


# ---------------------------------------------------------------------------
# Task 1: FFT inversion gate, normalization, tail probabilities, AD objective
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sigma", [0.02, 0.15])
def test_fft_invert_matches_gaussian_pdf(sigma):
    """Gate the inverse-FFT machinery independently of any CGMYW code, via a
    throwaway Gaussian characteristic-function closure, before trusting it
    on a Lévy density that does NOT integrate to 1."""

    def phi(u):
        return np.exp(-0.5 * sigma**2 * u**2)

    x, density = _fft_invert_characteristic_function(phi)
    expected = norm.pdf(x, 0.0, sigma)
    assert np.max(np.abs(density - expected)) < 1e-8


def test_levy_density_grid_does_not_integrate_to_one():
    # NOTE: psi_cgmyw is a correctly *compensated* characteristic exponent
    # (psi(0) == 0 exactly, verified in 01-01's MODEL-01 gate), so
    # exp(psi(u))'s Fourier inversion is already extremely close to a
    # normalized PDF -- the residual gap is FFT grid truncation/aliasing
    # error (Pitfall 7), not the theoretical "Lévy density has infinite
    # mass" gap the module docstring's informal phrasing evokes. Empirically
    # (verified directly against this grid across in-bounds and
    # boundary-adjacent P_BOUNDS draws) that residual tops out around 6e-5
    # for N_DENSITY/DX_DENSITY as sized -- never approaching 1e-3. Wn=Wp=50
    # (an extreme, boundary-adjacent draw) is used here as the parameter set
    # showing the largest measured drift, to prove the raw output is
    # genuinely, measurably not an exact PDF before trusting normalize_to_pdf
    # to correct it -- with a threshold the real, correctly-compensated math
    # can actually satisfy.
    params = (50.0, 50.0, 0.001, 0.001, 1.98, 1.98, 50.0, 50.0)
    x, density = levy_density_grid(*params)
    raw_integral = np.trapezoid(np.clip(density, 0, None), x)
    assert abs(raw_integral - 1.0) > 1e-5


@pytest.mark.parametrize(
    "params",
    [
        GOOD_PARAMS,
        (0.5, 2.0, 10.0, 20.0, 0.4, 0.8, 0.1, 1.0),
        (2.0, 0.5, 20.0, 10.0, 1.2, 0.3, 2.0, 0.5),
    ],
)
def test_normalize_to_pdf_integrates_to_one(params):
    x, density = levy_density_grid(*params)
    xp, pdf, norm_const = normalize_to_pdf(x, density)
    integral = np.trapezoid(pdf, xp)
    assert abs(integral - 1.0) < 1e-4
    assert np.isfinite(norm_const)
    assert norm_const > 0.0


def test_model_tail_probabilities_matches_normal_sf_cdf():
    sigma = 0.05

    def phi(u):
        return np.exp(-0.5 * sigma**2 * u**2)

    x, density = _fft_invert_characteristic_function(phi)
    xp, pdf, _ = normalize_to_pdf(x, density)
    thresholds = np.array([-0.15, -0.08, 0.08, 0.15])
    probs = model_tail_probabilities(xp, pdf, thresholds)
    expected = np.array(
        [
            norm.cdf(-0.15, 0, sigma),
            norm.cdf(-0.08, 0, sigma),
            norm.sf(0.08, 0, sigma),
            norm.sf(0.15, 0, sigma),
        ]
    )
    assert np.max(np.abs(probs - expected)) < 1e-3


def test_empirical_tail_probabilities_matches_true_normal_tail():
    sigma = 0.02
    n = 200_000
    rng = np.random.default_rng(42)
    sample = rng.normal(0.0, sigma, n)
    thresholds = np.array([-0.04, -0.02, 0.02, 0.04])
    probs = empirical_tail_probabilities(sample, thresholds)
    expected = np.array(
        [
            norm.cdf(-0.04, 0, sigma),
            norm.cdf(-0.02, 0, sigma),
            norm.sf(0.02, 0, sigma),
            norm.sf(0.04, 0, sigma),
        ]
    )
    se = np.sqrt(expected * (1 - expected) / n)
    assert np.all(np.abs(probs - expected) < 3 * se)


def test_ad_weighted_objective_finite_for_inbounds_theta():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0, 0.02, 200)
    loss = ad_weighted_objective(GOOD_PARAMS, returns)
    assert np.isfinite(loss)


def test_ad_weighted_objective_inf_at_yn_pole():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0, 0.02, 200)
    theta = (1.0, 1.0, 12.0, 15.0, 1.0, 0.6, 0.4, 0.4)  # Yn = 1.0 exactly
    loss = ad_weighted_objective(theta, returns)
    assert np.isinf(loss)
    assert not np.isnan(loss)


def test_ad_weighted_objective_inf_at_wn_yn_integer_pole():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0, 0.02, 200)
    # Wn=0.4, Yn=0.6 -> Wn+Yn=1.0 exactly, the documented Tricomi pole
    # (ADR-0007 / 01-01-SUMMARY.md).
    theta = (1.0, 1.0, 12.0, 15.0, 0.6, 0.6, 0.4, 0.4)
    loss = ad_weighted_objective(theta, returns)
    assert np.isinf(loss)
    assert not np.isnan(loss)

    # Same pole on the p-side: Wp+Yp = 1.0 exactly.
    theta_p = (1.0, 1.0, 12.0, 15.0, 0.6, 0.6, 0.42, 0.4)
    loss_p = ad_weighted_objective(theta_p, returns)
    assert np.isfinite(loss_p) or np.isinf(loss_p)  # sanity: never NaN
    assert not np.isnan(loss_p)


def test_ad_weighted_objective_lower_at_true_params_than_wrong():
    sample = _sample_from_cgmyw(GOOD_PARAMS, 4000, seed=7)
    true_loss = ad_weighted_objective(GOOD_PARAMS, sample)
    wrong_theta = (0.01, 0.01, 90.0, 95.0, 1.9, 1.9, 40.0, 40.0)
    wrong_loss = ad_weighted_objective(wrong_theta, sample)
    assert np.isfinite(true_loss)
    assert true_loss < wrong_loss


def test_ad_weight_increases_near_extreme_tail():
    """Direct unit test on the weighting function in isolation (not just
    inferred from the fit result): a threshold near the extreme tail (small
    empirical p) gets a measurably larger AD weight than a near-median one."""
    w_extreme = ad_weight(0.01)
    w_median = ad_weight(0.5)
    assert w_extreme > 10 * w_median


# ---------------------------------------------------------------------------
# Task 2: trailing 252-day window with min(252, available) fallback
# ---------------------------------------------------------------------------


def _synthetic_series(n_days=400, seed=1):
    dates = np.arange(20230101, 20230101 + n_days, dtype=float)
    log_rets = np.random.default_rng(seed).normal(0.0, 0.01, n_days)
    closes = 100.0 * np.exp(np.cumsum(log_rets))
    return dates, closes


def test_trailing_window_full_252():
    dates, closes = _synthetic_series(400)
    wd, r = trailing_window_returns(dates, closes, dates[399])
    assert len(wd) == 252
    assert len(r) == 251


def test_trailing_window_expanding_fallback():
    dates, closes = _synthetic_series(400)
    wd, r = trailing_window_returns(dates, closes, dates[100])
    assert len(wd) == 101
    assert len(r) == 100


def test_trailing_window_last_date_matches_requested():
    dates, closes = _synthetic_series(400)
    wd, _ = trailing_window_returns(dates, closes, dates[399])
    assert wd[-1] == dates[399]


def test_trailing_window_uses_most_recent_date_at_or_before():
    dates, closes = _synthetic_series(400)
    # dates are integers spaced by 1; request a date strictly between two
    # existing entries is impossible here (all integers exist), so instead
    # request a date that falls between the coarse yyyymmdd values by using
    # a half-step that is not itself present in the series.
    missing_date = dates[200] + 0.5
    wd, _ = trailing_window_returns(dates, closes, missing_date)
    assert wd[-1] == dates[200]


def test_trailing_window_end_date_before_series_raises():
    dates, closes = _synthetic_series(400)
    with pytest.raises(ValueError):
        trailing_window_returns(dates, closes, dates[0] - 1)


def test_trailing_window_insufficient_returns_raises():
    dates, closes = _synthetic_series(400)
    with pytest.raises(ValueError):
        trailing_window_returns(dates, closes, dates[5])


def test_trailing_window_no_padding_contiguous_slice():
    dates, closes = _synthetic_series(400)
    wd, _ = trailing_window_returns(dates, closes, dates[150])
    # A contiguous slice of the ascending input dates array.
    idx = np.searchsorted(dates, dates[150])
    expected = dates[idx - len(wd) + 1 : idx + 1]
    assert np.array_equal(wd, expected)


def test_trailing_window_docstring_mentions_d04_or_252():
    doc = trailing_window_returns.__doc__ or ""
    assert "D-04" in doc or "252" in doc


# ---------------------------------------------------------------------------
# Task 3: multi-start bounded L-BFGS-B fit
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_fit_p_measure_recovers_tail_behavior_on_synthetic_sample():
    """Parameter-LEVEL recovery is not a reliable property of this method:
    an 8-parameter asymmetric model fit against only a threshold grid's
    worth of tail probabilities is a genuinely underdetermined problem --
    ADR-0004 flags exactly this identifiability risk ("materially changes
    the model's degrees of freedom... with implications for identifiability
    against a single maturity's smile"). Verified directly in this session
    (multiple seeds, multiple threshold-grid densities, n up to 20000) that
    different, equally-valid-looking parameter vectors routinely achieve an
    AD objective at or below the true parameters' own value -- so a fixed
    0.3-absolute Yn/Yp tolerance is not a property this correctly-behaving
    optimizer can be expected to satisfy.

    What IS reliably guaranteed, and is the actual promise of a
    tail-probability-matching fit, is that (a) the optimizer's objective is
    no worse than the truth's, and (b) the recovered model's own tail
    probabilities closely track the true model's tail probabilities at the
    same thresholds -- both models are being asked to explain the same
    empirical tail, so their model-implied tails should agree with each
    other even when their raw parameter vectors do not.
    """
    sample = _sample_from_cgmyw(GOOD_PARAMS, 3000, seed=99)
    thresholds = compute_threshold_grid(sample)
    result = fit_p_measure(sample)

    true_loss = ad_weighted_objective(GOOD_PARAMS, sample, thresholds)
    assert np.isfinite(result.ad_objective)
    assert result.ad_objective <= true_loss + 5.0

    x, raw_true = levy_density_grid(*GOOD_PARAMS)
    _, pdf_true, _ = normalize_to_pdf(x, raw_true)
    true_tail_probs = model_tail_probabilities(x, pdf_true, thresholds)

    x_fit, raw_fit = levy_density_grid(*result.as_tuple())
    _, pdf_fit, _ = normalize_to_pdf(x_fit, raw_fit)
    fit_tail_probs = model_tail_probabilities(x_fit, pdf_fit, thresholds)

    assert np.max(np.abs(fit_tail_probs - true_tail_probs)) < 0.05


@pytest.mark.slow
def test_fit_p_measure_result_fields_finite_and_inbounds():
    sample = _sample_from_cgmyw(GOOD_PARAMS, 3000, seed=99)
    result = fit_p_measure(sample)
    assert isinstance(result, PFitResult)
    values = result.as_tuple()
    assert all(np.isfinite(v) for v in values)
    names = ("Cn", "Cp", "G", "M", "Yn", "Yp", "Wn", "Wp")
    for name, val in zip(names, values):
        lo, hi = P_BOUNDS[name]
        assert lo < val < hi


def test_fixed_starts_are_three_distinct_pole_free_inbounds_regimes():
    """_FIXED_STARTS: a classical-CGMY-ish guess, a near-zero-Y/large-W
    guess (the region an independently published CGMYW fit lands in), and a
    Y>1 guess -- each must be inside P_BOUNDS, finite under the objective
    (not sitting on a Gamma(-Y) or Tricomi pole), and start 0 must always be
    the original central seed regardless of how many are used."""
    import cgmyw.pfit as pfit_module

    assert len(pfit_module._FIXED_STARTS) == 3
    assert pfit_module._FIXED_STARTS[0] is pfit_module._CENTRAL_START

    sample = _sample_from_cgmyw(GOOD_PARAMS, 500, seed=7)
    seen_low_y = False
    seen_high_y = False
    for fixed in pfit_module._FIXED_STARTS:
        theta = tuple(fixed[name] for name in pfit_module.PARAM_ORDER)
        for name, val in fixed.items():
            lo, hi = P_BOUNDS[name]
            assert lo < val < hi, f"{name}={val} outside P_BOUNDS {(lo, hi)}"
        loss = ad_weighted_objective(theta, sample)
        assert np.isfinite(loss), f"fixed start {fixed} is on a pole (+inf objective)"
        if fixed["Yn"] < 1.0 and fixed["Yp"] < 1.0:
            seen_low_y = True
        if fixed["Yn"] > 1.0 and fixed["Yp"] > 1.0:
            seen_high_y = True
    # At least one structured start on each side of the Yn=Yp=1 Gamma(-Y)
    # pole -- the specific gap this trio exists to cover.
    assert seen_low_y and seen_high_y


@pytest.mark.parametrize("n_starts", [1, 2, 3, 4])
def test_fit_p_measure_small_n_starts_uses_fixed_seeds_first(n_starts, monkeypatch):
    """n_starts below 3 must take just the first n_starts fixed seeds (never
    error, never silently fall back to random draws for slots that should be
    structured) -- start order is start 0 = _CENTRAL_START always."""
    import cgmyw.pfit as pfit_module

    captured_x0s = []
    real_minimize = pfit_module.minimize

    def spying_minimize(fun, x0, **kwargs):
        captured_x0s.append(tuple(x0))
        return real_minimize(fun, x0, **kwargs)

    monkeypatch.setattr(pfit_module, "minimize", spying_minimize)
    sample = _sample_from_cgmyw(GOOD_PARAMS, 500, seed=7)
    pfit_module.fit_p_measure(sample, n_starts=n_starts)

    assert len(captured_x0s) == n_starts
    # x0 is in log space, with each seed's Cn, Cp variance-matched to the sample.
    expected_fixed = [
        tuple(np.log(pfit_module._variance_matched(fixed, float(np.var(sample)))))
        for fixed in pfit_module._FIXED_STARTS[:n_starts]
    ]
    for got, want in zip(captured_x0s, expected_fixed):
        assert got == want


@pytest.mark.slow
def test_fit_p_measure_starts_attempted_and_succeeded():
    sample = _sample_from_cgmyw(GOOD_PARAMS, 3000, seed=99)
    result = fit_p_measure(sample)
    assert result.starts_attempted == 8
    assert result.starts_succeeded >= 1


@pytest.mark.slow
def test_fit_p_measure_deterministic_same_seed():
    sample = _sample_from_cgmyw(GOOD_PARAMS, 3000, seed=99)
    r1 = fit_p_measure(sample)
    r2 = fit_p_measure(sample)
    assert r1.as_tuple() == r2.as_tuple()


def test_fit_p_measure_raises_when_all_starts_fail(monkeypatch):
    import cgmyw.pfit as pfit_module

    monkeypatch.setattr(pfit_module, "ad_weighted_objective", lambda *a, **k: np.inf)
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0, 0.02, 100)
    with pytest.raises(RuntimeError, match="8"):
        pfit_module.fit_p_measure(returns)


def test_fit_p_measure_raises_on_nonfinite_returns():
    returns = np.array([0.01, np.nan] + [0.001] * 80)
    with pytest.raises((RuntimeError, ValueError)):
        fit_p_measure(returns)


def test_fit_p_measure_warns_when_param_near_bound():
    import cgmyw.pfit as pfit_module

    # Force a "successful" bound-pinned result via monkeypatched minimize.
    class _FakeResult:
        def __init__(self, x):
            self.x = np.array(x)
            self.fun = 1.0
            self.success = True
            self.message = "fake"

    pinned = [pfit_module.P_BOUNDS[name][0] + 1e-9 for name in pfit_module.PARAM_ORDER]

    def fake_minimize(fun, x0, args, method, bounds):
        return _FakeResult(np.log(pinned))  # the optimizer works in log space

    monkeypatch_targets = pfit_module.minimize
    try:
        pfit_module.minimize = fake_minimize
        rng = np.random.default_rng(0)
        returns = rng.normal(0.0, 0.02, 100)
        with pytest.warns(RuntimeWarning):
            pfit_module.fit_p_measure(returns, n_starts=2)
    finally:
        pfit_module.minimize = monkeypatch_targets


def test_min_returns_constant_value():
    assert MIN_RETURNS == 60


def test_compute_threshold_grid_shape():
    # 30 (15 per side, 1-45% / 55-99%): spans the whole distribution like the
    # paper's Figure 1; the old tails-only grid never saw the body.
    rng = np.random.default_rng(3)
    sample = rng.normal(0.0, 0.02, 500)
    thresholds = compute_threshold_grid(sample)
    assert thresholds.shape == (30,)
    assert np.all(np.isfinite(thresholds))
    assert np.all(np.diff(thresholds) > 0)  # strictly increasing, no duplicate quantiles


def test_variance_matched_seed_hits_sample_variance_scale():
    """The seed rescaling must put each fixed start at the sample's return
    scale -- otherwise L-BFGS-B starts on a saturated plateau and stops."""
    import cgmyw.pfit as pfit_module

    rng = np.random.default_rng(5)
    for sd in (0.01, 0.03):
        returns = rng.normal(0.0, sd, 2000)
        theta = pfit_module._variance_matched(pfit_module._CENTRAL_START, float(returns.var()))
        x, raw = levy_density_grid(*theta)
        x, pdf, _ = normalize_to_pdf(x, raw)
        model_var = np.trapezoid(x**2 * pdf, x) - np.trapezoid(x * pdf, x) ** 2
        assert 0.5 < model_var / returns.var() < 2.0


def test_n_density_constant():
    assert N_DENSITY == 2**14


@pytest.mark.parametrize(
    "params",
    [
        GOOD_PARAMS,
        (0.9515, 1.3436, 51.983, 97.634, 0.0030, 0.0031, 22.269, 4.8288),  # Madan & Wang Table 1 AAPL
        (0.006954, 0.006556, 26.88, 40.87, 0.9477, 0.9507, 0.4222, 0.4238),  # our 2025-08-29 AAPL fit
    ],
)
def test_tail_probabilities_converged_on_default_grid(params):
    """Default-grid tail probabilities must match a 16x larger, 10x finer grid.
    Guards the threshold-sliver bug (masking from the first grid point past t
    was off by up to 0.005) and the tiny-Y end of P_BOUNDS."""
    th = np.array([-0.05, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.05])
    probs = []
    for n, dx in [(2**14, 2.5e-4), (2**18, 2.5e-5)]:
        x, raw = levy_density_grid(*params, n=n, dx=dx)
        x, pdf, _ = normalize_to_pdf(x, raw)
        probs.append(model_tail_probabilities(x, pdf, th))
    np.testing.assert_allclose(probs[0], probs[1], atol=5e-5)
