"""Gates for Q calibration (QFIT-01/02/03): smile selection on real AAPL
data, the measure-change rules, and parameter recovery on a synthetic smile."""

import numpy as np
import pytest

from cgmyw.data import COL
from cgmyw.pricing import cgmyw_prices, implied_vols
from cgmyw.qfit import (
    DTE_HI,
    DTE_LO,
    MIN_MID,
    TRADING_DAYS_PER_YEAR,
    SmileData,
    annualize,
    build_smile,
    calibrate_q,
    candidate_table,
    martingale_drift,
    risk_neutralize,
    select_date_maturity,
    solve_m,
)

# Madan & Wang (2026) Table 1, AAPL, in (Cn,Cp,G,M,Yn,Yp,Wn,Wp) daily units.
PAPER_AAPL_DAILY = (0.9515, 1.3436, 51.983, 97.634, 0.0030, 0.0031, 22.269, 4.8288)


def test_annualize_scales_only_arrival_rates():
    a = annualize(PAPER_AAPL_DAILY)
    assert a[:2] == pytest.approx(tuple(TRADING_DAYS_PER_YEAR * c for c in PAPER_AAPL_DAILY[:2]))
    assert a[2:] == PAPER_AAPL_DAILY[2:]
    # psi is linear in (Cn, Cp), so annual psi == 252 * daily psi exactly.
    assert martingale_drift(a) == pytest.approx(TRADING_DAYS_PER_YEAR * martingale_drift(PAPER_AAPL_DAILY), rel=1e-10)


def test_risk_neutralize_keeps_c_y_and_hits_martingale():
    """Paper Section 4: Cn, Cp, Yn, Yp untouched (Hellinger), psi_Q(-i) = target,
    and the move is small (the paper's AAPL G: 51.98 -> 51.20, M: 97.63 -> 98.18)."""
    p = annualize(PAPER_AAPL_DAILY)
    q = risk_neutralize(p, 0.0)
    assert (q[0], q[1], q[4], q[5]) == (p[0], p[1], p[4], p[5])
    assert abs(martingale_drift(q)) < 1e-8
    assert np.max(np.abs(np.array(q) - np.array(p))) < 5.0


def test_solve_m_is_the_unique_martingale_root():
    p = annualize(PAPER_AAPL_DAILY)
    m = solve_m(p, 0.04)
    assert m is not None
    q = (*p[:3], m, *p[4:])
    assert martingale_drift(q) == pytest.approx(0.04, abs=1e-9)
    assert solve_m(p, 1e6) is None  # no M can deliver an absurd drift


def test_calibrate_q_recovers_synthetic_free_params():
    """Price a smile from a known pinned-equivalent Q measure; calibration must
    reproduce it (IV RMSE ~ 0) with Cn, Cp, Yn, Yp unchanged."""
    p_daily = PAPER_AAPL_DAILY
    p_ann = annualize(p_daily)
    F, S0, T, df = 101.0, 100.0, 35 / 365, 0.996
    target = np.log(F / S0) / T
    truth = (*p_ann[:2], 40.0, 0.0, *p_ann[4:6], 15.0, 6.0)
    truth = (*truth[:3], solve_m(truth, target), *truth[4:])
    K = np.linspace(85, 115, 15)
    is_call = K >= F
    mid = cgmyw_prices(truth, F, df, T, K, is_call)
    iv = implied_vols(mid, F, K, T, df, is_call)
    smile = SmileData(20250101, 20250205, T, S0, F, df, K, is_call, mid, iv)

    fit = calibrate_q(p_daily, smile, n_starts=4)
    assert (fit.params[0], fit.params[1], fit.params[4], fit.params[5]) == (p_ann[0], p_ann[1], p_ann[4], p_ann[5])
    assert martingale_drift(fit.params) == pytest.approx(target, abs=1e-8)
    assert fit.iv_rmse < 2e-3


def test_select_date_maturity_respects_rules(aapl_outmm):
    qd, ex = select_date_maturity(aapl_outmm)
    cands = {(c["quote_date"], c["expiry"]): c for c in candidate_table(aapl_outmm)}
    best = cands[(qd, ex)]
    assert DTE_LO <= best["dte"] <= DTE_HI
    assert best["n_strikes"] == max(c["n_strikes"] for c in cands.values() if c["nan_iv_frac"] <= 0.5)


def test_build_smile_is_clean_otm(aapl_outmm):
    smile = build_smile(aapl_outmm, *select_date_maturity(aapl_outmm, 20250829))
    assert np.all(np.diff(smile.strikes) >= 0)
    assert np.all(np.where(smile.is_call, smile.strikes >= smile.F, smile.strikes < smile.F))
    assert np.all(smile.mid >= MIN_MID)
    assert np.all(np.isfinite(smile.market_iv)) and np.all(smile.market_iv < 5.0)
    # Our Black-76 IVs from mid track WRDS's own IV column for the same quotes.
    rows = aapl_outmm[(aapl_outmm[:, COL["date"]] == smile.quote_date) & (aapl_outmm[:, COL["expiry"]] == smile.expiry)]
    wrds_iv = {(r[COL["strike"]], r[COL["cp_flag"]] == 1): r[COL["iv"]] for r in rows}
    ref = np.array([wrds_iv[(k, c)] for k, c in zip(smile.strikes, smile.is_call)])
    assert np.median(np.abs(ref - smile.market_iv)) < 0.01


@pytest.mark.parametrize(
    "p_daily, paper_q",
    [
        # (Cn,Cp,G,M,Yn,Yp,Wn,Wp) from Madan & Wang Table 1 -> (G,M,Wn,Wp) from Table 2.
        (PAPER_AAPL_DAILY, (51.198, 98.183, 21.514, 5.3734)),
        ((3.5501, 4.5440, 73.823, 93.769, 0.0122, 0.0125, 12.840, 12.875), (72.387, 94.885, 11.421, 13.981)),  # NVDA
        ((1.3528, 1.4049, 75.518, 74.100, 0.0480, 0.0501, 4.1798, 4.9331), (74.043, 75.500, 2.7248, 6.3170)),  # META
    ],
)
def test_risk_neutralize_reproduces_published_table_2(p_daily, paper_q):
    """The whole chain -- psi_cgmyw, Kummer U, martingale condition, minimum
    norm -- against the paper's own risk-neutralization (daily units, r=q=0).
    0.5% tolerance: the published points themselves leave ~1e-5 residual
    drift, i.e. they are solver-tolerance accurate, not exact."""
    q = risk_neutralize(p_daily, 0.0)
    np.testing.assert_allclose((q[2], q[3], q[6], q[7]), paper_q, rtol=5e-3)
