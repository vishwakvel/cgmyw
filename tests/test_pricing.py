"""Gates for the Q-side pricer (MODEL-02): Black-76 round-trips, Carr-Madan
against Black-Scholes, and CGMYW Carr-Madan against independent quadrature."""

import numpy as np
import pytest
from scipy.integrate import quad

from cgmyw.pricing import (
    black76,
    carr_madan_calls,
    cgmyw_log_return_cf,
    cgmyw_prices,
    implied_vol,
)

# Annual-unit CGMYW vector in the paper's region (Madan & Wang 2026 Table 1
# AAPL, Cn/Cp scaled by 252), and one in the Y>1 regime.
PAPER_AAPL_ANNUAL = (0.9515 * 252, 1.3436 * 252, 51.983, 97.634, 0.0030, 0.0031, 22.269, 4.8288)
HIGH_Y_ANNUAL = (0.5, 0.4, 8.0, 12.0, 1.3, 1.2, 0.75, 0.9)  # W+Y = 2.05, 2.1: off the integer poles


@pytest.mark.parametrize("sigma", [0.05, 0.3, 0.9])
@pytest.mark.parametrize("K", [60.0, 95.0, 100.0, 110.0, 150.0])
@pytest.mark.parametrize("is_call", [True, False])
def test_implied_vol_round_trip(sigma, K, is_call):
    price = float(black76(100.0, K, 0.1, 0.99, sigma, is_call))
    iv = implied_vol(price, 100.0, K, 0.1, 0.99, is_call)
    # Deep-OTM/ITM low-vol prices carry no resolvable time value: NaN is the honest answer.
    assert np.isnan(iv) or abs(iv - sigma) < 1e-6


def test_implied_vol_nan_outside_attainable_range():
    assert np.isnan(implied_vol(1e-14, 100.0, 100.0, 0.1, 0.99, True))
    assert np.isnan(implied_vol(200.0, 100.0, 100.0, 0.1, 0.99, True))
    assert np.isnan(implied_vol(5.0, 100.0, 100.0, 0.0, 0.99, True))


@pytest.mark.parametrize("T,sigma", [(0.08, 0.15), (0.1, 0.25), (1.0, 0.4)])
def test_carr_madan_matches_black_scholes(T, sigma):
    F, df = 100.0, 0.99
    K = np.linspace(60, 160, 41)
    cf = lambda u: np.exp(-0.5 * sigma**2 * T * (u**2 + 1j * u))  # noqa: E731
    np.testing.assert_allclose(carr_madan_calls(cf, F, df, K), black76(F, K, T, df, sigma, True), atol=1e-5)


@pytest.mark.parametrize("params", [PAPER_AAPL_ANNUAL, HIGH_Y_ANNUAL])
def test_cgmyw_cf_is_forward_martingale(params):
    """phi(-i) == 1, i.e. E[S_T] = F, and phi(0) == 1."""
    cf = cgmyw_log_return_cf(params, 0.1)
    # 1e-9, not 1e-12: at Y ~ 0.003, Gamma(-Y) ~ -333 amplifies round-off.
    assert abs(cf(np.array([-1j]))[0] - 1) < 1e-9
    assert abs(cf(np.array([0.0]))[0] - 1) < 1e-9


@pytest.mark.parametrize("params", [PAPER_AAPL_ANNUAL, HIGH_Y_ANNUAL])
def test_cgmyw_carr_madan_matches_gil_pelaez_quadrature(params):
    """Independent check: C = df*(F*P1 - K*P2) with Gil-Pelaez probabilities
    from adaptive quadrature of the same characteristic function."""
    F, df, T = 100.0, 0.99, 0.1
    cf = cgmyw_log_return_cf(params, T)
    phi = lambda u: complex(cf(np.array([u], dtype=complex))[0])  # noqa: E731
    strikes = np.array([85.0, 100.0, 115.0])
    ref = []
    for K in strikes:
        k = np.log(K / F)
        p2 = 0.5 + quad(lambda u: (np.exp(-1j * u * k) * phi(u) / (1j * u)).real, 1e-10, 400, limit=2000)[0] / np.pi
        p1 = 0.5 + quad(lambda u: (np.exp(-1j * u * k) * phi(u - 1j) / (1j * u)).real, 1e-10, 400, limit=2000)[0] / np.pi
        ref.append(df * (F * p1 - K * p2))
    np.testing.assert_allclose(cgmyw_prices(params, F, df, T, strikes, True), ref, atol=1e-4)


def test_put_call_parity():
    F, df, T = 100.0, 0.99, 0.1
    K = np.array([90.0, 100.0, 110.0])
    c = cgmyw_prices(PAPER_AAPL_ANNUAL, F, df, T, K, True)
    p = cgmyw_prices(PAPER_AAPL_ANNUAL, F, df, T, K, False)
    np.testing.assert_allclose(c - p, df * (F - K), atol=1e-10)
    assert np.all(p > 0) and np.all(c > 0)
