"""Gates for surface calibration (2026-11-02): omega martingale correction,
Sato scaling, parity forward, and parameter recovery on a synthetic surface."""

import numpy as np
import pytest

from cgmyw.data import COL
from cgmyw.model import psi_cgmyw
from cgmyw.pricing import implied_vols
from cgmyw.qfit import SmileData
from cgmyw.surface import (
    TARGET_DTES,
    calibrate_surface,
    parity_forward,
    select_expiries,
    surface_cf,
    surface_prices,
)

# Annual X_1 in the regime the AAPL surface fits land in.
THETA = (40.0, 60.0, 20.0, 30.0, 0.3, 0.3, 3.0, 2.0)


@pytest.mark.parametrize("gamma", [None, 0.5])
@pytest.mark.parametrize("T", [0.08, 0.5, 1.0])
def test_cf_is_forward_martingale_for_any_params(gamma, T):
    """E[S_T/F] = 1 from omega alone -- no parameter is spent on it."""
    cf = surface_cf(THETA, T, gamma)
    assert abs(cf(np.array([-1j]))[0] - 1) < 1e-9
    assert abs(cf(np.array([0.0]))[0] - 1) < 1e-12


def test_sato_equals_levy_at_one_year():
    u = np.linspace(-20, 20, 41)
    np.testing.assert_allclose(surface_cf(THETA, 1.0, 0.7)(u), surface_cf(THETA, 1.0)(u), rtol=1e-12)


def test_sato_is_self_similar():
    """X_T = T^gamma X_1: E[e^{iuX_T}] = exp(psi(u T^gamma)), times the
    T-specific martingale constant (which does not scale like X)."""
    u, T, g = np.linspace(-15, 15, 31), 0.25, 0.6
    s = T**g
    want = np.exp(psi_cgmyw(u * s, *THETA) - 1j * u * psi_cgmyw(-1j * s, *THETA))
    np.testing.assert_allclose(surface_cf(THETA, T, g)(u), want, rtol=1e-10)


def test_parity_forward_tracks_wrds_forward(aapl_outmm):
    qd = 20250829
    for e in select_expiries(aapl_outmm, qd):
        F, q = parity_forward(aapl_outmm, qd, e)
        wrds_F = aapl_outmm[(aapl_outmm[:, COL["date"]] == qd) & (aapl_outmm[:, COL["expiry"]] == e)][0, COL["forward"]]
        assert abs(F / wrds_F - 1) < 5e-3
        assert -0.05 < q < 0.05


def test_select_expiries_hits_targets(aapl_outmm):
    ex = select_expiries(aapl_outmm, 20250829)
    assert len(ex) == len(TARGET_DTES) and ex == sorted(ex)


@pytest.mark.slow
def test_calibrate_surface_recovers_synthetic_sato_surface():
    truth, gamma = THETA, 0.55
    F, S0, df = 101.0, 100.0, 0.99
    smiles = []
    for T in (30 / 365, 91 / 365, 182 / 365):
        K = F * np.exp(np.linspace(-0.25, 0.2, 12) * np.sqrt(T / 0.25))
        is_call = K >= F
        sm = SmileData(20250101, 0, T, S0, F, df, K, is_call, np.zeros_like(K), np.ones_like(K))
        mid = surface_prices(truth, sm, gamma)
        smiles.append(SmileData(20250101, 0, T, S0, F, df, K, is_call, mid, implied_vols(mid, F, K, T, df, is_call)))
    fit = calibrate_surface(smiles, "Sato CGMYW", n_starts=3)
    assert fit.iv_rmse < 5e-3
    assert fit.gamma == pytest.approx(gamma, abs=0.1)
