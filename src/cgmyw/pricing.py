"""Option pricing for the Q stage (MODEL-02): Black-76 + implied vol, and the
Carr-Madan (1999) FFT call pricer driven by the CGMYW characteristic exponent.

Conventions, used by every function here:
- Prices are written against the forward F and discount factor df = exp(-rT),
  so r and q never appear separately (WRDS supplies F directly, column 7).
- T is in years. CGMYW parameters passed to `cgmyw_log_return_cf` must be in
  the same (annual) time unit -- see qfit.annualize for the daily->annual map.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.stats import norm

from .model import psi_cgmyw

IV_BRACKET: tuple[float, float] = (1e-4, 5.0)

# Carr-Madan grid. alpha=1.5 is the paper's own default damping; it requires
# E[S_T^(alpha+1)] < inf, i.e. M > alpha + 1 for CGMYW (the up-tail Kummer-U
# argument M - iu at u = v - (alpha+1)i must keep a positive real part).
CM_ALPHA: float = 1.5
CM_N: int = 4096
CM_ETA: float = 0.25  # v-grid spacing -> v_max = 1024, log-strike spacing 2pi/(N*eta) ~ 0.0061


def black76(F, K, T, df, sigma, is_call):
    """Black-76 price of a call (is_call True) or put, vectorised."""
    F, K, sigma = np.asarray(F, float), np.asarray(K, float), np.asarray(sigma, float)
    s = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * s**2) / s
    d2 = d1 - s
    call = df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    put = df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    return np.where(is_call, call, put)


def black76_vega(F, K, T, df, sigma):
    """dPrice/dsigma -- identical for calls and puts."""
    s = np.asarray(sigma, float) * np.sqrt(T)
    d1 = (np.log(F / np.asarray(K, float)) + 0.5 * s**2) / s
    return df * F * norm.pdf(d1) * np.sqrt(T)


def implied_vol(price, F, K, T, df, is_call, bracket=IV_BRACKET):
    """Scalar Black-76 implied vol via brentq; np.nan when the price is not
    attainable inside the bracket (never a fabricated endpoint)."""
    if not (T > 0 and price > 0 and np.isfinite(price)):
        return np.nan
    intrinsic = df * max(F - K, 0.0) if is_call else df * max(K - F, 0.0)
    if price - intrinsic <= 1e-10 * F:  # no resolvable time value -> IV unidentifiable
        return np.nan
    f = lambda s: float(black76(F, K, T, df, s, is_call)) - price  # noqa: E731
    lo, hi = bracket
    if f(lo) * f(hi) > 0:
        return np.nan
    return brentq(f, lo, hi, xtol=1e-10)


def implied_vols(prices, F, strikes, T, df, is_call):
    """Elementwise implied_vol over arrays of prices/strikes/flags."""
    return np.array([implied_vol(p, F, k, T, df, c) for p, k, c in zip(prices, strikes, is_call)])


def cgmyw_log_return_cf(params, T):
    """Characteristic function of ln(S_T / F) under CGMYW parameters `params`
    (Cn,Cp,G,M,Yn,Yp,Wn,Wp, annual units).

    phi(u) = exp(T * (psi(u) - i*u*psi(-i))). The -i*u*psi(-i) term is the
    martingale correction omega = -psi(-i) (forces phi(-i) == 1, i.e.
    E[S_T] = F). When Q parameters already satisfy the equivalence-preserving
    martingale condition T*psi(-i) = ln(F/S0) (qfit), this is algebraically the
    same process as S0*exp(X_T) -- the correction only removes solver-tolerance
    residue; it never adds a drift the measure change is not allowed to add.
    """
    omega = -psi_cgmyw(-1j, *params)
    return lambda u: np.exp(T * (psi_cgmyw(u, *params) + 1j * u * omega))


def carr_madan_calls(cf, F, df, strikes, alpha=CM_ALPHA, n=CM_N, eta=CM_ETA):
    """Carr-Madan FFT call prices at `strikes` for a forward-normalised
    characteristic function cf(u) = E[exp(iu ln(S_T/F))].

    Prices on the log-moneyness grid k = ln(K/F), Simpson-weighted, then
    cubic-spline interpolated to the requested strikes. Linear interpolation
    on the ~0.6% grid costs ~1e-3 * F near the money at T ~ 0.1 (curvature
    error lambda^2/8 * C''); the spline brings it to round-off.
    """
    lam = 2 * np.pi / (n * eta)
    b = n * lam / 2
    v = np.arange(n) * eta
    k = -b + lam * np.arange(n)
    psi_cm = df * cf(v - (alpha + 1) * 1j) / (alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v)
    simpson = (3 + (-1) ** np.arange(1, n + 1)) / 3.0
    simpson[0] = 1.0 / 3.0
    fft = np.fft.fft(np.exp(1j * v * b) * psi_cm * eta * simpson)
    calls_over_F = np.exp(-alpha * k) / np.pi * np.real(fft)
    log_m = np.log(np.asarray(strikes, float) / F)
    if not np.all(np.isfinite(calls_over_F)):
        return np.full(log_m.shape, np.nan)  # cf hit a pole: callers reject NaN, never a fake price
    return F * CubicSpline(k, calls_over_F)(log_m)


def cgmyw_prices(params, F, df, T, strikes, is_call):
    """CGMYW option prices: Carr-Madan calls, puts via put-call parity
    P = C - df*(F - K)."""
    strikes = np.asarray(strikes, float)
    calls = carr_madan_calls(cgmyw_log_return_cf(params, T), F, df, strikes)
    return np.where(is_call, calls, calls - df * (F - strikes))
