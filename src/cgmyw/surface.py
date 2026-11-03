"""Option-surface calibration (Dr. Madan, 2026-11-02): every model parameter
is free for pricing, several maturities are fit at once, and the time
dimension is either Lévy or Sato.

- Martingale by a constant, not by a parameter: ln S_T = ln F + X_T + omega_T
  with omega_T = -ln E[e^{X_T}], so E[S_T] = F for any parameters. Nothing is
  solved from the martingale condition (unlike qfit, which spends M on it).
- Forward from the options: F is where call and put mids cross (put-call
  parity C - P = df (F - K)); q is then implied by F = S0 e^{(r-q)T}.
- Time dimension. Lévy: X_T has exponent T psi(u), so skew decays like
  1/sqrt(T) and excess kurtosis like 1/T. Sato: X_T = T^gamma X_1 (X_1
  self-decomposable, which CGMYW is for non-negative G, M, Y, W), so
  E[e^{iuX_T}] = exp(psi(u T^gamma)) and skew/kurtosis are flat in T.
  CGMY stays CGMY under this scaling (G, M -> G/T^gamma, M/T^gamma); CGMYW
  does not, so pricing goes through the characteristic function, which needs
  no closed-form law at T.
- W = 0 variants answer the main question: does W add fitting power?

Parameters are in annual units, (Cn, Cp, G, M, Yn, Yp, Wn, Wp[, gamma]),
describing X_1 (one year).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import minimize

from .data import COL
from .model import psi_cgmyw
from .pfit import _FIXED_STARTS, PARAM_ORDER, _variance_matched
from .pricing import black76_vega, carr_madan_calls, implied_vols
from .qfit import MIN_STRIKES, Q_BOUNDS, TRADING_DAYS_PER_YEAR, SmileData, _dte, _usable, annualize, build_smile

# Calendar days: closest traded expiry to one month, a quarter, half a year, a year.
TARGET_DTES: tuple[int, ...] = (30, 91, 182, 365)
GAMMA_BOUNDS: tuple[float, float] = (0.05, 1.5)
PARITY_STRIKES: int = 4  # strikes nearest the call/put crossing used for F

# name -> (sato, w_free)
MODELS: dict[str, tuple[bool, bool]] = {
    "Lévy CGMY": (False, False),
    "Lévy CGMYW": (False, True),
    "Sato CGMY": (True, False),
    "Sato CGMYW": (True, True),
}


def parity_forward(outmm, quote_date, expiry):
    """(F, q) from put-call parity: F = K + (C - P)/df, taken as the median
    over the PARITY_STRIKES strikes where |C - P| is smallest (the crossing).
    Raises ValueError if fewer than 2 strikes carry both a live call and put."""
    rows = outmm[(outmm[:, COL["date"]] == quote_date) & (outmm[:, COL["expiry"]] == expiry) & (outmm[:, COL["bid"]] > 0)]
    T = int(_dte([quote_date], [expiry])[0]) / 365.0
    r, S0 = float(rows[0, COL["rate"]]), float(rows[0, COL["spot"]])
    df = np.exp(-r * T)
    mid = 0.5 * (rows[:, COL["bid"]] + rows[:, COL["ask"]])
    call = rows[:, COL["cp_flag"]] == 1
    calls = dict(zip(rows[call, COL["strike"]], mid[call]))
    puts = dict(zip(rows[~call, COL["strike"]], mid[~call]))
    K = np.array(sorted(set(calls) & set(puts)))
    if len(K) < 2:
        raise ValueError(f"put-call parity: < 2 strikes with both quotes on {quote_date}/{expiry}")
    diff = np.array([calls[k] - puts[k] for k in K])
    near = np.argsort(np.abs(diff))[:PARITY_STRIKES]
    F = float(np.median(K[near] + diff[near] / df))
    return F, r - np.log(F / S0) / T


def surface_smile(outmm, quote_date, expiry) -> SmileData:
    """build_smile's OTM quotes, re-based on the parity forward (IVs recomputed)."""
    s = build_smile(outmm, quote_date, expiry)
    F, _ = parity_forward(outmm, quote_date, expiry)
    # ponytail: OTM split stays at the WRDS forward; it differs from parity F by cents.
    iv = implied_vols(s.mid, F, s.strikes, s.T, s.df, s.is_call)
    ok = np.isfinite(iv)
    return replace(s, F=F, strikes=s.strikes[ok], is_call=s.is_call[ok], mid=s.mid[ok], market_iv=iv[ok])


def select_expiries(outmm, quote_date, targets=TARGET_DTES) -> list[int]:
    """The expiry closest to each target DTE among those with >= MIN_STRIKES
    usable OTM quotes; duplicates dropped, sorted by maturity."""
    rows = outmm[outmm[:, COL["date"]] == quote_date]
    exps = [e for e in np.unique(rows[:, COL["expiry"]]) if _usable(rows[rows[:, COL["expiry"]] == e]).sum() >= MIN_STRIKES]
    if not exps:
        raise ValueError(f"no expiry with >= {MIN_STRIKES} usable strikes on {quote_date}")
    dte = _dte([quote_date] * len(exps), exps)
    return sorted({int(exps[int(np.argmin(np.abs(dte - t)))]) for t in targets})


def surface_cf(params, T, gamma=None):
    """Forward-normalised cf of ln(S_T/F). gamma=None: Lévy, else Sato."""
    if gamma is None:
        omega = -T * psi_cgmyw(-1j, *params)
        return lambda u: np.exp(T * psi_cgmyw(u, *params) + 1j * u * omega)
    s = T**gamma
    omega = -psi_cgmyw(-1j * s, *params)
    return lambda u: np.exp(psi_cgmyw(u * s, *params) + 1j * u * omega)


def surface_prices(params, smile: SmileData, gamma=None):
    calls = carr_madan_calls(surface_cf(params, smile.T, gamma), smile.F, smile.df, smile.strikes)
    return np.where(smile.is_call, calls, calls - smile.df * (smile.F - smile.strikes))


@dataclass(frozen=True)
class SurfaceFit:
    model: str
    params: tuple  # (Cn,Cp,G,M,Yn,Yp,Wn,Wp), annual, X_1
    gamma: float | None
    objective: float  # mean ((model - mid)/vega)^2 over every option
    iv_rmse: float  # pooled over all maturities
    iv_rmse_by_T: tuple  # one per smile
    model_iv: tuple  # one array per smile


def _free_names(sato, w_free):
    return [n for n in PARAM_ORDER if w_free or n not in ("Wn", "Wp")] + (["gamma"] if sato else [])


def calibrate_surface(smiles: list[SmileData], model: str = "Sato CGMYW", n_starts: int = 6,
                      seed: int = 20261102, warm: SurfaceFit | None = None) -> SurfaceFit:
    """Joint fit to every smile in `smiles`, all parameters free (W fixed at 0
    for the CGMY models). Objective: mean squared vega-weighted price error,
    as in qfit. Log-space bounded Nelder-Mead from the P-fit's three seed
    shapes (variance-matched to the shortest smile's ATM IV, gamma = 0.5),
    seeded log-uniform draws, and `warm` (a fit of the nested W = 0 model:
    CGMYW contains CGMY, so starting there makes CGMYW at least as good up
    to the W floor), then one Nelder-Mead restart from the best point.
    Raises RuntimeError if no start gives a finite objective."""
    sato, w_free = MODELS[model]
    names = _free_names(sato, w_free)
    bounds = {**Q_BOUNDS, "gamma": GAMMA_BOUNDS}
    log_bounds = [tuple(np.log(bounds[n])) for n in names]
    lo, hi = np.array(log_bounds).T
    vegas = [black76_vega(s.F, s.strikes, s.T, s.df, s.market_iv) for s in smiles]

    def unpack(z):
        v = dict(zip(names, np.exp(z)))
        params = tuple(v.get(n, 0.0) for n in PARAM_ORDER)  # W absent -> 0
        return params, v.get("gamma")

    def objective(z):
        params, gamma = unpack(z)
        errs = [((surface_prices(params, s, gamma) - s.mid) / vg) ** 2 for s, vg in zip(smiles, vegas)]
        loss = float(np.mean(np.concatenate(errs)))
        return loss if np.isfinite(loss) else 1e6

    short = min(smiles, key=lambda s: s.T)
    atm_var_daily = float(short.market_iv[np.argmin(np.abs(np.log(short.strikes / short.F)))]) ** 2 / TRADING_DAYS_PER_YEAR
    rng = np.random.default_rng(seed)
    starts = []
    for shape in _FIXED_STARTS:
        p = dict(zip(PARAM_ORDER, annualize(_variance_matched(shape, atm_var_daily))), gamma=0.5)
        starts.append(np.clip(np.log([p[n] for n in names]), lo, hi))
    starts += [rng.uniform(lo, hi) for _ in range(max(0, n_starts - len(starts)))]
    if warm is not None:
        w = dict(zip(PARAM_ORDER, warm.params), gamma=warm.gamma or 0.5)
        starts.insert(0, np.clip(np.log([max(w[n], bounds[n][0]) for n in names]), lo, hi))

    def run(z0):
        return minimize(objective, z0, method="Nelder-Mead", bounds=log_bounds,
                        options={"maxiter": 400 * len(names), "xatol": 1e-4, "fatol": 1e-10})

    fits = [r for r in (run(z0) for z0 in starts if objective(z0) < 1e6) if r.fun < 1e6]
    if not fits:
        raise RuntimeError(f"surface calibration ({model}): no start gave a finite objective")
    best = min(fits, key=lambda r: r.fun)
    # Nelder-Mead's simplex collapses early in 9-D; a restart re-expands it.
    restart = run(best.x)
    best = restart if restart.fun < best.fun else best
    params, gamma = unpack(best.x)

    ivs, sq = [], []
    for s in smiles:
        iv = implied_vols(surface_prices(params, s, gamma), s.F, s.strikes, s.T, s.df, s.is_call)
        ivs.append(iv)
        sq.append((iv - s.market_iv) ** 2)
    by_T = tuple(float(np.sqrt(np.nanmean(e))) for e in sq)
    return SurfaceFit(model, params, gamma, float(best.fun), float(np.sqrt(np.nanmean(np.concatenate(sq)))), by_T, tuple(ivs))


__all__ = ["TARGET_DTES", "MODELS", "parity_forward", "surface_smile", "select_expiries", "surface_cf",
           "surface_prices", "SurfaceFit", "calibrate_surface"]
