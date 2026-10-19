"""Q-measure calibration (QFIT-01/02/03): pick a smile, risk-neutralize the
P-fit, calibrate to options.

Measure-change rules, from Madan & Wang, "Asset Pricing, Measure Changes, and
a New Lévy Model" (2026), Section 2/4:
- Equivalence (finite Hellinger distance) forbids changing Cn, Cp, Yn, Yp --
  they are pinned at their P values. Only G, M, Wn, Wp may move.
- No free drift either: the Q exponent is psi_cgmyw at the Q parameters, and
  the martingale condition is a constraint on those parameters,
  T * psi_Q(-i) = ln(F/S0) (the paper's eq. 7 with its r=q=0 simplification
  replaced by the market forward). This holds for every Y in (0,2), not just
  Y<1: psi_cgmyw's closed form carries the drift m(theta) = analytic mean of
  k, and equivalence moves the drift by exactly m(theta_Q) - m(theta_P).

Time units: the P-fit runs on daily returns, so its Cn, Cp are per trading
day. G, M, Y, W describe jump *sizes* and are time-free; only the arrival
rates scale, Cn/Cp * TRADING_DAYS_PER_YEAR, giving psi_annual = 252 *
psi_daily exactly. Option maturities are calendar days / 365.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize

from .data import COL, yyyymmdd_to_date
from .model import psi_cgmyw
from .pfit import _FIXED_STARTS, P_BOUNDS, PARAM_ORDER, _variance_matched
from .pricing import CM_ALPHA, black76_vega, cgmyw_prices, implied_vols

TRADING_DAYS_PER_YEAR: float = 252.0
DTE_LO, DTE_HI = 30, 45  # calendar days to expiry (D-01)
MIN_STRIKES: int = 6  # D-02
MAX_NAN_IV_FRAC: float = 0.5  # D-02
# Below ~10 cents a one-tick ($0.01) spread is >= 10% of the price, and the
# implied vol at that strike is quote noise (AAPL's auto-selected chain ran
# out to 1.5-cent puts at K/F = 0.44 with IV 0.83).
MIN_MID: float = 0.10

# Parameters equivalence lets Q change (paper Section 2). M's floor keeps the
# Carr-Madan damping integrable (needs M > alpha + 1).
FREE: tuple[str, ...] = ("G", "M", "Wn", "Wp")
# Annual units, like everything on the Q side: Cn/Cp are P_BOUNDS' daily
# bounds times TRADING_DAYS_PER_YEAR (only C_FREE / UNRESTRICTED search them).
Q_BOUNDS: dict[str, tuple[float, float]] = {
    **P_BOUNDS,
    "Cn": (P_BOUNDS["Cn"][0] * 252.0, P_BOUNDS["Cn"][1] * 252.0),
    "Cp": (P_BOUNDS["Cp"][0] * 252.0, P_BOUNDS["Cp"][1] * 252.0),
    "M": (CM_ALPHA + 1.1, 300.0),
}
_IDX = {n: i for i, n in enumerate(PARAM_ORDER)}

# What calibrate_q searches over; M is always solved from the martingale
# condition instead.
# - PINNED: the paper's strictly equivalent measure change (Cn, Cp, Yn, Yp
#   at their P values). Cannot fit AAPL's skew (ADR-0008).
# - C_FREE (the default): Cn, Cp move too, Yn, Yp stay pinned -- the
#   relaxation of Madan & Wang's Markovian-economies paper (Sec. 3), chosen
#   by Dr. Madan on 2026-10-19 ("keep it free for performance accuracy").
#   Not strictly equivalent; required returns then come from eq. 36's
#   kernel difference rather than a Radon-Nikodym density.
# - UNRESTRICTED: Yn, Yp free as well -- a diagnostic upper bound on fit.
PINNED: tuple[str, ...] = ("G", "Wn", "Wp")
C_FREE: tuple[str, ...] = ("Cn", "Cp", "G", "Wn", "Wp")
UNRESTRICTED: tuple[str, ...] = ("Cn", "Cp", "G", "Yn", "Yp", "Wn", "Wp")


def annualize(p_daily):
    """Daily-unit (Cn,Cp,G,M,Yn,Yp,Wn,Wp) -> annual units (scale Cn, Cp only)."""
    p = np.array(p_daily, float)
    p[:2] *= TRADING_DAYS_PER_YEAR
    return tuple(p)


def _with(params, names, values):
    p = np.array(params, float)
    p[[_IDX[n] for n in names]] = values
    return tuple(p)


def _get(params, names):
    return np.array([params[_IDX[n]] for n in names], float)


def martingale_drift(params) -> float:
    """psi(-i): the annual log-growth rate of E[S_t] under `params`."""
    return float(np.real(psi_cgmyw(-1j, *params)))


@dataclass(frozen=True)
class SmileData:
    quote_date: int
    expiry: int
    T: float  # years (calendar days / 365)
    S0: float
    F: float
    df: float  # exp(-r T)
    strikes: np.ndarray
    is_call: np.ndarray
    mid: np.ndarray
    market_iv: np.ndarray

    @property
    def target_drift(self) -> float:
        """ln(F/S0)/T: the annual psi_Q(-i) the martingale condition requires."""
        return float(np.log(self.F / self.S0) / self.T)


def _dte(quote_dates, expiries):
    days = {d: yyyymmdd_to_date(d) for d in np.unique(np.concatenate([quote_dates, expiries]))}
    return np.array([(days[e] - days[q]).days for q, e in zip(quote_dates, expiries)])


def _usable(rows):
    """OTM quotes split at the forward (calls K >= F, puts K < F), live bid,
    mid >= MIN_MID, WRDS IV computable. NaN-IV-flagged rows carry garbage `price` and a 50.0
    IV sentinel, so they never enter the smile."""
    K, F = rows[:, COL["strike"]], rows[:, COL["forward"]]
    call = rows[:, COL["cp_flag"]] == 1  # 1 = call (verified on AAPL: deep-ITM K=160 call ~ S-K)
    otm = np.where(call, K >= F, K < F)
    mid = 0.5 * (rows[:, COL["bid"]] + rows[:, COL["ask"]])
    return otm & (rows[:, COL["bid"]] > 0) & (mid >= MIN_MID) & (rows[:, COL["nan_iv_flag"]] != 1)


def candidate_table(outmm, quote_date=None):
    """Every (quote_date, expiry) with DTE in [DTE_LO, DTE_HI], with its usable
    OTM strike count and NaN-IV fraction. Restricted to one quote date if given."""
    rows = outmm if quote_date is None else outmm[outmm[:, COL["date"]] == quote_date]
    pairs, inv = np.unique(rows[:, [COL["date"], COL["expiry"]]], axis=0, return_inverse=True)
    inv = inv.ravel()
    dte = _dte(pairs[:, 0], pairs[:, 1])
    n_usable = np.bincount(inv, weights=_usable(rows), minlength=len(pairs))
    nan_frac = np.bincount(inv, weights=rows[:, COL["nan_iv_flag"]] == 1, minlength=len(pairs)) / np.bincount(inv)
    keep = (dte >= DTE_LO) & (dte <= DTE_HI)
    return [
        dict(quote_date=int(q), expiry=int(e), dte=int(d), n_strikes=int(n), nan_iv_frac=float(f))
        for q, e, d, n, f in zip(pairs[keep, 0], pairs[keep, 1], dte[keep], n_usable[keep], nan_frac[keep])
    ]


def select_date_maturity(outmm, quote_date=None) -> tuple[int, int]:
    """Data-quality choice (D-01..D-03): DTE window first, then >= MIN_STRIKES
    usable strikes and <= MAX_NAN_IV_FRAC flagged rows, ranked by strike count
    desc, NaN-IV fraction asc; latest date only as a final deterministic tie-break."""
    eligible = [
        c for c in candidate_table(outmm, quote_date)
        if c["n_strikes"] >= MIN_STRIKES and c["nan_iv_frac"] <= MAX_NAN_IV_FRAC
    ]
    if not eligible:
        raise ValueError(
            f"no (date, expiry) with {DTE_LO}-{DTE_HI} DTE, >= {MIN_STRIKES} usable "
            f"OTM strikes and <= {MAX_NAN_IV_FRAC:.0%} NaN-IV rows"
            + (f" on {quote_date}" if quote_date is not None else "")
        )
    best = max(eligible, key=lambda c: (c["n_strikes"], -c["nan_iv_frac"], c["quote_date"], -c["dte"]))
    return best["quote_date"], best["expiry"]


def build_smile(outmm, quote_date, expiry) -> SmileData:
    """The calibration target: usable OTM quotes at mid, sorted by strike,
    with Black-76 IVs recomputed from mid (unattainable mids dropped)."""
    rows = outmm[(outmm[:, COL["date"]] == quote_date) & (outmm[:, COL["expiry"]] == expiry)]
    rows = rows[_usable(rows)]
    rows = rows[np.argsort(rows[:, COL["strike"]])]
    if len(rows) == 0:
        raise ValueError(f"no usable OTM quotes for {quote_date} / {expiry}")
    T = int(_dte([quote_date], [expiry])[0]) / 365.0
    S0, F, r = (float(rows[0, COL[c]]) for c in ("spot", "forward", "rate"))
    df = float(np.exp(-r * T))
    K = rows[:, COL["strike"]]
    is_call = rows[:, COL["cp_flag"]] == 1
    mid = 0.5 * (rows[:, COL["bid"]] + rows[:, COL["ask"]])
    iv = implied_vols(mid, F, K, T, df, is_call)
    ok = np.isfinite(iv)
    return SmileData(int(quote_date), int(expiry), T, S0, F, df, K[ok], is_call[ok], mid[ok], iv[ok])


def risk_neutralize(p_annual, target_drift: float):
    """The paper's Section 4 measure change: the minimum-norm move of
    (G, M, Wn, Wp) away from P that satisfies the martingale condition.
    Uses no option data.

    Solved as min over (G, Wn, Wp) with M = solve_m(...) -- the constraint
    surface is exactly the graph of that root, so this is the constrained
    problem itself, minus SLSQP (whose finite-difference constraint gradients
    are noise-dominated at Y ~ 0.003, where Gamma(-Y) ~ -333 amplifies
    round-off; it hit its iteration limit on the paper's own AAPL vector).
    Raises RuntimeError if no feasible point is found.
    """
    p0 = _get(p_annual, FREE)
    names = ("G", "Wn", "Wp")

    def project(x):
        trial = _with(p_annual, names, x)
        m = solve_m(trial, target_drift)
        return None if m is None else _with(trial, ("M",), (m,))

    def norm2(x):
        q = project(x)
        return 1e12 if q is None else float(np.sum((_get(q, FREE) - p0) ** 2))

    res = minimize(norm2, _get(p_annual, names), method="Nelder-Mead",
                   bounds=[Q_BOUNDS[n] for n in names], options={"xatol": 1e-9, "fatol": 1e-14, "maxiter": 4000})
    q = project(res.x)
    if q is None:
        raise RuntimeError(f"risk-neutralization failed: no martingale measure found ({res.message})")
    return q


def solve_m(params, target_drift: float):
    """The M that makes psi(-i) == target_drift with everything else fixed,
    or None if no M in Q_BOUNDS does. psi(-i) is strictly decreasing in M
    (the up-tail term is int (e^x - 1) e^{-Mx} ... dx with a positive
    integrand), so the root is unique when bracketed."""
    lo, hi = Q_BOUNDS["M"]
    f = lambda m: martingale_drift(_with(params, ("M",), (m,))) - target_drift  # noqa: E731
    f_lo, f_hi = f(lo), f(hi)
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or f_lo * f_hi > 0:
        return None
    return brentq(f, lo, hi, xtol=1e-12, rtol=1e-14)


@dataclass(frozen=True)
class QFitResult:
    params: tuple  # annual (Cn,Cp,G,M,Yn,Yp,Wn,Wp); Cn,Cp,Yn,Yp == P values
    rn_params: tuple  # martingale-only (option-free) risk-neutral measure
    objective: float  # vega-weighted price error -- what the fit minimizes
    rn_objective: float  # same objective at rn_params (always >= objective: rn is start 0)
    iv_rmse: float  # over strikes where the model price has an implied vol
    rn_iv_rmse: float
    model_iv: np.ndarray
    rn_iv: np.ndarray
    n_options: int
    starts_succeeded: int


def _iv_rmse(params, smile):
    model = cgmyw_prices(params, smile.F, smile.df, smile.T, smile.strikes, smile.is_call)
    iv = implied_vols(model, smile.F, smile.strikes, smile.T, smile.df, smile.is_call)
    return float(np.sqrt(np.nanmean((iv - smile.market_iv) ** 2))), iv


def calibrate_q(p_daily, smile: SmileData, search: tuple[str, ...] = C_FREE,
                n_starts: int = 6, seed: int = 20261018) -> QFitResult:
    """Fit the Q measure to the smile. `search` parameters move (log space,
    bounded Nelder-Mead, Q_BOUNDS); M is solved from the martingale condition
    T*psi(-i) = ln(F/S0) (solve_m), so every visited point is a martingale
    measure by construction; everything else stays at the annualized P value.
    Default C_FREE (Cn, Cp, G, Wn, Wp; Yn, Yp pinned) per Dr. Madan; PINNED
    is the paper's strictly equivalent measure change.

    Objective: mean squared vega-weighted price error, ((model - mid)/vega)^2,
    vega taken once at the market IV -- a first-order IV-error loss without a
    Black-Scholes inversion per evaluation (2026-08-07 meeting). Starts: the
    option-free risk-neutralized point (risk_neutralize, the paper's own
    measure); the P-fit's three structured seed shapes (_FIXED_STARTS), C
    scaled so annual variance = ATM market IV^2; then seeded log-uniform
    draws in Q_BOUNDS. A start that is not a martingale measure is skipped
    (Nelder-Mead would only wander the 1e6 plateau). Raises RuntimeError if
    no start ends at a martingale measure with a finite objective.
    """
    p_ann = annualize(p_daily)
    rn = risk_neutralize(p_ann, smile.target_drift)
    vega = black76_vega(smile.F, smile.strikes, smile.T, smile.df, smile.market_iv)

    def to_params(z):
        trial = _with(p_ann, search, np.exp(z))
        m = solve_m(trial, smile.target_drift)
        return None if m is None else _with(trial, ("M",), (m,))

    def objective(z):
        q = to_params(z)
        if q is None:
            return 1e6  # no martingale measure here; finite so finite-difference gradients survive
        prices = cgmyw_prices(q, smile.F, smile.df, smile.T, smile.strikes, smile.is_call)
        loss = float(np.mean(((prices - smile.mid) / vega) ** 2))
        return loss if np.isfinite(loss) else 1e6

    log_bounds = [tuple(np.log(Q_BOUNDS[n])) for n in search]
    lo, hi = np.array(log_bounds).T
    rng = np.random.default_rng(seed)
    atm_var_daily = float(smile.market_iv[np.argmin(np.abs(np.log(smile.strikes / smile.F)))]) ** 2 / TRADING_DAYS_PER_YEAR
    shaped = [annualize(_variance_matched(seed_shape, atm_var_daily)) for seed_shape in _FIXED_STARTS]
    starts = [np.log(_get(p, search)) for p in [rn, *shaped]]
    starts += [rng.uniform(lo, hi) for _ in range(max(0, n_starts - len(starts)))]

    rn_obj = objective(np.log(_get(rn, search)))
    fits = []
    for z0 in starts:
        if objective(np.clip(z0, lo, hi)) >= 1e6:
            continue
        # Nelder-Mead, not L-BFGS-B: finite-difference gradients through the
        # FFT + root-solve are noisy, and on AAPL L-BFGS-B stalled at 7 IV
        # points where Nelder-Mead reached 0.4 on the same unrestricted smile.
        res = minimize(objective, np.clip(z0, lo, hi), method="Nelder-Mead", bounds=log_bounds,
                       options={"maxiter": 400 * len(search), "xatol": 1e-4, "fatol": 1e-10})
        q = to_params(res.x)
        if q is not None and res.fun < 1e6:
            fits.append((float(res.fun), q))
    if not fits:
        raise RuntimeError(f"Q-calibration: none of {n_starts} starts reached a martingale measure")
    best_obj, best = min(fits, key=lambda f: f[0])

    iv_rmse, model_iv = _iv_rmse(best, smile)
    rn_rmse, rn_iv = _iv_rmse(rn, smile)
    return QFitResult(best, rn, best_obj, rn_obj, iv_rmse, rn_rmse, model_iv, rn_iv, len(smile.strikes), len(fits))


__all__ = [
    "TRADING_DAYS_PER_YEAR", "Q_BOUNDS", "PINNED", "C_FREE", "UNRESTRICTED", "SmileData", "QFitResult", "annualize",
    "martingale_drift", "candidate_table", "select_date_maturity", "build_smile",
    "risk_neutralize", "solve_m", "calibrate_q",
]
