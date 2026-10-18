"""Console report and diagnostic plots for the P-then-Q pipeline (D-05/D-06,
QFIT-03). Agg backend and explicit figure close, so a 10-stock loop never
needs a display or leaks figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .pfit import (  # noqa: E402
    PARAM_ORDER,
    compute_threshold_grid,
    empirical_tail_probabilities,
    levy_density_grid,
    model_tail_probabilities,
    normalize_to_pdf,
)


def _fmt(params) -> str:
    return "  ".join(f"{n}={v:.4g}" for n, v in zip(PARAM_ORDER, params))


def format_report(symbol, p_fit, smile, q_fit, q_free=None) -> str:
    lines = [
        f"=== {symbol}: P-measure (daily units, window {p_fit.window_start_date}-{p_fit.window_end_date}, "
        f"{p_fit.n_returns} returns) ===",
        _fmt(p_fit.as_tuple()),
        f"AD objective {p_fit.ad_objective:.5g}; starts {p_fit.starts_succeeded}/{p_fit.starts_attempted}",
        "",
        f"=== {symbol}: Q-measure (annual units; quote {smile.quote_date}, expiry {smile.expiry}, "
        f"T={smile.T:.4f}y, {q_fit.n_options} OTM options) ===",
        f"martingale-only (paper Sec. 4): {_fmt(q_fit.rn_params)}",
        f"  IV RMSE {q_fit.rn_iv_rmse:.4f}; objective {q_fit.rn_objective:.5g}",
        f"smile-calibrated, Cn/Cp/Yn/Yp pinned: {_fmt(q_fit.params)}",
        f"  IV RMSE {q_fit.iv_rmse:.4f}; objective {q_fit.objective:.5g}; starts ok {q_fit.starts_succeeded}",
        "  (objective = mean ((model - mid)/vega)^2, the vega-weighted price error minimized;",
        "   it tracks IV error only to first order, so with errors this large the two can rank fits differently)",
    ]
    if q_free is not None:
        lines += [
            f"unrestricted (NOT equivalent to P; restriction-cost diagnostic): {_fmt(q_free.params)}",
            f"  IV RMSE {q_free.iv_rmse:.4f}",
        ]
    return "\n".join(lines)


def plot_tail_fit(p_fit, returns, out_path, symbol: str) -> Path:
    """Empirical vs. model tail probabilities at the fit's own threshold grid."""
    thresholds = compute_threshold_grid(returns)
    x, raw = levy_density_grid(*p_fit.as_tuple())
    x, pdf, _ = normalize_to_pdf(x, raw)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(thresholds, empirical_tail_probabilities(returns, thresholds), "o", label="empirical")
    ax.plot(thresholds, model_tail_probabilities(x, pdf, thresholds), "r.", ms=10, label="CGMYW (P)")
    ax.set(xlabel="daily log-return threshold", ylabel="tail probability", title=f"{symbol} P-fit tail probabilities")
    ax.legend()
    return _save(fig, out_path)


def plot_smile(smile, q_fit, out_path, symbol: str, q_free=None) -> Path:
    """Market vs. model implied vols across log-moneyness."""
    k = np.log(smile.strikes / smile.F)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(k, smile.market_iv, "ko", ms=4, label="market (mid)")
    ax.plot(k, q_fit.model_iv, "r-", label=f"Q pinned to P (RMSE {q_fit.iv_rmse:.3f})")
    ax.plot(k, q_fit.rn_iv, "b--", label=f"martingale-only (RMSE {q_fit.rn_iv_rmse:.3f})")
    if q_free is not None:
        ax.plot(k, q_free.model_iv, "g:", label=f"unrestricted (RMSE {q_free.iv_rmse:.3f})")
    ax.set(xlabel="ln(K/F)", ylabel="implied vol",
           title=f"{symbol} {smile.quote_date} -> {smile.expiry} smile")
    ax.legend()
    return _save(fig, out_path)


def _save(fig, out_path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
