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


def format_report(symbol, p_fit, smile, q_fits: dict) -> str:
    """q_fits: label -> QFitResult, main fit first. The martingale-only
    measure (paper Sec. 4) is shown once, from the first fit."""
    first = next(iter(q_fits.values()))
    lines = [
        f"=== {symbol}: P-measure (daily units, window {p_fit.window_start_date}-{p_fit.window_end_date}, "
        f"{p_fit.n_returns} returns) ===",
        _fmt(p_fit.as_tuple()),
        f"AD objective {p_fit.ad_objective:.5g}; starts {p_fit.starts_succeeded}/{p_fit.starts_attempted}",
        "",
        f"=== {symbol}: Q-measure (annual units; quote {smile.quote_date}, expiry {smile.expiry}, "
        f"T={smile.T:.4f}y, {first.n_options} OTM options) ===",
        f"martingale-only (paper Sec. 4, no option data): {_fmt(first.rn_params)}",
        f"  IV RMSE {first.rn_iv_rmse:.4f}",
    ]
    for label, q in q_fits.items():
        lines += [
            f"{label}: {_fmt(q.params)}",
            f"  IV RMSE {q.iv_rmse:.4f}; objective {q.objective:.5g}; starts ok {q.starts_succeeded}",
        ]
    lines.append("(objective = mean ((model - mid)/vega)^2, the vega-weighted price error minimized)")
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


def plot_smile(smile, q_fits: dict, out_path, symbol: str) -> Path:
    """Market vs. model implied vols across log-moneyness, one line per fit."""
    k = np.log(smile.strikes / smile.F)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(k, smile.market_iv, "ko", ms=4, label="market (mid)")
    first = next(iter(q_fits.values()))
    ax.plot(k, first.rn_iv, "b--", label=f"martingale-only (RMSE {first.rn_iv_rmse:.3f})")
    for (label, q), style in zip(q_fits.items(), ("r-", "m-.", "g:")):
        ax.plot(k, q.model_iv, style, label=f"{label} (RMSE {q.iv_rmse:.3f})")
    ax.set(xlabel="ln(K/F)", ylabel="implied vol",
           title=f"{symbol} {smile.quote_date} -> {smile.expiry} smile")
    ax.legend(fontsize=8)
    return _save(fig, out_path)


def _save(fig, out_path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
