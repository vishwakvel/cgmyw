"""Lévy vs. Sato, with and without W, on 4-maturity AAPL surfaces (2026-11-02).

    uv run python scripts/run_surface.py            # the 18 report dates
    uv run python scripts/run_surface.py --quick    # the 2 headline dates

Writes report/surface.json and report/figures/surface_*.png.
"""

from __future__ import annotations

import argparse
import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cgmyw.data import COL, load_wrds  # noqa: E402
from cgmyw.pfit import PARAM_ORDER  # noqa: E402
from cgmyw.surface import MODELS, calibrate_surface, parity_forward, select_expiries, surface_smile  # noqa: E402

WRDS = "WRDSdata/aapldata20230103to20250829.dat"
OUT, FIG = Path("report"), Path("report/figures")
HEADLINE = (20230113, 20250829)
STYLE = {"Lévy CGMY": "k:", "Lévy CGMYW": "k-.", "Sato CGMY": "k--", "Sato CGMYW": "k-"}
plt.rcParams.update({"font.family": "serif", "font.size": 9, "figure.dpi": 150})


def report_dates(outmm, step=42):
    qd = np.unique(outmm[:, COL["date"]]).astype(int)
    picks = list(qd[::step]) + ([qd[-1]] if (len(qd) - 1) % step else [])
    return sorted({int(d) for d in picks} | set(HEADLINE))


def _smiles(outmm, qd):
    return [surface_smile(outmm, qd, e) for e in select_expiries(outmm, qd)]


def fit_family(qd, family):
    """CGMY then CGMYW (warm-started from it) for one time dimension."""
    warnings.filterwarnings("ignore")
    outmm = load_wrds(WRDS)
    smiles = _smiles(outmm, qd)
    q = [parity_forward(outmm, qd, s.expiry)[1] for s in smiles]
    base = calibrate_surface(smiles, f"{family} CGMY", n_starts=4)
    full = calibrate_surface(smiles, f"{family} CGMYW", n_starts=4, warm=base)
    return [dict(quote_date=qd, model=f.model, params=dict(zip(PARAM_ORDER, f.params)), gamma=f.gamma,
                 iv_rmse=f.iv_rmse, iv_rmse_by_T=list(f.iv_rmse_by_T), objective=f.objective,
                 T=[s.T for s in smiles], n=[len(s.strikes) for s in smiles], q=q,
                 model_iv=[iv.tolist() for iv in f.model_iv]) for f in (base, full)]


def plot_surface(outmm, qd, fits):
    smiles = _smiles(outmm, qd)
    fig, axs = plt.subplots(1, len(smiles), figsize=(7.2, 2.6), sharey=True)
    for i, (ax, s) in enumerate(zip(axs, smiles)):
        k = np.log(s.strikes / s.F)
        ax.plot(k, s.market_iv, "ko", ms=2.5, mfc="none", label="market")
        for name in ("Lévy CGMYW", "Sato CGMYW"):
            ax.plot(k, fits[name]["model_iv"][i], STYLE[name], lw=0.9, label=name)
        ax.set_title(f"{round(s.T * 365)} days", fontsize=8)
        ax.set_xlabel("ln(K/F)")
    axs[0].set_ylabel("implied vol")
    axs[0].legend(fontsize=6)
    fig.suptitle(f"AAPL {qd}: one parameter set for all four maturities", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / f"surface_{qd}.png")
    plt.close(fig)


def plot_summary(rows, dates):
    by = {(r["quote_date"], r["model"]): r for r in rows}
    x = np.arange(len(dates))
    fig, (a, b) = plt.subplots(2, 1, figsize=(7, 4.6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for m in MODELS:
        a.plot(x, [100 * by[d, m]["iv_rmse"] for d in dates], STYLE[m], marker="o", ms=2.5, lw=0.9, label=m)
    a.set(ylabel="surface IV RMSE (vol pts)", yscale="log")
    a.legend(fontsize=7, ncol=2)
    b.plot(x, [by[d, "Sato CGMYW"]["gamma"] for d in dates], "ko-", ms=3, label="Sato CGMYW")
    b.plot(x, [by[d, "Sato CGMY"]["gamma"] for d in dates], "ks--", ms=3, mfc="none", label="Sato CGMY")
    b.set(ylabel=r"$\gamma$", xticks=x)
    b.set_xticklabels(dates, rotation=60, fontsize=7)
    b.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "surface_summary.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    for m in MODELS:
        ax.plot(range(4), [100 * np.mean([by[d, m]["iv_rmse_by_T"][j] for d in dates if len(by[d, m]["T"]) == 4])
                           for j in range(4)], STYLE[m], marker="o", ms=3, lw=0.9, label=m)
    ax.set(xticks=range(4), xticklabels=["~1 month", "~3 months", "~6 months", "~1 year"],
           ylabel="mean IV RMSE (vol pts)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "surface_by_maturity.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=9)
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore")
    outmm = load_wrds(WRDS)
    dates = list(HEADLINE) if args.quick else report_dates(outmm)
    tasks = [(d, fam) for d in dates for fam in ("Lévy", "Sato")]
    with ProcessPoolExecutor(args.workers) as ex:
        rows = [r for pair in ex.map(fit_family, *zip(*tasks)) for r in pair]
    (OUT / "surface.json").write_text(json.dumps(rows, indent=1, default=float))
    by = {(r["quote_date"], r["model"]): r for r in rows}
    for d in HEADLINE:
        plot_surface(outmm, d, {m: by[d, m] for m in MODELS})
    plot_summary(rows, dates)
    for d in dates:
        print(d, "  ".join(f"{m} {100 * by[d, m]['iv_rmse']:.2f}" for m in MODELS),
              f"gamma {by[d, 'Sato CGMYW']['gamma']:.3f}")


if __name__ == "__main__":
    main()
