"""Every number and figure in report/report.tex, from one command.

    uv run python scripts/make_report.py            # AAPL, ~15 quote dates
    uv run python scripts/make_report.py --quick    # 2 headline dates only

Writes report/figures/*.png and report/results.json. Three parts:
1. Correctness checks against independent references (no market data).
2. Headline AAPL dates: P-fit, all three Q searches, and an out-of-sample
   maturity priced with the main fit's parameters.
3. Stability: the main P-then-Q fit repeated across quote dates.
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

from cgmyw.data import COL, daily_close_series_for_pfit, load_wrds, total_return_index  # noqa: E402
from cgmyw.model import psi_cgmy_classical, psi_cgmyw  # noqa: E402
from cgmyw.pfit import (  # noqa: E402
    PARAM_ORDER,
    compute_threshold_grid,
    empirical_tail_probabilities,
    fit_p_measure,
    levy_density_grid,
    model_tail_probabilities,
    normalize_to_pdf,
    trailing_window_returns,
)
from cgmyw.pricing import black76, carr_madan_calls, cgmyw_prices, implied_vols  # noqa: E402
from cgmyw.qfit import (  # noqa: E402
    C_FREE,
    PINNED,
    UNRESTRICTED,
    _dte,
    _usable,
    build_smile,
    calibrate_q,
    risk_neutralize,
    select_date_maturity,
)

OUT = Path("report")
FIG = OUT / "figures"
WRDS = "WRDSdata/aapldata20230103to20250829.dat"
CRSP = "WRDSdata/aapl_crsp_returns.csv"
HEADLINE = (20230113, 20250829)
plt.rcParams.update({"font.family": "serif", "font.size": 9, "figure.dpi": 150})

# Madan & Wang (2026), Table 1 (Cn,Cp,G,M,Yn,Yp,Wn,Wp, daily) and Table 2 (G,M,Wn,Wp).
PAPER = {
    "AMZN": ((2.9224, 3.2505, 109.07, 118.77, 0.0125, 0.0128, 7.7120, 8.2070), (108.00, 119.74, 6.6417, 9.1668)),
    "GOOGL": ((2.7891, 3.3206, 112.92, 131.78, 0.0138, 0.0142, 14.354, 14.329), (111.17, 133.24, 12.620, 15.783)),
    "META": ((1.3528, 1.4049, 75.518, 74.100, 0.0480, 0.0501, 4.1798, 4.9331), (74.043, 75.500, 2.7248, 6.3170)),
    "MSFT": ((1.4915, 1.9141, 101.76, 120.6, 0.0042, 0.0042, 0.0001, 7.6325), (100.55, 121.64, 0.0000, 8.5691)),
    "NVDA": ((3.5501, 4.5440, 73.823, 93.769, 0.0122, 0.0125, 12.840, 12.875), (72.387, 94.885, 11.421, 13.981)),
    "AAPL": ((0.9515, 1.3436, 51.983, 97.634, 0.0030, 0.0031, 22.269, 4.8288), (51.198, 98.183, 21.514, 5.3734)),
    "IBM": ((1.4140, 0.9536, 108.36, 121.22, 0.0018, 0.1111, 7.0068, 10.629), (105.31, 123.44, 3.9886, 12.826)),
    "CSCO": ((1.2301, 1.3965, 109.20, 118.81, 0.0062, 0.0129, 7.8315, 8.2510), (106.44, 121.17, 5.0940, 10.587)),
    "INTC": ((1.7506, 1.6823, 63.443, 60.158, 0.0002, 0.0002, 1.0097, 1.0438), (62.965, 60.625, 0.6385, 1.4847)),
    "ORCL": ((1.3745, 1.9320, 51.973, 97.615, 0.0030, 0.0031, 22.259, 4.8127), (51.289, 98.098, 21.601, 5.2944)),
}


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name)
    plt.close(fig)


# ---------------------------------------------------------------- 1. checks
def check_model_reduction():
    """Wn=Wp=0, Cn=Cp, Yn=Yp must give classical CGMY exactly (MODEL-01)."""
    u = np.linspace(-300, 300, 2001)
    cases = [(1.0, 12.0, 15.0, 0.6), (0.5, 8.0, 10.0, 1.4), (2.0, 20.0, 25.0, 0.95)]
    fig, ax = plt.subplots(figsize=(5.5, 3))
    out = []
    for (C, G, M, Y), ls in zip(cases, ("-", "--", ":")):
        err = np.abs(psi_cgmyw(u, C, C, G, M, Y, Y, 0.0, 0.0) - psi_cgmy_classical(u, C, G, M, Y))
        out.append(dict(C=C, G=G, M=M, Y=Y, max_abs_err=float(err.max())))
        ax.semilogy(u, np.maximum(err, 1e-17), "k" + ls, lw=0.8, label=f"C={C}, G={G}, M={M}, Y={Y}")
    ax.axhline(1e-8, color="0.5", lw=0.6)
    ax.text(-290, 1.6e-8, "pass threshold 1e-8", fontsize=7, color="0.4")
    ax.set(xlabel="u", ylabel=r"$|\psi_{CGMYW}-\psi_{CGMY}|$", ylim=(1e-17, 1e-6))
    ax.legend(fontsize=7, loc="lower right")
    _save(fig, "check_reduction.png")
    return out


def check_power_tilt():
    """W != 0: closed form vs. direct mpmath quadrature of int (e^{iux}-1) k(x) dx."""
    import mpmath as mp

    mp.mp.dps = 30
    p = dict(Cn=1.0, Cp=1.3, G=9.7, M=6.2, Yn=0.6, Yp=0.7, Wn=0.5, Wp=0.9)
    split = [0, mp.mpf("1e-8"), mp.mpf("1e-4"), mp.mpf("1e-2"), mp.mpf("0.1"), 1, 10, mp.inf]

    def numeric(u):
        def side(C, lam, Y, W, sign):
            f = lambda x: (mp.e ** (sign * 1j * u * x) - 1) * C * mp.e ** (-lam * x) / (x ** (1 + Y) * (1 + x) ** W)  # noqa: E731
            return mp.quad(lambda x: mp.re(f(x)), split) + 1j * mp.quad(lambda x: mp.im(f(x)), split)

        return complex(side(p["Cn"], p["G"], p["Yn"], p["Wn"], -1) + side(p["Cp"], p["M"], p["Yp"], p["Wp"], 1))

    us = [0.25, 0.5, 1, 2, 4, 8, 16, 32]
    rel = [abs(complex(psi_cgmyw(u, *p.values())) - numeric(u)) / abs(numeric(u)) for u in us]
    return dict(params=p, u=us, rel_err=[float(r) for r in rel])


def check_carr_madan():
    """FFT pricer vs. closed-form Black-Scholes, across strikes."""
    F, df, T, sig = 100.0, 0.99, 0.1, 0.25
    K = np.linspace(60, 160, 201)
    cf = lambda u: np.exp(-0.5 * sig**2 * T * (u**2 + 1j * u))  # noqa: E731
    err = np.abs(carr_madan_calls(cf, F, df, K) - black76(F, K, T, df, sig, True))
    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    ax.semilogy(K, np.maximum(err, 1e-16), "k-", lw=0.8)
    ax.set(xlabel="strike (F = 100)", ylabel="|Carr-Madan - Black-Scholes|")
    _save(fig, "check_carr_madan.png")
    return dict(max_abs_err=float(err.max()))


def check_table2():
    """Our risk-neutralization from the paper's Table 1 vs. its Table 2."""
    rows = {}
    for tic, (p, q_paper) in PAPER.items():
        q = risk_neutralize(p, 0.0)
        ours = (q[2], q[3], q[6], q[7])
        rows[tic] = dict(paper=q_paper, ours=[float(v) for v in ours],
                         max_rel_err=float(max(abs(a - b) / max(abs(b), 1e-2) for a, b in zip(ours, q_paper))))
    return rows


# --------------------------------------------------------------- 2/3. AAPL
def _load():
    outmm = load_wrds(WRDS)
    dates, _, lr, source = daily_close_series_for_pfit(outmm, "AAPL", crsp_path=CRSP)
    return outmm, dates, total_return_index(lr), source


def _tail_rmse(p_fit, returns):
    t = compute_threshold_grid(returns)
    x, pdf, _ = normalize_to_pdf(*levy_density_grid(*p_fit.as_tuple()))
    return float(np.sqrt(np.mean((empirical_tail_probabilities(returns, t) - model_tail_probabilities(x, pdf, t)) ** 2)))


def _oos_expiry(outmm, quote_date, used_expiry):
    """Longest-chain expiry 60-120 calendar days out: a maturity the Q fit never saw."""
    rows = outmm[outmm[:, COL["date"]] == quote_date]
    best = None
    for e in np.unique(rows[:, COL["expiry"]]):
        dte = int(_dte([quote_date], [e])[0])
        n = int(_usable(rows[rows[:, COL["expiry"]] == e]).sum())
        if e != used_expiry and 60 <= dte <= 120 and n >= 6 and (best is None or n > best[1]):
            best = (int(e), n)
    return None if best is None else best[0]


def _iv_rmse(params, smile):
    iv = implied_vols(cgmyw_prices(params, smile.F, smile.df, smile.T, smile.strikes, smile.is_call),
                      smile.F, smile.strikes, smile.T, smile.df, smile.is_call)
    return float(np.sqrt(np.nanmean((iv - smile.market_iv) ** 2))), iv


def study_date(quote_date, all_searches=False):
    warnings.filterwarnings("ignore")
    outmm, dates, tri, _ = _load()
    qd, expiry = select_date_maturity(outmm, quote_date)
    smile = build_smile(outmm, qd, expiry)
    wd, returns = trailing_window_returns(dates, tri, qd)
    p = fit_p_measure(returns, window_dates=wd)
    searches = {"main": C_FREE} | ({"pinned": PINNED, "unrestricted": UNRESTRICTED} if all_searches else {})
    q = {k: calibrate_q(p.as_tuple(), smile, search=s) for k, s in searches.items()}
    res = dict(
        quote_date=qd, expiry=expiry, T=smile.T, n_options=len(smile.strikes), atm_iv=float(
            smile.market_iv[np.argmin(np.abs(np.log(smile.strikes / smile.F)))]),
        p=dict(zip(PARAM_ORDER, p.as_tuple())), p_ad=p.ad_objective, p_tail_rmse=_tail_rmse(p, returns),
        p_starts=f"{p.starts_succeeded}/{p.starts_attempted}", rn_iv_rmse=q["main"].rn_iv_rmse,
        q={k: dict(params=dict(zip(PARAM_ORDER, v.params)), iv_rmse=v.iv_rmse) for k, v in q.items()},
    )
    oos = _oos_expiry(outmm, qd, expiry)
    if oos is not None:
        s2 = build_smile(outmm, qd, oos)
        rmse, iv2 = _iv_rmse(q["main"].params, s2)
        res["oos"] = dict(expiry=oos, T=s2.T, n_options=len(s2.strikes), iv_rmse=rmse)
    if all_searches:
        _plot_headline(qd, p, returns, smile, q, s2 if oos is not None else None, iv2 if oos is not None else None)
    return res


def _plot_headline(qd, p, returns, smile, q, s2, iv2):
    t = compute_threshold_grid(returns)
    x, pdf, _ = normalize_to_pdf(*levy_density_grid(*p.as_tuple()))
    fig, (a, b) = plt.subplots(1, 2, figsize=(7, 2.8))
    a.plot(t, empirical_tail_probabilities(returns, t), "ko", mfc="none", ms=5, label="empirical")
    a.plot(t, model_tail_probabilities(x, pdf, t), "k.", ms=4, label="CGMYW (P)")
    a.set(xlabel="daily log-return threshold", ylabel="tail probability")
    a.legend(fontsize=7)
    b.hist(returns, bins=50, density=True, color="0.8", edgecolor="0.5", lw=0.3, label="returns")
    m = (x > returns.min() * 1.3) & (x < returns.max() * 1.3)
    b.plot(x[m], pdf[m], "k-", lw=0.9, label="CGMYW density")
    b.set(xlabel="daily log return", ylabel="density")
    b.legend(fontsize=7)
    _save(fig, f"pfit_{qd}.png")

    k = np.log(smile.strikes / smile.F)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(k, smile.market_iv, "ko", ms=3.5, mfc="none", label="market (mid)")
    styles = {"main": ("k-", "main fit: $C_n,C_p$ free"), "pinned": ("k--", "strictly equivalent"),
              "unrestricted": ("k:", "unrestricted (diagnostic)")}
    for key, (ls, lab) in styles.items():
        ax.plot(k, q[key].model_iv, ls, lw=1, label=f"{lab} ({100 * q[key].iv_rmse:.2f} vol pts)")
    ax.set(xlabel="log-moneyness ln(K/F)", ylabel="implied volatility",
           title=f"AAPL {qd}, {round(smile.T * 365)}-day smile")
    ax.legend(fontsize=7)
    _save(fig, f"smile_{qd}.png")

    if s2 is not None:
        fig, ax = plt.subplots(figsize=(5.5, 2.8))
        ax.plot(np.log(s2.strikes / s2.F), s2.market_iv, "ko", ms=3.5, mfc="none", label="market (mid)")
        ax.plot(np.log(s2.strikes / s2.F), iv2, "k-", lw=1, label="main-fit parameters, not refit")
        ax.set(xlabel="log-moneyness ln(K/F)", ylabel="implied volatility",
               title=f"Out of sample: {round(s2.T * 365)}-day smile, same date")
        ax.legend(fontsize=7)
        _save(fig, f"oos_{qd}.png")


def plot_stability(rows):
    d = [str(r["quote_date"]) for r in rows]
    x = np.arange(len(rows))
    fig, (a, b) = plt.subplots(2, 1, figsize=(7, 4.6), sharex=True)
    a.plot(x, [100 * r["q"]["main"]["iv_rmse"] for r in rows], "ko-", ms=3, label="main Q fit (in sample)")
    a.plot(x, [100 * r["oos"]["iv_rmse"] if "oos" in r else np.nan for r in rows], "ks--", ms=3, mfc="none",
           label="longer maturity, not refit (out of sample)")
    a.plot(x, [100 * r["rn_iv_rmse"] for r in rows], "k^:", ms=3, mfc="none", label="P + martingale only (no options)")
    a.set(ylabel="IV RMSE (vol points)", yscale="log")
    a.legend(fontsize=7)
    b.plot(x, [r["p_tail_rmse"] for r in rows], "ko-", ms=3)
    b.axhline(0.0067, color="0.5", lw=0.7)
    b.text(0, 0.0069, "Madan-Wang Table 1, AAPL (1033-day window)", fontsize=7, color="0.4")
    b.set(ylabel="P tail-prob. RMSE", xticks=x)
    b.set_xticklabels(d, rotation=60, fontsize=7)
    _save(fig, "stability.png")

    fig, axs = plt.subplots(2, 2, figsize=(7, 4.2), sharex=True)
    for ax, (names, lab) in zip(axs.flat, [(("Cn", "Cp"), "C (daily)"), (("G", "M"), "G, M"),
                                          (("Yn", "Yp"), "Y"), (("Wn", "Wp"), "W")]):
        for n, ls in zip(names, ("o-", "s--")):
            ax.plot(x, [r["p"][n] for r in rows], "k" + ls, ms=3, mfc="none" if "s" in ls else "k", label=n)
        ax.set(ylabel=lab, xticks=x)
        ax.set_xticklabels(d, rotation=60, fontsize=6)
        ax.legend(fontsize=7)
    _save(fig, "p_params.png")


def stability_dates(outmm, step=42):
    """Every `step`-th quote date from the first with a 252-day P window."""
    qd = np.unique(outmm[:, COL["date"]]).astype(int)
    picks = list(qd[::step]) + ([qd[-1]] if (len(qd) - 1) % step else [])
    return [int(d) for d in picks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore")

    outmm, _, _, source = _load()
    res = dict(data_source=source, checks=dict(
        reduction=check_model_reduction(), power_tilt=check_power_tilt(),
        carr_madan=check_carr_madan(), table2=check_table2()))
    print(json.dumps(res["checks"], indent=1, default=str))

    dates = [] if args.quick else [d for d in stability_dates(outmm) if d not in HEADLINE]
    with ProcessPoolExecutor(args.workers) as ex:
        heads = [ex.submit(study_date, d, True) for d in HEADLINE]
        rest = list(ex.map(study_date, dates))
        res["headline"] = [h.result() for h in heads]
    res["stability"] = sorted(rest + res["headline"], key=lambda r: r["quote_date"])
    plot_stability(res["stability"])
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float))
    for r in res["stability"]:
        print(r["quote_date"], r["expiry"], f"P {r['p_tail_rmse']:.4f}", f"Q {100 * r['q']['main']['iv_rmse']:.2f}",
              f"oos {100 * r['oos']['iv_rmse']:.2f}" if "oos" in r else "")


if __name__ == "__main__":
    main()
