"""One command, raw WRDS/CRSP files -> P-fit -> Q-fit -> report + plots.

    uv run python scripts/run_pipeline.py                      # AAPL, auto-selected smile
    uv run python scripts/run_pipeline.py --quote-date 20250829
    uv run python scripts/run_pipeline.py --symbol MSFT

Nothing here is AAPL-specific: input paths default from --symbol, so a
stock never silently picks up another stock's files.
"""

from __future__ import annotations

import argparse
import logging

from cgmyw.data import daily_close_series_for_pfit, load_wrds, total_return_index
from cgmyw.pfit import fit_p_measure, trailing_window_returns
from cgmyw.qfit import C_FREE, PINNED, UNRESTRICTED, build_smile, calibrate_q, select_date_maturity
from cgmyw.report import format_report, plot_smile, plot_tail_fit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="AAPL")
    ap.add_argument("--wrds", help="default: WRDSdata/<symbol>data20230103to20250829.dat")
    ap.add_argument("--crsp", help="default: WRDSdata/<symbol>_crsp_returns.csv")
    ap.add_argument("--quote-date", type=int, help="yyyymmdd; default: best data-quality date in the file")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--main-only", action="store_true", help="skip the strict-equivalence and unrestricted comparison fits")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    sym = args.symbol.lower()
    args.wrds = args.wrds or f"WRDSdata/{sym}data20230103to20250829.dat"
    args.crsp = args.crsp or f"WRDSdata/{sym}_crsp_returns.csv"

    outmm = load_wrds(args.wrds)
    quote_date, expiry = select_date_maturity(outmm, args.quote_date)
    smile = build_smile(outmm, quote_date, expiry)

    dates, _, log_returns, source = daily_close_series_for_pfit(outmm, args.symbol, crsp_path=args.crsp)
    # Window the total-return index, never raw prices (CRSP DlyPrc drops dividends).
    window_dates, returns = trailing_window_returns(dates, total_return_index(log_returns), quote_date)
    p_fit = fit_p_measure(returns, window_dates=window_dates)

    # Main fit: Cn, Cp free, Yn, Yp pinned (Dr. Madan, 2026-10-19). The other
    # two bracket it: strict equivalence (the paper) and fully unrestricted.
    searches = {"Q, Cn/Cp free (main)": C_FREE}
    if not args.main_only:
        searches |= {"Q, strictly equivalent": PINNED, "Q, unrestricted (diagnostic)": UNRESTRICTED}
    q_fits = {label: calibrate_q(p_fit.as_tuple(), smile, search=s) for label, s in searches.items()}

    print(f"P-data source: {source}")
    print(format_report(args.symbol, p_fit, smile, q_fits))
    stem = f"{args.out}/{args.symbol.lower()}_{quote_date}"
    print(plot_tail_fit(p_fit, returns, f"{stem}_pfit_tails.png", args.symbol))
    print(plot_smile(smile, q_fits, f"{stem}_smile.png", args.symbol))


if __name__ == "__main__":
    main()
