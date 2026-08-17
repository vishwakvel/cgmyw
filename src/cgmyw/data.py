"""Data ingestion for the CGMYW pipeline.

Three independent daily-return sources feed the P-fit (plan 01-02), in priority
order per ADR-0006 (second revision, 2026-08-17):

1. CRSP's daily stock file, exported from WRDS by the user (`load_crsp_returns`) --
   primary. Dividend-inclusive, split-corrected `DlyRet`.
2. `yfinance`'s dividend/split-adjusted close (`fetch_yfinance_close_series`) --
   first fallback and cross-check.
3. The WRDS options-file spot column (`daily_close_series`) -- second fallback
   and further cross-check. This is the *only* series Q-measure/options code
   (Phase 1.5) touches; it is decoupled from the P-fit's primary source.

`daily_close_series_for_pfit` is the three-tier resolver plan 01-02 actually calls.
"""

from __future__ import annotations

import csv
import datetime
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("cgmyw.data")

# 0-indexed column map for the (N,13) `outmm` matrix. Single source of truth for
# column indices -- no other module may hardcode an integer column index.
COL = dict(
    date=0,
    expiry=1,
    strike=2,
    cp_flag=3,
    price=4,
    iv=5,
    spot=6,
    forward=7,
    rate=8,
    div_yield=9,
    bid=10,
    ask=11,
    nan_iv_flag=12,
)

DEFAULT_CRSP_PATH = "WRDSdata/aapl_crsp_returns.csv"

# PFIT-02's rolling-window floor: a source must supply at least this many points
# to be accepted by daily_close_series_for_pfit's tier resolution.
_MIN_POINTS = 252

# Cross-check agreement threshold: Pearson correlation of log returns on the
# overlap below this logs a warning (never a hard failure -- ADR-0006).
_CROSS_CHECK_CORR_THRESHOLD = 0.9


def load_wrds(path):
    """Load a WRDS MATLAB v5 .dat file's `outmm` matrix.

    Returns an (N, 13) float64 array. Raises FileNotFoundError naming the
    expected WRDSdata/ location if the path does not exist (ASVS V12: fail
    with a clear error, not a raw stack trace).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"WRDS data file not found at {path}. Expected a MATLAB v5 .dat file "
            "under WRDSdata/ containing a single 'outmm' variable, an (N,13) "
            "float64 matrix."
        )
    import scipy.io as sio

    d = sio.loadmat(str(path))
    outmm = d["outmm"]
    assert outmm.ndim == 2 and outmm.shape[1] == 13, (
        f"expected outmm to be (N,13), got shape {outmm.shape}"
    )
    assert outmm.dtype == np.float64, f"expected outmm dtype float64, got {outmm.dtype}"
    return outmm


def daily_close_series(outmm):
    """DATA-02 (second fallback/cross-check per ADR-0006): derive a daily
    close/log-return series from the WRDS options file's spot column.

    Returns (unique_dates, closes, log_returns) with len(log_returns) ==
    len(closes) - 1. Spot is verified bit-exact constant within every unique
    date on AAPL's file (research: max intra-date spread == 0.0 across all
    667 dates) -- asserted here as a cheap guard so a different stock's file
    cannot silently violate the invariant.
    """
    dates = outmm[:, COL["date"]]
    spot = outmm[:, COL["spot"]]
    uniq_dates, first_idx = np.unique(dates, return_index=True)

    # Cheap guard: spot must be bit-exact constant within each unique date.
    # Vectorized via a stable sort + reduceat rather than an O(N*D) per-date
    # boolean-mask scan, which is too slow at real-file scale (N ~ 1.16M rows).
    sort_idx = np.argsort(dates, kind="stable")
    sorted_dates = dates[sort_idx]
    sorted_spot = spot[sort_idx]
    group_starts = np.searchsorted(sorted_dates, uniq_dates)
    group_max = np.maximum.reduceat(sorted_spot, group_starts)
    group_min = np.minimum.reduceat(sorted_spot, group_starts)
    max_spread = float(np.max(group_max - group_min))
    assert max_spread == 0.0, (
        f"spot price is not constant within every unique date (max spread {max_spread}); "
        "daily_close_series's first-of-date shortcut is unsafe for this file"
    )

    closes = spot[first_idx]
    log_returns = np.diff(np.log(closes))
    return uniq_dates, closes, log_returns


def exclude_nan_iv(outmm):
    """Filter rows where the NaN-IV flag is set (Phase 1.5 Q-side use).

    Filters on the flag column (COL['nan_iv_flag'] != 1), never on the 50.0
    sentinel in the IV column -- the flag is the documented contract.
    """
    return outmm[outmm[:, COL["nan_iv_flag"]] != 1]


def yyyymmdd_to_date(value):
    """Convert a float yyyymmdd encoding (scalar or array) to datetime.date."""
    if np.isscalar(value):
        v = int(value)
        return datetime.date(v // 10000, (v // 100) % 100, v % 100)
    return np.array(
        [yyyymmdd_to_date(v) for v in np.asarray(value).ravel()], dtype=object
    )


def _date_to_yyyymmdd(d) -> float:
    """Convert a datetime.date/datetime.datetime/pandas.Timestamp to a float yyyymmdd."""
    return float(f"{d.year:04d}{d.month:02d}{d.day:02d}")


def fetch_yfinance_close_series(ticker, start, end, _downloader=None):
    """Fetch a dividend/split-adjusted daily close series from yfinance.

    Returns (dates, closes, log_returns) with the same shape contract as
    daily_close_series -- dates encoded as float yyyymmdd so all three P-data
    sources intersect with a single np.intersect1d call.

    _downloader: optional callable(ticker, start, end) -> a yfinance-shaped
    frame, injected in tests so the default suite never touches the network.

    Raises RuntimeError naming the ticker and date range on an empty/all-NaN
    result. Real network exceptions propagate as-is -- this function's job is
    just "fetch or raise"; daily_close_series_for_pfit decides what to do
    about a failure.
    """
    if _downloader is not None:
        raw = _downloader(ticker, start, end)
    else:
        import yfinance as yf

        # auto_adjust=True requests the dividend/split-adjusted series -- under
        # this flag the returned "Close" column *is* the adjusted close (yfinance
        # does not return a separate "Adj Close" column when auto_adjust=True).
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if raw is None or len(raw) == 0:
        raise RuntimeError(
            f"yfinance returned no data for {ticker!r} between {start} and {end}"
        )

    close = raw["Close"]
    if hasattr(close, "columns"):
        # download() with multi_level_index=True (the default, even for a single
        # ticker) returns a DataFrame here, not a Series -- take the one column.
        close = close.iloc[:, 0]

    close = close.dropna()
    if len(close) == 0:
        raise RuntimeError(
            f"yfinance returned all-NaN close data for {ticker!r} between {start} and {end}"
        )

    close = close.sort_index()
    dates = np.array([_date_to_yyyymmdd(d) for d in close.index], dtype=float)
    closes = close.to_numpy(dtype=float)
    log_returns = np.diff(np.log(closes))
    return dates, closes, log_returns


def _normalize_crsp_date(raw: str) -> float:
    """Accept both 'YYYY-MM-DD' (verified real-export format) and bare
    'YYYYMMDD', normalizing to the same float yyyymmdd encoding used
    elsewhere in this module."""
    s = raw.strip()
    if "-" in s:
        y, m, d = s.split("-")
        return float(f"{int(y):04d}{int(m):02d}{int(d):02d}")
    return float(s)


def load_crsp_returns(path: str | Path = DEFAULT_CRSP_PATH):
    """Read CRSP's daily stock file (exported from WRDS's CIZ-format query) as
    the P-fit's primary data source (ADR-0006, second revision).

    Returns (dates, prc, log_returns) with len(log_returns) == len(dates) - 1.
    dates/prc are full-length (N); log_returns[i] is ln(1 + DlyRet) for the
    return realized on dates[i+1] -- the first row's DlyRet is dropped because
    its prior close lies outside the exported window (a deliberate one-element
    drop, not an off-by-one).

    Matches columns by header name, not position or count: the real export has
    8 columns (PERMNO,HdrCUSIP,Ticker,PERMCO,DlyCalDt,DlyPrc,DlyRet,
    DlyRetMissFlg); HdrCUSIP/Ticker/PERMCO are present and ignored.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"CRSP export not found at {path}. Expected a CSV with (at least) "
            "columns PERMNO, DlyCalDt, DlyRet, DlyPrc (optionally "
            "DlyRetMissFlg) -- a WRDS CIZ-format CRSP daily stock file export."
        )

    permnos: set[str] = set()
    rows: list[tuple[float, float, float]] = []
    n_dropped_parse = 0
    n_dropped_missflag = 0
    n_negative_prc = 0

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            permno_raw = row.get("PERMNO")
            if permno_raw is not None:
                permnos.add(permno_raw.strip())

            missflag = (row.get("DlyRetMissFlg") or "").strip()
            if missflag and missflag.upper() != "NA":
                n_dropped_missflag += 1
                continue

            raw_ret = (row.get("DlyRet") or "").strip()
            raw_prc = (row.get("DlyPrc") or "").strip()
            try:
                ret = float(raw_ret)
                prc = float(raw_prc)
            except (TypeError, ValueError):
                n_dropped_parse += 1
                continue

            if prc < 0:
                n_negative_prc += 1
            prc = abs(prc)

            raw_date = (row.get("DlyCalDt") or "").strip()
            date_f = _normalize_crsp_date(raw_date)

            rows.append((date_f, prc, ret))

    if len(permnos) > 1:
        raise ValueError(
            f"CRSP export at {path} contains multiple PERMNOs {sorted(permnos)}; "
            "expected a single-security export. A multi-PERMNO export means the "
            "wrong WRDS query was run."
        )

    if n_dropped_parse:
        logger.warning(
            "load_crsp_returns(%s): dropped %d row(s) with unparseable DlyRet/DlyPrc",
            path, n_dropped_parse,
        )
    if n_dropped_missflag:
        logger.warning(
            "load_crsp_returns(%s): dropped %d row(s) flagged by DlyRetMissFlg",
            path, n_dropped_missflag,
        )
    if n_negative_prc:
        logger.warning(
            "load_crsp_returns(%s): %d row(s) had negative DlyPrc (CRSP's "
            "bid/ask-midpoint marker for an absent closing trade); took abs()",
            path, n_negative_prc,
        )

    rows.sort(key=lambda r: r[0])

    dates = np.array([r[0] for r in rows], dtype=float)
    prc = np.array([r[1] for r in rows], dtype=float)
    ret = np.array([r[2] for r in rows], dtype=float)

    # Deliberate one-element drop: the first row's DlyRet spans a period
    # beginning before the export window, so it is not used to compute a
    # within-window log return. dates/prc stay full-length.
    log_returns = np.log(1.0 + ret[1:])

    return dates, prc, log_returns


def _cross_check(dates_a, log_returns_a, dates_b, log_returns_b, name_a, name_b):
    """Soft cross-check: Pearson correlation of log returns on overlapping
    dates. Disagreement always logs a warning, never raises (ADR-0006)."""
    dates_a_ret = np.asarray(dates_a[1:])
    dates_b_ret = np.asarray(dates_b[1:])
    common, idx_a, idx_b = np.intersect1d(dates_a_ret, dates_b_ret, return_indices=True)
    if len(common) < 10:
        logger.warning(
            "cross-check %s vs %s: only %d overlapping dates, skipping correlation check",
            name_a, name_b, len(common),
        )
        return
    a = np.asarray(log_returns_a)[idx_a]
    b = np.asarray(log_returns_b)[idx_b]
    corr = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(corr) or corr < _CROSS_CHECK_CORR_THRESHOLD:
        logger.warning(
            "cross-check %s vs %s: correlation %.4f below threshold %.2f on %d overlapping dates",
            name_a, name_b, corr, _CROSS_CHECK_CORR_THRESHOLD, len(common),
        )
    return corr


def daily_close_series_for_pfit(
    outmm, ticker, start=None, end=None, crsp_path=DEFAULT_CRSP_PATH, _downloader=None
):
    """Three-tier resolver for the P-fit's daily return series (ADR-0006,
    second revision, 2026-08-17). This is what plan 01-02's P-fit actually calls.

    Tries load_crsp_returns(crsp_path) first; accepts it when it yields at
    least 252 points (PFIT-02's rolling-window floor) and returns
    source="crsp". On failure/insufficiency, tries fetch_yfinance_close_series
    (passing _downloader through) with the same floor, returning
    source="yfinance". On failure/insufficiency there too, falls back to
    daily_close_series(outmm), returning source="wrds_fallback".

    Cross-checks whichever tiers are simultaneously available on overlapping
    dates (soft warning on mismatch, never a hard failure). When CRSP
    succeeds, cross-checks against the WRDS-derived series (always local,
    zero network) and against yfinance only if a _downloader was injected --
    never initiates a live network fetch purely to cross-check a series that
    already succeeded.
    """
    wrds_dates, wrds_closes, wrds_log_returns = daily_close_series(outmm)

    if start is None or end is None:
        wrds_dates_as_dates = [yyyymmdd_to_date(d) for d in wrds_dates]
        if start is None:
            start = min(wrds_dates_as_dates)
        if end is None:
            end = max(wrds_dates_as_dates)

    # Tier 1: CRSP
    crsp_result = None
    try:
        c_dates, c_prc, c_log_returns = load_crsp_returns(crsp_path)
        if len(c_dates) < _MIN_POINTS:
            logger.warning(
                "load_crsp_returns(%s): only %d points (< %d floor), trying fallback",
                crsp_path, len(c_dates), _MIN_POINTS,
            )
        else:
            crsp_result = (c_dates, c_prc, c_log_returns)
    except FileNotFoundError as e:
        logger.warning("load_crsp_returns unavailable (%s), trying fallback", e)
    except ValueError as e:
        logger.warning("load_crsp_returns parse failure (%s), trying fallback", e)

    if crsp_result is not None:
        dates, prc, log_returns = crsp_result
        _cross_check(dates, log_returns, wrds_dates, wrds_log_returns, "crsp", "wrds_fallback")
        if _downloader is not None:
            try:
                y_dates, _y_closes, y_log_returns = fetch_yfinance_close_series(
                    ticker, start, end, _downloader=_downloader
                )
                _cross_check(dates, log_returns, y_dates, y_log_returns, "crsp", "yfinance")
            except Exception as e:  # noqa: BLE001 - cross-check is best-effort
                logger.warning("yfinance cross-check fetch failed: %s", e)
        return dates, prc, log_returns, "crsp"

    # Tier 2: yfinance
    try:
        y_dates, y_closes, y_log_returns = fetch_yfinance_close_series(
            ticker, start, end, _downloader=_downloader
        )
        if len(y_dates) >= _MIN_POINTS:
            _cross_check(y_dates, y_log_returns, wrds_dates, wrds_log_returns, "yfinance", "wrds_fallback")
            return y_dates, y_closes, y_log_returns, "yfinance"
        logger.warning(
            "fetch_yfinance_close_series: only %d points (< %d floor), falling back to wrds",
            len(y_dates), _MIN_POINTS,
        )
    except Exception as e:  # noqa: BLE001 - any fetch failure falls through to tier 3
        logger.warning("fetch_yfinance_close_series failed (%s), falling back to wrds", e)

    # Tier 3: WRDS options-file spot column
    return wrds_dates, wrds_closes, wrds_log_returns, "wrds_fallback"
