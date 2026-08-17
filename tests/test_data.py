"""Tests for cgmyw.data: WRDS .dat loading, CRSP CSV reading, yfinance fetch,
and the three-tier P-fit data-source resolver (ADR-0006, second revision)."""

import csv
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cgmyw.data import (
    COL,
    DEFAULT_CRSP_PATH,
    daily_close_series,
    daily_close_series_for_pfit,
    exclude_nan_iv,
    fetch_yfinance_close_series,
    load_crsp_returns,
    yyyymmdd_to_date,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_CRSP_PATH = REPO_ROOT / DEFAULT_CRSP_PATH


# ---------------------------------------------------------------------------
# WRDS .dat loading / column-mapped access
# ---------------------------------------------------------------------------


def test_load_wrds_shape_dtype(aapl_outmm):
    assert aapl_outmm.shape == (1161438, 13)
    assert aapl_outmm.dtype == np.float64


def test_spot_constant_within_date(aapl_outmm):
    dates = aapl_outmm[:, COL["date"]]
    spot = aapl_outmm[:, COL["spot"]]
    uniq_dates = np.unique(dates)
    sort_idx = np.argsort(dates, kind="stable")
    sorted_dates = dates[sort_idx]
    sorted_spot = spot[sort_idx]
    starts = np.searchsorted(sorted_dates, uniq_dates)
    gmax = np.maximum.reduceat(sorted_spot, starts)
    gmin = np.minimum.reduceat(sorted_spot, starts)
    assert float(np.max(gmax - gmin)) == 0.0


def test_daily_close_series_shapes(aapl_outmm):
    dates, closes, log_returns = daily_close_series(aapl_outmm)
    assert len(dates) == 667
    assert len(closes) == 667
    assert len(log_returns) == 666
    assert np.all(closes > 0)
    assert np.all(np.abs(log_returns) < 0.5)


def test_exclude_nan_iv(aapl_outmm):
    filtered = exclude_nan_iv(aapl_outmm)
    n_total = aapl_outmm.shape[0]
    n_kept = filtered.shape[0]
    frac_dropped = 1.0 - n_kept / n_total
    assert 0.0 < frac_dropped < 1.0

    flag_col = aapl_outmm[:, COL["nan_iv_flag"]]
    dropped_rows = aapl_outmm[flag_col == 1]
    assert dropped_rows.shape[0] > 0
    assert np.all(dropped_rows[:, COL["nan_iv_flag"]] == 1)
    assert np.all(filtered[:, COL["nan_iv_flag"]] != 1)


def test_yyyymmdd_to_date_scalar():
    d = yyyymmdd_to_date(20230103.0)
    assert d == datetime.date(2023, 1, 3)


def test_yyyymmdd_to_date_array():
    arr = yyyymmdd_to_date(np.array([20230103.0, 20230104.0]))
    assert list(arr) == [datetime.date(2023, 1, 3), datetime.date(2023, 1, 4)]


# ---------------------------------------------------------------------------
# yfinance fetch (via injected _downloader -- default suite never hits network)
# ---------------------------------------------------------------------------


def _fake_downloader(dates, closes):
    idx = pd.to_datetime(dates)
    frame = pd.DataFrame({"Close": closes}, index=idx)

    def _downloader(ticker, start, end):
        return frame

    return _downloader


def test_fetch_yfinance_close_series_fake_downloader():
    dates = [datetime.date(2024, 1, d) for d in range(1, 11)]
    closes = 100.0 + np.cumsum(np.full(10, 0.5))
    downloader = _fake_downloader(dates, closes)

    out_dates, out_closes, out_log_returns = fetch_yfinance_close_series(
        "AAPL", "2024-01-01", "2024-01-10", _downloader=downloader
    )
    assert len(out_dates) == 10
    assert len(out_closes) == 10
    assert len(out_log_returns) == 9
    assert out_dates[0] == 20240101.0
    assert out_dates[-1] == 20240110.0


def test_fetch_yfinance_close_series_empty_downloader_raises():
    def empty_downloader(ticker, start, end):
        return pd.DataFrame({"Close": []})

    with pytest.raises(RuntimeError):
        fetch_yfinance_close_series("AAPL", "2024-01-01", "2024-01-10", _downloader=empty_downloader)


@pytest.mark.network
def test_fetch_yfinance_close_series_real_network_smoke():
    dates, closes, log_returns = fetch_yfinance_close_series("AAPL", "2024-01-01", "2024-02-01")
    assert len(dates) > 0
    assert len(closes) == len(dates)
    assert len(log_returns) == len(dates) - 1


# ---------------------------------------------------------------------------
# CRSP CSV reader (load_crsp_returns)
# ---------------------------------------------------------------------------

_CRSP_HEADER = ["PERMNO", "HdrCUSIP", "Ticker", "PERMCO", "DlyCalDt", "DlyPrc", "DlyRet", "DlyRetMissFlg"]


def _write_crsp_csv(tmp_path, rows, permno="14593", filename="crsp.csv"):
    """rows: list of (date_str, prc, ret, missflag='NA')."""
    path = tmp_path / filename
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_CRSP_HEADER)
        for row in rows:
            date_s, prc, ret = row[0], row[1], row[2]
            missflag = row[3] if len(row) > 3 else "NA"
            row_permno = row[4] if len(row) > 4 else permno
            writer.writerow([row_permno, "03783310", "AAPL", "7", date_s, prc, ret, missflag])
    return path


def test_load_crsp_returns_happy_path_shapes(tmp_path):
    rows = [
        ("2024-01-01", 100.0, 0.0),
        ("2024-01-02", 101.0, 0.01),
        ("2024-01-03", 99.0, -0.0198),
        ("2024-01-04", 100.0, 0.0101),
    ]
    path = _write_crsp_csv(tmp_path, rows)
    dates, prc, log_returns = load_crsp_returns(path)
    assert len(dates) == 4
    assert len(prc) == 4
    assert len(log_returns) == 3
    assert np.all(prc > 0)


def test_load_crsp_returns_alignment(tmp_path):
    rows = [
        ("2024-01-01", 100.0, 0.0),  # first row's DlyRet must be dropped
        ("2024-01-02", 101.0, 0.01),
        ("2024-01-03", 99.0, -0.0198),
    ]
    path = _write_crsp_csv(tmp_path, rows)
    dates, prc, log_returns = load_crsp_returns(path)
    assert dates[1] == 20240102.0
    assert dates[2] == 20240103.0
    assert np.isclose(log_returns[0], np.log(1 + 0.01))
    assert np.isclose(log_returns[1], np.log(1 - 0.0198))


def test_load_crsp_returns_file_not_found(tmp_path):
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError, match=str(missing)):
        load_crsp_returns(missing)


def test_load_crsp_returns_multi_permno_raises(tmp_path):
    rows = [
        ("2024-01-01", 100.0, 0.0, "NA", "14593"),
        ("2024-01-02", 101.0, 0.01, "NA", "99999"),
    ]
    path = _write_crsp_csv(tmp_path, rows)
    with pytest.raises(ValueError):
        load_crsp_returns(path)


def test_load_crsp_returns_drops_bad_rows(tmp_path):
    rows = [
        ("2024-01-01", 100.0, 0.0),
        ("2024-01-02", 101.0, "NOT_A_NUMBER"),  # unparseable DlyRet -> dropped
        ("2024-01-03", 99.0, -0.02, "S"),  # non-NA missflag -> dropped
        ("2024-01-04", 100.0, 0.0101),
    ]
    path = _write_crsp_csv(tmp_path, rows)
    dates, prc, log_returns = load_crsp_returns(path)
    # survivors: 2024-01-01, 2024-01-04 (rows 2 and 3 dropped), then first row's
    # own DlyRet is also dropped by the alignment rule -> 1 log return remains
    assert len(dates) == 2
    assert dates[0] == 20240101.0
    assert dates[1] == 20240104.0
    assert len(log_returns) == 1
    assert np.isclose(log_returns[0], np.log(1 + 0.0101))


def test_load_crsp_returns_negative_prc_becomes_positive(tmp_path):
    rows = [
        ("2024-01-01", -100.0, 0.0),
        ("2024-01-02", 101.0, 0.01),
    ]
    path = _write_crsp_csv(tmp_path, rows)
    dates, prc, log_returns = load_crsp_returns(path)
    assert prc[0] == 100.0
    assert np.all(prc > 0)


def test_load_crsp_returns_accepts_both_date_formats(tmp_path):
    rows_iso = [("2024-01-01", 100.0, 0.0), ("2024-01-02", 101.0, 0.01)]
    rows_bare = [("20240101", 100.0, 0.0), ("20240102", 101.0, 0.01)]
    path_iso = _write_crsp_csv(tmp_path, rows_iso, filename="iso.csv")
    path_bare = _write_crsp_csv(tmp_path, rows_bare, filename="bare.csv")
    dates_iso, _, _ = load_crsp_returns(path_iso)
    dates_bare, _, _ = load_crsp_returns(path_bare)
    assert list(dates_iso) == list(dates_bare) == [20240101.0, 20240102.0]


def test_load_crsp_returns_real_file():
    if not REAL_CRSP_PATH.exists():
        pytest.skip("CRSP export not present")
    dates, prc, log_returns = load_crsp_returns()
    assert len(dates) >= 252
    assert len(log_returns) == len(dates) - 1
    assert np.all(prc > 0)


# ---------------------------------------------------------------------------
# daily_close_series_for_pfit: three-tier resolver
# ---------------------------------------------------------------------------


def _yyyymmdd_float_to_iso(d: float) -> str:
    v = int(d)
    return f"{v // 10000:04d}-{(v // 100) % 100:02d}-{v % 100:02d}"


def test_pfit_resolver_yfinance_tier_when_crsp_absent(tmp_path, aapl_outmm):
    missing_crsp = tmp_path / "missing_crsp.csv"
    rng = np.random.default_rng(0)
    n = 300
    dates = [datetime.date(2023, 1, 1) + datetime.timedelta(days=i) for i in range(n)]
    closes = 150.0 + np.cumsum(rng.normal(0, 0.1, size=n))
    downloader = _fake_downloader(dates, closes)

    out_dates, out_closes, out_log_returns, source = daily_close_series_for_pfit(
        aapl_outmm, "AAPL", crsp_path=missing_crsp, _downloader=downloader
    )
    assert source == "yfinance"
    assert len(out_dates) == n
    assert len(out_log_returns) == n - 1


def test_pfit_resolver_wrds_fallback_when_crsp_absent_and_downloader_fails(tmp_path, aapl_outmm):
    missing_crsp = tmp_path / "missing_crsp.csv"

    def failing_downloader(ticker, start, end):
        raise RuntimeError("network down")

    wrds_dates, wrds_closes, wrds_log_returns = daily_close_series(aapl_outmm)
    out_dates, out_closes, out_log_returns, source = daily_close_series_for_pfit(
        aapl_outmm, "AAPL", crsp_path=missing_crsp, _downloader=failing_downloader
    )
    assert source == "wrds_fallback"
    assert len(out_dates) == len(wrds_dates)
    assert len(out_log_returns) == len(wrds_log_returns)


def test_pfit_resolver_wrds_fallback_when_downloader_too_short(tmp_path, aapl_outmm):
    missing_crsp = tmp_path / "missing_crsp.csv"
    n = 10  # below the 252-point floor
    dates = [datetime.date(2023, 1, 1) + datetime.timedelta(days=i) for i in range(n)]
    closes = 150.0 + np.arange(n) * 0.1
    downloader = _fake_downloader(dates, closes)

    out_dates, out_closes, out_log_returns, source = daily_close_series_for_pfit(
        aapl_outmm, "AAPL", crsp_path=missing_crsp, _downloader=downloader
    )
    assert source == "wrds_fallback"


def test_pfit_resolver_crsp_tier_with_valid_cross_check(tmp_path, aapl_outmm):
    wrds_dates, wrds_closes, _ = daily_close_series(aapl_outmm)
    rng = np.random.default_rng(1)
    noisy_closes = wrds_closes * (1.0 + rng.normal(0, 1e-4, size=len(wrds_closes)))

    rows = []
    for i, (d, p) in enumerate(zip(wrds_dates, noisy_closes)):
        if i == 0:
            ret = 0.0
        else:
            ret = p / noisy_closes[i - 1] - 1.0
        rows.append((_yyyymmdd_float_to_iso(d), p, ret))
    crsp_path = _write_crsp_csv(tmp_path, rows)

    out_dates, out_prc, out_log_returns, source = daily_close_series_for_pfit(
        aapl_outmm, "AAPL", crsp_path=crsp_path, _downloader=None
    )
    assert source == "crsp"
    assert len(out_dates) == len(wrds_dates)
    assert len(out_log_returns) == len(wrds_dates) - 1
