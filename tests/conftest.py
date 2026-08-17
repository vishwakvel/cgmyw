"""Shared pytest fixtures for the cgmyw test suite."""

from pathlib import Path

import pytest

from cgmyw.data import load_wrds

REPO_ROOT = Path(__file__).resolve().parent.parent
AAPL_DAT_PATH = REPO_ROOT / "WRDSdata" / "aapldata20230103to20250829.dat"


@pytest.fixture(scope="session")
def aapl_outmm():
    """Session-scoped fixture: the AAPL WRDS options file's outmm matrix.

    Loading is ~24MB, so session scope matters -- every data-touching test in
    this phase reuses this one load. WRDSdata/ is gitignored, so skip rather
    than fail when the file is absent (e.g. a fresh clone).
    """
    if not AAPL_DAT_PATH.exists():
        pytest.skip("WRDS data file not present")
    return load_wrds(AAPL_DAT_PATH)
