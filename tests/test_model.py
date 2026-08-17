"""Tests for cgmyw.model: hybrid Kummer-U and the 8-parameter CGMYW
characteristic exponent (MODEL-01).

Ground truth for the Kummer-U hybrid is cross-validated against mpmath.hyperu
(arbitrary precision) -- a dev-only test dependency, never imported by
src/cgmyw/. The MODEL-01 correctness gate is the symmetric-specialization
reduction to classical CGMY at 1e-8.
"""

import numpy as np
import pytest

from cgmyw.model import KUMMER_Z_THRESHOLD, kummer_u, psi_cgmy_classical, psi_cgmyw


# ---------------------------------------------------------------------------
# kummer_u
# ---------------------------------------------------------------------------


def test_kummer_u_a_zero_is_identity():
    """DLMF 13.2.9: U(0, b, z) == 1 for any b, z."""
    for b, z in [(1.6, 3.0 + 4.0j), (2.3, 10.0), (0.5, 50.0 + 5.0j)]:
        out = kummer_u(0.0, b, z)
        assert np.allclose(out, 1.0)


@pytest.mark.parametrize("z_mag", [3.0, 8.0])
def test_kummer_u_tricomi_branch_matches_mpmath(z_mag):
    mpmath = pytest.importorskip("mpmath")
    a, b = 0.4, 1.0  # Wn=0.4, Wn+Yn=1.0 -- plausible CGMYW (W, W+Y) pair
    z = z_mag + 2.0j
    assert abs(z) < KUMMER_Z_THRESHOLD  # confirm this exercises the Tricomi branch

    got = complex(kummer_u(a, b, z)[0])
    want = complex(mpmath.hyperu(a, b, complex(z)))
    rel_err = abs(got - want) / abs(want)
    assert rel_err < 1e-9


@pytest.mark.parametrize("z_mag", [35.0, 80.0])
def test_kummer_u_asymptotic_branch_matches_mpmath(z_mag):
    mpmath = pytest.importorskip("mpmath")
    a, b = 0.4, 1.0
    z = z_mag + 2.0j
    assert abs(z) >= KUMMER_Z_THRESHOLD  # confirm this exercises the asymptotic branch

    got = complex(kummer_u(a, b, z)[0])
    want = complex(mpmath.hyperu(a, b, complex(z)))
    rel_err = abs(got - want) / abs(want)
    assert rel_err < 1e-7


def test_kummer_u_agrees_with_scipy_hyperu_real_z():
    from scipy.special import hyperu as scipy_hyperu

    rng = np.random.default_rng(42)
    for _ in range(20):
        a = rng.uniform(0.1, 2.0)
        b = rng.uniform(1.1, 3.0)  # keep b away from small/negative poles
        z = rng.uniform(0.5, 15.0)  # real z, small-|z| Tricomi regime
        got = complex(kummer_u(a, b, z)[0])
        want = complex(scipy_hyperu(a, b, z))
        if want == 0 or not np.isfinite(want):
            continue
        rel_err = abs(got - want) / abs(want)
        assert rel_err < 1e-6


def test_kummer_u_accepts_complex_array_without_raising():
    z = np.array([1.0 + 2.0j, 25.0 + 3.0j, 0.5 - 1.0j])
    out = kummer_u(0.6, 1.6, z)
    assert out.dtype == complex
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# psi_cgmyw / psi_cgmy_classical (MODEL-01)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "C,G,M,Y",
    [
        (1.0, 12.0, 15.0, 0.6),   # Y < 1
        (0.5, 8.0, 10.0, 1.4),    # Y > 1
        (2.0, 20.0, 25.0, 0.95),  # Y near 1 but not on the pole
    ],
)
def test_psi_cgmyw_symmetric_reduction_matches_classical_cgmy(C, G, M, Y):
    u = np.linspace(-300, 300, 2001)
    lhs = psi_cgmyw(u, C, C, G, M, Y, Y, 0.0, 0.0)
    rhs = psi_cgmy_classical(u, C, G, M, Y)
    max_abs_err = np.max(np.abs(lhs - rhs))
    assert max_abs_err < 1e-8


def test_psi_cgmyw_distinct_tail_params_differ_from_symmetric_call():
    u = np.linspace(-50, 50, 501)
    symmetric = psi_cgmyw(u, 1.0, 1.0, 12.0, 15.0, 0.6, 0.6, 0.0, 0.0)
    asymmetric = psi_cgmyw(u, 1.0, 1.5, 12.0, 15.0, 0.6, 0.4, 0.3, 0.2)
    assert np.max(np.abs(symmetric - asymmetric)) > 1e-3


def test_psi_cgmyw_accepts_complex_u():
    for u in (3.0 - 1.75j, -1j):
        v = psi_cgmyw(u, 1.0, 1.2, 12.0, 15.0, 0.6, 0.55, 0.4, 0.35)
        assert np.isfinite(v)


def test_psi_cgmyw_at_minus_i_is_real():
    w = psi_cgmyw(-1j, 1.0, 1.2, 12.0, 15.0, 0.6, 0.55, 0.4, 0.35)
    assert abs(complex(w).imag) < 1e-10


def test_psi_cgmyw_vanishes_at_origin():
    w = psi_cgmyw(0.0, 1.0, 1.2, 12.0, 15.0, 0.6, 0.55, 0.4, 0.35)
    assert abs(complex(w)) < 1e-12
