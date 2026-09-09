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
    # a=Wn=0.4, b=Wn+Yn=1.3 -- plausible CGMYW (W, W+Y) pair, deliberately off
    # any integer b (RESEARCH.md Pitfall 3: gamma(1-b)/gamma(b-1) pole at b=1).
    a, b = 0.4, 1.3
    z = z_mag + 2.0j
    assert abs(z) < KUMMER_Z_THRESHOLD  # confirm this exercises the Tricomi branch

    got = complex(kummer_u(a, b, z)[0])
    want = complex(mpmath.hyperu(a, b, complex(z)))
    rel_err = abs(got - want) / abs(want)
    assert rel_err < 1e-9


@pytest.mark.parametrize("z_mag", [35.0, 80.0])
def test_kummer_u_asymptotic_branch_matches_mpmath(z_mag):
    mpmath = pytest.importorskip("mpmath")
    a, b = 0.4, 1.3
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


def _psi_cgmyw_numeric(u, Cn, Cp, G, M, Yn, Yp, Wn, Wp, dps=30):
    """psi(u) = int_{-inf}^{inf} (e^{iux} - 1) k(x) dx computed by direct
    high-precision quadrature of the CGMYW Levy density, with no Kummer-U
    involved. Independent ground truth for the closed form's power-tilt term.

    Down tail (x<0), substituting x -> -x: int_0^inf (e^{-iux}-1) *
    Cn e^{-Gx}/(x^{1+Yn}(1+x)^{Wn}) dx.  Up tail (x>0): int_0^inf
    (e^{iux}-1) * Cp e^{-Mx}/(x^{1+Yp}(1+x)^{Wp}) dx.  Both integrands are
    O(x^{-Y}) near 0, integrable for Y < 1 (kept below 1 in the callers).
    """
    import mpmath as mp

    mp.mp.dps = dps
    u = complex(u)
    split = [0, mp.mpf("1e-8"), mp.mpf("1e-4"), mp.mpf("1e-2"), mp.mpf("0.1"), 1, 10, mp.inf]

    def dens_down(x):
        return Cn * mp.e ** (-G * x) / (x ** (1 + Yn) * (1 + x) ** Wn)

    def dens_up(x):
        return Cp * mp.e ** (-M * x) / (x ** (1 + Yp) * (1 + x) ** Wp)

    down = mp.quad(lambda x: mp.re((mp.e ** (-1j * u * x) - 1) * dens_down(x)), split) + 1j * mp.quad(
        lambda x: mp.im((mp.e ** (-1j * u * x) - 1) * dens_down(x)), split
    )
    up = mp.quad(lambda x: mp.re((mp.e ** (1j * u * x) - 1) * dens_up(x)), split) + 1j * mp.quad(
        lambda x: mp.im((mp.e ** (1j * u * x) - 1) * dens_up(x)), split
    )
    return complex(down) + complex(up)


@pytest.mark.parametrize(
    "u,p",
    [
        (2.0, dict(Cn=1.0, Cp=1.3, G=9.7, M=6.2, Yn=0.6, Yp=0.7, Wn=0.5, Wp=0.9)),
        (0.8, dict(Cn=0.7, Cp=0.7, G=12.0, M=15.0, Yn=0.4, Yp=0.55, Wn=1.0, Wp=0.3)),
        (-1.5, dict(Cn=1.1, Cp=0.9, G=8.0, M=10.0, Yn=0.75, Yp=0.5, Wn=2.0, Wp=1.4)),
    ],
)
def test_psi_cgmyw_power_tilt_matches_numerical_integration(u, p):
    """The W != 0 correctness gate the MODEL-01 symmetric reduction cannot be:
    at Wn=Wp=0, U(0, b, z) == 1 for every b, so that reduction passes whether
    the Kummer-U second argument is W+Y or W+Y+1. Only a W != 0 check against
    an independent computation of psi(u) can pin the tilt term. The printed
    closed form's U(W, W+Y, z) is wrong by 1-10% here; U(W, W+Y+1, z) matches
    the direct integral to ~1e-14 (well inside this 1e-6 bar).
    """
    pytest.importorskip("mpmath")
    got = complex(psi_cgmyw(u, p["Cn"], p["Cp"], p["G"], p["M"], p["Yn"], p["Yp"], p["Wn"], p["Wp"]))
    want = _psi_cgmyw_numeric(u, **p)
    assert abs(got - want) / abs(want) < 1e-6


def test_psi_cgmyw_distinct_tail_params_differ_from_symmetric_call():
    u = np.linspace(-50, 50, 501)
    symmetric = psi_cgmyw(u, 1.0, 1.0, 12.0, 15.0, 0.6, 0.6, 0.0, 0.0)
    asymmetric = psi_cgmyw(u, 1.0, 1.5, 12.0, 15.0, 0.6, 0.4, 0.3, 0.2)
    assert np.max(np.abs(symmetric - asymmetric)) > 1e-3


def test_psi_cgmyw_accepts_complex_u():
    # Wn=0.42 (not 0.4): Wn+Yn=0.4+0.6 would land exactly on the documented
    # Tricomi-formula pole at integer b (RESEARCH.md Pitfall 3, reproduced
    # directly: gamma(1-b)/gamma(b-1) both blow up at b=1.0, producing NaN --
    # a real, expected singularity model.py deliberately does not trap).
    for u in (3.0 - 1.75j, -1j):
        v = psi_cgmyw(u, 1.0, 1.2, 12.0, 15.0, 0.6, 0.55, 0.42, 0.35)
        assert np.isfinite(v)


def test_psi_cgmyw_at_minus_i_is_real():
    w = psi_cgmyw(-1j, 1.0, 1.2, 12.0, 15.0, 0.6, 0.55, 0.42, 0.35)
    assert abs(complex(w).imag) < 1e-10


def test_psi_cgmyw_vanishes_at_origin():
    w = psi_cgmyw(0.0, 1.0, 1.2, 12.0, 15.0, 0.6, 0.55, 0.42, 0.35)
    assert abs(complex(w)) < 1e-12
