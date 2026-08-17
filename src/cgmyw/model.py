"""The CGMYW characteristic exponent: a hybrid complex-argument Kummer-U
reconstruction plus the 8-parameter asymmetric psi_cgmyw (MODEL-01).

Deviation from `.claude/CLAUDE.md`'s original "use scipy.special.hyperu"
constraint, recorded in full in docs/adr/0007-hybrid-kummer-u.md:
`scipy.special.hyperu` is a real-argument-only ufunc and raises TypeError on
complex z. psi(u) requires U(Wn, Wn+Yn, G+iu) and U(Wp, Wp+Yp, M-iu) --
complex arguments by construction. The replacement below is built entirely
from scipy.special primitives (hyp1f1, gamma), never a hand-rolled
hypergeometric series solver, preserving the constraint's intent.

Requires scipy>=1.14: `hyp1f1`'s complex-argument support was broken prior to
that release (scipy issue #20797) and this reconstruction depends on it.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma, hyp1f1

# |z| dispatch threshold between the Tricomi (small-|z|) and asymptotic
# (large-|z|) branches of kummer_u, and the number of terms retained in the
# asymptotic series. Module-level constants rather than inline magic numbers
# (T-01-04: reproducibility).
KUMMER_Z_THRESHOLD: float = 20.0  # empirically cross-validated against mpmath (RESEARCH.md)
# RESEARCH.md's own validated recipe used 10 terms; this ships 12, independently
# re-validated by tests/test_model.py's mpmath cross-check (<1e-7 rel. error at
# |z|=35,80) but not re-derived from first principles -- asymptotic series don't
# necessarily improve monotonically with more terms, so "12 works" is tested,
# not proven optimal.
KUMMER_ASYMPTOTIC_TERMS: int = 12


def _u_tricomi(a, b, z):
    """DLMF 13.2.42 (Tricomi's relation), built on hyp1f1 + gamma.

    Accurate for |z| below ~15-20; suffers catastrophic cancellation between
    two exponentially-growing terms above that (research: relative error
    ~2e3x at Re(z)=40, ~5e7x at Re(z)=50, with no warning emitted). Never
    call this alone for large |z| -- see kummer_u's dispatch.
    """
    # Two terms, each individually huge (grows like e^Re(z)) and of opposite
    # sign -- their SUM is the small, correct answer, which is exactly what
    # makes this formula lose precision once Re(z) gets large (float64 can't
    # resolve a ~15-digit cancellation between two ~1e17-magnitude numbers).
    term1 = (gamma(1 - b) / gamma(a - b + 1)) * hyp1f1(a, b, z)
    term2 = (gamma(b - 1) / gamma(a)) * z ** (1 - b) * hyp1f1(a - b + 1, 2 - b, z)
    return term1 + term2


def _u_asymptotic(a, b, z, nterms):
    """DLMF 13.7.3, the large-|z| asymptotic series for U(a,b,z).

    Valid for |arg z| < 3*pi/2. Pure elementary complex arithmetic (no
    special-function calls) -- accurate for large |z| where _u_tricomi
    cancels catastrophically, diverges for small |z|.
    """
    total = np.ones_like(z)  # running sum of the series (term_0 = 1)
    term = np.ones_like(z)  # the current term, built incrementally from the previous one
    ak = complex(a)  # tracks (a + k - 1) as k increases, starts at k=1 -> a
    abk = complex(a - b + 1)  # tracks (a - b + k), starts at k=1 -> a-b+1
    for k in range(1, nterms):
        # Each term is the previous term times a ratio -- this recurrence is
        # algebraically equivalent to the series' explicit k-th coefficient,
        # just cheaper to compute step-by-step than re-deriving it from scratch.
        term = term * ak * abk / k * (-1.0 / z)
        total = total + term
        ak += 1
        abk += 1
    # z**(-a) is the series' overall prefactor, applied once at the end.
    return z ** (-a) * total


def kummer_u(a, b, z, threshold: float = KUMMER_Z_THRESHOLD, nterms: int = KUMMER_ASYMPTOTIC_TERMS):
    """Confluent hypergeometric function of the second kind, U(a, b, z),
    complex z supported (unlike scipy.special.hyperu, which is real-only).

    Dispatches elementwise on |z| between the small-|z| Tricomi relation
    (_u_tricomi, DLMF 13.2.42) and the large-|z| asymptotic series
    (_u_asymptotic, DLMF 13.7.3). This dispatch is the whole point of the
    function and must not be simplified away: using Tricomi unconditionally
    returns finite, plausible-looking, WRONG numbers once Re(z) passes
    roughly 25 (research: ~2e3x relative error at Re(z)=40, ~5e7x at
    Re(z)=50) -- with no exception, no warning, nothing to catch it. A
    wrong-but-believable option smile is a far worse failure mode than a
    crash, so both branches are always evaluated by |z| region, never one
    alone.

    Both branches independently satisfy U(0, b, z) == 1 (DLMF 13.2.9)
    without any special-casing of a == 0 -- this is the identity the
    MODEL-01 symmetric-reduction gate in tests/test_model.py depends on.
    """
    # Force z into a 1-D complex array regardless of whether the caller passed
    # a scalar, a real number, or an array -- this makes the boolean-mask
    # dispatch below work uniformly for every call shape.
    z = np.atleast_1d(np.asarray(z, dtype=complex))
    out = np.empty_like(z)
    small = np.abs(z) < threshold  # boolean mask: True where Tricomi applies
    if np.any(small):
        out[small] = _u_tricomi(a, b, z[small])
    if np.any(~small):  # ~small = the large-|z| points
        out[~small] = _u_asymptotic(a, b, z[~small], nterms=nterms)
    return out


def _scalarize(u, result):
    """Squeeze a length-1 array result back to a bare scalar when the caller
    passed a scalar `u` -- shared by psi_cgmyw and psi_cgmy_classical so a
    caller doing `psi_cgmyw(-1j, ...)` gets a scalar back, not a 1-element
    array."""
    return result[0] if np.ndim(u) == 0 else result


def psi_cgmyw(u, Cn, Cp, G, M, Yn, Yp, Wn, Wp):
    """The 8-parameter asymmetric CGMYW characteristic exponent
    psi(u; Cn,Cp,G,M,Yn,Yp,Wn,Wp), such that E[e^{iuX_t}] = exp(t*psi(u))
    (ADR-0004's locked parameterization -- Cn,Cp,Yn,Wn independent per side,
    G,M shared exactly as in classical CGMY).

    u may be real or complex (Phase 1.5's Carr-Madan pricer evaluates this
    at u = v - (alpha+1)j). Each tail's constant term and FFT-grid term
    route through the same kummer_u wrapper -- there is exactly one code
    path for U in this pipeline. Wn=Wp=0 is not special-cased:
    kummer_u(0, b, z) is identically 1 on its own, so the general formula
    self-reduces to classical CGMY when additionally Cn=Cp and Yn=Yp -- this
    self-reduction is exactly what the MODEL-01 1e-8 gate measures; a
    shortcut here would make that gate vacuous.

    The principal complex branch of zg**bn / zm**bp is valid only while
    Re(zg) > 0 and Re(zm) > 0; for Carr-Madan's damped argument
    u = v - (alpha+1)j this is exactly the alpha < M-1 integrability
    condition Phase 1.5 will enforce.

    No NaN/inf trapping here by design: poles at Yn/Yp in {0,1} and at
    integer Wn+Yn / Wp+Yp are real and must surface here, not be silently
    swallowed. The P-fit's objective (plan 01-02) owns rejecting a NaN/inf
    result so L-BFGS-B steps away cleanly.
    """
    u_arr = np.atleast_1d(np.asarray(u, dtype=complex))

    # zg/zm are the complex Kummer-U arguments for the down-tail (G-keyed) and
    # up-tail (M-keyed) terms respectively -- the +iu / -iu sign split is what
    # makes psi(u) the Fourier dual of a two-sided (negative + positive jump)
    # Levy density.
    zg = G + 1j * u_arr
    zm = M - 1j * u_arr
    bn = Wn + Yn  # Kummer-U's second parameter for the down-tail term
    bp = Wp + Yp  # ... and for the up-tail term

    # Each tail's term is (value at the FFT-grid point zg/zm) minus (value at
    # the real "constant" point G/M) -- this difference is what makes psi(0)
    # vanish exactly (a required property of a characteristic exponent).
    term_n = Cn * gamma(-Yn) * (zg ** bn * kummer_u(Wn, bn, zg) - G ** bn * kummer_u(Wn, bn, G))
    term_p = Cp * gamma(-Yp) * (zm ** bp * kummer_u(Wp, bp, zm) - M ** bp * kummer_u(Wp, bp, M))
    result = term_n + term_p

    return _scalarize(u, result)


def psi_cgmy_classical(u, C, G, M, Y):
    """Classical CGMY (Carr-Geman-Madan-Yor 2002) closed-form characteristic
    exponent -- the independent reference implementation the MODEL-01 gate
    compares psi_cgmyw's symmetric specialization against. Shares no code
    with psi_cgmyw beyond scipy.special.gamma, so the gate does not compare
    a function to itself.
    """
    u_arr = np.atleast_1d(np.asarray(u, dtype=complex))

    # Classical CGMY closed form: no Kummer-U involved (W=0 collapses it away
    # analytically), which is exactly why this is a valid independent check --
    # psi_cgmyw at Wn=Wp=0 should reduce to this via a genuinely different
    # code path, not the same formula in disguise.
    result = C * gamma(-Y) * ((M - 1j * u_arr) ** Y - M ** Y + (G + 1j * u_arr) ** Y - G ** Y)

    return _scalarize(u, result)
