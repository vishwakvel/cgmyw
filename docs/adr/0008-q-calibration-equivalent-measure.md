# Q calibration: pin Cn, Cp, Yn, Yp; solve M from the martingale condition; no free drift

Status: accepted (2026-10-18). Resolves the three Phase 1.5 questions left open by the 2026-08-07 meeting.

**Pinning split (8 params).** Madan & Wang, "Asset Pricing, Measure Changes, and a New Lévy Model" (2026), Section 2: equivalence (finite Hellinger distance) forbids changing C and Y on either side. Cn, Cp, Yn, Yp are pinned at their P values; G, M, Wn, Wp move.

**Martingale condition, not a drift correction.** The paper imposes `psi_Q(-i) = 0` (its r=q=0 simplification) as a constraint on the Q parameters. We use the market forward instead: `T * psi_Q(-i) = ln(F/S0)`. There is no free drift: `psi_cgmyw`'s closed form carries the drift `m(theta)` (analytic continuation of the mean of k) for every Y in (0,2), and equivalence moves the drift by exactly `m(theta_Q) - m(theta_P)`, so the Q exponent is `psi_cgmyw` at the Q parameters, nothing added. Since `psi(-i)` is strictly decreasing in M, M is solved by a 1-D root-find (`qfit.solve_m`) rather than a constrained optimizer: every point the optimizer visits is an equivalent martingale measure. The Carr-Madan pricer still applies `omega = -psi(-i)`; under the constraint this is the identity up to solver round-off.

**Time units.** P is fit on daily returns, so Cn, Cp are per trading day. G, M, Y, W describe jump sizes and are time-free; only arrival rates scale: `C_annual = 252 * C_daily`, giving `psi_annual = 252 * psi_daily` exactly. Option maturity is calendar days / 365.

**Objective.** Mean squared vega-weighted price error `((model - mid)/vega)^2`, vega at the market IV, per the meeting (no Black-Scholes inversion per evaluation). IV RMSE is reported alongside it.

**Validation.** From the paper's Table 1 P-parameters, `qfit.risk_neutralize` reproduces its Table 2 risk-neutral (G, M, Wp, Wn) for all ten tickers to about 3 significant figures (the published points leave ~1e-5 residual drift, i.e. solver tolerance). An unrestricted CGMYW fit (all of Cn, Cp, Yn, Yp freed too) reproduces AAPL smiles to 0.1-0.4 vol points, so the pricer and optimizer are not the bottleneck.

**Finding: the equivalent Q measure cannot fit AAPL's skew, and this is not a code defect.** IV RMSE with Cn, Cp, Yn, Yp pinned: 6.4 vol points on 2023-01-13, 8.7 on 2025-08-29. Unrestricted CGMYW fits the same smiles to 0.4 and 0.14. Diagnosis (2026-10-18):

1. *Martingale lock.* With no diffusion, equivalence also fixes the drift, so the martingale condition is a constraint on the jump parameters alone. Each tail's contribution to psi(-i) is of order C*Gamma(1-Y)*G^(Y-1) (the mean jump rate), far larger than ln(F/S0)/T, so the condition forces the down and up contributions to nearly cancel: M tracks G (scan on 2025-08-29: M ~ 1.3 G at every G), and at a realistic ATM vol the attainable put skew is ~2 vol points against the market's 17.5. Freeing the drift alone (C, Y still pinned) cuts 2025-08-29 from 9.5 to 1.8 vol points, M running to its bound; a free drift is only equivalent if the model gains a Brownian component (Girsanov) -- a model change for Dr. Madan to decide.
2. *P and Q want incompatible (C, Y).* Refitting P with C, Y fixed at the option-implied values raises tail-probability RMSE 0.008 -> 0.049 (6x). A joint fit with shared (C, Y) has no interior compromise: it lands either at P-good / Q 8.6 vol points or at Q-good / P 25x worse.
3. *(C, Y) are weakly identified from 251 daily returns.* Different windows land in different regimes (Y ~ 1.67 symmetric on 2023-01-13; Yn ~ 0.15 / Yp ~ 0.86 on 2025-08-29) at similar objective. Since these are exactly the pinned parameters, P-fit identification drives Q.

The paper itself notes its measure change "has only addressed the stock martingale condition and not the wider implications of option markets." Open questions for Dr. Madan: add a diffusion component; accept the unrestricted Q fit as the deliverable (not equivalent to P); or estimate jointly with an explicit P-vs-Q weight.

Consequence: `qfit.calibrate_q(search=PINNED)` is the deliverable; `search=UNRESTRICTED` is a diagnostic only (not equivalent to P) and is labelled as such everywhere it is reported.
