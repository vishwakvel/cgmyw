# Surface calibration: omega martingale constant, parity forward, Sato time scaling

Status: accepted (2026-11-02, meeting with Dr. Madan). Supersedes ADR-0008's "M from the martingale condition" for option calibration; `qfit` keeps that scheme as the record of the P-then-Q work.

**Martingale by a constant.** Calibrating to options should not spend a parameter on the martingale condition. Practice in every model: ln S_T = ln F + X_T + omega_T with omega_T = -ln E[e^{X_T}], so E[S_T] = F for any parameters and all of them are free to fit the surface. ADR-0008's scheme was the theoretical (equivalent-measure) view.

**Forward from the options.** F = S0 e^{(r-q)T}: S0 is the close, r comes from the rates market, T is known, and q is implied so F sits where call and put mids cross (put-call parity, median over the 4 strikes nearest the crossing). On AAPL it lands within 0.5% of WRDS's forward column.

**Sato process.** For any Lévy process, skew decays like 1/sqrt(T) and excess kurtosis like 1/T, so no Lévy model, however many parameters, matches an option surface's term structure. If X_1 is self-decomposable (|x| k(x) decreasing in |x| on each side; true for CGMYW with G, M, Y, W >= 0), X_T = T^gamma X_1 defines an additive (Sato) process with flat skew and kurtosis. 9 parameters: the 8 CGMYW plus gamma. CGMY stays CGMY under the scaling (C, Y fixed; G, M -> G/T^gamma, M/T^gamma); CGMYW does not (the (1+|x|)^W factor does not scale), so pricing uses the characteristic function exp(psi(u T^gamma)) directly.

**Maturities.** Fit jointly on the traded expiries closest to 1 month, 3 months, 6 months and 1 year.

**The question this answers.** Does W add fitting power? Compared head to head: Lévy CGMY (6), Lévy CGMYW (8), Sato CGMY (7), Sato CGMYW (9), same surfaces, same objective (vega-weighted price error), all parameters free.

Consequence: `cgmyw.surface` holds this calibration; `scripts/run_surface.py` runs the four-model comparison.
