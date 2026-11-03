# CGMYW Option Pricing

Empirical project: implement the 8-parameter CGMYW Lévy jump model (CGMY extended with a power-law tilt per tail), fit it to daily returns (P), calibrate it to WRDS equity option data via FFT (Q), and report estimation results.

## Language

**CGMY**:
The classical 4-parameter Lévy density (C, G, M, Y): exponential decay rate G on down-jumps, M on up-jumps, shared activity level C and small-jump index Y.
_Avoid_: Variance gamma (VG is the special case Y=0, not this model)

**CGMYW** (this project's model):
CGMY with a power-law tilt per tail: `k(x) = Cn·exp(-G|x|)/(|x|^(1+Yn)(1+|x|)^Wn)` for x<0, `Cp·exp(-Mx)/(x^(1+Yp)(1+x)^Wp)` for x>0. 8 parameters (Cn,Cp,G,M,Yn,Yp,Wn,Wp), per ADR-0004 (supersedes ADR-0001's symmetric 5-parameter reading). Published as Madan & Wang, "Asset Pricing, Measure Changes, and a New Lévy Model" (2026).
_Avoid_: "the new model", "5-param CGMYW" (superseded)

**Lévy density** `k(x)`:
Expected arrival rate of jumps of size in `[x, x+dx]` per unit time. Not a probability density — does not integrate to 1.

**Characteristic exponent** `ψ(u)`:
The function such that `E[e^{iuX_t}] = exp(t·ψ(u))`; determines option prices via Fourier inversion (Carr–Madan FFT).

**P-measure**:
The physical/statistical measure, estimated from historical return time series.

**Q-measure**:
The risk-neutral measure, calibrated from option prices; discounted stock is a Q-martingale.

**Hellinger condition**:
The constraint that C and Y must be identical under P and Q for the two measures to be equivalent; here Cn, Cp, Yn, Yp are pinned and only G, M, Wn, Wp move (ADR-0008).

**Kummer U** `U(a,b,z)`:
Confluent hypergeometric function of the second kind (Tricomi function); appears in the CGMYW characteristic exponent as U(W, W+Y+1, z) because of the power-law tilt. Evaluated at complex z via a hyp1f1/asymptotic reconstruction, since `scipy.special.hyperu` is real-only (ADR-0007).

## Data

**Quote row**:
One option observation: current date, expiration date, strike, call/put flag, price, implied vol, spot, forward, rate, dividend yield, bid, ask, NaN-IV flag. 13 columns, stored in WRDSdata/*.dat (MATLAB v5 mat-files, variable name `outmm`).

**NaN-IV flag**:
Column 13; 1 means the implied vol for that quote could not be computed by WRDS. Not necessarily a reason to exclude the row from price-based calibration.
