# CGMYW Option Pricing

Empirical project: implement a 5-parameter Lévy jump model (CGMY extended with a shared power-law tilt), calibrate it to WRDS equity option data via FFT, and report estimation results.

## Language

**CGMY**:
The classical 4-parameter Lévy density (C, G, M, Y): exponential decay rate G on down-jumps, M on up-jumps, shared activity level C and small-jump index Y.
_Avoid_: Variance gamma (VG is the special case Y=0, not this model)

**CGMYW** (this project's model):
CGMY with one added shared power-law tilt parameter W, applied symmetrically to both tails: `k(x) = C·exp(-G|x|)/(|x|^(1+Y)(1+|x|)^W)` for x<0, mirrored with M for x>0. 5 parameters total: C, G, M, Y, W. This is the "new 5-parameter model" referred to in the project brief — a deliberately simplified, symmetric case of the fuller asymmetric model in the attached note (see ADR-0001).
_Avoid_: "the new model" (ambiguous with the paper's full asymmetric version), "5-param CGMY" (drop — could be confused with plain CGMY)

**Full asymmetric CGMYW**:
The 8-parameter version from the attached note (Cn,Yn,Wn,G,Cp,Yp,Wp,M), where each tail gets independent C, Y, W. Not the initial implementation target — a possible later extension.

**Lévy density** `k(x)`:
Expected arrival rate of jumps of size in `[x, x+dx]` per unit time. Not a probability density — does not integrate to 1.

**Characteristic exponent** `ψ(u)`:
The function such that `E[e^{iuX_t}] = exp(t·ψ(u))`; determines option prices via Fourier inversion (Carr–Madan FFT).

**P-measure**:
The physical/statistical measure, estimated from historical return time series.

**Q-measure**:
The risk-neutral measure, calibrated from option prices; discounted stock is a Q-martingale.

**Hellinger condition**:
The constraint that C and Y must be identical under P and Q for the two measures to be equivalent; only G, M (and here, W) are free to move between measures.

**Kummer U** `U(a,b,z)`:
Confluent hypergeometric function of the second kind (Tricomi function); appears in the CGMYW characteristic exponent because of the power-law tilt. Evaluated via `scipy.special.hyperu`.

## Data

**Quote row**:
One option observation: current date, expiration date, strike, call/put flag, price, implied vol, spot, forward, rate, dividend yield, bid, ask, NaN-IV flag. 13 columns, stored in WRDSdata/*.dat (MATLAB v5 mat-files, variable name `outmm`).

**NaN-IV flag**:
Column 13; 1 means the implied vol for that quote could not be computed by WRDS. Not necessarily a reason to exclude the row from price-based calibration.
