# CGMYW Option Pricing

## What This Is

An empirical research project (prof-directed) that implements a new 5-parameter Lévy jump model — CGMY extended with a shared power-law tail tilt (C, G, M, Y, W) — in Python, calibrates it against real WRDS equity option data for 10 stocks, and writes up how it performs. The model comes from an unpublished research note (attached PDF, likely Dilip Madan) generalizing the CGMY/variance-gamma family that has priced Morgan Stanley's option books for a decade.

## Core Value

A working, generic P-then-Q calibration pipeline (time series → risk-neutral option smile) for one stock that produces a defensible parameter fit and error report — everything else (scaling to 10 stocks, polished write-up) builds on this working core.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Load and parse WRDS option data (MATLAB v5 `.dat` files, 13-column `outmm` matrix) for any of the 10 stocks
- [ ] Derive a daily closing-price / log-return time series per stock directly from the option data (column 7, verified one price per date)
- [ ] Implement the CGMYW characteristic exponent ψ(u; C,G,M,Y,W) using `scipy.special.hyperu` (Kummer U)
- [ ] P-measure: fit (C,G_P,M_P,Y,W_P) via characteristic-function MLE (FFT-invert ψ to get return density, maximize log-likelihood) on a trailing 252-trading-day window ending at a given date
- [ ] Q-measure: implement Carr-Madan FFT option pricer from ψ(u), including the martingale drift correction ω = -ψ(-i) (requires M > 1)
- [ ] Q-measure: calibrate free (G_Q,M_Q,W_Q) to the option smile at one chosen maturity, holding C,Y fixed from the P-fit, minimizing squared IV error (BS-inverted mid prices, NaN-IV rows excluded)
- [ ] Multi-start (n=8) bounded L-BFGS-B optimization for both P and Q fits
- [ ] Auto-select a good (date, maturity) for the first result based on data quality (strike count, NaN-IV fraction, days-to-expiry), not hardcoded
- [ ] Working end-to-end pipeline on one stock (AAPL) as proof of correctness
- [ ] Generalize the pipeline to run on all 10 stocks (AAPL, AMZN, CSCO, GOOGL, IBM, INTC, META, MSFT, NVDA, ORCL) and produce a cross-stock results table
- [ ] `cgmyw_estimation.ipynb` notebook: narrative + code + smile-fit plots + P-vs-Q parameter comparison + residual/error diagnostics
- [ ] `REPORT.md`/PDF distilled write-up of estimation results, once results are stable (loosely aimed at the professor's 5-page Rapid Communications in Mathematical Finance format)

### Out of Scope

- Full asymmetric 8-parameter CGMYW (Cn,Yn,Wn,G,Cp,Yp,Wp,M) from the attached note — deferred; the 5-param symmetric-W version is the initial target (ADR-0001)
- Bilateral gamma model — considered as an alternative starting point per the professor's email, rejected in favor of going straight to CGMYW (ADR-0001)
- External price/return data sources (e.g. yfinance, CRSP) — daily closes are derived directly from the WRDS option data itself
- Publication-ready 5-page paper — the notebook + REPORT.md write-up is the deliverable for this milestone; formal submission to FMF's Rapid Communications section is a future, separate step

## Context

- Repo: `/Users/vishwakvelamuri/cgmyw` — greenfield, no code yet as of project init.
- `CGMYW.pdf`: the source research note deriving the Lévy density, characteristic exponent, and Kummer U closed form. See `CONTEXT.md` for the reconstructed math and glossary.
- `WRDSdata/*.dat` (10 files, one per stock): MATLAB v5 mat-files (misleading `.dat` extension), each a single `outmm` variable, an (N,13) float64 matrix. Verified format: date, expiry (yyyymmdd), strike, call/put flag, option price, IV, spot, forward, rate, div yield, bid, ask, NaN-IV flag. Row counts range 416K (CSCO) to 3.0M (NVDA). Date range 2023-01-03 to 2025-08-29, 667 unique trading days per stock, spot price verified constant within a date.
- A `grill-with-docs` session (superpowers `grilling` + `domain-modeling` skills) preceded this GSD init and resolved the major architecture ambiguities — see `docs/adr/0001` through `0003` and `CONTEXT.md` for the full glossary and rationale. GSD planning should build on these, not re-derive them.
- The professor's email is the authoritative source when it conflicts with the paraphrased task brief (e.g. the P-then-Q pipeline order — ADR-0002).

## Constraints

- **Data source**: Only the 10 provided WRDS `.dat` files — no live data fetching.
- **Numerical**: CGMYW's characteristic exponent requires evaluating the Kummer U (confluent hypergeometric) function at complex arguments across an FFT grid; must use `scipy.special.hyperu` (handles complex input) rather than a hand-rolled series.
- **Faithfulness to source**: The professor's email takes precedence over the summarized task brief wherever they diverge (see ADR-0002).
- **Tooling**: Python + scipy/numpy stack (no new heavy dependencies) per prior conversation — `scipy.io.loadmat`, `scipy.optimize`, `scipy.special.hyperu`, `numpy.fft`.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Symmetric 5-param CGMYW (C,G,M,Y,W), not full 8-param asymmetric or bilateral gamma | Matches brief's "5-parameter model" phrasing; minimal extension of classical CGMY | — Pending (ADR-0001) |
| Two-stage P-then-Q pipeline | Professor's email is authoritative and explicit on ordering | — Pending (ADR-0002) |
| Rolling 252-day P-window per quote date, not full-sample | Tracks regime shifts (e.g. NVDA 2023 vs 2025); matches live book-marking framing | — Pending (ADR-0003) |
| Derive time series from option data itself (col 7) | Verified one consistent spot price per date; avoids external data dependency | ✓ Good |
| Q-calibration loss = squared IV error | Comparable across strikes/moneyness; standard practice | — Pending |
| Multi-start (n=8) L-BFGS-B optimizer | Balance of robustness vs. speed for nonconvex fits across 10 stocks | — Pending |
| Notebook first, REPORT.md later | Avoid premature formatting work before results are stable | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-08-05 after initialization*
