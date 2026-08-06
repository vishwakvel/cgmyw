# Re-fit the P-measure on a trailing 252-day window per quote date, not once per stock

Status: accepted

Considered fitting (C,G_P,M_P,Y,W_P) once per stock on the full ~2.6-year return history, which would be simpler and give more stable estimates of C,Y (the parameters the Hellinger condition treats as fixed, pathwise-observable quantities — arguably they *shouldn't* be regime-sensitive). Rejected in favor of a rolling trailing-252-trading-day window re-fit at each option quote date, to better match the professor's description of a live book-marking workflow and to let the model track regime shifts (e.g. NVDA's return distribution changed materially between 2023 and 2025).

Consequence: the P-fit is no longer a one-time-per-stock computation — it runs once per (stock, quote date) used in Q-calibration, so the P-step's cost (FFT density inversion inside an MLE loop) is incurred repeatedly. Dates near the start of the sample (before ~252 trading days of history accumulate, roughly the first calendar year from 2023-01-03) won't have a full window; needs a minimum-window fallback rule, decided at implementation time.
