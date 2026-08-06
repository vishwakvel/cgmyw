# Follow the two-stage P-then-Q pipeline, not a Q-only calibration

Status: accepted

The main task brief, read alone, only asks for option-data (Q-measure) calibration and a results write-up — no time series step. But the professor's email is explicit: "You can start with time series data and then move to option data." When the two sources conflicted, we deferred to the professor's email as the authoritative source over the summarized task brief.

Decided pipeline: (1) fit (C, G_P, M_P, Y, W_P) to a historical return time series under P; (2) hold C and Y fixed (Hellinger equivalence — see CONTEXT.md) and calibrate (G_Q, M_Q, W_Q) to the option smile under Q via FFT, subject to the martingale condition. This roughly doubles phase-1 scope versus Q-only, and requires picking a time-series estimation method (separate decision) before the option calibration can be scoped.

Consequence: the option-calibration step is no longer a free 5-parameter fit — it's a constrained 3-parameter fit (G,M,W) with C,Y pinned from the P step. The optimizer and its identifiability properties differ from a naive Q-only calibration.
