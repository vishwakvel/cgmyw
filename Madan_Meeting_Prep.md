# Meeting Prep — Dr. Madan, [Meeting Date]

---

## 1. Background: What CGMYW Is and Why It Exists

### The problem with existing jump models

Standard option pricing models (Black-Scholes, and even the base CGMY jump model) impose measure changes between the physical (P) measure and the risk-neutral (Q) measure that are too restrictive. Specifically: a pure exponential tilt (changing only G and M, the tail-decay rate parameters) can only produce a **monotone** relationship between P and Q densities — either P is above Q at every jump size, or Q is above P at every jump size. It can't do both.

But empirically, that's not how markets behave. Every market event has two probabilities: the actual likelihood of it happening (physical/P), and the price the market charges to bet on it (risk-neutral/Q). The empirical pattern is:
- **Small moves**: physical probability > risk-neutral price (people underpay for everyday risk — this is the "investment" regime)
- **Large moves (tails)**: risk-neutral price > physical probability (people overpay for tail insurance — this is the "insurance" regime)

A model restricted to exponential tilts alone cannot produce this crossing behavior no matter how you set the parameters.

### The CGMYW fix

The CGMYW Lévy density adds a **power-law tilt** on top of the standard CGMY exponential tilt:

```
k(x) = C · exp((G-M)x/2 - (G+M)|x|/2) / |x|^(1+Y) · (1+|x|)^(-W)
```

The power tilt `(1+|x|)^(-W)` is the key addition. It's **inactive near zero** (so it doesn't disturb the Hellinger equivalence condition — see below) but **reshapes the tails** at large |x|. Because it grows differently than the exponential tilt, it allows the density *ratio* between P and Q to cross — P above Q for small x, Q above P for large x (or vice versa) — which a pure exponential tilt structurally cannot do.

### The Hellinger equivalence constraint

For P and Q to be *equivalent* measures (a technical requirement for the change of measure to be valid — no measure can assign positive probability to something the other assigns zero probability to), the Hellinger distance between the two Lévy densities must be finite. For CGMY-family densities, this condition forces **C and Y to stay fixed** across P and Q — they cannot change under the measure change. Only the tail-decay parameters (G, M, and now W) are free to move between P and Q.

This is why the estimation naturally splits into two steps:
- **P-measure step**: estimate all parameters (C, Y, G, M, W) from historical return data
- **Q-measure step**: keep C, Y fixed at their P-measure values; recalibrate only G, M, W to fit the option surface

### The closed-form solution

Madan derived the characteristic exponent for this density in closed form using the **Kummer confluent hypergeometric function U(a,b,z)**. This matters practically because a closed-form characteristic function is what makes Fourier-based option pricing (FFT / Carr-Madan methods) tractable — without it, you'd need slow numerical integration for every price evaluation.

**Confirmed: it's a 5-parameter model — C, Y, G, M, W (one shared W)**, not separate Wn/Wp on each side. This was ambiguous from the paper's general notation but Madan confirmed 5 params in the plan-email exchange.

---

## 2. My Implementation Plan (Already Sent to Him — Confirm/Revisit)

I sent Madan a detailed plan email covering the following. Worth walking through it with him to confirm nothing needs to change, especially now that the required-returns extension (Section 4 below) might affect sequencing.

**Scope**: Implement CGMYW in Python, calibrate against WRDS option data for 10 stocks (AAPL, AMZN, CSCO, GOOGL, IBM, INTC, META, MSFT, NVDA, ORCL), write up results.

**4 Phases (strictly sequential):**
1. **AAPL MVP** — full pipeline on one stock: data load → characteristic function → P-fit → Q-calibration → smile plot with error metric. Proves correctness before scaling.
2. **Scale to 10 stocks** — generalize with zero stock-specific code, produce one cross-stock results table.
3. **Estimation notebook** (`cgmyw_estimation.ipynb`) — narrative, smile-fit plots, P-vs-Q parameter comparison, error diagnostics across all 10 names.
4. **Report** — distilled write-up targeting his new "Rapid Communications in Mathematical Finance" section (~5 pages), deferred until results are stable.

**Key methodology decisions made (paper is silent on these — worth flagging again in person):**

- **Date/maturity auto-selection**: filter to ~30-45 calendar-day tenor, require ≥6 strikes and ≤50% missing IVs, rank by strike count then data quality. Chosen so it still works on thinner-volume names like CSCO.
- **P-measure estimation**: characteristic-function MLE (not moment matching) — chosen because it's the estimator that actually uses the closed-form characteristic function he derived, and it's more reliable for estimating the tail-shape parameters (Y, W) that matter most for the paper's thesis. Uses a **rolling 252-trading-day window per quote date**, not one fixed estimate per stock — this keeps P and Q measure-matched in time (see rationale below), with a fallback to `min(252, available history)` for early dates.
- **Q-measure calibration**: minimizing squared **implied-vol error** (not raw price error), inverting both model and market prices to IVs via Black-Scholes and matching there. Chosen over raw price error because IV-space normalizes across strikes/maturities and is the field-standard convention; chosen over bid-ask-weighted price error because IV-error is more standard/defensible for a first empirical pass, even though bid-ask weighting is arguably more in line with his own conic finance philosophy on spreads.
- **Optimization**: 8 random-start bounded L-BFGS-B fits for both P and Q stages (multi-start chosen over single local run because this is a brand-new model with no prior literature giving good starting points — a single run risks landing in a bad local optimum unnoticed; chosen over a global optimizer like differential evolution because the dimensionality is low (5-6 params) and the objective is smooth, so paying for population-based global search isn't worth it, especially since Q-step evaluations already involve expensive Kummer-function calls). If none of the 8 converges to a finite, in-bounds result, the pipeline **raises an error with diagnostics** rather than silently reporting a bad fit.
- **Units**: trading-day conventions throughout — P-fit uses daily returns directly, Q-fit converts calendar days to trading days (`calendar_days × 252/365`) and the risk-free rate to a daily rate on the same convention, so P and Q parameters stay comparable.
- **Deliverable format for MVP**: a script with reusable functions (not a notebook) — notebook is Phase 3's job and will import these functions rather than re-deriving the pipeline.

**Why rolling P-window instead of one fixed estimate per stock**: since the whole point of the model is comparing P against Q on the same options, using a stale/fixed P estimate that doesn't line up with when the option quotes were trading would contaminate the comparison — any crossing behavior found (or not found) could be an artifact of regime mismatch rather than a real feature of the model. NVDA is a good example: its 2023 return distribution looks very different from its 2025 one, so a single full-sample estimate blends two different worlds.

**Tooling note worth mentioning**: `scipy.special.hyperu` (the obvious tool for the Kummer-U function) turns out not to support complex arguments at all — it's real-argument-only. Since the characteristic exponent needs complex evaluation on an FFT grid, I reconstructed it from `scipy.special.hyp1f1` instead (stable small-argument and large-argument asymptotic branches), validated against arbitrary-precision results to ~1e-14. Worth flagging since it's a deviation from the literal tooling, though it stays within scipy.

**Explicitly out of scope for this milestone** (mention only if relevant): the full 8-parameter asymmetric CGMYW (independent C, Y, W per tail) is tracked as a possible v2 extension; bilateral gamma as an alternative model; external price data (using WRDS's own spot column instead); formal journal submission.

---

## 3. The New Piece: Required Returns via the Signed Kernel-Difference Operator

Madan separately sent the **"Asset Pricing in Multi-Asset Markovian Economies"** paper (with King Wang), pointing specifically to **Section 4**, and said: *"This is explained in Section 4 with results later in the paper. We could do this with the CGMYW model."*

### What that paper does, briefly

It builds high-dimensional Markovian models for multi-asset economies where states are observable (asset returns are linear combinations of independent Lévy "driver" processes, and the drivers themselves are kernel-weighted integrals of past Lévy shocks — this construction is what gives the Markovian structure). It solves for state-dependent risk premia via minimum-norm solutions to the martingale equations, and uses those to derive **state-dependent required returns** on each asset. Applied empirically to US/EU/UK sector ETF economies.

### Section 4 specifically — why it matters for CGMYW

Normally, required returns are computed as **covariation with the pricing-kernel density process** — this only works when P and Q are *equivalent* measures (i.e., when the Radon-Nikodym derivative between them exists). That's the case when only G, M move under the measure change and C, Y stay fixed (the "clean" Hellinger-equivalent case).

But later in the paper, Madan also lets the **C parameters change** under the measure change (going from 2m to 4m free parameters in the multi-asset setting). When C changes, equivalence can fail — the Radon-Nikodym interpretation breaks down. His key result: the required-return formula **still works** even without equivalence. It generalizes to the action of a **signed kernel-difference operator** — literally the risk-neutral Lévy density minus the physical Lévy density, `k̃(y) − k(y)`, applied to the asset's jump transform `(e^(B_ij·y) − 1)`. This is Equation (36) in the paper, and it's the general formula that survives outside equivalence.

### Why this connects directly to CGMYW

CGMYW's entire reason for existing is the **power tilt (W)**, which is a *much* bigger departure from the base CGMY density than just letting C change. So there's a live question of whether CGMYW's measure change actually preserves the Hellinger equivalence condition at all — and if it doesn't (which seems plausible, since the tilt reshapes the tails so substantially), then the signed kernel-difference operator (eq. 36) would be the *only* valid way to compute required returns for this model, not the standard covariation formula.

Practically: once I have calibrated CGMYW parameters (C, Y, G, M, W) on both P and Q sides for a stock, I could apply the same eq. 36 machinery — the kernel-difference operator acting on the jump transform — to compute the model's implied required return for that asset. This would be a genuinely new empirical result: not just "does CGMYW fit the option smile well" but "what required returns does CGMYW imply, and how do they compare to what the SD1CGMY model produced in this paper."

---

## 4. Questions to Ask Tomorrow

### On the required-returns extension (new, most important to resolve)

1. **Scope**: Is this meant as a **univariate/single-asset** application (just using CGMYW's own G, M, W kernel difference for one stock at a time), or does it require embedding CGMYW into the **full multi-asset driver framework** from this paper (the B-matrix, m independent Lévy drivers, spectral diagonalization, etc.)? This matters enormously for how much additional work this is — univariate is a contained add-on to the existing pipeline; multi-asset is close to a second project.
2. **Equivalence question**: Does he expect CGMYW's power tilt to break Hellinger equivalence (meaning eq. 36's kernel-difference operator is *required*, not optional), or does he think equivalence might still hold under some parameter regime for this model? I'd like to understand his intuition here since it determines which formula is even valid.
3. **Relationship to the existing deliverable**: Does this extend/replace the WRDS calibration project, or is it a separate output/writeup layered on top once calibration is done?
4. **Approach**: Same phased style as the base project (get a working single-stock version first, then scale), or does he have a different sequencing in mind given this consumes already-fitted parameters rather than starting from data?
5. **Priority**: Should I treat this as the next thing to work on, or does finishing the base WRDS calibration (Phases 1-2) come first, with this as Phase 5 afterward?

### On the base WRDS/CGMYW plan (confirm/revisit given the new piece may affect sequencing)

6. Any feedback on the specific methodology decisions in the plan email — auto date/maturity selection, rolling P-window, 8-multistart optimizer policy, trading-day unit convention, IV-space Q calibration? (Asking directly since these were judgment calls where the paper didn't specify an approach.)
7. Confirm: 5 parameters with **one shared W** (not separate Wn/Wp per side) — matches what I've built so far, just want to make sure that's definitely his intent given the paper's notation is written generally enough to support either reading.
8. **General sequencing**: given the new required-returns piece, does he want me to finish scaling to all 10 stocks before touching this, or interleave/prioritize this first?

---

## 5. My Current Technical Approach (for reference if he asks)

- **Language/tools**: Python, `scipy` throughout (including the reconstructed Kummer-U workaround above), FFT-based characteristic function inversion for option pricing.
- **Data**: WRDS options database (OptionMetrics/IvyDB US).
- **Structure**: reusable functions (data load → P-fit → Q-fit → plot) rather than notebook-first, so Phase 3's notebook can import from a tested pipeline rather than duplicating logic.
- **Repo**: private GitHub repo; licensing/visibility deferred pending his guidance, given the IP is his framework.

---

## 6. One-Line Summary to Open the Meeting With

*"I've got the base CGMYW plan built out — AAPL MVP is the current focus — and I read through the Multi-Asset Markovian paper you sent. I'd like to walk through the plan details briefly, then figure out how the required-returns piece fits in — scope, sequencing, and whether CGMYW's tilt actually preserves equivalence or needs the kernel-difference approach from Section 4."*
