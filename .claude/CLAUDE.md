<!-- GSD:project-start source:PROJECT.md -->

## Project

**CGMYW Option Pricing**

An empirical research project (prof-directed) that implements a new 8-parameter Lévy jump model — CGMY extended with an independent power-law tail tilt per side (Cn, Cp, G, M, Yn, Yp, Wn, Wp) — in Python, calibrates it against real WRDS equity option data for 10 stocks, and writes up how it performs. The model comes from an unpublished research note (attached PDF, likely Dilip Madan) generalizing the CGMY/variance-gamma family that has priced Morgan Stanley's option books for a decade. Parameter count corrected 2026-08-07 per a direct meeting with Dr. Madan (see `.planning/PROJECT.md`, ADR-0004).

**Core Value:** A working, generic P-then-Q calibration pipeline (time series → risk-neutral option smile) for one stock that produces a defensible parameter fit and error report — everything else (scaling to 10 stocks, polished write-up) builds on this working core.

### Constraints

- **Data source**: WRDS `.dat` files remain the sole source for Q-measure/options data (no live fetching for that side). For P-measure's daily return series, the primary source is CRSP via WRDS (`DlyRet`, dividend-inclusive and split-corrected, per the professor's 2026-08-17 guidance), with live fetching via `yfinance` (dividend/split-adjusted close) permitted as a fallback and cross-check, and the WRDS options-file spot column as a second fallback and cross-check — see ADR-0006.
- **Numerical**: CGMYW's characteristic exponent requires evaluating the Kummer U (confluent hypergeometric) function at complex arguments across an FFT grid; must use `scipy.special.hyperu` (handles complex input) rather than a hand-rolled series.
- **Faithfulness to source**: The professor's email takes precedence over the summarized task brief wherever they diverge (see ADR-0002).
- **Tooling**: Python + scipy/numpy stack (no new heavy dependencies) per prior conversation — `scipy.io.loadmat`, `scipy.optimize`, `scipy.special.hyperu`, `numpy.fft`. `yfinance` is an approved dependency in its fallback/cross-check role for the P-data source (ADR-0006) — no other new heavy dependencies.

<!-- GSD:project-end -->

<!-- GSD:stack-start source:STACK.md -->

## Technology Stack

Technology stack not yet documented. Will populate after codebase mapping or first phase.
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->

## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->

## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->

## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->

## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:

- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->

## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
