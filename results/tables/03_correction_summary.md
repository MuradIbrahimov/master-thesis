# Stage 03 correction audit — 3 November 2026

The original Stage 03 comparison remains on the **same 3,699 Brent complete-case training dates**. No observation was added or removed, so B0, Gaussian B1, M1, and M2 information criteria remain sample-comparable. The reproducible audit is code/stage03_corrections.py; detailed CSV outputs in this directory start with 03_correction_.

## Early Brent coverage

| Year | Raw weekday rows | Nonmissing raw prices = maximum returns | Complete-case returns | Lost at GPR/VIX alignment | Missing raw prices | Maximum return gap |
|---|---:|---:|---:|---:|---:|---:|
| 2000 | 260 | 202 | 197 | 5 | 58 | 10 calendar days |
| 2001 | 261 | 194 | 189 | 5 | 67 | 8 calendar days |
| 2002 | 261 | 194 | 188 | 6 | 67 | 11 calendar days |
| 2003 | 261 | 216 | 210 | 6 | 45 | 7 calendar days |
| 2004 | 262 | 230 | 223 | 7 | 32 | 6 calendar days |
| 2005 | 260 | 224 | 217 | 7 | 36 | 9 calendar days |
| 2006 | 260 | 229 | 223 | 6 | 31 | 9 calendar days |

For every year 2000–2015, the dates with nonmissing raw Brent prices exactly equal the dates in the maximum-return file. The first in-window return (4 January 2000) uses the previous observed price from 30 December 1999. The complete-case losses exactly equal missing same-day VIX entries in the maximum file; the GPR/VIX join did not cause the large 2000–2006 shortfall. The lower counts primarily originate in the raw Brent series. Returns compare consecutive observed positive prices. A return following an irregular gap is the cumulative log change over that recorded gap, **not a literal one-calendar-day return**. No price was forward-filled or zero holiday return created. This unequal spacing remains a limitation of the daily econometric models and warrants sensitivity analysis before final thesis claims.

## GARCH distribution and remaining dependence

| Model | AIC | BIC | alpha + beta | 10-lag squared standardized Ljung–Box p | 10-lag residual ARCH-LM p |
|---|---:|---:|---:|---:|---:|
| Gaussian GARCH(1,1) | 15,895.50 | 15,920.36 | 0.9999999996 | 0.000220 | 0.000120 |
| Student-t GARCH(1,1) | 15,715.18 | 15,746.26 | 0.9992463860 | 0.000076 | 0.000040 |

Both fits converged on identical observations. Student-t improves in-sample AIC/BIC, but both persistence estimates are effectively near-integrated and both leave significant squared-residual dependence and ARCH effects. The signed standardized-residual Ljung–Box p-values at lags 5, 10, and 20 exceed 0.1 for both models. Full diagnostics and Student-t degrees of freedom are in the companion CSV files.

## M1 and M2

M1 preserves the original switching-mean, switching-variance specification. Its three original starts converged with a likelihood spread below 0.00002; transition rows and filtered/smoothed probabilities normalize, and both states have material occupancy. The exact-sample M1 is refitted in Notebook 04.

The expanded M2 audit used ten independent seeded starts. All converged; log-likelihood spread was 0.00625. Estimated daily state standard deviations are 1.055, 2.097, and 4.485 percentage points, with mean smoothed occupancies of 20.9%, 67.9%, and 11.2%. This is a useful retrospective three-level description. The direct high-to-low transition is estimated essentially at zero, and its standard error is unavailable. M2 is therefore **not reliable for conventional inference about every transition coefficient**, even though the likelihood surface appears numerically stable at the reported precision.

## Gate

**PASS for Stage 04:** training dates and gaps are documented, M1 is stable, and transition/probability checks pass. The GARCH and M2 limitations remain unresolved and explicit. Stage 04 uses the unchanged 3,699-date training sample to retain comparability.
