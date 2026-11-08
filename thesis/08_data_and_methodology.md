# Data and Methodology — chapter draft

## Research design

The study asks whether geopolitical risk (GPR) and financial-market uncertainty (VIX) help identify and predict crude-oil volatility regimes when they enter the transition probabilities of a Markov-switching model. Brent is the primary series; West Texas Intermediate (WTI) supplies a cross-market robustness check. The models are a constant-variance Gaussian benchmark (B0), Gaussian and Student-t GARCH(1,1) benchmarks (B1 and B1-t), a two-state constant-transition Gaussian Markov model (M1), a descriptive three-state counterpart (M2), and two-state transition models with lagged GPR (M3), lagged VIX (M4), or both (M5). All Markov observation equations permit both the conditional return mean and variance to differ by state. The appendix extension E1–E3 is specified in [08_stage07_appendix.md](08_stage07_appendix.md).

The working research plan established five hypotheses, reproduced in the [evidence matrix](08_research_evidence_matrix.md): H1 concerns distinguishable and persistent oil-volatility regimes; H2 concerns information in GPR about entry into or persistence of high volatility; H3 concerns information in VIX about high-volatility transitions; H4 concerns the incremental information in GPR beyond VIX, or vice versa; and H5 concerns out-of-sample gains from time-varying transition models over simpler models. The supervisor may refine their final wording, but the analysis is mapped to these existing hypotheses. This is an observational forecasting design; a transition association does not identify a causal effect.

## Data, sample construction and timing

The unchanged raw price files are `DCOILBRENTEU.csv` and `DCOILWTICO.csv` (US dollars per barrel), with `VIXCLS.csv` for VIX and `data_gpr_daily_recent.xls` for the recent daily GPR index. The price and VIX series identifiers correspond to the US Energy Information Administration series distributed by FRED and the Cboe VIX series, respectively; the local files' precise download dates are **not recorded**. [FRED Brent](https://fred.stlouisfed.org/series/DCOILBRENTEU), [FRED WTI](https://fred.stlouisfed.org/series/DCOILWTICO), [FRED VIX](https://fred.stlouisfed.org/series/VIXCLS), and [Cboe's VIX methodology](https://cdn.cboe.com/resources/vix/VIX_Methodology.pdf) document the underlying series. VIX measures expected 30-day S&P 500 volatility inferred from option prices; it is a broad financial uncertainty proxy, not oil-specific implied volatility. The daily GPR index is a news-based measure of geopolitical risk; its source, construction and revision caveat are documented by [Caldara and Iacoviello (2022)](https://www.matteoiacoviello.com/gpr_files/GPR_PAPER.pdf) and the [maintainer's data page](https://www.matteoiacoviello.com/gpr.htm). The index is not an observed geopolitical shock or an instrument.

Let $\mathcal O$ be the set of dates with an **observed oil price**, including a nonpositive price. For an observed price on date $t$, define the immediately preceding observed-price date and the percentage log return as

$$
\tau(t)=\max\{s\in\mathcal O:s<t\},\qquad
r_t=\begin{cases}
100\log\!\left(P_t/P_{\tau(t)}\right),&P_t>0\text{ and }P_{\tau(t)}>0,\\
\text{undefined},&\text{otherwise}.
\end{cases}
$$

This matches Notebook 01's implementation: it first drops **missing** prices, then uses a one-row shift to select the previous observed price, and only then checks that both endpoints are positive. It does not search backward for a more recent *positive* price. If no preceding observed price exists, the return is also undefined. The date interval $t-\tau(t)$ is retained. An observation after a missing-price gap measures the cumulative change over that interval and is treated as one observation by the daily models; it is not literally a one-calendar-day return. Prices are neither carried forward nor interpolated, and no holiday zero return is manufactured. The raw Brent file's missing prices explain most of the sparse early counts: complete-case training returns number 197 in 2000, 189 in 2001 and 188 in 2002; only 5, 5 and 6 returns, respectively, are lost at the VIX/GPR complete-case alignment. The maximum training gap is 11 calendar days; 167 training intervals exceed three days. These facts constrain the daily-time interpretation of variance recursions and state durations. The 2007–2015 re-estimation is a separately labelled sensitivity exercise. [Notebook 01](../notebooks/01_data_preparation.ipynb), [coverage audit](../results/tables/03_correction_annual_coverage.csv) and [forecast data audit](../results/tables/05_data_audit.json) retain the details.

WTI's negative observed price of −36.98 dollars per barrel on 20 April 2020 remains in the price audit. The 20 April return is undefined because its current price is negative; the 21 April return is undefined because its **immediately preceding observed** price is −36.98. On 22 April, both observed endpoints are positive (13.64 and 8.91 dollars per barrel), so the +42.583% log return is retained. The implementation and saved data therefore agree with the stated policy. The WTI complete-case data contain 6,336 returns, including 3,633 training, 1,000 validation and 1,703 test returns. [Notebook 01](../notebooks/01_data_preparation.ipynb), [WTI data audit](../results/tables/06_wti_data_audit.json) and [price audit](../data/processed/01_wti_price_return_audit.csv) provide the trace.

The maximum Brent return file has 6,495 rows. Requiring an oil return and the original and lagged predictors produces 6,410 common Brent rows through **26 October 2026**: 3,699 training (4 January 2000–31 December 2015), 999 validation (4 January 2016–31 December 2019), and 1,712 final test observations (2 January 2020–26 October 2026). The 2026 year is partial. This complete-case restriction discards 85 otherwise usable Brent returns. Forecast comparisons within each market and period use identical evaluation dates. See [sample summary](../data/processed/01_sample_summary.csv) and [data audit](../results/tables/05_data_audit.json).

For return date $t$, GPR is taken from **exactly the preceding calendar date**. VIX is taken from the most recent nonmissing **previous VIX observation**, which may be more than one calendar day earlier. Notebook 01 constructs the VIX lag on rows where a same-day VIX value exists, then requires both original and lagged VIX/GPR fields for the shared complete-case sample. Same-day VIX is retained for descriptive alignment and common-date comparability, but it is **not** a transition predictor and is **not mathematically required** by the M3–M5 observation or transition equations. This is a conservative sample-design and lag-construction choice. It excludes 85 otherwise valid Brent returns: 51 training, 21 validation and 13 test. A read-only check of the raw VIX file found an earlier observed VIX quote for every one of those dates, so their exclusion is not forced by absence of historical VIX information; using them would require rebuilding lag fields and re-estimating all comparison models on a newly specified common sample. The established results are left unchanged. This restriction can affect estimation, forecast-date coverage and comparability to an analysis using all usable oil returns; it is distinct from the much larger 2000–06 Brent price gaps. [Notebook 01](../notebooks/01_data_preparation.ipynb), [maximum Brent data](../data/processed/01_brent_max.csv), [complete-case Brent data](../data/processed/01_brent_complete_case.csv) and [sample summary](../data/processed/01_sample_summary.csv) document the design.

Each predictor's source date precedes its target return date. Historical GPR reference dates do not establish when those data were published, and the supplied series is a revised vintage. Accordingly, GPR-containing forecasts are **historical pseudo-out-of-sample**, not verified real-time predictions. Predictor $x$ is standardized as

$$z_t=(x_t-\bar x_{\mathcal T})/s_{\mathcal T},$$

where $\bar x_{\mathcal T}$ and the population standard deviation $s_{\mathcal T}$ use only the estimation window $\mathcal T$. Training statistics are used for validation; training-plus-validation statistics are used for the locked test fits. No test observation contributes to these moments. [Scaling records](../results/tables/05_scaling.csv) preserve the exact values.

## Preliminary diagnostics

Descriptive returns are reported in percentage points. Training-sample Jarque–Bera tests address marginal Gaussian normality; the augmented Dickey–Fuller (ADF) and KPSS tests have complementary nulls of a unit root and level stationarity; Ljung–Box tests examine serial dependence in signed and squared returns at 5, 10 and 20 observation lags; and Engle's ARCH-LM test examines squared-return dependence at 10 lags (with 5 and 20 as checks). The training period, rather than later data, determines diagnostic interpretation. Their rejection or non-rejection is evidence about specific nulls, not proof that one proposed volatility model is correct. [Tables 02](../results/tables/02_test_summary.csv) record the settings and statistics. ARCH's original formulation is due to [Engle (1982)](https://doi.org/10.2307/1912773).

## Benchmark conditional distributions

B0 uses a constant mean $\mu$ and variance $\sigma^2$: $r_t\sim N(\mu,\sigma^2)$. B1 and B1-t use the same constant-mean GARCH(1,1) recursion, following [Bollerslev (1986)](https://doi.org/10.1016/0304-4076(86)90063-1):

$$
r_t=\mu+\varepsilon_t,\qquad
\varepsilon_t=\sqrt{h_t}\,z_t,\qquad
h_t=\omega+\alpha\varepsilon_{t-1}^2+\beta h_{t-1},
$$

with $\omega>0$, $\alpha,\beta\geq0$. B1 assumes $z_t\sim N(0,1)$. B1-t assumes a Student-t innovation standardized to unit variance, with degrees of freedom $\nu>2$. Thus $h_t$ is the conditional **variance** under both distributions. The one-step conditional **second moment** is $h_t+\mu^2$. A value of $\alpha+\beta$ near one signals very persistent estimated variance; it does not by itself establish an exact integrated process. The Gaussian and Student-t fits use the same observations. Standardized and squared-standardized residual Ljung–Box tests and ARCH-LM checks assess what the fitted recursion leaves unexplained. See [GARCH comparison](../results/tables/03_correction_garch_comparison.csv) and [residual diagnostics](../results/tables/03_correction_garch_residual_tests.csv).

## Markov observation and transition equations

Following the latent-state filtering framework of [Hamilton (1989)](https://doi.org/10.2307/1912559), for $K=2$ in M1 and M3–M5 (or $K=3$ in M2),

$$
S_t\in\{0,\ldots,K-1\},\qquad
r_t\mid(S_t=j,\mathcal F_{t-1})\sim N(\mu_j,\sigma_j^2).
$$

Both $\mu_j$ and $\sigma_j^2$ are state-specific and constant within an estimation fit. States are relabelled by increasing fitted variance: `Low`, `High` (and `Medium` for M2). State labels represent statistical conditional variance, not a directly observed market classification. For constant transitions, $p_{ij}=\Pr(S_t=j\mid S_{t-1}=i)$ and each origin row sums to one. Expected duration in state $i$ at a fixed transition probability is $1/(1-p_{ii})$ **retained observations**, not calendar days. M2 is retained for retrospective description because a near-zero fitted transition lacks a standard error, making conventional coefficient inference unreliable.

M3–M5 change **only** transition probabilities, following the time-varying probability idea of [Filardo (1994)](https://doi.org/10.1080/07350015.1994.10524545). For origin $i\in\{0,1\}$, the implementation's logit is

$$
\Pr(S_t=0\mid S_{t-1}=i,z_t)
=\operatorname{logit}^{-1}\!\left(a_i+b_{i,G}z_{G,t}+b_{i,V}z_{V,t}\right),\quad
\Pr(S_t=1\mid S_{t-1}=i,z_t)=1-\Pr(S_t=0\mid S_{t-1}=i,z_t).
$$

M3 includes $b_{i,G}$ only, M4 includes $b_{i,V}$ only, and M5 includes both; $a_i$ is an explicit transition intercept. A coefficient on the probability of **destination 0** must be translated carefully before describing low→high entry or high→high persistence. The fitted `statsmodels` transition tensor is stored as destination × origin × time and is transposed to origin × destination before interpretation. [Stage 04 transition outputs](../results/tables/04_average_transitions.csv) and [parameter estimates](../results/tables/04_m5_parameters.csv) are the checked source. Parameter fitting uses repeated independent starts, convergence and likelihood-spread checks, finite standard errors where available, positive variances, valid transition rows, and nontrivial state occupancy. Near-zero probabilities and multiple local optima remain potential sources of fragile inference.

For economic interpretation, the predictor under study is changed from its training 10th to 90th percentile while the other is held at its training median. The difference in the fitted probability is reported in percentage points; its 95% interval is a first-order delta-method calculation using the fitted covariance matrix. These intervals are conditional on model specification and are not causal confidence intervals. The p10–p90 changes of different GPR indices represent different raw shifts and must not be treated as equivalent shocks.

## Chronological one-step forecasting

All primary models first fit only the 2000–2015 training returns and produce 2016–2019 validation forecasts. The final model specifications are then locked. For the 2020–26 test, parameters are re-estimated on returns **through 31 December 2019** and held fixed while the recursion processes later observations. Validation and test predictor scaling use their corresponding estimation windows. The forecasts are one step ahead on the sequence of retained observations, subject to the irregular-calendar limitation; no expanding-window parameter re-estimation is used.

Let $\pi_{t-1\mid t-1,i}$ be the filtered state probability after observing the previous retained return and $P_t=(p_{ij,t})$ the origin-row transition matrix calculated with lagged predictors. The predictive state distribution is

$$q_{t\mid t-1,j}=\sum_i\pi_{t-1\mid t-1,i}p_{ij,t}.$$

Its Gaussian mixture predictive mean, second moment, variance, and density are

$$
m_t=\sum_jq_{t\mid t-1,j}\mu_j,\qquad
s_t=\operatorname E(r_t^2\mid\mathcal F_{t-1})
=\sum_jq_{t\mid t-1,j}(\sigma_j^2+\mu_j^2),
$$
$$
v_t=s_t-m_t^2,\qquad
f_t(r)=\sum_jq_{t\mid t-1,j}\,\phi(r;\mu_j,\sigma_j^2).
$$

The state-mean differences therefore contribute to mixture variance. After storing the forecast, the observed $r_t$ updates the filtered distribution:

$$\pi_{t\mid t,j}=\frac{q_{t\mid t-1,j}\phi(r_t;\mu_j,\sigma_j^2)}{f_t(r_t)}.$$

Only **filtered** probabilities enter this sequence. Smoothed probabilities use later returns and are restricted to retrospective figures and occupancy descriptions. GARCH forecasts calculate $h_t$ from the previous observed residual and previous conditional variance, then update the recursion after $r_t$ is observed. Log densities use a Gaussian for B0/B1, the standardized Student-t for B1-t, and a numerically stable log-sum-exp Gaussian mixture for Markov models. The saved forecast code asserts predictor chronology, identical dates, probability normalization, positive finite moments, and forecast-before-update order. [Forecast code](../code/stage05_forecast.py) and [forecast audit](../results/tables/05_data_audit.json) are the reproducible record.

## Forecast loss and uncertainty

The common noisy proxy is the squared observed percentage return $y_t=r_t^2$. The primary score compares it with the predicted conditional **second moment** $s_t>0$:

$$
L^{Q}_t=\log s_t+\frac{y_t}{s_t},\qquad
\overline{L}^{Q}=\frac1n\sum_{t=1}^nL^{Q}_t.
$$

When $m_t\ne0$, $s_t=v_t+m_t^2$; using $v_t$ instead would evaluate a different forecast target. Secondary losses are $n^{-1}\sum_t(s_t-y_t)^2$ (MSE, in fourth powers of return percentage points), $-n^{-1}\sum_t\log f_t(r_t)$ (negative predictive log score), and mean absolute second-moment error (in squared percentage points). QLIKE is motivated by proxy-robust volatility forecast comparison in [Patton (2011)](https://doi.org/10.1016/j.jeconom.2010.03.034); a single squared daily return remains a very noisy proxy and can make event-specific loss differences unstable. All losses in each model comparison use the same retained dates and return scaling. Density scores assess the full predictive distribution and need not rank models as QLIKE does.

For selected paired QLIKE differences $d_t=L_{A,t}^Q-L_{B,t}^Q$, the project resamples **circular blocks of 10 consecutive observation dates**, with 2,000 bootstrap replicates and recorded seeds, preserving local dependence. A positive mean favours B. The saved outputs contain basic 95% intervals and a centered absolute-tail two-sided p-value for Stages 05–06. With asymmetric bootstrap distributions, that p-value need not give the same 5% decision as an equal-tail basic interval; the WTI M1–M4 case is corrected coherently in [the appendix](08_stage07_appendix.md). The procedure does not adjust for multiple model comparisons, chosen block length, structural breaks, or the nonregularity of some nested comparisons. Therefore point loss differences are assessed alongside interval width, validation consistency and episode sensitivity rather than converted mechanically into declarations of superiority. [Comparison code](../code/stage05_forecast.py) and [bootstrap table](../results/tables/05_paired_block_bootstrap.csv) specify the implementation.

The 2020 and 2022 disruption windows are full calendar years, defined before period-specific interpretation. Top-1%-absolute-return influence is diagnostic only; no extreme observation is removed from primary evaluation. WTI, five-observation VIX and five-calendar-day GPR lags, $\log(1+\mathrm{GPR})$, and a 2007-start Brent fit are sensitivity exercises rather than replacements for the primary specification. Stage 07's AI and oil-specific GPR comparisons are retrospective extensions because those indices were developed in 2026 and 2020–26 outcomes had already been examined in prior stages.

## Verified methodological references

The bibliographic entries below were checked against publisher, author, index-provider or official data pages. They support methods and data definitions; they are not a substitute for a full literature review.

- Bollerslev, T. (1986). “Generalized autoregressive conditional heteroskedasticity.” *Journal of Econometrics*, 31(3), 307–327. [DOI](https://doi.org/10.1016/0304-4076(86)90063-1).
- Caldara, D., and Iacoviello, M. (2022). “Measuring Geopolitical Risk.” *American Economic Review*, 112(4), 1194–1225. [Author-hosted paper](https://www.matteoiacoviello.com/gpr_files/GPR_PAPER.pdf).
- Engle, R. F. (1982). “Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation.” *Econometrica*, 50(4), 987–1007. [DOI](https://doi.org/10.2307/1912773).
- Filardo, A. J. (1994). “Business-cycle phases and their transitional dynamics.” *Journal of Business & Economic Statistics*, 12(3), 299–308. [DOI](https://doi.org/10.1080/07350015.1994.10524545).
- Hamilton, J. D. (1989). “A new approach to the economic analysis of nonstationary time series and the business cycle.” *Econometrica*, 57(2), 357–384. [DOI](https://doi.org/10.2307/1912559).
- Patton, A. J. (2011). “Volatility forecast comparison using imperfect volatility proxies.” *Journal of Econometrics*, 160(1), 246–256. [DOI and author version](https://public.econ.duke.edu/~ap172/Patton_vol_proxies_JoE_2011.pdf).
