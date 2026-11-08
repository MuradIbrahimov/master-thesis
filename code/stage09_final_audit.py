"""Stage 09: independent checks and bounded validation-only sensitivities.

Run via notebooks/09_final_econometric_audit.ipynb. Existing outputs are read-only.
New outputs have the 09_ prefix. Returns are in percentage points.
"""
from __future__ import annotations

import json
import hashlib
import re
import math
import platform
import warnings
from pathlib import Path

import arch
from arch import arch_model
import numpy as np
import pandas as pd
import scipy
import statsmodels
from scipy.special import expit, logsumexp
from scipy.stats import chi2, norm, t as student_t
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from PIL import Image

from stage05_forecast import (ROOT, TABLES, SPECS, SEEDS, MARKOV_SETTINGS,
                              audit_data, forecast_block, performance,
                              scale_predictors, fit_markov)

AUDIT_SEED = 9092026
BOOTSTRAP_REPS = 120
BOOTSTRAP_MAXITER = 140


def save(frame, name):
    frame.to_csv(TABLES / f"09_{name}.csv", index=False, float_format="%.12g",
                 date_format="%Y-%m-%d")


def raw_calendar_and_exclusions(d):
    raw = pd.read_csv(ROOT / "data/raw/DCOILBRENTEU.csv", parse_dates=["observation_date"])
    raw = raw.rename(columns={"observation_date": "date", "DCOILBRENTEU": "price"})
    raw.price = pd.to_numeric(raw.price, errors="coerce")
    assert raw.date.is_unique and raw.date.is_monotonic_increasing
    # Retain pre-2000 source rows when auditing the first 2000 return, whose
    # previous observed price is in 1999; annual summaries are merged by year.
    raw = raw.loc[raw.date.le(d.date.max())].copy()
    maximum = pd.read_csv(ROOT / "data/processed/01_brent_max.csv",
                          parse_dates=["date", "previous_observed_date"])
    assert len(maximum) == 6495 and len(d) == 6410
    assert set(d.date).issubset(set(maximum.date))
    rdates = raw.date.to_numpy(dtype="datetime64[ns]")
    blank = raw.loc[raw.price.isna(), "date"].to_numpy(dtype="datetime64[ns]")
    dates = d.date.to_numpy(dtype="datetime64[ns]")
    previous = d.previous_observed_date.to_numpy(dtype="datetime64[ns]")
    # The FRED extract does not prove whether each absent weekday was a market
    # holiday, a publication closure, or a missing source quote.
    missing_rows_between = np.searchsorted(blank, dates, side="left") - np.searchsorted(blank, previous, side="right")
    raw_rows_between = np.searchsorted(rdates, dates, side="left") - np.searchsorted(rdates, previous, side="right")
    business_exposure = np.busday_count(previous.astype("datetime64[D]"), dates.astype("datetime64[D]"))
    assert np.all(business_exposure >= 1)
    extra_unlisted_weekdays = business_exposure - 1 - raw_rows_between
    assert np.all(extra_unlisted_weekdays >= 0)
    ledger = d[["date", "split", "previous_observed_date", "observation_gap_calendar_days"]].copy()
    ledger["business_day_exposure"] = business_exposure
    ledger["intermediate_raw_blank_rows"] = missing_rows_between
    ledger["intermediate_unlisted_weekdays"] = extra_unlisted_weekdays
    ledger["year"] = ledger.date.dt.year
    ledger["long_gap_gt3_calendar_days"] = ledger.observation_gap_calendar_days.gt(3)
    ledger["extra_business_exposure"] = ledger.business_day_exposure.gt(1)
    assert ledger.loc[ledger.split.eq("train"), "long_gap_gt3_calendar_days"].sum() == 167
    assert ledger.loc[ledger.split.eq("train"), "observation_gap_calendar_days"].max() == 11
    save(ledger, "calendar_interval_ledger")
    annual = ledger.groupby(["year", "split"], as_index=False).agg(
        complete_case_returns=("date", "size"),
        gaps_gt3=("long_gap_gt3_calendar_days", "sum"),
        extra_business_exposure=("extra_business_exposure", "sum"),
        blank_source_rows_within_intervals=("intermediate_raw_blank_rows", "sum"),
        unlisted_weekdays_within_intervals=("intermediate_unlisted_weekdays", "sum"),
        max_calendar_gap=("observation_gap_calendar_days", "max"),
        max_business_exposure=("business_day_exposure", "max"))
    raw_annual = raw.assign(year=raw.date.dt.year).groupby("year").agg(
        raw_date_rows=("date", "size"), raw_positive_prices=("price", lambda x: int((x > 0).sum())),
        raw_blank_prices=("price", lambda x: int(x.isna().sum())))
    max_annual = maximum.assign(year=maximum.date.dt.year).groupby("year").size().rename("maximum_usable_returns")
    annual = annual.merge(raw_annual.reset_index(), on="year").merge(max_annual.reset_index(), on="year")
    annual["lost_to_common_sample"] = annual.maximum_usable_returns - annual.complete_case_returns
    save(annual, "annual_calendar_coverage")
    missing = maximum.loc[~maximum.date.isin(d.date)].copy()
    assert missing.split.value_counts().to_dict() == {"train": 51, "validation": 21, "test": 13}
    assert missing.vix_index.isna().all() and missing.vix_lag1_observation.isna().all()
    assert missing.gpr_lag1_calendar_day.notna().all()
    save(missing[["date", "split", "brent_return_pct", "observation_gap_calendar_days",
                  "gpr_lag1_calendar_day", "vix_index", "vix_lag1_observation"]],
         "same_day_vix_exclusions")
    return maximum, ledger, annual


def scenario(fit, est, predictor, transition):
    """Independent analytic logistic contrast and gradient, low/high ordered."""
    assert fit["diag"]["model"] in ("M4", "M5")
    names = fit["result"].model.param_names
    params = np.asarray(fit["params"], float)
    cov = np.asarray(fit["result"].cov_params(), float)
    order = np.asarray(fit["order"], int)
    origin = int(order[0 if transition == "low_to_high" else 1])
    destination = int(order[1])
    assert destination in (0, 1)
    spec = SPECS[fit["diag"]["model"]]
    levels = {key: np.quantile(est[col].to_numpy(float), [.1, .5, .9])
              for key, col in [("gpr", "gpr_lag1_calendar_day"),
                               ("vix", "vix_lag1_observation")]}
    scales = {key: (float(est[col].mean()), float(est[col].std(ddof=0)))
              for key, col in [("gpr", "gpr_lag1_calendar_day"),
                               ("vix", "vix_lag1_observation")]}

    def at(quantile_index):
        z = {key: (levels[key][quantile_index if key == predictor else 1] - scales[key][0]) / scales[key][1]
             for key in levels}
        x = np.array([1.0] + [z[key] for key in spec])
        ix = [names.index(f"p[{origin}->0].tvtp{k}") for k in range(len(x))]
        q_to_zero = float(expit(params[ix] @ x))
        result = q_to_zero if destination == 0 else 1 - q_to_zero
        grad = np.zeros(len(params))
        grad[ix] = (1 if destination == 0 else -1) * q_to_zero * (1-q_to_zero) * x
        assert 0 <= result <= 1
        return result, grad

    p10, g10 = at(0); p90, g90 = at(2)
    delta = p90-p10; grad = g90-g10
    se2 = float(grad @ cov @ grad)
    assert se2 >= 0 and np.isfinite(se2)
    se = math.sqrt(se2)
    return dict(model=fit["diag"]["model"], predictor=predictor, transition=transition,
                p10_value=float(levels[predictor][0]), p90_value=float(levels[predictor][2]),
                probability_p10=p10, probability_p90=p90, difference=delta,
                delta_se=se, ci95_low=delta-1.96*se, ci95_high=delta+1.96*se)


def summarize_fit(case, fits, est):
    params, regimes, effects, diag = [], [], [], []
    for model, fit in fits.items():
        diagnostics = dict(fit["diag"]); diagnostics["case"] = case; diag.append(diagnostics)
        if model in ("B1", "B1-t"):
            for key, value in fit["params"].items():
                params.append(dict(case=case, model=model, parameter=key, estimate=value))
            continue
        for name, value, se in zip(fit["result"].model.param_names, fit["params"], fit["result"].bse):
            params.append(dict(case=case, model=model, parameter=name, estimate=value,
                               std_error=se if np.isfinite(se) else np.nan))
        for state in (0, 1):
            regimes.append(dict(case=case, model=model, state=("low", "high")[state],
                                original_state=int(fit["order"][state]),
                                mean_pct=float(fit["means"][state]),
                                sd_pct=float(np.sqrt(fit["variances"][state]))))
        for predictor in SPECS[model]:
            for transition in ("low_to_high", "high_to_high"):
                effects.append(dict(case=case, **scenario(fit, est, predictor, transition)))
    return [pd.DataFrame(v) for v in (params, regimes, effects, diag)]


def gap_sensitivity(d, ledger):
    work = d.copy()
    exposure = ledger.business_day_exposure.to_numpy(float)
    assert (exposure >= 1).all()
    original_return = work.brent_return_pct.to_numpy(float)
    work.brent_return_pct = original_return / np.sqrt(exposure)
    train = work.loc[work.split.eq("train")].reset_index(drop=True)
    val = work.loc[work.split.eq("validation")].reset_index(drop=True)
    model_set = ("M1", "M4", "M5")
    fc, scales, attempts, diagnostics, parameters, fits = forecast_block(
        train, val, "gap_exposure_validation", model_set=model_set, return_fits=True)
    # Stage 05 enforces alpha+beta<1 and aborts if the exposure-adjusted GARCH
    # estimate reaches that boundary. Keep the estimate and its diagnostic flag;
    # the finite-horizon conditional-variance recursion remains computable.
    extra = []
    y_train = train.brent_return_pct.to_numpy(float)
    y_val = val.brent_return_pct.to_numpy(float)
    for name, dist in (("B1", "normal"), ("B1-t", "t")):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = arch_model(y_train, mean="Constant", vol="GARCH", p=1, o=0, q=1,
                             dist=dist, rescale=False).fit(disp="off", show_warning=True,
                                                             options={"maxiter": 1000})
        p = res.params
        persistence = float(p["alpha[1]"]+p["beta[1]"])
        valid = bool(res.convergence_flag == 0 and np.isfinite(p).all() and p["omega"] > 0 and
                     p["alpha[1]"] >= 0 and p["beta[1]"] >= 0 and persistence <= 1+1e-7 and
                     (dist == "normal" or p["nu"] > 2))
        attempts.append(dict(period="gap_exposure_validation", model=name, seed=np.nan,
                             valid=valid, log_likelihood=res.loglikelihood,
                             warning_count=len(caught),
                             warnings=" | ".join(f"{w.category.__name__}: {w.message}" for w in caught),
                             error="" if valid else "Invalid or explosive GARCH estimate"))
        if not valid:
            raise RuntimeError(f"Gap sensitivity {name}: unusable fitted recursion; see attempt ledger")
        hprev=float(np.asarray(res.conditional_volatility)[-1]**2)
        yprev=float(y_train[-1])
        for i, (date, realized) in enumerate(zip(val.date, y_val)):
            h=float(p["omega"]+p["alpha[1]"]*(yprev-p["mu"])**2+p["beta[1]"]*hprev)
            assert h>0 and np.isfinite(h)
            if name == "B1":
                ld=float(norm.logpdf(realized, p["mu"], np.sqrt(h)))
            else:
                nu=float(p["nu"])
                z=(realized-p["mu"])/np.sqrt(h)*np.sqrt(nu/(nu-2))
                ld=float(student_t.logpdf(z,df=nu)+.5*np.log(nu/(nu-2))-.5*np.log(h))
            second=float(h+p["mu"]**2)
            extra.append(dict(period="gap_exposure_validation",date=date,
                              forecast_information_through=train.date.iloc[-1] if i==0 else val.date.iloc[i-1],
                              observed_return_pct=realized,observed_squared_return_pct2=realized**2,
                              observation_gap_calendar_days=int(val.observation_gap_calendar_days.iloc[i]),
                              model=name,predicted_mean_pct=float(p["mu"]),predicted_variance_pct2=h,
                              predicted_second_moment_pct2=second,predicted_volatility_pct=np.sqrt(h),
                              predicted_low_probability=np.nan,predicted_high_probability=np.nan,
                              log_predictive_density=ld,qlike=np.log(second)+realized**2/second,
                              squared_error=(second-realized**2)**2,
                              absolute_error=abs(second-realized**2),negative_log_score=-ld))
            hprev=h;yprev=float(realized)
        fits[name]=dict(params=p,diag=dict(period="gap_exposure_validation",model=name,n=len(y_train),
                                           llf=res.loglikelihood,converged=True,successful_starts=1,
                                           ll_spread=0,seed=np.nan,missing_se=int(np.isnan(res.std_err).sum()),
                                           min_variance=float(np.min(np.asarray(res.conditional_volatility)**2)),
                                           min_transition=np.nan,alpha_plus_beta=persistence,
                                           near_integrated=persistence>=.995,
                                           boundary_or_nonstationary=persistence>=1))
        diagnostics.append(fits[name]["diag"])
        parameters.extend(dict(period="gap_exposure_validation",model=name,parameter=k,estimate=float(v))
                          for k,v in p.items())
    fc=pd.concat([fc,pd.DataFrame(extra)],ignore_index=True)
    exp_map = dict(zip(ledger.date, ledger.business_day_exposure))
    raw_map = dict(zip(d.date, d.brent_return_pct))
    e = fc.date.map(exp_map).to_numpy(float)
    fc["business_day_exposure"] = e
    fc.observed_return_pct = fc.date.map(raw_map).to_numpy(float)
    fc.observed_squared_return_pct2 = fc.observed_return_pct ** 2
    fc.predicted_mean_pct *= np.sqrt(e)
    fc.predicted_variance_pct2 *= e
    fc.predicted_second_moment_pct2 *= e
    fc.predicted_volatility_pct *= np.sqrt(e)
    fc.log_predictive_density -= .5*np.log(e)
    fc.negative_log_score = -fc.log_predictive_density
    fc.qlike = np.log(fc.predicted_second_moment_pct2) + fc.observed_squared_return_pct2/fc.predicted_second_moment_pct2
    fc.squared_error = (fc.predicted_second_moment_pct2-fc.observed_squared_return_pct2)**2
    fc.absolute_error = abs(fc.predicted_second_moment_pct2-fc.observed_squared_return_pct2)
    assert np.isfinite(fc.qlike).all() and fc.groupby("model").date.nunique().eq(999).all()
    save(fc, "gap_exposure_validation_forecasts")
    save(performance(fc, ["period", "model"]), "gap_exposure_validation_performance")
    return summarize_fit("gap_exposure", fits, train), attempts, diagnostics, parameters


def vix_exclusion_sensitivity(maximum, d):
    raw = pd.read_csv(ROOT / "data/raw/VIXCLS.csv", parse_dates=["observation_date"])
    raw = raw.rename(columns={"observation_date": "vix_source_date", "VIXCLS": "vix_quote"})
    raw.vix_quote = pd.to_numeric(raw.vix_quote, errors="coerce")
    raw = raw.dropna().sort_values("vix_source_date")
    alt = pd.merge_asof(maximum.sort_values("date"), raw,
                        left_on="date", right_on="vix_source_date",
                        direction="backward", allow_exact_matches=False)
    assert len(alt) == 6495 and alt.vix_quote.notna().all()
    assert (alt.vix_source_date < alt.date).all()
    overlap = alt.loc[alt.date.isin(d.date)].merge(
        d[["date", "vix_lag1_observation", "gpr_lag1_calendar_day"]], on="date", suffixes=("_alt", "_old"))
    assert np.allclose(overlap.vix_quote, overlap.vix_lag1_observation_old, atol=1e-10)
    assert np.allclose(overlap.gpr_lag1_calendar_day_alt, overlap.gpr_lag1_calendar_day_old, atol=1e-10)
    alt["vix_lag1_observation"] = alt.vix_quote
    alt["vix_lag_source_date"] = alt.vix_source_date
    added = alt.loc[~alt.date.isin(d.date), ["date", "split", "vix_quote", "vix_source_date",
                                            "observation_gap_calendar_days"]].copy()
    assert added.split.value_counts().to_dict() == {"train": 51, "validation": 21, "test": 13}
    save(added, "recovered_vix_lag_dates")
    train = alt.loc[alt.split.eq("train")].reset_index(drop=True)
    val = alt.loc[alt.split.eq("validation")].reset_index(drop=True)
    fc, scales, attempts, diagnostics, parameters, fits = forecast_block(
        train, val, "vix_lag_only_validation", model_set=("M1", "M4", "M5"), return_fits=True)
    assert fc.groupby("model").date.nunique().eq(1020).all()
    save(fc, "vix_lag_only_validation_forecasts")
    full = performance(fc, ["period", "model"])
    matched = performance(fc.loc[fc.date.isin(d.loc[d.split.eq("validation"), "date"])], ["period", "model"])
    full["evaluation_set"] = "all_1020_dates"
    matched["evaluation_set"] = "original_999_dates"
    save(pd.concat([full, matched], ignore_index=True), "vix_lag_only_validation_performance")
    return summarize_fit("vix_lag_only", fits, train), attempts, diagnostics, parameters


def gpr_delay_sensitivity(d):
    raw = pd.read_excel(ROOT / "data/raw/data_gpr_daily_recent.xls", engine="xlrd")
    lookup = pd.Series(pd.to_numeric(raw.GPRD, errors="coerce").to_numpy(),
                       index=pd.to_datetime(raw.DAY.astype(str), format="%Y%m%d"))
    work = d.copy()
    work["gpr_lag1_calendar_day"] = (work.date-pd.Timedelta(days=7)).map(lookup)
    assert work.gpr_lag1_calendar_day.notna().all()
    train = work.loc[work.split.eq("train")].reset_index(drop=True)
    val = work.loc[work.split.eq("validation")].reset_index(drop=True)
    fc, scales, attempts, diagnostics, parameters, fits = forecast_block(
        train, val, "gpr_7_calendar_day_proxy_validation", model_set=("M5",), return_fits=True)
    save(fc, "gpr_7day_proxy_validation_forecasts")
    save(performance(fc, ["period", "model"]), "gpr_7day_proxy_validation_performance")
    return summarize_fit("gpr_7day_proxy", fits, train), attempts, diagnostics, parameters


def original_fit_and_scenarios(d):
    train = d.loc[d.split.eq("train")].reset_index(drop=True)
    val = d.loc[d.split.eq("validation")].reset_index(drop=True)
    columns, _ = scale_predictors(train, val, "original_audit")
    attempts = []
    fits = {name: fit_markov(name, train.brent_return_pct.to_numpy(float), columns,
                             "original_audit", attempts) for name in ("M4", "M5")}
    saved = pd.read_csv(TABLES / "04_model_comparison.csv").set_index("model")
    for name, fit in fits.items():
        assert np.isclose(fit["result"].llf, saved.loc[name, "log_likelihood"], atol=.002)
        assert len(fit["params"]) == int(saved.loc[name, "parameters"])
        assert np.array_equal(fit["order"], [0, 1])
    lr = 2*(fits["M5"]["result"].llf-fits["M4"]["result"].llf)
    assert np.isclose(lr, 8.812976, atol=.01)
    effects = []
    for predictor, transition in (("gpr", "low_to_high"), ("vix", "high_to_high")):
        effects.append(scenario(fits["M5"], train, predictor, transition))
    effects = pd.DataFrame(effects)
    old = pd.read_csv(TABLES / "06_transition_scenarios.csv")
    for i, row in enumerate(effects.itertuples()):
        ref = old.loc[old.case.eq("brent_original") & old.model.eq("M5") &
                      old.predictor.eq(row.predictor) & old.transition.eq(row.transition)]
        assert len(ref) == 1
        effects.loc[i, "saved_difference"] = ref.iloc[0].p90_minus_p10
        effects.loc[i, "saved_ci95_low"] = ref.iloc[0].delta_ci95_low
        effects.loc[i, "saved_ci95_high"] = ref.iloc[0].delta_ci95_high
        effects.loc[i, "absolute_difference_discrepancy"] = abs(row.difference-ref.iloc[0].p90_minus_p10)
        effects.loc[i, "max_ci_bound_discrepancy"] = max(abs(row.ci95_low-ref.iloc[0].delta_ci95_low),
                                                        abs(row.ci95_high-ref.iloc[0].delta_ci95_high))
        assert effects.loc[i, "absolute_difference_discrepancy"] < .005
        assert effects.loc[i, "max_ci_bound_discrepancy"] < .02
    save(effects, "original_transition_contrast_verification")
    save(pd.DataFrame(attempts), "original_markov_fit_attempts")
    result = pd.DataFrame([dict(n=len(train), ll_m4=fits["M4"]["result"].llf,
                                ll_m5=fits["M5"]["result"].llf,
                                parameters_m4=8, parameters_m5=10,
                                lr=lr, nominal_chi2_df=2, nominal_chi2_p=chi2.sf(lr, 2),
                                inferential_status="nominal only; parametric bootstrap reported separately")])
    save(result, "m4_m5_likelihood_ratio")
    return train, columns, fits, effects, result


def _simulate_markov(fit, transitions, rng):
    """Conditional on the observed exogenous path; initial state stationary at P0."""
    P = transitions
    assert P.shape[1:] == (2, 2) and np.allclose(P.sum(axis=2), 1)
    n = len(P)
    states = np.empty(n, int)
    p01, p10 = P[0, 0, 1], P[0, 1, 0]
    pi_high = p01 / (p01+p10)
    states[0] = rng.choice(2, p=[1-pi_high, pi_high])
    for t in range(1, n):
        states[t] = rng.choice(2, p=P[t, states[t-1]])
    y = rng.normal(fit["means"][states], np.sqrt(fit["variances"][states]))
    return y


def _sim_transition_path(fit, columns, n):
    spec = SPECS[fit["diag"]["model"]]
    X = np.column_stack([np.ones(n)] + [columns[key][0] for key in spec])
    raw = np.asarray(fit["model"].regime_transition_matrix(fit["params"], exog_tvtp=X))
    order = fit["order"]
    P = raw.transpose(2, 1, 0)[:, order][:, :, order]
    assert P.shape == (n, 2, 2) and np.allclose(P.sum(axis=2), 1, atol=1e-9)
    return P


def _embedded_null(m4):
    p = np.asarray(m4["params"])
    names = list(m4["result"].model.param_names)
    assert names[:4] == ["p[0->0].tvtp0", "p[1->0].tvtp0", "p[0->0].tvtp1", "p[1->0].tvtp1"]
    return np.r_[p[:2], 0., 0., p[2:4], p[4:]]


def _refit_sim(name, y, columns, starts, rep, experiment):
    spec = SPECS[name]
    X = np.column_stack([np.ones(len(y))] + [columns[key][0] for key in spec])
    mod = MarkovRegression(y, k_regimes=2, trend="c", switching_trend=True,
                           switching_variance=True, exog_tvtp=X)
    attempts = []
    winners = []
    for start_id, start in enumerate(starts):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                res = mod.fit(start_params=start, transformed=True, em_iter=0,
                              search_reps=0, maxiter=BOOTSTRAP_MAXITER, disp=False,
                              cov_type="none")
            p = pd.Series(res.params, index=res.model.param_names)
            valid = bool(res.mle_retvals.get("converged", False) and np.isfinite(res.llf) and
                         np.isfinite(res.params).all() and
                         all(p[f"sigma2[{j}]"] > 0 for j in (0, 1)))
            if valid:
                raw = np.asarray(res.regime_transition)
                valid = (np.isfinite(raw).all() and raw.min() >= -1e-10 and
                         raw.max() <= 1+1e-10 and np.allclose(raw.sum(axis=0), 1, atol=1e-8))
            attempts.append(dict(experiment=experiment, replication=rep, model=name, start=start_id,
                                 converged=valid, llf=res.llf,
                                 warnings=" | ".join(f"{w.category.__name__}: {w.message}" for w in caught),
                                 error=""))
            if valid:
                winners.append((res.llf, res))
        except Exception as exc:
            attempts.append(dict(experiment=experiment, replication=rep, model=name, start=start_id,
                                 converged=False, llf=np.nan, warnings="", error=repr(exc)))
    if not winners:
        return None, attempts
    res = max(winners, key=lambda item: item[0])[1]
    return res, attempts


def _effect_from_result(res, est, predictor, transition):
    names = list(res.model.param_names)
    p = np.asarray(res.params)
    order = np.argsort([p[names.index(f"sigma2[{j}]")] for j in (0, 1)])
    origin = int(order[0 if transition == "low_to_high" else 1])
    destination = int(order[1])
    spec = ("gpr", "vix")
    levels = {key: np.quantile(est[col], [.1, .5, .9]) for key, col in
              [("gpr", "gpr_lag1_calendar_day"), ("vix", "vix_lag1_observation")]}
    scales = {key: (est[col].mean(), est[col].std(ddof=0)) for key, col in
              [("gpr", "gpr_lag1_calendar_day"), ("vix", "vix_lag1_observation")]}
    def prob(qindex):
        x = np.array([1.] + [(levels[key][qindex if key == predictor else 1]-scales[key][0])/scales[key][1]
                              for key in spec])
        ix = [names.index(f"p[{origin}->0].tvtp{k}") for k in range(3)]
        qzero = expit(p[ix] @ x)
        return qzero if destination == 0 else 1-qzero
    return float(prob(2)-prob(0))


def parametric_bootstraps(train, columns, fits, original_effects, original_lr):
    """Re-estimate both nested models under M4 and M5 under M5 simulations.

    Conditions on historical predictor paths. Optimization failures are separate
    from null-tail evidence. Intervals are percentile, not bias corrected.
    """
    n = len(train)
    rng = np.random.default_rng(AUDIT_SEED)
    m4, m5 = fits["M4"], fits["M5"]
    P4 = _sim_transition_path(m4, columns, n)
    P5 = _sim_transition_path(m5, columns, n)
    null_start = _embedded_null(m4)
    rows, attempts = [], []
    for rep in range(BOOTSTRAP_REPS):
        y = _simulate_markov(m4, P4, rng)
        r4, a4 = _refit_sim("M4", y, columns, [m4["params"], m4["params"] +
                                np.r_[rng.normal(0,.08,4), rng.normal(0,.08,2), rng.normal(0,.15,2)]],
                            rep, "lr_under_m4")
        r5, a5 = _refit_sim("M5", y, columns, [null_start, m5["params"]], rep, "lr_under_m4")
        attempts.extend(a4+a5)
        if r4 is None or r5 is None:
            rows.append(dict(experiment="lr_under_m4", replication=rep, valid=False,
                             lr=np.nan, ll_m4=np.nan if r4 is None else r4.llf,
                             ll_m5=np.nan if r5 is None else r5.llf))
        else:
            lr = 2*(r5.llf-r4.llf)
            # Negative LR beyond optimizer tolerance indicates unreliable fits.
            valid = bool(lr >= -.02)
            rows.append(dict(experiment="lr_under_m4", replication=rep, valid=valid,
                             lr=lr, ll_m4=r4.llf, ll_m5=r5.llf))
    for rep in range(BOOTSTRAP_REPS):
        y = _simulate_markov(m5, P5, rng)
        jitter = np.asarray(m5["params"]).copy()
        jitter[:6] += rng.normal(0, .07, 6)
        r5, a5 = _refit_sim("M5", y, columns, [m5["params"], jitter], rep, "effect_under_m5")
        attempts.extend(a5)
        if r5 is None:
            rows.append(dict(experiment="effect_under_m5", replication=rep, valid=False,
                             gpr_low_high=np.nan, vix_high_high=np.nan))
        else:
            g = _effect_from_result(r5, train, "gpr", "low_to_high")
            v = _effect_from_result(r5, train, "vix", "high_to_high")
            rows.append(dict(experiment="effect_under_m5", replication=rep,
                             valid=bool(np.isfinite([g, v]).all()),
                             gpr_low_high=g, vix_high_high=v))
    sims = pd.DataFrame(rows); att = pd.DataFrame(attempts)
    save(sims, "parametric_bootstrap_replications")
    save(att, "parametric_bootstrap_fit_attempts")
    lr_sim = sims.loc[sims.experiment.eq("lr_under_m4") & sims.valid, "lr"].to_numpy(float)
    n_lr = len(lr_sim)
    lr_observed = float(original_lr.iloc[0].lr)
    tail = int((lr_sim >= lr_observed).sum())
    p = (tail+1)/(n_lr+1) if n_lr else np.nan
    mc_se = np.sqrt(p*(1-p)/(n_lr+1)) if n_lr else np.nan
    summary = [dict(experiment="lr_under_m4", attempted=BOOTSTRAP_REPS,
                    successful=n_lr, failures=BOOTSTRAP_REPS-n_lr,
                    observed=lr_observed, bootstrap_p=p, monte_carlo_se=mc_se,
                    bootstrap_ci95_low=np.nan, bootstrap_ci95_high=np.nan,
                    tail_count=tail)]
    eff = sims.loc[sims.experiment.eq("effect_under_m5") & sims.valid]
    for col, predictor, transition in (("gpr_low_high", "gpr", "low_to_high"),
                                       ("vix_high_high", "vix", "high_to_high")):
        vals = eff[col].to_numpy(float)
        observed = original_effects.loc[original_effects.predictor.eq(predictor) &
                                        original_effects.transition.eq(transition), "difference"].iloc[0]
        lo, hi = np.quantile(vals, [.025, .975]) if len(vals) else (np.nan, np.nan)
        summary.append(dict(experiment=col, attempted=BOOTSTRAP_REPS, successful=len(vals),
                            failures=BOOTSTRAP_REPS-len(vals), observed=observed,
                            bootstrap_p=np.nan, monte_carlo_se=np.nan,
                            bootstrap_ci95_low=lo, bootstrap_ci95_high=hi, tail_count=np.nan))
    result = pd.DataFrame(summary)
    save(result, "parametric_bootstrap_summary")
    return result


def _saved_transition(model, p, scales, date, data):
    if model == "M1":
        q0 = float(p["p[0->0]"]); q1 = float(p["p[1->0]"])
    else:
        spec = SPECS[model]
        row = data.loc[date]
        x = [1.]
        for key in spec:
            col = "gpr_lag1_calendar_day" if key == "gpr" else "vix_lag1_observation"
            mu, sd = scales[key]
            x.append((row[col]-mu)/sd)
        q0, q1 = [expit(sum(p[f"p[{j}->0].tvtp{k}"]*x[k] for k in range(len(x))))
                  for j in (0, 1)]
    raw = np.array([[q0, 1-q0], [q1, 1-q1]])
    order = np.argsort([p["sigma2[0]"], p["sigma2[1]"]])
    out = raw[np.ix_(order, order)]
    assert np.isfinite(out).all() and np.allclose(out.sum(axis=1), 1, atol=1e-8)
    return out


def verify_forecasts(d):
    saved = pd.read_csv(TABLES / "05_forecasts.csv", parse_dates=["date", "forecast_information_through"])
    pfile = pd.read_csv(TABLES / "05_fitted_parameters.csv")
    sfile = pd.read_csv(TABLES / "05_scaling.csv")
    original_perf = pd.read_csv(TABLES / "05_performance.csv")
    data = d.set_index("date")
    assert saved.groupby(["period", "model"]).size().eq([1712]*7+[999]*7).all() or \
        sorted(saved.groupby(["period", "model"]).size().unique()) == [999, 1712]
    assert saved.groupby(["period", "date"]).model.nunique().eq(7).all()
    assert (saved.forecast_information_through < saved.date).all()
    assert np.isfinite(saved[["predicted_variance_pct2", "predicted_second_moment_pct2",
                              "log_predictive_density", "qlike"]]).all().all()
    assert saved.predicted_variance_pct2.gt(0).all() and saved.predicted_second_moment_pct2.gt(0).all()
    assert np.allclose(saved.predicted_second_moment_pct2,
                       saved.predicted_variance_pct2+saved.predicted_mean_pct**2, rtol=1e-9, atol=1e-8)
    assert np.allclose(saved.qlike, np.log(saved.predicted_second_moment_pct2)+
                       saved.observed_return_pct**2/saved.predicted_second_moment_pct2, rtol=1e-9, atol=1e-8)
    assert np.allclose(saved.negative_log_score, -saved.log_predictive_density, rtol=1e-9, atol=1e-8)
    assert np.allclose(saved.squared_error,
                       (saved.predicted_second_moment_pct2-saved.observed_return_pct**2)**2,
                       rtol=1e-9, atol=1e-7)
    rows=[]
    for period in ("validation", "test"):
        fitting = d.loc[d.split.eq("train") if period == "validation" else d.split.isin(["train", "validation"])]
        scales = {}
        for key, col in (("gpr", "gpr_lag1_calendar_day"), ("vix", "vix_lag1_observation")):
            r = sfile.loc[sfile.period.eq(period)&sfile.predictor.eq(key)].iloc[0]
            assert np.isclose(r["mean"], fitting[col].mean(), atol=1e-8)
            assert np.isclose(r.sd, fitting[col].std(ddof=0), atol=1e-8)
            scales[key] = (float(r["mean"]), float(r.sd))
        for model in ("B0", "B1", "B1-t", "M1", "M3", "M4", "M5"):
            group = saved.loc[saved.period.eq(period)&saved.model.eq(model)].sort_values("date").reset_index(drop=True)
            p = pfile.loc[pfile.period.eq(period)&pfile.model.eq(model)].set_index("parameter").estimate.to_dict()
            assert group.date.is_unique and group.date.is_monotonic_increasing
            assert group.forecast_information_through.iloc[0] == fitting.date.max()
            assert group.forecast_information_through.iloc[1:].reset_index(drop=True).equals(
                group.date.iloc[:-1].reset_index(drop=True))
            assert np.allclose(group.observed_return_pct, group.date.map(data.brent_return_pct))
            y = group.observed_return_pct.to_numpy(float)
            mu = group.predicted_mean_pct.to_numpy(float)
            h = group.predicted_variance_pct2.to_numpy(float)
            if model == "B1-t":
                nu = p["nu"]
                z = (y-mu)/np.sqrt(h)*np.sqrt(nu/(nu-2))
                ld = student_t.logpdf(z, df=nu)+.5*np.log(nu/(nu-2))-.5*np.log(h)
            elif model.startswith("M"):
                order = np.argsort([p["sigma2[0]"], p["sigma2[1]"]])
                means = np.array([p[f"const[{j}]"] for j in order])
                variances = np.array([p[f"sigma2[{j}]"] for j in order])
                q = group[["predicted_low_probability", "predicted_high_probability"]].to_numpy(float)
                assert np.isfinite(q).all() and np.all((q>=0)&(q<=1)) and np.allclose(q.sum(axis=1),1,atol=1e-8)
                check_mu = q@means
                check_second = q@(variances+means**2)
                assert np.allclose(group.predicted_mean_pct, check_mu, atol=1e-7)
                assert np.allclose(group.predicted_second_moment_pct2, check_second, atol=1e-7)
                log_emission = norm.logpdf(y[:,None], means, np.sqrt(variances))
                ld = logsumexp(np.log(q)+log_emission, axis=1)
                for i in range(1, len(group)):
                    posterior = np.exp(np.log(q[i-1])+log_emission[i-1]-ld[i-1])
                    P = _saved_transition(model, p, scales, group.date.iloc[i], data)
                    assert np.allclose(posterior@P, q[i], atol=2e-6)
            else:
                ld = norm.logpdf(y, mu, np.sqrt(h))
            assert np.allclose(ld, group.log_predictive_density, atol=2e-7)
            if model in ("B1", "B1-t"):
                expected = p["omega"] + p["alpha[1]"]*(y[:-1]-p["mu"])**2 + p["beta[1]"]*h[:-1]
                assert np.allclose(expected, h[1:], rtol=1e-7, atol=1e-7)
            perf = original_perf.loc[original_perf.period.eq(period)&original_perf.model.eq(model)].iloc[0]
            assert np.isclose(group.qlike.mean(), perf.qlike, atol=1e-8)
            assert np.isclose(group.negative_log_score.mean(), perf.log_score, atol=1e-8)
            rows.append(dict(period=period, model=model, n=len(group),
                             max_density_absolute_error=float(np.max(abs(ld-group.log_predictive_density))),
                             qlike_recomputed=float(group.qlike.mean()),
                             target_return_leakage_check="passed: prior filtered posterior predicts next state",
                             scaling_check="passed", matched_dates="passed"))
    result = pd.DataFrame(rows)
    save(result, "forecast_consistency_checks")
    return saved, result


def verify_loss_bootstraps(saved):
    original = pd.read_csv(TABLES / "05_paired_block_bootstrap.csv")
    rows=[]
    for period in ("validation", "test"):
        wide = saved.loc[saved.period.eq(period)].pivot(index="date", columns="model", values="qlike")
        n=len(wide); B=2000; block=10; seed=7205
        rng=np.random.default_rng(seed)
        starts=rng.integers(0,n,size=(B,int(np.ceil(n/block))))
        ix=((starts[:,:,None]+np.arange(block))%n).reshape(B,-1)[:,:n]
        for old in original.loc[original.period.eq(period)].itertuples():
            d=(wide[old.left]-wide[old.right]).to_numpy(float)
            observed=float(d.mean()); boot=d[ix].mean(axis=1); centered=boot-observed
            qlo,qhi=np.quantile(boot,[.025,.975]); lo,hi=2*observed-qhi,2*observed-qlo
            abs_p=(1+int((abs(centered)>=abs(observed)).sum()))/(B+1)
            lower=(1+int((centered<=observed).sum()))/(B+1)
            upper=(1+int((centered>=observed).sum()))/(B+1)
            equal=min(1.,2*min(lower,upper))
            assert np.isclose(observed,old.mean_qlike_left_minus_right,atol=1e-8)
            assert np.isclose([lo,hi],[old.basic_ci95_low,old.basic_ci95_high],atol=1e-8).all()
            assert np.isclose(abs_p,old.centered_block_bootstrap_p,atol=1e-12)
            rows.append(dict(period=period,left=old.left,right=old.right,n=n,
                             mean_difference=observed,basic_ci95_low=lo,basic_ci95_high=hi,
                             original_absolute_tail_p=abs_p,equal_tail_p=equal,
                             interval_excludes_zero=bool(lo>0 or hi<0)))
    wti=pd.read_csv(TABLES/"06_wti_forecasts.csv",parse_dates=["date"])
    w=wti.loc[wti.period.eq("wti_test") & wti.model.isin(["M1","M4"])].pivot(index="date",columns="model",values="qlike")
    d=(w.M1-w.M4).to_numpy(float);n=len(d);B=2000;block=10;seed=7205
    rng=np.random.default_rng(seed); starts=rng.integers(0,n,size=(B,int(np.ceil(n/block))))
    ix=((starts[:,:,None]+np.arange(block))%n).reshape(B,-1)[:,:n]
    observed=float(d.mean());boot=d[ix].mean(axis=1);centered=boot-observed
    qlo,qhi=np.quantile(boot,[.025,.975]);lo,hi=2*observed-qhi,2*observed-qlo
    equal=min(1.,2*min((1+sum(centered<=observed))/(B+1),(1+sum(centered>=observed))/(B+1)))
    correction=pd.read_csv(TABLES/"07_stage06_bootstrap_correction.csv").iloc[0]
    assert np.isclose(equal,correction.corrected_equal_tail_p_consistent_with_basic_ci,atol=1e-12)
    assert np.isclose([lo,hi],[correction.basic_ci95_low,correction.basic_ci95_high],atol=1e-12).all()
    rows.append(dict(period="wti_test",left="M1",right="M4",n=n,mean_difference=observed,
                     basic_ci95_low=lo,basic_ci95_high=hi,original_absolute_tail_p=correction.original_absolute_tail_p,
                     equal_tail_p=equal,interval_excludes_zero=bool(lo>0 or hi<0)))
    result=pd.DataFrame(rows);save(result,"paired_bootstrap_verification")
    return result


def reproducibility_and_timing():
    files = [ROOT/"data/raw/DCOILBRENTEU.csv", ROOT/"data/raw/DCOILWTICO.csv",
             ROOT/"data/raw/VIXCLS.csv", ROOT/"data/raw/data_gpr_daily_recent.xls",
             ROOT/"data/processed/01_brent_max.csv", ROOT/"data/processed/01_brent_complete_case.csv",
             ROOT/"data/processed/01_wti_robustness.csv",
             ROOT/"results/tables/04_model_comparison.csv", ROOT/"results/tables/05_forecasts.csv",
             ROOT/"results/tables/05_performance.csv", ROOT/"results/tables/07_stage06_bootstrap_correction.csv"]
    hashes = []
    for path in files:
        assert path.exists()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes.append(dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size, sha256=digest))
    save(pd.DataFrame(hashes), "input_sha256")
    notebook_rows=[]
    for j in range(1, 8):
        paths=list((ROOT/"notebooks").glob(f"{j:02d}_*.ipynb"))
        assert len(paths)==1
        nb=json.loads(paths[0].read_text(encoding="utf-8"))
        code=[cell for cell in nb["cells"] if cell["cell_type"]=="code"]
        executed=sum(cell.get("execution_count") is not None for cell in code)
        errors=sum(out.get("output_type")=="error" for cell in code for out in cell.get("outputs",[]))
        notebook_rows.append(dict(notebook=paths[0].name,code_cells=len(code),
                                  executed_code_cells=executed,saved_error_outputs=errors,
                                  all_code_cells_executed=executed==len(code)))
    save(pd.DataFrame(notebook_rows),"notebook_execution_check")
    figures=[]
    for path in sorted((ROOT/"results/figures").glob("0[1-7]_*.png")):
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            width,height=im.size
        assert width>0 and height>0
        figures.append(dict(path=path.relative_to(ROOT).as_posix(),width_px=width,
                            height_px=height,bytes=path.stat().st_size,valid_png=True))
    assert figures
    save(pd.DataFrame(figures),"figure_integrity")
    links=[]
    for path in sorted((ROOT/"thesis").glob("08_*.md")):
        for target in re.findall(r"\]\(([^)]+)\)",path.read_text(encoding="utf-8")):
            if target.startswith(("http:","https:","#")):
                continue
            exists=(path.parent/target.split("#")[0]).resolve().exists()
            links.append(dict(chapter=path.name,target=target,exists=exists))
    assert all(row["exists"] for row in links)
    save(pd.DataFrame(links),"stage08_local_reference_check")
    checklist=[
        ("Source and processed data", "passed", "Input SHA-256 ledger and Stage 01/05 return assertions"),
        ("Missing-data and return-interval rules", "passed with qualification", "Raw weekday blanks, previous-observed-price return rule and 09 interval ledger; holiday cause unverified"),
        ("Sample chronology and common dates", "passed", "Stage 05 and 09 forecast assertions"),
        ("Model parameter order and transition orientation", "passed", "Analytic reconstruction and filter recursion"),
        ("Filtered forecast states", "passed", "Independent posterior/next-prior recurrence"),
        ("Random seeds and settings", "passed", "Stage 05 metadata and Stage 09 constants"),
        ("Warnings and failed fits", "reported", "Stage 09 attempt ledgers; legacy Stage 04/05 ledgers retained"),
        ("Software versions", "passed", "Stage 09 environment manifest"),
        ("Saved figures and manuscript table/figure links", "passed", "09 figure integrity and Stage 08 local-link ledgers"),
        ("Relative project paths", "passed", "Code resolves from Path(__file__)"),
        ("Historical GPR point-in-time vintages 2000–2019", "unavailable", "Official archive listing starts 2022-02-24"),
    ]
    save(pd.DataFrame(checklist,columns=["item","status","evidence"]),"reproducibility_checklist")
    versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
                  scipy=scipy.__version__,statsmodels=statsmodels.__version__,arch=arch.__version__,
                  seeds_stage05=list(SEEDS),markov_settings_stage05=MARKOV_SETTINGS,
                  audit_seed=AUDIT_SEED,parametric_bootstrap_attempts_per_experiment=BOOTSTRAP_REPS,
                  parametric_bootstrap_maxiter=BOOTSTRAP_MAXITER)
    (TABLES/"09_environment.json").write_text(json.dumps(versions,indent=2),encoding="utf-8")
    timing = """# GPR publication and vintage assessment

The [official GPR page](https://www.matteoiacoviello.com/gpr.htm) says the recent daily index is updated every Monday and that latest values are preliminary and subject to revision. The page links to its [GitHub archive](https://github.com/iacoviel/iacoviel.github.io/tree/master/gpr_archive_files). An 8 November 2026 inspection of the [archive listing API](https://api.github.com/repos/iacoviel/iacoviel.github.io/contents/gpr_archive_files) found its earliest named daily recent-GPR vintage on **2022-02-24**. Filenames alone do not authenticate precise release timestamps. No contemporaneous 2000–2015 training or 2016–2019 validation vintages were found in this official archive. The present-day source also reflects methodological and historical revisions; earlier source-matched values cannot simply be substituted for the established index.

The established previous-calendar-day GPR value is a reference-date lag, not a verified publication lag. Stage 09 therefore treats the original GPR exercise as historical association and revised-data pseudo-out-of-sample prediction. A seven-calendar-day lag of the same revised series is an **approximate publication-delay sensitivity**; it does not remove retrospective revisions or establish genuine real-time performance. M4 is the unchanged VIX comparator on the same validation dates. The previously examined 2020–2026 period is not used to choose or test the new convention.
"""
    (TABLES/"09_gpr_publication_timing.md").write_text(timing,encoding="utf-8")


def thesis_change_table():
    rows=[
        ("Irregular Brent intervals", "Major", "Explain source blanks and unlisted weekdays; distinguish observation-step GARCH/Markov clock from calendar time; report exposure sensitivity", "Open: add audit findings to methodology and discussion"),
        ("Same-day VIX common-case restriction", "Moderate", "State 85 usable returns were conservatively excluded and show lag-only matched-date sensitivity", "Open: annotate sample construction"),
        ("Historical GPR availability", "Major", "Replace real-time predictive language with revised-data pseudo-out-of-sample; state vintages unavailable for 2000–2019", "Open: qualify abstract, results and conclusions"),
        ("Incremental GPR LR", "Moderate", "Update H4: conditional parametric bootstrap supports in-sample incremental fit; BIC and validation do not support a forecast gain beyond VIX", "Open: split in-sample association from prediction"),
        ("Transition contrast uncertainty", "Moderate", "Report bootstrap percentile intervals and failures beside delta intervals; avoid causal wording", "Open: update uncertainty paragraph"),
        ("WTI paired loss comparison", "Moderate", "Retain Stage 06 absolute-tail p=.0475 as audit trail; use coherent equal-tail p=.0820 and CI containing zero", "Already corrected in Stage 07/08; verify final manuscript"),
        ("Stage 07 repeated test", "Moderate", "Keep later 2020–2026 reuse explicitly exploratory", "Already in Stage 08; verify final manuscript"),
    ]
    save(pd.DataFrame(rows,columns=["issue","severity","recommended_thesis_text_change","status"]),
         "thesis_text_changes")


def run():
    TABLES.mkdir(parents=True,exist_ok=True)
    d=audit_data(write_audit=False)
    maximum,ledger,annual=raw_calendar_and_exclusions(d)
    saved,checks=verify_forecasts(d)
    paired=verify_loss_bootstraps(saved)
    reproducibility_and_timing()
    thesis_change_table()
    all_param=[];all_regimes=[];all_effects=[];all_diag=[];all_attempts=[]
    for output,attempts,diagnostics,parameters in (
        gap_sensitivity(d,ledger),vix_exclusion_sensitivity(maximum,d),gpr_delay_sensitivity(d)):
        params,regimes,effects,diags=output
        all_param.append(params);all_regimes.append(regimes);all_effects.append(effects);all_diag.append(diags)
        all_attempts.extend(attempts)
    save(pd.concat(all_param,ignore_index=True),"sensitivity_parameters")
    save(pd.concat(all_regimes,ignore_index=True),"sensitivity_regimes")
    save(pd.concat(all_effects,ignore_index=True),"sensitivity_transition_contrasts")
    save(pd.concat(all_diag,ignore_index=True),"sensitivity_diagnostics")
    save(pd.DataFrame(all_attempts),"sensitivity_fit_attempts")
    train,columns,fits,effects,lr=original_fit_and_scenarios(d)
    boot=parametric_bootstraps(train,columns,fits,effects,lr)
    return dict(annual_coverage=annual,forecast_checks=checks,paired_bootstrap=paired,
                original_lr=lr,original_effects=effects,parametric_bootstrap=boot,
                sensitivity_performance=pd.concat([
                    pd.read_csv(TABLES/"09_gap_exposure_validation_performance.csv"),
                    pd.read_csv(TABLES/"09_vix_lag_only_validation_performance.csv"),
                    pd.read_csv(TABLES/"09_gpr_7day_proxy_validation_performance.csv")],ignore_index=True))



