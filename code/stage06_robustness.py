"""Stage 06 controlled oil-benchmark, predictor, and sample robustness checks."""
from __future__ import annotations

import json
import platform
from pathlib import Path

import arch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import statsmodels

from stage05_forecast import (ROOT, TABLES, FIGURES, SPECS, SEEDS,
                              audit_data, compare, fit_markov, forecast_block,
                              performance, scale_predictors)


def read_inputs():
    brent = audit_data(write_audit=False)
    wti = pd.read_csv(ROOT / "data/processed/01_wti_robustness.csv",
                      parse_dates=["date", "previous_observed_date", "vix_lag_source_date"])
    audit = pd.read_csv(ROOT / "data/processed/01_wti_price_return_audit.csv",
                        parse_dates=["date", "previous_observed_date"])
    assert wti.date.is_unique and wti.date.is_monotonic_increasing
    assert wti.split.value_counts().to_dict() == {"train":3633,"validation":1000,"test":1703}
    assert wti.date.max() == brent.date.max()
    assert np.isfinite(wti.wti_return_pct).all()
    assert np.allclose(wti.wti_return_pct,100*np.log(
        wti.wti_price_usd_per_bbl/wti.wti_price_usd_per_bbl_previous_observed),atol=1e-8)
    assert (wti.wti_price_usd_per_bbl > 0).all() and (wti.wti_price_usd_per_bbl_previous_observed > 0).all()
    assert (wti.vix_lag_source_date < wti.date).all()
    assert set(wti.date).issubset(set(audit.date))
    crisis = audit.loc[audit.date.between("2020-04-16","2020-04-24")]
    bad = crisis.loc[crisis.wti_return_pct.isna()]
    assert bad.date.dt.strftime("%Y-%m-%d").tolist() == ["2020-04-20","2020-04-21"]
    assert bad.return_exclusion_reason.tolist() == ["current price nonpositive","previous price nonpositive"]
    assert crisis.loc[crisis.date.eq("2020-04-22"),"wti_return_pct"].notna().all()
    # The source audit also contains other unavailable returns; preserve an exact ledger.
    unavailable = audit.loc[audit.wti_return_pct.isna(),
                            ["date","wti_price_usd_per_bbl","wti_price_usd_per_bbl_previous_observed",
                             "return_exclusion_reason"]].copy()
    unavailable.to_csv(TABLES/"06_wti_unavailable_returns.csv",index=False,date_format="%Y-%m-%d")
    coverage = wti.groupby([wti.date.dt.year,"split"]).agg(
        n=("date","size"), max_gap_days=("observation_gap_calendar_days","max"),
        gaps_over_3_days=("observation_gap_calendar_days",lambda x:int((x>3).sum()))).reset_index(names=["year","split"])
    coverage.to_csv(TABLES/"06_wti_coverage.csv",index=False)
    data_audit={"observed_price_rows_in_window":len(audit),
                "valid_positive_price_log_returns":int(audit.wti_return_pct.notna().sum()),
                "complete_case_returns":len(wti),
                "valid_returns_excluded_by_predictor_complete_cases":int(audit.wti_return_pct.notna().sum()-len(wti)),
                "undefined_nonpositive_price_return_dates":bad.date.dt.strftime("%Y-%m-%d").tolist(),
                "split_counts":wti.split.value_counts().to_dict(),
                "maximum_calendar_gap_days":int(wti.observation_gap_calendar_days.max()),
                "gaps_over_3_calendar_days":int((wti.observation_gap_calendar_days>3).sum())}
    (TABLES/"06_wti_data_audit.json").write_text(json.dumps(data_audit,indent=2),encoding="utf-8")
    # Stage 05 and Stage 04 saved outputs are audited, not refit merely to recreate them.
    p5 = pd.read_csv(TABLES/"05_performance.csv")
    s5 = pd.read_csv(TABLES/"05_2007_start_validation_performance.csv")
    m4 = pd.read_csv(TABLES/"04_model_comparison.csv")
    assert set(p5.model) == {"B0","B1","B1-t","M1","M3","M4","M5"}
    assert set(s5.model).issuperset(SPECS) and set(m4.model) == set(SPECS)
    assert m4.observations.eq(3699).all()
    return brent,wti,audit,p5,s5,m4


def prepared(frame, return_col):
    return frame.rename(columns={return_col:"brent_return_pct"}).copy()


def training_scale(est):
    return {key:(float(est[col].mean()),float(est[col].std(ddof=0)))
            for key,col in [("gpr","gpr_lag1_calendar_day"),("vix","vix_lag1_observation")]}


def transition_matrix(fit, z):
    name=fit["diag"]["model"]
    cols=SPECS[name]
    x=np.array([[1.0]+[z[c] for c in cols]]) if cols else None
    raw=np.asarray(fit["model"].regime_transition_matrix(fit["params"],exog_tvtp=x))
    order=fit["order"]
    P=raw[:,:,0].T[np.ix_(order,order)]
    assert np.isfinite(P).all() and np.all((P>=0)&(P<=1)) and np.allclose(P.sum(axis=1),1,atol=1e-9)
    return P


def contrast_with_uncertainty(fit,zlow,zhigh,cell):
    """Delta-method interval for a p90-minus-p10 transition contrast."""
    def target(params):
        original=fit["params"]
        fit["params"]=params
        try:
            value=transition_matrix(fit,zhigh)[cell]-transition_matrix(fit,zlow)[cell]
        finally:
            fit["params"]=original
        return float(value)
    value=target(fit["params"])
    cov=np.asarray(fit["result"].cov_params(),dtype=float)
    grad=[]
    for i,p in enumerate(fit["params"]):
        step=1e-5*max(1,abs(p))
        plus=fit["params"].copy();minus=fit["params"].copy()
        plus[i]+=step;minus[i]-=step
        grad.append((target(plus)-target(minus))/(2*step))
    variance=float(np.asarray(grad)@cov@np.asarray(grad))
    if not np.isfinite(variance) or variance<0:
        return value,np.nan,np.nan,np.nan
    se=np.sqrt(variance)
    return value,se,value-1.96*se,value+1.96*se


def fit_outputs(case,fitset,est,raw_gpr_quantiles=None):
    """All scenarios condition on the other predictor's fitting-sample median."""
    scales=training_scale(est)
    raw_gpr=est.gpr_lag1_calendar_day.to_numpy(float)
    # For log(1+GPR), quantiles are translated from original GPR levels.
    if raw_gpr_quantiles is None: raw_gpr_quantiles=np.quantile(raw_gpr,[.1,.5,.9])
    raw_levels=np.asarray(raw_gpr_quantiles,dtype=float)
    gpr_levels=raw_levels.copy()
    if case=="gpr_log1p": gpr_levels=np.log1p(gpr_levels)
    vix_levels=np.quantile(est.vix_lag1_observation,[.1,.5,.9])
    scenario_rows=[];coef_rows=[];regime_rows=[];diag_rows=[]
    for name,fit in fitset.items():
        diag=dict(fit["diag"]);diag["case"]=case;diag_rows.append(diag)
        p=fit["params"];res=fit["result"]
        bse=np.asarray(res.bse)
        for k,(value,se) in enumerate(zip(p,bse)):
            coef_rows.append(dict(case=case,model=name,parameter=res.model.param_names[k],
                                  estimate=float(value),std_error=float(se) if np.isfinite(se) else np.nan))
        zmedian={"gpr":(gpr_levels[1]-scales["gpr"][0])/scales["gpr"][1],
                 "vix":(vix_levels[1]-scales["vix"][0])/scales["vix"][1]}
        base=transition_matrix(fit,zmedian)
        filtered=np.asarray(res.filtered_marginal_probabilities)[:,fit["order"]]
        smoothed=np.asarray(res.smoothed_marginal_probabilities)[:,fit["order"]]
        raw_trans=np.asarray(res.regime_transition)
        average_P=raw_trans.transpose(2,1,0)[:,fit["order"]][:,:,fit["order"]].mean(axis=0)
        assert np.allclose(average_P.sum(axis=1),1,atol=1e-8)
        for state,label in enumerate(("Low","High")):
            stay=float(base[state,state])
            regime_rows.append(dict(case=case,model=name,state=label,
                                    original_state=int(fit["order"][state]),
                                    mean_pct=float(fit["means"][state]),
                                    sd_pct=float(np.sqrt(fit["variances"][state])),
                                    stay_probability_at_median=stay,
                                    stay_probability_training_average=float(average_P[state,state]),
                                    mean_filtered_occupancy=float(filtered[:,state].mean()),
                                    mean_smoothed_occupancy_retrospective=float(smoothed[:,state].mean()),
                                    expected_duration_observations_at_median=1/(1-stay)))
        for focal in SPECS[name]:
            values=gpr_levels if focal=="gpr" else vix_levels
            low=dict(zmedian);high=dict(zmedian)
            low[focal]=(values[0]-scales[focal][0])/scales[focal][1]
            high[focal]=(values[2]-scales[focal][0])/scales[focal][1]
            Plow=transition_matrix(fit,low);Phigh=transition_matrix(fit,high)
            for label,cell in [("low_to_high",(0,1)),("high_to_high",(1,1))]:
                effect,se,lo,hi=contrast_with_uncertainty(fit,low,high,cell)
                scenario_rows.append(dict(case=case,model=name,predictor=focal,transition=label,
                                          raw_p10=float(raw_levels[0] if focal=="gpr" else values[0]),
                                          raw_p90=float(raw_levels[2] if focal=="gpr" else values[2]),
                                          probability_p10=float(Plow[cell]),probability_p90=float(Phigh[cell]),
                                          p90_minus_p10=effect,delta_se=se,delta_ci95_low=lo,delta_ci95_high=hi))
    return (pd.DataFrame(scenario_rows),pd.DataFrame(coef_rows),
            pd.DataFrame(regime_rows),pd.DataFrame(diag_rows))


def alternative_columns(brent):
    vix=pd.read_csv(ROOT/"data/raw/VIXCLS.csv",dtype="string")
    vix["date"]=pd.to_datetime(vix.observation_date)
    vix["value"]=pd.to_numeric(vix.VIXCLS,errors="coerce")
    observed=vix.loc[vix.value.notna(),["date","value"]].sort_values("date").copy()
    observed["lag5"] = observed.value.shift(5)
    observed["lag5_source_date"] = observed.date.shift(5)
    d=brent.merge(observed[["date","lag5","lag5_source_date"]],on="date",how="left",validate="one_to_one")
    import xlrd
    gpr=pd.read_excel(ROOT/"data/raw/data_gpr_daily_recent.xls",engine="xlrd")
    lookup=pd.Series(pd.to_numeric(gpr.GPRD,errors="coerce").to_numpy(),
                     index=pd.to_datetime(gpr.DAY.astype(str),format="%Y%m%d"))
    d["gpr_lag5_calendar_day"]=(d.date-pd.Timedelta(days=5)).map(lookup)
    d=d.rename(columns={"lag5":"vix_lag5_observation","lag5_source_date":"vix_lag5_source_date"})
    assert d[["gpr_lag5_calendar_day","vix_lag5_observation"]].notna().all().all()
    assert (d.vix_lag5_source_date < d.date).all()
    assert np.allclose(d.gpr_lag5_calendar_day,(d.date-pd.Timedelta(days=5)).map(lookup))
    # Check that 'five observations' is exactly the fifth prior nonmissing VIX quote.
    assert np.allclose(d.vix_lag5_observation,d.vix_lag5_source_date.map(observed.set_index("date").value))
    return d


def write_assessment():
    """Summarize the fixed Stage 06 evidence with explicit qualifications."""
    rows=[
        ("Distinct low/high WTI volatility regimes",
         "WTI training state SDs are about 1.8 versus 4.4–4.9 percentage points; both states have meaningful occupancy.",
         "Occupancy and persistence vary with the transition specification.",
         "supported — descriptive two-state distinction"),
        ("GPR and entry to high volatility",
         "Brent M5 p90–p10 GPR contrast is about +0.036 (delta 95% CI +0.011 to +0.061); positive with five-day lag and log transform.",
         "WTI M5 contrast is about +0.012 with interval crossing zero; 2007-start Brent contrast is near zero and uncertain.",
         "mixed — sample and benchmark dependent"),
        ("VIX and high-state persistence",
         "M5 p90–p10 VIX contrast is positive in Brent (+0.193) and WTI (+0.211); Brent lag and transform variants retain positive contrasts.",
         "WTI M5 interval narrowly crosses zero; 2007-start interval is wide and crosses zero.",
         "mixed — direction stable, precision uneven"),
        ("Incremental GPR forecasting value beyond VIX",
         "Brent M5 numerically improves on M1 in the Stage 05 test.",
         "Brent M5 versus M4 test difference is under 0.001 with interval crossing zero; WTI M5 is worse than M4 on validation and test QLIKE.",
         "not supported — no consistent predictive gain"),
        ("Alternative lag and GPR transform forecasts",
         "Five-lag and log(1+GPR) Brent M5 validation QLIKE stays near 2.445 on identical dates.",
         "GPR probability-contrast magnitude changes; original level has slightly better AIC/BIC than log transform.",
         "mixed — forecast ranking stable, effect size changes"),
        ("Shorter Brent training period",
         "Regime variance ordering survives the 2007–2015 fit.",
         "M5 validation QLIKE worsens from about 2.445 to 2.502; GPR contrast loses precision and VIX entry contrast changes sharply.",
         "mixed — early coverage materially affects transitions"),
        ("Real-time GPR predictive claim",
         "Lagged revised historical GPR is associated with some fitted Brent transition probabilities.",
         "Historical publication vintages and release timing are unverified; held-out incremental GPR gains are not reliable.",
         "not supported — pseudo-out-of-sample evidence only"),
    ]
    table=pd.DataFrame(rows,columns=["Finding","Supporting evidence","Contradicting evidence","Robustness assessment"])
    table.to_csv(TABLES/"06_robustness_assessment.csv",index=False)
    return table


def run():
    TABLES.mkdir(parents=True,exist_ok=True);FIGURES.mkdir(parents=True,exist_ok=True)
    brent,wti,wti_audit,p5,s5,stage4=read_inputs()
    all_scenarios=[];all_coef=[];all_regimes=[];all_diag=[];all_attempts=[]
    wti_fc=[];wti_scales=[];all_scaling=[]
    wti_train=prepared(wti.loc[wti.split.eq("train")].reset_index(drop=True),"wti_return_pct")
    wti_val=prepared(wti.loc[wti.split.eq("validation")].reset_index(drop=True),"wti_return_pct")
    wti_est=prepared(wti.loc[wti.split.isin(["train","validation"])].reset_index(drop=True),"wti_return_pct")
    wti_test=prepared(wti.loc[wti.split.eq("test")].reset_index(drop=True),"wti_return_pct")
    wti_fit={}
    for period,est,ev in [("wti_validation",wti_train,wti_val),("wti_test",wti_est,wti_test)]:
        fc,sc,at,di,pa,fits=forecast_block(est,ev,period,tuple(SPECS),return_fits=True)
        assert set(fits)==set(SPECS) and fc.groupby("model").date.nunique().eq(len(ev)).all()
        wti_fc.append(fc);wti_scales.extend(sc);all_attempts.extend(at)
        all_scaling.extend(sc)
        wti_fit[period]=fits
        if period=="wti_validation":
            scenarios,coefficients,regimes,diagnostics=fit_outputs("wti_training",fits,est)
            all_scenarios.append(scenarios);all_coef.append(coefficients)
            all_regimes.append(regimes);all_diag.append(diagnostics)
        else:
            all_diag.append(pd.DataFrame([dict(f["diag"],case="wti_test") for f in fits.values()]))
        pd.DataFrame(all_attempts).to_csv(TABLES/"06_fit_attempts.csv",index=False)
    wti_forecasts=pd.concat(wti_fc,ignore_index=True)
    wti_forecasts.to_csv(TABLES/"06_wti_forecasts.csv",index=False,float_format="%.12g",date_format="%Y-%m-%d")
    wti_performance=performance(wti_forecasts,["period","model"])
    wti_performance.to_csv(TABLES/"06_wti_forecast_performance.csv",index=False)
    wti_comparison=pd.concat([compare(wti_forecasts.loc[wti_forecasts.period.eq(k)],k,
                                     pairs=[("M1","M3"),("M1","M4"),("M1","M5"),("M4","M5")])
                              for k in ("wti_validation","wti_test")],ignore_index=True)
    wti_comparison.to_csv(TABLES/"06_wti_paired_bootstrap.csv",index=False)
    pd.DataFrame(wti_scales).to_csv(TABLES/"06_wti_scaling.csv",index=False)
    # Benchmark comparison uses within-benchmark ranks; likelihoods are not directly compared.
    bcmp=stage4[["model","observations","log_likelihood","aic","bic"]].copy()
    bcmp["benchmark"]="Brent"
    wcmp=pd.DataFrame([dict(model=n,observations=len(wti_train),log_likelihood=f["result"].llf,
                            aic=f["result"].aic,bic=f["result"].bic,benchmark="WTI")
                       for n,f in wti_fit["wti_validation"].items()])
    comparison=pd.concat([bcmp,wcmp],ignore_index=True)
    comparison["aic_rank_within_benchmark"]=comparison.groupby("benchmark").aic.rank(method="min").astype(int)
    comparison["bic_rank_within_benchmark"]=comparison.groupby("benchmark").bic.rank(method="min").astype(int)
    comparison.to_csv(TABLES/"06_brent_wti_model_comparison.csv",index=False)
    brent_coef=[]
    for name in SPECS:
        saved=pd.read_csv(TABLES/f"04_{name.lower()}_parameters.csv")
        for row in saved.itertuples():
            brent_coef.append(dict(case="brent_original",model=name,parameter=row.parameter,
                                   estimate=row.estimate,std_error=row.std_error))
    pd.concat([pd.DataFrame(brent_coef),all_coef[0]],ignore_index=True).to_csv(
        TABLES/"06_brent_wti_coefficients.csv",index=False)
    # Refit only the original Brent M5 once to recover its full covariance;
    # Stage 04 exported marginal standard errors but not the covariance matrix.
    brent_train=brent.loc[brent.split.eq("train")].reset_index(drop=True)
    basecols,_=scale_predictors(brent_train,brent_train,"brent_original")
    baseline_fit={"M5":fit_markov("M5",brent_train.brent_return_pct.to_numpy(float),
                                   basecols,"brent_original",all_attempts)}
    outputs=fit_outputs("brent_original",baseline_fit,brent_train)
    for target,item in zip((all_scenarios,all_coef,all_regimes,all_diag),outputs):target.append(item)
    pd.DataFrame(all_attempts).to_csv(TABLES/"06_fit_attempts.csv",index=False)
    # Lag experiments: same Brent dates, original specification held as saved Stage 04/05 reference.
    lagged=alternative_columns(brent)
    assert len(lagged)==len(brent) and lagged.date.equals(brent.date)
    lag_results=[]
    for case,replace,models in [
        ("vix_lag5_observations",{"vix_lag1_observation":"vix_lag5_observation"},("M4","M5")),
        ("gpr_lag5_calendar_days",{"gpr_lag1_calendar_day":"gpr_lag5_calendar_day"},("M3","M5"))]:
        alt=lagged.copy()
        for original,source in replace.items():alt[original]=alt[source]
        est=alt.loc[alt.split.eq("train")].reset_index(drop=True)
        ev=alt.loc[alt.split.eq("validation")].reset_index(drop=True)
        fc,sc,at,di,pa,fits=forecast_block(est,ev,case,models,return_fits=True)
        assert fc.groupby("model").date.nunique().eq(999).all()
        lag_results.append(performance(fc,["period","model"]))
        all_scaling.extend(sc)
        all_attempts.extend(at)
        outputs=fit_outputs(case,fits,est,
                            raw_gpr_quantiles=np.quantile(brent.loc[brent.split.eq("train"),"gpr_lag1_calendar_day"],[.1,.5,.9])
                            if case=="gpr_lag5_calendar_days" else None)
        for target,item in zip((all_scenarios,all_coef,all_regimes,all_diag),outputs):target.append(item)
        fc.to_csv(TABLES/f"06_{case}_validation_forecasts.csv",index=False,float_format="%.12g",date_format="%Y-%m-%d")
        pd.DataFrame(all_attempts).to_csv(TABLES/"06_fit_attempts.csv",index=False)
    # Log transform changes only the GPR regressor; retain every original Brent date.
    logdata=brent.copy()
    assert logdata.gpr_lag1_calendar_day.ge(0).all()
    original_gpr_quantiles=np.quantile(brent.loc[brent.split.eq("train"),"gpr_lag1_calendar_day"],[.1,.5,.9])
    logdata["gpr_lag1_calendar_day"]=np.log1p(logdata.gpr_lag1_calendar_day)
    log_est=logdata.loc[logdata.split.eq("train")].reset_index(drop=True)
    log_ev=logdata.loc[logdata.split.eq("validation")].reset_index(drop=True)
    logfc,logsc,at,di,pa,logfits=forecast_block(log_est,log_ev,"gpr_log1p",("M5",),return_fits=True)
    assert logfc.date.equals(brent.loc[brent.split.eq("validation"),"date"].reset_index(drop=True))
    all_attempts.extend(at)
    all_scaling.extend(logsc)
    outputs=fit_outputs("gpr_log1p",logfits,log_est,original_gpr_quantiles)
    for target,item in zip((all_scenarios,all_coef,all_regimes,all_diag),outputs):target.append(item)
    logfc.to_csv(TABLES/"06_gpr_log1p_validation_forecasts.csv",index=False,float_format="%.12g",date_format="%Y-%m-%d")
    pd.DataFrame(all_attempts).to_csv(TABLES/"06_fit_attempts.csv",index=False)
    pd.DataFrame(all_scaling).to_csv(TABLES/"06_scaling.csv",index=False)
    # Reuse Stage 05 2007-start validation performance; estimate coefficients
    # because that stage did not export uncertainty or transition scenarios.
    short=brent.loc[brent.date.between("2007-01-01","2015-12-31")].reset_index(drop=True)
    y=short.brent_return_pct.to_numpy(float)
    cols,_=scale_predictors(short,short,"brent_2007_training")
    shortfits={}
    for name in ("M3","M4","M5"):
        shortfits[name]=fit_markov(name,y,cols,"brent_2007_training",all_attempts)
        shortfits[name]["diag"]["fitting_start"]=str(short.date.min().date())
        shortfits[name]["diag"]["fitting_end"]=str(short.date.max().date())
    outputs=fit_outputs("brent_2007_training",shortfits,short)
    for target,item in zip((all_scenarios,all_coef,all_regimes,all_diag),outputs):target.append(item)
    pd.DataFrame(all_attempts).to_csv(TABLES/"06_fit_attempts.csv",index=False)
    # Saved original Brent scenarios and parameters anchor controlled comparisons.
    original_scenarios=pd.read_csv(TABLES/"04_transition_scenarios.csv")
    scenario_table=pd.concat(all_scenarios,ignore_index=True)
    scenario_table.to_csv(TABLES/"06_transition_scenarios.csv",index=False)
    coefficients=pd.concat(all_coef,ignore_index=True)
    coefficients.to_csv(TABLES/"06_all_coefficients.csv",index=False)
    regimes=pd.concat(all_regimes,ignore_index=True)
    regimes.to_csv(TABLES/"06_regime_summary.csv",index=False)
    diagnostics=pd.concat(all_diag,ignore_index=True)
    diagnostics["parameters"]=diagnostics.param_names.str.split(" | ",regex=False).str.len()
    diagnostics["aic"]=-2*diagnostics.llf+2*diagnostics.parameters
    diagnostics["bic"]=-2*diagnostics.llf+diagnostics.parameters*np.log(diagnostics.n)
    diagnostics.to_csv(TABLES/"06_numerical_stability.csv",index=False)
    lag_performance=pd.concat([p5.loc[p5.period.eq("validation") & p5.model.isin(SPECS)].assign(period="original_lags")]
                              +lag_results,ignore_index=True)
    lag_performance.to_csv(TABLES/"06_alternative_lag_performance.csv",index=False)
    lag_fit=pd.concat([
        stage4.loc[stage4.model.isin(["M3","M4","M5"]),["model","observations","log_likelihood","aic","bic"]].rename(
            columns={"observations":"n","log_likelihood":"llf"}).assign(case="brent_original"),
        diagnostics.loc[diagnostics.case.isin(["vix_lag5_observations","gpr_lag5_calendar_days"]),
                        ["case","model","n","llf","aic","bic"]]],ignore_index=True)
    lag_fit.to_csv(TABLES/"06_alternative_lag_fit.csv",index=False)
    transform_performance=pd.concat([
        p5.loc[p5.period.eq("validation") & p5.model.eq("M5")].assign(period="gpr_level"),
        performance(logfc,["period","model"])],ignore_index=True)
    transform_performance.to_csv(TABLES/"06_gpr_transform_performance.csv",index=False)
    transform_fit=pd.concat([
        stage4.loc[stage4.model.eq("M5"),["model","observations","log_likelihood","aic","bic"]].rename(
            columns={"observations":"n","log_likelihood":"llf"}).assign(case="gpr_level"),
        diagnostics.loc[diagnostics.case.eq("gpr_log1p"),["case","model","n","llf","aic","bic"]]],ignore_index=True)
    transform_fit.to_csv(TABLES/"06_gpr_transform_fit.csv",index=False)
    early=pd.concat([
        p5.loc[p5.period.eq("validation") & p5.model.isin(SPECS)].assign(training="2000-2015"),
        s5.loc[s5.model.isin(SPECS)].assign(training="2007-2015")],ignore_index=True)
    early.to_csv(TABLES/"06_early_sample_performance.csv",index=False)
    # A compact comparison of original and alternate scenario contrasts.
    baseline=[]
    for r in original_scenarios.itertuples():
        if r.scenario in ("gpr_p10_other_median","gpr_p90_other_median","vix_p10_other_median","vix_p90_other_median"):
            baseline.append(dict(model=r.model,predictor=r.scenario.split("_")[0],
                                 percentile=r.scenario.split("_")[1],low_to_high=r.low_to_high,
                                 high_to_high=r.high_to_high))
    base=pd.DataFrame(baseline)
    baseline_effect=[]
    for (model,predictor),part in base.groupby(["model","predictor"]):
        if model=="M5": continue  # original M5 refit above supplies intervals
        if set(part.percentile)=={"p10","p90"}:
            lo=part.loc[part.percentile.eq("p10")].iloc[0];hi=part.loc[part.percentile.eq("p90")].iloc[0]
            for transition in ("low_to_high","high_to_high"):
                baseline_effect.append(dict(case="brent_original",model=model,predictor=predictor,
                                            transition=transition,p90_minus_p10=hi[transition]-lo[transition]))
    effects=pd.concat([pd.DataFrame(baseline_effect),scenario_table],ignore_index=True)
    effects.to_csv(TABLES/"06_effect_comparison.csv",index=False)
    metadata=dict(software={"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,
                            "scipy":scipy.__version__,"statsmodels":statsmodels.__version__,"arch":arch.__version__},
                  seeds=list(SEEDS),uncertainty="delta-method Wald intervals for scenario contrasts; paired 10-observation block-bootstrap for losses",
                  alternative_lags={"VIX":"fifth preceding nonmissing market observation, assigned to current date with observed VIX",
                                    "GPR":"exactly five calendar days before current return"},
                  forecasts="fixed parameters; sequential filtered probabilities, never smoothed",
                  gpr_limitation="revised historical values; unverified release timing; pseudo-out-of-sample")
    (TABLES/"06_reproducibility.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    # A single figure relates effect magnitude and uncertainty across controlled cases.
    graph=scenario_table.loc[(scenario_table.model.eq("M5")) & scenario_table.case.ne("wti_test")].copy()
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,predictor,title in zip(axes,("gpr","vix"),("GPR: low to high","VIX: high to high")):
        transition="low_to_high" if predictor=="gpr" else "high_to_high"
        sub=graph.loc[graph.predictor.eq(predictor)&graph.transition.eq(transition)].reset_index(drop=True)
        x=np.arange(len(sub))
        ax.errorbar(x,sub.p90_minus_p10,
                    yerr=np.vstack([sub.p90_minus_p10-sub.delta_ci95_low,sub.delta_ci95_high-sub.p90_minus_p10]),
                    fmt="o",capsize=3,color="#315f78")
        ax.axhline(0,color="0.5",lw=.8)
        ax.set_xticks(x,sub.case,rotation=55,ha="right")
        ax.set(title=title,ylabel="p90 minus p10 transition probability")
        ax.grid(axis="y",alpha=.15)
    fig.tight_layout();fig.savefig(FIGURES/"06_transition_effect_robustness.png",dpi=180,bbox_inches="tight");plt.close(fig)
    assessment=write_assessment()
    return dict(comparison=comparison,wti_performance=wti_performance,wti_comparison=wti_comparison,
                lag_performance=lag_performance,transform_performance=transform_performance,
                early_performance=early,scenarios=scenario_table,effects=effects,
                coefficients=coefficients,regimes=regimes,diagnostics=diagnostics,
                attempts=pd.DataFrame(all_attempts),assessment=assessment)
