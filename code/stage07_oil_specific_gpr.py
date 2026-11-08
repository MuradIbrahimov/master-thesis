"""Retrospective AI-GPR oil-specific Brent Markov transition experiment."""
from __future__ import annotations

import hashlib
import json
import platform
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import statsmodels
from scipy.special import logsumexp
from scipy.stats import norm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

ROOT=Path(__file__).resolve().parents[1]
TABLES=ROOT/"results"/"tables"
FIGURES=ROOT/"results"/"figures"
RAW=ROOT/"data"/"raw"/"ai_gpr_data_daily.csv"
SEEDS=(41,67,103)
FIT_SETTINGS=dict(em_iter=8,search_reps=6,search_iter=8,maxiter=300,disp=False)
SPECS={"M4":("vix",),"M5":("original_gpr","vix"),
       "E1":("ai_gpr","vix"),"E2":("oil_gpr","vix"),
       "E3":("matched_keyword_gpr","vix")}
COLS={"original_gpr":"gpr_lag1_calendar_day", "vix":"vix_lag1_observation",
      "ai_gpr":"gpr_ai_lag1_calendar_day", "oil_gpr":"gpr_oil_lag1_calendar_day",
      "matched_keyword_gpr":"gpr_aer_lag1_calendar_day"}
NEW={"GPR_AI":"gpr_ai_lag1_calendar_day","GPR_OIL":"gpr_oil_lag1_calendar_day",
     "GPR_AER":"gpr_aer_lag1_calendar_day",
     "GPR_OIL_THREATS":"gpr_oil_threats_lag1_calendar_day",
     "GPR_OIL_ACTS":"gpr_oil_acts_lag1_calendar_day"}


def bootstrap_correction():
    old=pd.read_csv(TABLES/"06_wti_paired_bootstrap.csv")
    old=old.loc[old.period.eq("wti_test") & old.left.eq("M1") & old.right.eq("M4")].iloc[0]
    f=pd.read_csv(TABLES/"06_wti_forecasts.csv",parse_dates=["date"])
    a=f.loc[f.period.eq("wti_test") & f.model.isin(["M1","M4"])].pivot(index="date",columns="model",values="qlike")
    d=(a.M1-a.M4).to_numpy(float);n=len(d);B=2000;block=10;seed=7205
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,n,size=(B,int(np.ceil(n/block))))
    index=((starts[:,:,None]+np.arange(block))%n).reshape(B,-1)[:,:n]
    observed=float(d.mean()); boot=d[index].mean(axis=1); null=boot-observed
    q025,q975=np.quantile(boot,[.025,.975])
    basic_low=2*observed-q975;basic_high=2*observed-q025
    absolute_tail=(1+int((abs(null)>=abs(observed)).sum()))/(B+1)
    upper=(1+int((null>=observed).sum()))/(B+1)
    lower=(1+int((null<=observed).sum()))/(B+1)
    equal_tail=min(1.0,2*min(upper,lower))
    assert n==1703 and np.isclose(observed,old.mean_qlike_left_minus_right,atol=1e-6)
    assert np.isclose(basic_low,old.basic_ci95_low,atol=1e-6)
    assert np.isclose(basic_high,old.basic_ci95_high,atol=1e-6)
    assert np.isclose(absolute_tail,old.centered_block_bootstrap_p,atol=1e-10)
    assert basic_low<0<basic_high and equal_tail>.05
    out=pd.DataFrame([dict(comparison="WTI test M1 minus M4 QLIKE",n=n,
                           mean_difference=observed,sign="positive favors M4",
                           original_absolute_tail_p=absolute_tail,
                           corrected_equal_tail_p_consistent_with_basic_ci=equal_tail,
                           basic_ci95_low=basic_low,basic_ci95_high=basic_high,
                           lower_tail_probability=lower,upper_tail_probability=upper,
                           block_length=block,replications=B,seed=seed)])
    out.to_csv(TABLES/"07_stage06_bootstrap_correction.csv",index=False)
    (TABLES/"07_stage06_bootstrap_correction.md").write_text(
        "The Stage 06 calculation and sign are correct: M1 minus M4 is positive when M4 has lower QLIKE. "
        "The reported two-sided p used the absolute centered-bootstrap tail, while the basic 95% interval "
        "uses equal tails of an asymmetric bootstrap distribution. These conventions need not agree. "
        f"The equal-tail two-sided p matching that interval is {equal_tail:.4f}; the original absolute-tail "
        f"p is {absolute_tail:.4f}. The interval [{basic_low:.4f}, {basic_high:.4f}] contains zero. "
        "We retain the Stage 06 file and use the equal-tail result for a coherent paired interpretation. "
        "Neither result is adjusted for multiple comparisons or resolves the nonregularity of nested forecasts.\n",
        encoding="utf-8")
    return out


def prepare_data():
    assert RAW.exists(),f"Official CSV unavailable: {RAW}"
    raw=pd.read_csv(RAW,parse_dates=["Date"])
    assert raw.shape[1]==17 and raw.Date.is_unique and raw.Date.is_monotonic_increasing
    assert raw.Date.min()==pd.Timestamp("1960-01-01") and raw.Date.max()>=pd.Timestamp("2026-10-26")
    assert (raw.Date.diff().dropna().dt.days==1).all()
    assert set(NEW).issubset(raw.columns) and raw[list(NEW)].notna().all().all()
    assert raw[list(NEW)].ge(0).all().all()
    brent=pd.read_csv(ROOT/"data/processed/01_brent_complete_case.csv",
                      parse_dates=["date","vix_lag_source_date","previous_observed_date"])
    assert len(brent)==6410 and brent.date.is_monotonic_increasing
    assert brent.split.value_counts().to_dict()=={"train":3699,"validation":999,"test":1712}
    lookup=raw.set_index("Date")
    d=brent.copy()
    source_date=d.date-pd.Timedelta(days=1)
    for source,target in NEW.items():d[target]=source_date.map(lookup[source])
    required=[COLS[x] for x in COLS]
    assert d[required].notna().all().all() and (source_date<d.date).all()
    assert (d.vix_lag_source_date<d.date).all()
    assert len(d)==len(brent) and d.date.equals(brent.date)
    d.to_csv(ROOT/"data/processed/07_brent_ai_gpr_matched.csv",index=False,date_format="%Y-%m-%d")
    coverage=[]
    for source,target in NEW.items():
        for split,part in d.groupby("split"):
            x=part[target]
            coverage.append(dict(index=source,split=split,n=len(x),missing=int(x.isna().sum()),
                                 zero=int(x.eq(0).sum()),mean=float(x.mean()),sd=float(x.std(ddof=0)),
                                 p10=float(x.quantile(.1)),median=float(x.median()),p90=float(x.quantile(.9)),
                                 maximum=float(x.max()),first_date=str(part.date.min().date()),
                                 last_date=str(part.date.max().date())))
    pd.DataFrame(coverage).to_csv(TABLES/"07_data_coverage.csv",index=False)
    definitions=[
        ("GPRD","Original broad daily GPR (Stages 01–06)","10-newspaper recent keyword index; revised daily XLS", "data_gpr_daily_recent.xls"),
        ("GPR_AI","General AI-GPR","Intensity-weighted LLM scores divided by all articles in 3 newspapers",RAW.name),
        ("GPR_OIL","Oil GPR","LLM-classified geopolitical oil/energy supply disruption articles, weighted by GPR intensity",RAW.name),
        ("GPR_OIL_THREATS","Oil Threats GPR","Oil disruption articles classified as threats or both threats and acts",RAW.name),
        ("GPR_OIL_ACTS","Oil Acts GPR","Oil disruption articles classified as acts or both threats and acts",RAW.name),
        ("GPR_AER","Source-matched keyword GPR","Proximity keyword method applied to same 3 newspapers as AI-GPR",RAW.name),
    ]
    pd.DataFrame(definitions,columns=["column","meaning","construction","file"]).to_csv(
        TABLES/"07_index_definitions.csv",index=False)
    metadata=dict(source_url="https://www.matteoiacoviello.com/ai_gpr_files/ai_gpr_data_daily.csv",
                  methodology_url="https://www.matteoiacoviello.com/research_files/AI_GPR_PAPER.pdf",
                  downloaded_sha256=hashlib.sha256(RAW.read_bytes()).hexdigest(),
                  raw_rows=len(raw),raw_first=str(raw.Date.min().date()),raw_last=str(raw.Date.max().date()),
                  raw_columns=raw.columns.tolist(),raw_missing=raw.isna().sum().to_dict(),
                  brent_matched_rows=len(d),new_predictor_missing_on_brent_dates=0,
                  lag="previous calendar day; revised historical series; publication timing unverified",
                  historical_test_status="exploratory; 2020 onward already examined in earlier stages")
    (TABLES/"07_source_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    return d,raw


def scaled(est,ev,period):
    arrays={};rows=[]
    for name,col in COLS.items():
        mu=float(est[col].mean());sd=float(est[col].std(ddof=0))
        assert np.isfinite(mu) and np.isfinite(sd) and sd>0
        arrays[name]=((est[col].to_numpy(float)-mu)/sd,(ev[col].to_numpy(float)-mu)/sd)
        rows.append(dict(period=period,predictor=name,column=col,mean=mu,sd=sd,n=len(est),
                         fitting_end=str(est.date.max().date())))
    return arrays,pd.DataFrame(rows)


def fit_one(name,y,X,period,attempts):
    model=MarkovRegression(y,k_regimes=2,trend="c",switching_trend=True,
                           switching_variance=True,exog_tvtp=X)
    candidates=[]
    for seed in SEEDS:
        np.random.seed(seed)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                res=model.fit(**FIT_SETTINGS)
            params=pd.Series(res.params,index=res.model.param_names)
            valid=bool(res.mle_retvals.get("converged",False) and
                       np.isfinite(res.llf) and np.isfinite(res.params).all() and
                       all(params[f"sigma2[{j}]"]>0 for j in (0,1)))
            attempts.append(dict(period=period,model=name,seed=seed,valid=valid,llf=res.llf,
                                 warning_count=len(caught),warnings=" | ".join(
                                     f"{w.category.__name__}: {w.message}" for w in caught),error=""))
            if valid:candidates.append((res.llf,seed,res))
        except Exception as exc:
            attempts.append(dict(period=period,model=name,seed=seed,valid=False,llf=np.nan,
                                 warning_count=0,warnings="",error=repr(exc)))
    pd.DataFrame(attempts).to_csv(TABLES/"07_fit_attempts.csv",index=False)
    if not candidates:raise RuntimeError(f"{period} {name}: no valid fit; inspect 07_fit_attempts.csv")
    ll,seed,res=max(candidates,key=lambda x:x[0])
    p=pd.Series(res.params,index=res.model.param_names)
    order=np.argsort([p[f"sigma2[{j}]"] for j in (0,1)])
    means=np.array([p[f"const[{j}]"] for j in order]);vars_=np.array([p[f"sigma2[{j}]"] for j in order])
    raw=np.asarray(res.regime_transition)
    assert raw.shape==(2,2,len(y))
    P=raw.transpose(2,1,0)[:,order][:,:,order]
    assert np.allclose(raw.sum(axis=0),1,atol=1e-8)
    assert np.all((P>=0)&(P<=1)) and np.allclose(P.sum(axis=2),1,atol=1e-8)
    filt=np.asarray(res.filtered_marginal_probabilities)[:,order]
    predicted=np.asarray(res.predicted_marginal_probabilities)[:,order]
    smoothed=np.asarray(res.smoothed_marginal_probabilities)[:,order]
    assert np.allclose(filt.sum(axis=1),1,atol=1e-7)
    assert np.allclose(smoothed.sum(axis=1),1,atol=1e-7)
    for j in (1,len(y)//2,len(y)-1):assert np.allclose(filt[j-1]@P[j],predicted[j],atol=1e-7)
    log_emission=norm.logpdf(y[:,None],loc=means,scale=np.sqrt(vars_))
    log_joint=np.log(predicted)+log_emission
    recovered=np.exp(log_joint-logsumexp(log_joint,axis=1)[:,None])
    assert np.allclose(recovered,filt,atol=2e-6)
    assert np.isclose(logsumexp(log_joint,axis=1).sum(),ll,atol=1e-4)
    fit=dict(name=name,model=model,result=res,params=np.asarray(res.params),order=order,
             means=means,variances=vars_,filtered_last=filt[-1].copy())
    diag=dict(period=period,model=name,n=len(y),llf=ll,aic=res.aic,bic=res.bic,
              successful_starts=len(candidates),failed_starts=len(SEEDS)-len(candidates),
              ll_spread=max(z[0] for z in candidates)-min(z[0] for z in candidates),
              chosen_seed=seed,missing_se=int(np.isnan(res.bse).sum()),
              minimum_transition=float(P.min()),maximum_transition=float(P.max()),
              minimum_variance=float(vars_.min()),warning_count=sum(
                  a["warning_count"] for a in attempts if a["period"]==period and a["model"]==name))
    regimes=[dict(period=period,model=name,state=label,original_state=int(order[j]),
                  mean_pct=float(means[j]),sd_pct=float(np.sqrt(vars_[j])),
                  filtered_occupancy=float(filt[:,j].mean()),
                  smoothed_occupancy_retrospective=float(smoothed[:,j].mean()),
                  average_stay_probability=float(P[:,j,j].mean()))
             for j,label in enumerate(("Low","High"))]
    parameters=[dict(period=period,model=name,parameter=k,estimate=float(v),
                     std_error=float(se) if np.isfinite(se) else np.nan)
                for k,v,se in zip(res.model.param_names,res.params,res.bse)]
    return fit,diag,regimes,parameters


def transition(fit,x):
    raw=np.asarray(fit["model"].regime_transition_matrix(fit["params"],exog_tvtp=x))
    assert raw.shape==(2,2,len(x))
    order=fit["order"]
    P=raw.transpose(2,1,0)[:,order][:,:,order]
    assert np.isfinite(P).all() and np.all((P>=0)&(P<=1)) and np.allclose(P.sum(axis=2),1,atol=1e-9)
    return P


def forecast_group(est,ev,period):
    assert est.date.max()<ev.date.min() and ev.date.is_monotonic_increasing
    arrays,scaling=scaled(est,ev,period)
    y=est.brent_return_pct.to_numpy(float);observed=ev.brent_return_pct.to_numpy(float)
    attempts=[];diags=[];regimes=[];params=[];fits={};transitions={}
    for name,features in SPECS.items():
        X=np.column_stack([np.ones(len(est))]+[arrays[k][0] for k in features])
        fit,diag,regime,parameter=fit_one(name,y,X,period,attempts)
        fits[name]=fit;diags.append(diag);regimes.extend(regime);params.extend(parameter)
        Xev=np.column_stack([np.ones(len(ev))]+[arrays[k][1] for k in features])
        transitions[name]=transition(fit,Xev)
    states={name:fit["filtered_last"].copy() for name,fit in fits.items()}
    rows=[]
    for i,(date,r) in enumerate(zip(ev.date,observed)):
        info_through=est.date.iloc[-1] if i==0 else ev.date.iloc[i-1]
        assert info_through<date
        updated={}
        for name,fit in fits.items():
            q=states[name]@transitions[name][i]
            assert np.isfinite(q).all() and np.all((q>=0)&(q<=1)) and np.isclose(q.sum(),1,atol=1e-9)
            mean=float(q@fit["means"])
            second=float(q@(fit["variances"]+fit["means"]**2))
            variance=float(second-mean**2)
            assert np.isclose(mean,np.sum(q*fit["means"]))
            assert np.isclose(second,np.sum(q*(fit["variances"]+fit["means"]**2)))
            assert np.isfinite([mean,second,variance]).all() and second>0 and variance>0
            logjoint=np.log(q)+norm.logpdf(r,fit["means"],np.sqrt(fit["variances"]))
            logdensity=float(logsumexp(logjoint))
            rows.append(dict(period=period,date=date,model=name,information_through=info_through,
                             return_pct=r,predicted_low=q[0],predicted_high=q[1],
                             predicted_mean_pct=mean,predicted_variance_pct2=variance,
                             predicted_second_moment_pct2=second,log_predictive_density=logdensity,
                             qlike=np.log(second)+r*r/second,mse=(second-r*r)**2,
                             log_score=-logdensity))
            updated[name]=np.exp(logjoint-logdensity)
        # Assimilate the current return only after every forecast has been saved.
        states=updated
    out=pd.DataFrame(rows)
    assert out.groupby("model").date.nunique().eq(len(ev)).all()
    assert out.groupby("date").size().eq(len(SPECS)).all()
    assert (out.information_through<out.date).all()
    assert np.allclose(out.predicted_low+out.predicted_high,1,atol=1e-9)
    assert out[["qlike","mse","log_score"]].notna().all().all()
    assert not any("smoothed" in key for f in fits.values() for key in f)
    return out,scaling,pd.DataFrame(attempts),pd.DataFrame(diags),pd.DataFrame(regimes),pd.DataFrame(params),fits


def scenario_rows(fits,est,period):
    means={key:float(est[col].mean()) for key,col in COLS.items()}
    sds={key:float(est[col].std(ddof=0)) for key,col in COLS.items()}
    med={key:float(est[col].median()) for key,col in COLS.items()}
    rows=[]
    for name,fit in fits.items():
        focal="vix" if name=="M4" else SPECS[name][0]
        raw=est[COLS[focal]].to_numpy(float)
        p10,p90=np.quantile(raw,[.1,.9])
        def matrix(params,raw_value):
            z={key:(med[key]-means[key])/sds[key] for key in SPECS[name]}
            z[focal]=(raw_value-means[focal])/sds[focal]
            X=np.array([[1.0]+[z[key] for key in SPECS[name]]])
            original=fit["params"]
            fit["params"]=params
            try:return transition(fit,X)[0]
            finally:fit["params"]=original
        low=matrix(fit["params"],p10);high=matrix(fit["params"],p90)
        cov=np.asarray(fit["result"].cov_params(),dtype=float)
        for label,cell in [("low_to_high",(0,1)),("high_to_high",(1,1))]:
            def target(p):return float(matrix(p,p90)[cell]-matrix(p,p10)[cell])
            delta=target(fit["params"])
            grad=[]
            for j,p in enumerate(fit["params"]):
                step=1e-5*max(1,abs(p));plus=fit["params"].copy();minus=fit["params"].copy()
                plus[j]+=step;minus[j]-=step
                grad.append((target(plus)-target(minus))/(2*step))
            var=float(np.asarray(grad)@cov@np.asarray(grad))
            se=np.sqrt(var) if np.isfinite(var) and var>=0 else np.nan
            rows.append(dict(period=period,model=name,predictor=focal,transition=label,
                             raw_p10=p10,raw_p90=p90,probability_p10=float(low[cell]),
                             probability_p90=float(high[cell]),difference_pp=100*delta,
                             delta_se_pp=100*se,ci95_low_pp=100*(delta-1.96*se),
                             ci95_high_pp=100*(delta+1.96*se)))
    return pd.DataFrame(rows)


def score_table(f):
    return f.groupby(["period","model"],as_index=False).agg(n=("qlike","size"),
           qlike=("qlike","mean"),mse=("mse","mean"),log_score=("log_score","mean"))


def save_standardized_predictors(d,scaling):
    blocks=[]
    for period,part in scaling.groupby("period"):
        subset=d.loc[~d.split.eq("test")].copy() if period=="validation" else d.copy()
        subset["scaling_period"]=period
        subset["gpr_source_date"]=subset.date-pd.Timedelta(days=1)
        keep=["scaling_period","date","split","gpr_source_date","vix_lag_source_date"]
        for row in part.itertuples():
            subset[f"raw_{row.predictor}"]=subset[row.column]
            subset[f"z_{row.predictor}"]=(subset[row.column]-row.mean)/row.sd
            keep.extend([f"raw_{row.predictor}",f"z_{row.predictor}"])
        blocks.append(subset[keep])
    standardized=pd.concat(blocks,ignore_index=True)
    assert standardized.filter(regex="^z_").notna().all().all()
    standardized.to_csv(TABLES/"07_standardized_predictors.csv",index=False,date_format="%Y-%m-%d")


def paired(f,period,replications=2000,block=10,seed=7207):
    pairs=[("M4","E2"),("M5","E2"),("E1","E2"),("E3","E2"),("M5","E1")]
    wide=f.pivot(index="date",columns="model",values="qlike")
    assert wide.notna().all().all() and set(wide)==set(SPECS)
    n=len(wide);rng=np.random.default_rng(seed)
    starts=rng.integers(0,n,size=(replications,int(np.ceil(n/block))))
    idx=((starts[:,:,None]+np.arange(block))%n).reshape(replications,-1)[:,:n]
    rows=[]
    for left,right in pairs:
        d=(wide[left]-wide[right]).to_numpy(float)
        effect=float(d.mean());boot=d[idx].mean(axis=1);null=boot-effect
        qlow,qhigh=np.quantile(boot,[.025,.975])
        tail_up=(1+int((null>=effect).sum()))/(replications+1)
        tail_down=(1+int((null<=effect).sum()))/(replications+1)
        rows.append(dict(period=period,left=left,right=right,n=n,
                         mean_qlike_left_minus_right=effect,
                         basic_ci95_low=2*effect-qhigh,basic_ci95_high=2*effect-qlow,
                         equal_tail_bootstrap_p=min(1.0,2*min(tail_up,tail_down)),
                         block_length=block,replications=replications,seed=seed))
    return pd.DataFrame(rows)


def run():
    TABLES.mkdir(parents=True,exist_ok=True);FIGURES.mkdir(parents=True,exist_ok=True)
    correction=bootstrap_correction()
    d,raw=prepare_data()
    train=d.loc[d.split.eq("train")].reset_index(drop=True)
    val=d.loc[d.split.eq("validation")].reset_index(drop=True)
    combined=d.loc[d.split.isin(["train","validation"])].reset_index(drop=True)
    test=d.loc[d.split.eq("test")].reset_index(drop=True)
    all_f=[];all_sc=[];all_at=[];all_di=[];all_reg=[];all_params=[];all_scenarios=[]
    for period,est,ev in [("validation",train,val),("exploratory_test",combined,test)]:
        f,sc,at,di,reg,params,fits=forecast_group(est,ev,period)
        all_f.append(f);all_sc.append(sc);all_at.append(at);all_di.append(di)
        all_reg.append(reg);all_params.append(params)
        if period=="validation":all_scenarios.append(scenario_rows(fits,est,period))
        pd.concat(all_at,ignore_index=True).to_csv(TABLES/"07_fit_attempts.csv",index=False)
    forecasts=pd.concat(all_f,ignore_index=True)
    forecasts.to_csv(TABLES/"07_forecasts.csv",index=False,float_format="%.12g",date_format="%Y-%m-%d")
    scaling=pd.concat(all_sc,ignore_index=True);scaling.to_csv(TABLES/"07_scaling.csv",index=False)
    save_standardized_predictors(d,scaling)
    diagnostics=pd.concat(all_di,ignore_index=True);diagnostics.to_csv(TABLES/"07_model_stability.csv",index=False)
    diagnostics.loc[diagnostics.period.eq("validation"),
                    ["model","n","llf","aic","bic","successful_starts","ll_spread"]].to_csv(
                        TABLES/"07_model_comparison.csv",index=False)
    regimes=pd.concat(all_reg,ignore_index=True);regimes.to_csv(TABLES/"07_regime_summary.csv",index=False)
    parameters=pd.concat(all_params,ignore_index=True);parameters.to_csv(TABLES/"07_parameters.csv",index=False)
    scenarios=pd.concat(all_scenarios,ignore_index=True);scenarios.to_csv(TABLES/"07_transition_scenarios.csv",index=False)
    score=score_table(forecasts);score.to_csv(TABLES/"07_forecast_performance.csv",index=False)
    score.loc[score.period.eq("validation")].to_csv(TABLES/"07_validation_forecast_comparison.csv",index=False)
    score.loc[score.period.eq("exploratory_test")].to_csv(TABLES/"07_exploratory_test_comparison.csv",index=False)
    comparisons=pd.concat([paired(forecasts.loc[forecasts.period.eq(p)],p) for p in ("validation","exploratory_test")],ignore_index=True)
    comparisons.to_csv(TABLES/"07_paired_qlike.csv",index=False)
    # An interpretable figure, with each index's own p10-to-p90 change.
    chart=scenarios.loc[scenarios.model.ne("M4")]
    fig,axes=plt.subplots(1,2,figsize=(10,4.2))
    for ax,t in zip(axes,("low_to_high","high_to_high")):
        part=chart.loc[chart.transition.eq(t)].reset_index(drop=True)
        x=np.arange(len(part));y=part.difference_pp.to_numpy()
        lo=part.ci95_low_pp.to_numpy();hi=part.ci95_high_pp.to_numpy()
        ax.errorbar(x,y,yerr=np.vstack([y-lo,hi-y]),fmt="o",capsize=3,color="#315f78")
        ax.axhline(0,color="0.5",lw=.8)
        ax.set_xticks(x,part.model)
        ax.set(title=t.replace("_"," ").title(),ylabel="p90 − p10 (percentage points)")
        ax.grid(axis="y",alpha=.15)
    fig.suptitle("Training-sample geopolitical-risk transition contrasts")
    fig.tight_layout();fig.savefig(FIGURES/"07_transition_contrasts.png",dpi=180,bbox_inches="tight");plt.close(fig)
    metadata=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
                  scipy=scipy.__version__,statsmodels=statsmodels.__version__,seeds=list(SEEDS),
                  fit_settings=FIT_SETTINGS,validation_n=len(val),exploratory_test_n=len(test),
                  historical_reconstruction="AI-GPR constructed in 2026; no verified historical real-time vintages",
                  test_status="exploratory because 2020–2026 outcomes were previously examined")
    (TABLES/"07_reproducibility.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    return dict(correction=correction,coverage=pd.read_csv(TABLES/"07_data_coverage.csv"),
                definitions=pd.read_csv(TABLES/"07_index_definitions.csv"),
                diagnostics=diagnostics,regimes=regimes,parameters=parameters,
                scenarios=scenarios,score=score,comparisons=comparisons,
                attempts=pd.concat(all_at,ignore_index=True))
