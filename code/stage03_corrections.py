"""Reproducible Stage 03 coverage audit, Student-t GARCH, and 10-start M2 audit."""
from pathlib import Path
import json, warnings
import numpy as np, pandas as pd
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox,het_arch
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

root=Path(__file__).resolve().parents[1]
data=root/"data"; out=root/"results"/"tables"; out.mkdir(parents=True,exist_ok=True)
raw=pd.read_csv(data/"raw"/"DCOILBRENTEU.csv",parse_dates=["observation_date"])
raw["price"]=pd.to_numeric(raw.DCOILBRENTEU,errors="coerce")
maximum=pd.read_csv(data/"processed"/"01_brent_max.csv",parse_dates=["date","previous_observed_date"])
complete=pd.read_csv(data/"processed"/"01_brent_complete_case.csv",parse_dates=["date"])
assert len(complete)==6410 and complete.split.eq("train").sum()==3699
rows=[]
for year in range(2000,2016):
    r=raw.loc[raw.observation_date.dt.year.eq(year)]
    m=maximum.loc[maximum.date.dt.year.eq(year)]
    c=complete.loc[complete.date.dt.year.eq(year)]
    assert set(r.loc[r.price.notna(),"observation_date"])==set(m.date)
    assert set(c.date).issubset(set(m.date))
    assert len(m)-len(c)==m.vix_index.isna().sum()
    rows.append({"year":year,"raw_weekday_rows":len(r),"raw_nonmissing_prices":r.price.notna().sum(),
                 "raw_missing_prices":r.price.isna().sum(),"maximum_returns":len(m),
                 "complete_case_returns":len(c),"lost_to_missing_vix":len(m)-len(c),
                 "return_gaps_gt_3_days":int(m.observation_gap_calendar_days.gt(3).sum()),
                 "maximum_return_gap_days":int(m.observation_gap_calendar_days.max())})
coverage=pd.DataFrame(rows)
coverage.to_csv(out/"03_correction_annual_coverage.csv",index=False)

y=complete.loc[complete.split.eq("train"),"brent_return_pct"].to_numpy()
assert len(y)==3699 and np.isfinite(y).all()
grows=[]
for dist in ["normal","t"]:
    fit=arch_model(y,mean="Constant",vol="GARCH",p=1,o=0,q=1,dist=dist,rescale=False).fit(
        disp="off",show_warning=False,options={"maxiter":1000})
    p=fit.params
    z=np.asarray(fit.std_resid)
    rows=[]
    for label,v in [("standardized",z),("squared_standardized",z*z)]:
        lb=acorr_ljungbox(v,lags=[5,10,20],return_df=True,model_df=0)
        for lag,row in lb.iterrows():
            rows.append({"distribution":dist,"series":label,"test":"Ljung-Box","lag":lag,
                         "statistic":row.lb_stat,"p_value":row.lb_pvalue})
    lm,lmp,_,_=het_arch(z-z.mean(),nlags=10,ddof=1,result_object=False)
    rows.append({"distribution":dist,"series":"standardized","test":"ARCH-LM","lag":10,
                 "statistic":lm,"p_value":lmp})
    grows.extend(rows)
    pd.DataFrame({"parameter":p.index,"estimate":p.values,"std_error":fit.std_err.values,
                  "t_stat":fit.tvalues.values,"p_value":fit.pvalues.values}).to_csv(
        out/f"03_correction_garch_{dist}_parameters.csv",index=False)
    near=float(p["alpha[1]"]+p["beta[1]"])
    record={"distribution":dist,"n":len(y),"log_likelihood":fit.loglikelihood,"aic":fit.aic,
            "bic":fit.bic,"alpha_plus_beta":near,"near_unit_persistence":near>=.995,
            "convergence_flag":fit.convergence_flag,
            "residual_arch_lm_p10":lmp}
    if dist=="normal": normal=record
    else: student=record
pd.DataFrame([normal,student]).to_csv(out/"03_correction_garch_comparison.csv",index=False)
pd.DataFrame(grows).to_csv(out/"03_correction_garch_residual_tests.csv",index=False)

m=MarkovRegression(y,k_regimes=3,trend="c",switching_trend=True,switching_variance=True)
attempts=[]; fits=[]
for seed in [41,67,103,131,173,211,257,307,353,401]:
    np.random.seed(seed)
    try:
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            fit=m.fit(em_iter=10,search_reps=8,search_iter=8,maxiter=300,disp=False)
        p=pd.Series(fit.params,index=fit.model.param_names)
        v=np.array([p[f"sigma2[{i}]"] for i in range(3)])
        good=bool(fit.mle_retvals.get("converged",False) and np.isfinite(fit.llf)
                  and np.isfinite(fit.params).all() and np.all(v>0))
        attempts.append({"seed":seed,"log_likelihood":fit.llf,"converged":good,
                         "warnings":" | ".join(str(w.message) for w in ws),"error":""})
        if good: fits.append((fit.llf,seed,fit))
    except Exception as e:
        attempts.append({"seed":seed,"log_likelihood":np.nan,"converged":False,
                         "warnings":"","error":repr(e)})
pd.DataFrame(attempts).to_csv(out/"03_correction_m2_ten_starts.csv",index=False)
if not fits: raise RuntimeError("No valid M2 fit in ten starts")
ll,seed,best=max(fits,key=lambda q:q[0])
p=pd.Series(best.params,index=best.model.param_names)
order=np.argsort([p[f"sigma2[{i}]"] for i in range(3)])
P=best.regime_transition[:,:,0].T[np.ix_(order,order)]
q=np.asarray(best.smoothed_marginal_probabilities)[:,order]
f=np.asarray(best.filtered_marginal_probabilities)[:,order]
assert np.allclose(P.sum(axis=1),1) and np.allclose(q.sum(axis=1),1) and np.allclose(f.sum(axis=1),1)
summary=pd.DataFrame([{"label":label,"original_state":int(i),"mean":p[f"const[{i}]"],
                       "variance":p[f"sigma2[{i}]"],"sd":np.sqrt(p[f"sigma2[{i}]"]),
                       "stay_probability":P[j,j],"expected_duration":1/(1-P[j,j]),
                       "mean_smoothed_occupancy":q[:,j].mean(),
                       "most_likely_days":int((q.argmax(axis=1)==j).sum())}
                      for j,(i,label) in enumerate(zip(order,["Low","Medium","High"]))])
summary.to_csv(out/"03_correction_m2_regimes.csv",index=False)
pd.DataFrame(P,index=["Low","Medium","High"],columns=["Low","Medium","High"]).to_csv(
    out/"03_correction_m2_transition_matrix.csv")
pd.DataFrame({"parameter":p.index,"estimate":p.values,"std_error":best.bse}).to_csv(
    out/"03_correction_m2_parameters.csv",index=False)
audit={"successful_starts":len(fits),"attempted_starts":10,"chosen_seed":seed,
       "loglikelihood_spread":max(q[0] for q in fits)-min(q[0] for q in fits),
       "unavailable_standard_errors":int(np.isnan(best.bse).sum()),
       "minimum_transition":float(P.min()),"min_occupancy":float(q.mean(axis=0).min())}
(out/"03_correction_gate.json").write_text(json.dumps(audit,indent=2))
print("coverage early:",coverage.head(3).to_string(index=False))
print("GARCH:",normal,student)
print("M2 gate:",audit)
