"""Stage 05: fixed-parameter, sequential one-step Brent forecast experiment.

Run from the project root. The accompanying notebook calls ``run()`` and
displays the saved diagnostics. All returns are percentage points.
"""
from __future__ import annotations

import json
import platform
import warnings
from pathlib import Path

import arch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import statsmodels
from arch import arch_model
from scipy.special import logsumexp
from scipy.stats import norm, t as student_t
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
SEEDS = (41, 67, 103)
MARKOV_SETTINGS = dict(em_iter=8, search_reps=6, search_iter=8,
                       maxiter=300, disp=False)
SPECS = {"M1": (), "M3": ("gpr",), "M4": ("vix",),
         "M5": ("gpr", "vix")}
MODELS = ("B0", "B1", "B1-t", "M1", "M3", "M4", "M5")


def audit_data(write_audit=True):
    d = pd.read_csv(ROOT / "data/processed/01_brent_complete_case.csv",
                    parse_dates=["date", "previous_observed_date", "vix_lag_source_date"])
    m = pd.read_csv(ROOT / "data/processed/01_brent_max.csv", parse_dates=["date"])
    assert len(d) == 6410 and d.date.is_unique and d.date.is_monotonic_increasing
    assert d.date.min() == pd.Timestamp("2000-01-04")
    assert d.date.max() == pd.Timestamp("2026-10-26")
    assert d.split.value_counts().to_dict() == {"train": 3699, "validation": 999, "test": 1712}
    assert d.brent_return_pct.notna().all() and np.isfinite(d.brent_return_pct).all()
    assert (d.vix_lag_source_date < d.date).all()
    assert (d.previous_observed_date < d.date).all()
    assert (d.observation_gap_calendar_days == (d.date - d.previous_observed_date).dt.days).all()
    assert np.allclose(d.brent_return_pct, 100 * np.log(
        d.brent_price_usd_per_bbl / d.brent_price_usd_per_bbl_previous_observed), atol=1e-8)
    assert d.date.eq(d.date.shift()).sum() == 0
    assert d.date.equals(m.loc[m.date.isin(d.date), "date"].reset_index(drop=True))
    assert d.loc[d.split.eq("train"), "date"].max() <= pd.Timestamp("2015-12-31")
    assert d.loc[d.split.eq("validation"), "date"].between("2016-01-01", "2019-12-31").all()
    assert d.loc[d.split.eq("test"), "date"].ge("2020-01-01").all()
    # Source-date check for GPR previous calendar day, including weekends.
    import xlrd  # required by pandas for the original .xls source
    raw_gpr = pd.read_excel(ROOT / "data/raw/data_gpr_daily_recent.xls", engine="xlrd")
    lookup = pd.Series(pd.to_numeric(raw_gpr.GPRD, errors="coerce").to_numpy(),
                       index=pd.to_datetime(raw_gpr.DAY.astype(str), format="%Y%m%d"))
    expected = (d.date - pd.Timedelta(days=1)).map(lookup)
    assert np.allclose(d.gpr_lag1_calendar_day, expected, atol=1e-5)
    assert np.isfinite(d[["gpr_lag1_calendar_day", "vix_lag1_observation"]]).all().all()
    stage4 = pd.read_csv(TABLES / "04_model_comparison.csv")
    assert set(stage4.model) == set(SPECS)
    assert stage4.observations.eq(3699).all() and stage4.converged.all()
    stage3 = pd.read_csv(TABLES / "03_model_comparison.csv")
    garch_audit = pd.read_csv(TABLES / "03_correction_garch_comparison.csv")
    coverage = pd.read_csv(TABLES / "03_correction_annual_coverage.csv")
    assert stage3.training_observations.eq(3699).all()
    assert set(garch_audit.distribution) == {"normal", "t"} and garch_audit.n.eq(3699).all()
    assert coverage.loc[coverage.year.isin([2000, 2001, 2002]), "complete_case_returns"].tolist() == [197, 189, 188]
    for name, spec in SPECS.items():
        names = set(pd.read_csv(TABLES / f"04_{name.lower()}_parameters.csv").parameter)
        assert all(f"const[{j}]" in names and f"sigma2[{j}]" in names for j in (0, 1))
        if spec:
            assert all(f"p[{j}->0].tvtp0" in names for j in (0, 1))
            assert len(names) == 6 + 2*len(spec)
        else:
            assert {"p[0->0]", "p[1->0]"}.issubset(names) and len(names) == 6
    saved_scale = pd.read_csv(TABLES / "04_training_scaling.csv").set_index("predictor")
    for key, col in [("gpr", "gpr_lag1_calendar_day"), ("vix", "vix_lag1_observation")]:
        x = d.loc[d.split.eq("train"), col]
        assert np.isclose(x.mean(), saved_scale.loc[key, "mean_training"], atol=1e-7)
        assert np.isclose(x.std(ddof=0), saved_scale.loc[key, "sd_training"], atol=1e-7)
    audit = {
        "return_column": "brent_return_pct", "return_units": "percentage points; 100*log(P_t/P_previous observed)",
        "last_common_date": str(d.date.max().date()),
        "sample_counts": d.split.value_counts().to_dict(),
        "gap_over_3_calendar_days": d.groupby("split").observation_gap_calendar_days.apply(lambda x: int((x > 3).sum())).to_dict(),
        "gap_over_5_calendar_days": d.groupby("split").observation_gap_calendar_days.apply(lambda x: int((x > 5).sum())).to_dict(),
        "max_gap_calendar_days": d.groupby("split").observation_gap_calendar_days.max().to_dict(),
        "max_to_complete_case_loss": int(len(m) - len(d)),
        "gpr_lag": "previous calendar day; historical revised series, release timing unverified",
        "vix_lag": "previous nonmissing VIX observation, strictly earlier than target return",
        "stage04_observation_equation": "Gaussian, switching intercept and variance, two states for M1/M3/M4/M5",
        "stage03_audit": "same 3699 training observations in original comparison and corrected normal/t GARCH; 2000-2002 coverage 197/189/188",
        "stage04_transition_orientation": "statsmodels: destination x origin x time; converted to origin x destination",
        "unresolved": ["Irregular observed-price intervals are treated as one observation in the daily return model.",
                       "Revised GPR history has no verified real-time release vintage."],
    }
    if write_audit:
        (TABLES / "05_data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return d


def scale_predictors(est, evaluation, period):
    scales = []
    columns = {}
    for key, col in [("gpr", "gpr_lag1_calendar_day"), ("vix", "vix_lag1_observation")]:
        mean = float(est[col].mean()); sd = float(est[col].std(ddof=0))
        assert sd > 0 and np.isfinite(mean)
        columns[key] = ((est[col].to_numpy() - mean) / sd,
                        (evaluation[col].to_numpy() - mean) / sd)
        scales.append(dict(period=period, predictor=key, mean=mean, sd=sd,
                           fitting_start=str(est.date.min().date()),
                           fitting_end=str(est.date.max().date()), n=len(est)))
    return columns, scales


def fit_markov(name, y, columns, period, attempts):
    spec = SPECS[name]
    X = np.column_stack([np.ones(len(y))] + [columns[c][0] for c in spec]) if spec else None
    model = MarkovRegression(y, k_regimes=2, trend="c", switching_trend=True,
                             switching_variance=True, exog_tvtp=X)
    candidates = []
    for seed in SEEDS:
        np.random.seed(seed)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                res = model.fit(**MARKOV_SETTINGS)
            p = pd.Series(res.params, index=res.model.param_names)
            valid = bool(res.mle_retvals.get("converged", False) and
                         np.isfinite(res.llf) and np.isfinite(res.params).all() and
                         all(p[f"sigma2[{j}]"] > 0 for j in (0, 1)))
            attempts.append(dict(period=period, model=name, seed=seed, valid=valid,
                                 log_likelihood=res.llf, warning_count=len(caught),
                                 warnings=" | ".join(f"{w.category.__name__}: {w.message}" for w in caught), error=""))
            if valid: candidates.append((res.llf, seed, res))
        except Exception as exc:
            attempts.append(dict(period=period, model=name, seed=seed, valid=False,
                                 log_likelihood=np.nan, warning_count=0, warnings="", error=repr(exc)))
    if not candidates:
        raise RuntimeError(f"{period} {name}: no valid converged estimate. See 05_fit_attempts.csv")
    _, seed, res = max(candidates, key=lambda x: x[0])
    p = pd.Series(res.params, index=res.model.param_names)
    order = np.argsort([p[f"sigma2[{j}]"] for j in (0, 1)])
    means = np.array([p[f"const[{j}]"] for j in order])
    variances = np.array([p[f"sigma2[{j}]"] for j in order])
    filtered = np.asarray(res.filtered_marginal_probabilities)[:, order]
    predicted = np.asarray(res.predicted_marginal_probabilities)[:, order]
    raw = np.asarray(res.regime_transition)
    assert raw.shape == (2, 2, 1 if name == "M1" else len(y))
    transition = raw.transpose(2, 1, 0)[:, order][:, :, order]
    assert np.allclose(transition.sum(axis=2), 1, atol=1e-8)
    assert np.allclose(filtered.sum(axis=1), 1, atol=1e-7)
    assert np.allclose(predicted.sum(axis=1), 1, atol=1e-7)
    # This establishes the statsmodels index convention empirically. t uses X_t.
    for j in (1, len(y)//2, len(y)-1):
        P = transition[0 if name == "M1" else j]
        assert np.allclose(filtered[j-1] @ P, predicted[j], atol=1e-7)
    emission = norm.logpdf(y[:, None], loc=means, scale=np.sqrt(variances))
    reconstructed = np.exp(np.log(predicted) + emission -
                           logsumexp(np.log(predicted) + emission, axis=1)[:, None])
    assert np.allclose(reconstructed, filtered, atol=2e-6)
    assert np.isclose(logsumexp(np.log(predicted)+emission, axis=1).sum(), res.llf, atol=1e-4)
    diag = dict(period=period, model=name, n=len(y), fitting_start=None, fitting_end=None,
                llf=float(res.llf), converged=True, successful_starts=len(candidates),
                ll_spread=max(c[0] for c in candidates)-min(c[0] for c in candidates),
                seed=seed, missing_se=int(np.isnan(res.bse).sum()),
                min_variance=float(min(variances)), min_transition=float(transition.min()),
                param_names=" | ".join(res.model.param_names))
    return dict(model=model, result=res, order=order, means=means, variances=variances,
                filtered_last=filtered[-1].copy(), params=np.asarray(res.params), diag=diag)


def fit_garch(name, y, period, attempts):
    dist = "t" if name == "B1-t" else "normal"
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = arch_model(y, mean="Constant", vol="GARCH", p=1, o=0, q=1,
                             dist=dist, rescale=False).fit(disp="off", show_warning=True,
                                                             options={"maxiter": 1000})
        p = res.params
        valid = (res.convergence_flag == 0 and np.isfinite(p).all() and
                 p["omega"] > 0 and p["alpha[1]"] >= 0 and p["beta[1]"] >= 0 and
                 p["alpha[1]"] + p["beta[1]"] < 1 and
                 (dist == "normal" or p["nu"] > 2))
        attempts.append(dict(period=period, model=name, seed=np.nan, valid=valid,
                             log_likelihood=res.loglikelihood, warning_count=len(caught),
                             warnings=" | ".join(f"{w.category.__name__}: {w.message}" for w in caught), error=""))
        if not valid: raise RuntimeError(f"{period} {name}: invalid GARCH estimate")
        h = np.asarray(res.conditional_volatility)**2
        assert np.isfinite(h).all() and np.all(h > 0)
        if dist == "normal":
            check_ll = np.sum(norm.logpdf(y, loc=p["mu"], scale=np.sqrt(h)))
        else:
            nu = float(p["nu"])
            z = (y-p["mu"])/np.sqrt(h)*np.sqrt(nu/(nu-2))
            check_ll = np.sum(student_t.logpdf(z, df=nu) + .5*np.log(nu/(nu-2)) - .5*np.log(h))
        assert np.isclose(check_ll, res.loglikelihood, atol=1e-5)
        for j in (1, len(y)//2, len(y)-1):
            expected = p["omega"] + p["alpha[1]"]*(y[j-1]-p["mu"])**2 + p["beta[1]"]*h[j-1]
            assert np.isclose(h[j], expected, rtol=1e-6, atol=1e-6)
        diag = dict(period=period, model=name, n=len(y), fitting_start=None, fitting_end=None,
                    llf=float(res.loglikelihood), converged=True, successful_starts=1,
                    ll_spread=0, seed=np.nan, missing_se=int(np.isnan(res.std_err).sum()),
                    min_variance=float(h.min()), min_transition=np.nan,
                    param_names=" | ".join(p.index),
                    alpha_plus_beta=float(p["alpha[1]"]+p["beta[1]"]),
                    near_integrated=bool(p["alpha[1]"]+p["beta[1]"] >= .995),
                    student_t_df=float(p["nu"]) if dist == "t" else np.nan)
        return dict(result=res, params=p, last_h=h[-1], diag=diag)
    except Exception:
        if not attempts or attempts[-1]["model"] != name or attempts[-1]["period"] != period:
            attempts.append(dict(period=period, model=name, seed=np.nan, valid=False,
                                 log_likelihood=np.nan, warning_count=0, warnings="", error="fit exception"))
        raise


def markov_transitions(fit, columns, n):
    spec = SPECS[fit["diag"]["model"]]
    X = np.column_stack([np.ones(n)] + [columns[c][1] for c in spec]) if spec else None
    raw = np.asarray(fit["model"].regime_transition_matrix(fit["params"], exog_tvtp=X))
    assert raw.shape == (2, 2, n if spec else 1)
    P = raw.transpose(2, 1, 0)[:, fit["order"]][:, :, fit["order"]]
    if not spec: P = np.repeat(P, n, axis=0)
    assert np.isfinite(P).all() and np.all((P >= 0) & (P <= 1))
    assert np.allclose(P.sum(axis=2), 1, atol=1e-9)
    return P


def forecast_block(est, ev, period, model_set=MODELS, return_fits=False):
    assert est.date.max() < ev.date.min() and est.date.is_monotonic_increasing and ev.date.is_monotonic_increasing
    y = est.brent_return_pct.to_numpy(float)
    y_ev = ev.brent_return_pct.to_numpy(float)
    columns, scales = scale_predictors(est, ev, period)
    attempts, diagnostics, parameters, rows = [], [], [], []
    fits = {}
    mu0 = float(y.mean()); v0 = float(np.mean((y-mu0)**2))
    if "B0" in model_set:
        fits["B0"] = dict(mu=mu0, variance=v0)
        diagnostics.append(dict(period=period, model="B0", n=len(y), fitting_start=str(est.date.min().date()),
                                fitting_end=str(est.date.max().date()), llf=float(np.sum(norm.logpdf(y, mu0, np.sqrt(v0)))),
                                converged=True, successful_starts=1, ll_spread=0, seed=np.nan,
                                missing_se=0, min_variance=v0, min_transition=np.nan, param_names="mu | sigma2"))
        parameters.extend([dict(period=period, model="B0", parameter="mu", estimate=mu0),
                           dict(period=period, model="B0", parameter="sigma2", estimate=v0)])
    for name in ("B1", "B1-t"):
        if name in model_set:
            fits[name] = fit_garch(name, y, period, attempts)
            diagnostics.append(fits[name]["diag"])
            parameters.extend(dict(period=period, model=name, parameter=k, estimate=float(v))
                              for k, v in fits[name]["params"].items())
    for name in SPECS:
        if name in model_set:
            fits[name] = fit_markov(name, y, columns, period, attempts)
            diagnostics.append(fits[name]["diag"])
            parameters.extend(dict(period=period, model=name, parameter=k, estimate=float(v))
                              for k, v in zip(fits[name]["result"].model.param_names,
                                              fits[name]["params"]))
    for diag in diagnostics:
        diag["fitting_start"] = str(est.date.min().date())
        diag["fitting_end"] = str(est.date.max().date())
    transitions = {name: markov_transitions(fits[name], columns, len(ev))
                   for name in SPECS if name in fits}
    states = {name: fits[name]["filtered_last"].copy() for name in transitions}
    h_previous = {name: fits[name]["last_h"] for name in ("B1", "B1-t") if name in fits}
    previous_y = float(y[-1])
    for i, (date, realized) in enumerate(zip(ev.date, y_ev)):
        forecast_at = est.date.iloc[-1] if i == 0 else ev.date.iloc[i-1]
        common = dict(period=period, date=date, forecast_information_through=forecast_at,
                      observed_return_pct=realized, observed_squared_return_pct2=realized**2,
                      observation_gap_calendar_days=int(ev.observation_gap_calendar_days.iloc[i]))
        for name in model_set:
            if name == "B0":
                mu, variance = mu0, v0
                logdensity = float(norm.logpdf(realized, mu, np.sqrt(variance)))
                q = [np.nan, np.nan]
            elif name in ("B1", "B1-t"):
                p = fits[name]["params"]
                h = float(p["omega"] + p["alpha[1]"]*(previous_y-p["mu"])**2 + p["beta[1]"]*h_previous[name])
                mu, variance = float(p["mu"]), h
                if name == "B1": logdensity = float(norm.logpdf(realized, mu, np.sqrt(h)))
                else:
                    nu = float(p["nu"])
                    z = (realized-mu)/np.sqrt(h)*np.sqrt(nu/(nu-2))
                    logdensity = float(student_t.logpdf(z, df=nu) + .5*np.log(nu/(nu-2)) - .5*np.log(h))
                q = [np.nan, np.nan]
            else:
                fit = fits[name]
                q = states[name] @ transitions[name][i]
                assert np.isfinite(q).all() and np.all((q >= 0) & (q <= 1)) and np.isclose(q.sum(), 1, atol=1e-9)
                means, vars_ = fit["means"], fit["variances"]
                mu = float(q @ means)
                second = float(q @ (vars_ + means**2))
                variance = float(second - mu**2)
                assert np.isclose(mu, np.sum(q*means))
                assert np.isclose(second, np.sum(q*(vars_+means**2)))
                assert np.isclose(variance, second-mu**2)
                logweights = np.log(q) + norm.logpdf(realized, means, np.sqrt(vars_))
                logdensity = float(logsumexp(logweights))
            if name in ("B0", "B1", "B1-t"): second = variance + mu**2
            assert np.isfinite([mu, variance, second, logdensity]).all() and variance > 0 and second > 0
            rows.append(dict(**common, model=name, predicted_mean_pct=mu,
                             predicted_variance_pct2=variance, predicted_second_moment_pct2=second,
                             predicted_volatility_pct=np.sqrt(variance),
                             predicted_low_probability=q[0], predicted_high_probability=q[1],
                             log_predictive_density=logdensity,
                             qlike=np.log(second)+realized**2/second,
                             squared_error=(second-realized**2)**2,
                             absolute_error=abs(second-realized**2),
                             negative_log_score=-logdensity))
        # All model forecasts at t have already been recorded. Now assimilate y_t.
        for name in transitions:
            q = states[name] @ transitions[name][i]
            fit = fits[name]
            logw = np.log(q) + norm.logpdf(realized, fit["means"], np.sqrt(fit["variances"]))
            states[name] = np.exp(logw-logsumexp(logw))
            assert np.isclose(states[name].sum(), 1, atol=1e-9)
        for name in h_previous:
            h_previous[name] = next(r["predicted_variance_pct2"] for r in rows[-len(model_set):] if r["model"] == name)
        previous_y = float(realized)
    out = pd.DataFrame(rows)
    assert out.groupby("model").date.nunique().eq(len(ev)).all()
    assert all(out.loc[out.model.eq(name), "date"].reset_index(drop=True).equals(ev.date.reset_index(drop=True)) for name in model_set)
    assert (out.forecast_information_through < out.date).all()
    assert out.groupby("date").size().eq(len(model_set)).all()
    assert out[["predicted_variance_pct2", "predicted_second_moment_pct2", "qlike", "negative_log_score"]].notna().all().all()
    assert not any("smoothed" in key for fit in fits.values() for key in fit.keys())
    outputs = (out, scales, attempts, diagnostics, parameters)
    return (*outputs, fits) if return_fits else outputs


def performance(forecasts, groupcols):
    return forecasts.groupby(groupcols, as_index=False).agg(
        n=("qlike", "size"), qlike=("qlike", "mean"),
        mse=("squared_error", "mean"), log_score=("negative_log_score", "mean"),
        mae=("absolute_error", "mean"))


def compare(forecasts, period, pairs=None, bootstrap=2000, block=10, seed=7205):
    if pairs is None:
        pairs = [("M1", "M3"), ("M1", "M4"), ("M1", "M5"),
                 ("M4", "M5"), ("B1", "M5"), ("B1-t", "M5")]
    wide = forecasts.pivot(index="date", columns="model", values="qlike")
    rng = np.random.default_rng(seed)
    n = len(wide)
    assert wide.notna().all().all()
    starts = rng.integers(0, n, size=(bootstrap, int(np.ceil(n/block))))
    indices = (starts[:,:,None] + np.arange(block)) % n
    indices = indices.reshape(bootstrap,-1)[:,:n]
    rows=[]
    for left,right in pairs:
        if left not in wide or right not in wide: continue
        d = (wide[left]-wide[right]).to_numpy()
        observed = float(d.mean())
        boot = d[indices].mean(axis=1)
        qlow,qhigh = np.quantile(boot,[.025,.975])
        # Basic interval inverts the same centered bootstrap distribution as p.
        lo,hi = 2*observed-qhigh, 2*observed-qlow
        null = (d-d.mean())[indices].mean(axis=1)
        p = (1+int((np.abs(null)>=abs(observed)).sum()))/(bootstrap+1)
        rows.append(dict(period=period, left=left, right=right, n=n,
                         mean_qlike_left_minus_right=observed, basic_ci95_low=lo, basic_ci95_high=hi,
                         centered_block_bootstrap_p=p, block_length=block,
                         bootstrap_replicates=bootstrap, seed=seed))
    return pd.DataFrame(rows)


def figures(forecasts, yearly, comparisons):
    test = forecasts.loc[forecasts.period.eq("test")].copy()
    colors={"B1":"#22577a","B1-t":"#4d908e","M1":"#bb6b39","M5":"#7a4e86"}
    p = test.pivot(index="date", columns="model", values="predicted_second_moment_pct2")
    actual = test.drop_duplicates("date").set_index("date").observed_squared_return_pct2
    fig,ax=plt.subplots(figsize=(12,4))
    ax.plot(actual.index, actual.clip(upper=actual.quantile(.99)),color="0.55",lw=.6,label="Actual squared return (display capped at p99)")
    for name in ("B1-t","M1","M5"):
        ax.plot(p.index,p[name],lw=.9,label=name,color=colors[name])
    ax.set(title="Test-period squared Brent returns and one-step second-moment forecasts",ylabel="Return squared (percentage points²)",xlabel="Date")
    ax.legend(ncol=2);ax.grid(alpha=.15);fig.tight_layout();fig.savefig(FIGURES/"05_actual_vs_forecast.png",dpi=180);plt.close(fig)
    v=test.pivot(index="date",columns="model",values="predicted_volatility_pct")
    fig,ax=plt.subplots(figsize=(12,4))
    for name in ("B1","B1-t","M1","M5"):
        ax.plot(v.index,v[name],lw=.9,label=name,color=colors[name])
    ax.set(title="One-step predicted Brent return volatility, fixed 2019-end parameters",ylabel="SD (percentage points)",xlabel="Date")
    ax.legend(ncol=4);ax.grid(alpha=.15);fig.tight_layout();fig.savefig(FIGURES/"05_predicted_volatility.png",dpi=180);plt.close(fig)
    loss=test.pivot(index="date",columns="model",values="qlike")
    fig,ax=plt.subplots(figsize=(12,4))
    for name in ("B1","B1-t","M3","M4","M5"):
        ax.plot(loss.index,(loss[name]-loss["M1"]).cumsum(),lw=.95,label=name)
    ax.axhline(0,color="0.4",lw=.7)
    ax.set(title="Cumulative QLIKE difference versus M1 (negative favors comparison model)",ylabel="Cumulative loss difference",xlabel="Date")
    ax.legend(ncol=5);ax.grid(alpha=.15);fig.tight_layout();fig.savefig(FIGURES/"05_cumulative_qlike_difference.png",dpi=180);plt.close(fig)
    sub=yearly.loc[yearly.model.isin(["B1-t","M1","M4","M5"])]
    pivot=sub.pivot(index="year",columns="model",values="qlike")
    fig,ax=plt.subplots(figsize=(10,4))
    pivot.plot.bar(ax=ax,width=.75)
    ax.set(title="Test QLIKE by calendar year",ylabel="Mean QLIKE",xlabel="Year")
    ax.grid(axis="y",alpha=.15);fig.tight_layout();fig.savefig(FIGURES/"05_yearly_qlike.png",dpi=180);plt.close(fig)
    dis=test.assign(disruption=np.select([test.date.dt.year.eq(2020),test.date.dt.year.eq(2022)],
                                          ["2020 oil-market disruption","2022 geopolitical disruption"],default="Other test years"))
    d=performance(dis,["disruption","model"])
    d=d.loc[d.model.isin(["B1-t","M1","M4","M5"])]
    fig,ax=plt.subplots(figsize=(10,4))
    d.pivot(index="disruption",columns="model",values="qlike").plot.bar(ax=ax,rot=0,width=.7)
    ax.set(title="Forecast losses in predefined disruption years",ylabel="Mean QLIKE",xlabel="")
    ax.grid(axis="y",alpha=.15);fig.tight_layout();fig.savefig(FIGURES/"05_disruption_qlike.png",dpi=180);plt.close(fig)
    return d


def run():
    TABLES.mkdir(parents=True,exist_ok=True); FIGURES.mkdir(parents=True,exist_ok=True)
    d=audit_data()
    train=d.loc[d.split.eq("train")].reset_index(drop=True)
    val=d.loc[d.split.eq("validation")].reset_index(drop=True)
    est_final=d.loc[d.split.isin(["train","validation"])].reset_index(drop=True)
    test=d.loc[d.split.eq("test")].reset_index(drop=True)
    all_fc=[];all_scale=[];all_attempts=[];all_diag=[];all_params=[]
    for name,est,ev in [("validation",train,val),("test",est_final,test)]:
        fc,sc,at,di,pa=forecast_block(est,ev,name)
        all_fc.append(fc);all_scale.extend(sc);all_attempts.extend(at);all_diag.extend(di);all_params.extend(pa)
        pd.DataFrame(all_attempts).to_csv(TABLES/"05_fit_attempts.csv",index=False)
    forecasts=pd.concat(all_fc,ignore_index=True)
    forecasts.to_csv(TABLES/"05_forecasts.csv",index=False,float_format="%.12g",date_format="%Y-%m-%d")
    pd.DataFrame(all_scale).to_csv(TABLES/"05_scaling.csv",index=False)
    pd.DataFrame(all_diag).to_csv(TABLES/"05_fit_diagnostics.csv",index=False)
    pd.DataFrame(all_params).to_csv(TABLES/"05_fitted_parameters.csv",index=False)
    perf=performance(forecasts,["period","model"])
    perf.to_csv(TABLES/"05_performance.csv",index=False)
    yearly=performance(forecasts.loc[forecasts.period.eq("test")].assign(year=lambda x:x.date.dt.year),["year","model"])
    yearly.to_csv(TABLES/"05_test_yearly_performance.csv",index=False)
    comparisons=pd.concat([compare(forecasts.loc[forecasts.period.eq(k)],k) for k in ("validation","test")],ignore_index=True)
    comparisons.to_csv(TABLES/"05_paired_block_bootstrap.csv",index=False)
    disruptions=figures(forecasts,yearly,comparisons)
    disruptions.to_csv(TABLES/"05_disruption_performance.csv",index=False)
    # Extreme-return influence is descriptive; primary scores retain every observation.
    cutoff=test.brent_return_pct.abs().quantile(.99)
    test_fc=forecasts.loc[forecasts.period.eq("test")].copy()
    test_fc["top_one_percent_abs_return"] = test_fc.observed_return_pct.abs().ge(cutoff)
    influence=performance(test_fc,["top_one_percent_abs_return","model"])
    influence.to_csv(TABLES/"05_extreme_return_influence.csv",index=False)
    impact=[]
    for model,part in test_fc.groupby("model"):
        top=part.loc[part.top_one_percent_abs_return]
        year2020=part.loc[part.date.dt.year.eq(2020)]
        impact.append(dict(model=model, full_test_n=len(part), top_one_percent_n=len(top),
                           top_one_percent_qlike_sum_share=float(top.qlike.sum()/part.qlike.sum()),
                           qlike_full=float(part.qlike.mean()),
                           qlike_excluding_top_one_percent_diagnostic=float(part.loc[~part.top_one_percent_abs_return,"qlike"].mean()),
                           year2020_n=len(year2020), year2020_qlike_sum_share=float(year2020.qlike.sum()/part.qlike.sum()),
                           qlike_excluding_2020_diagnostic=float(part.loc[~part.date.dt.year.eq(2020),"qlike"].mean())))
    pd.DataFrame(impact).to_csv(TABLES/"05_extreme_return_impact.csv",index=False)
    # Sensitivity is developed solely against the validation period, without test data.
    clean=train.loc[train.date.ge("2007-01-01")].reset_index(drop=True)
    sens,sc,at,di,pa=forecast_block(clean,val,"validation_2007_start")
    sensperf=performance(sens,["period","model"])
    sensperf.to_csv(TABLES/"05_2007_start_validation_performance.csv",index=False)
    sens.to_csv(TABLES/"05_2007_start_validation_forecasts.csv",index=False,float_format="%.12g",date_format="%Y-%m-%d")
    pd.DataFrame(at).to_csv(TABLES/"05_2007_start_fit_attempts.csv",index=False)
    pd.DataFrame(di).to_csv(TABLES/"05_2007_start_fit_diagnostics.csv",index=False)
    pd.DataFrame(sc).to_csv(TABLES/"05_2007_start_scaling.csv",index=False)
    metadata=dict(software={"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,
                            "scipy":scipy.__version__,"statsmodels":statsmodels.__version__,"arch":arch.__version__,
                            "matplotlib":matplotlib.__version__},
                  seeds=list(SEEDS),markov_fit_settings=MARKOV_SETTINGS,bootstrap_seed=7205,
                  forecast_method="fixed parameters; sequential filtered Markov update and GARCH variance recursion",
                  gpr_limitation="Historical revised GPR; pseudo-out-of-sample, not verified real-time",
                  primary_metric="mean(log(predicted conditional second moment) + return squared / predicted conditional second moment)",
                  disruption_periods={"2020":"2020-01-01 through 2020-12-31", "2022":"2022-01-01 through 2022-12-31"},
                  extreme_cutoff_abs_return_pct=float(cutoff),
                  figure_squared_return_display_cap="p99 for visibility only; no observations excluded from metrics")
    (TABLES/"05_reproducibility.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    return dict(performance=perf,comparisons=comparisons,yearly=yearly,
                diagnostics=pd.DataFrame(all_diag), sensitivity=sensperf,
                disruptions=disruptions,extremes=influence)
