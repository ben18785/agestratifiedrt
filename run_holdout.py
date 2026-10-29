"""Hold-out check of the England fits that use the ONS Infection Survey.

For each period the survey windows are split into two halves (odd and even windows). Each
model is fitted with one half left out (those windows get 0 people tested, so they drop out of
the likelihood) and predicts the positives in the windows it did not see. Every window is
predicted once, by a fit that never saw it.

Models (keys from FITS in run_england.py): by default totals_ons and comix_weekly_ons; any fit
that uses the survey can be named with --models. Predictions are scored against the
observed positives, and compared with a baseline that interpolates the observed positivity
(log-linearly) between the neighbouring windows, which were seen.

Usage:
  python run_holdout.py PERIOD SPLIT [--models KEY,KEY]   # fit one half (SPLIT 0 or 1)
  python run_holdout.py --summarise      # combine all periods and splits into a table
"""
import sys

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import binom

from common import ROOT

DEFAULT_MODELS = ["totals_ons", "comix_weekly_ons"]


def score(draws_pct, n_tested, n_pos, rng):
    """Predictive checks for one window and group from draws of positivity (%)."""
    p = np.clip(draws_pct / 100, 1e-9, 1 - 1e-9)
    y = rng.binomial(n_tested, p)                       # posterior predictive positives
    lpd = logsumexp(binom.logpmf(n_pos, n_tested, p)) - np.log(len(p))
    lo, hi = np.quantile(y, [0.025, 0.975])
    pit = np.mean(y < n_pos) + 0.5 * np.mean(y == n_pos)
    return dict(pred_median_pct=100 * np.median(y) / n_tested, pred_lo_pct=100 * lo / n_tested,
                pred_hi_pct=100 * hi / n_tested, covered=bool(lo <= n_pos <= hi), pit=pit, log_pred_density=lpd)


def fit_split(split, keys):
    import run_england as e                             # reads the period from sys.argv
    dat = e.load()
    rng = np.random.default_rng(10 + split)
    for key in keys:
        rows = []
        name, opt = e.NAME[key], e.FITS[key]["opt"]
        data = e.stan_data(dat, **opt)
        K = data["K"]
        held = [k for k in range(K) if k % 2 == split]
        n_tst, n_pos = np.array(data["n_tested"]), np.array(data["n_pos"])
        fit_data = dict(data, n_tested=n_tst.copy(), n_pos=n_pos.copy())
        fit_data["n_tested"][held] = 0
        fit_data["n_pos"][held] = 0
        fit_data["n_tested"], fit_data["n_pos"] = fit_data["n_tested"].tolist(), fit_data["n_pos"].tolist()
        print(e.PERIOD, split, name, "holding out windows", held, flush=True)
        f, _ = e.fit(dat, **opt, data=fit_data)
        prev = f.stan_variable("prev_pct")              # draws x K x N
        obs_pct = 100 * n_pos / np.maximum(n_tst, 1)
        groups = ["all"] if opt["N"] == 1 else e.GROUPS
        for k in held:
            for j, g in enumerate(groups):
                if n_tst[k, j] == 0:
                    continue
                # baseline: log-linear interpolation between the nearest seen windows with data
                seen = [kk for kk in range(K) if kk not in held and n_tst[kk, j] > 0]
                before, after = [kk for kk in seen if kk < k], [kk for kk in seen if kk > k]
                lp = lambda kk: np.log((n_pos[kk, j] + 0.5) / n_tst[kk, j])
                if before and after:
                    a, b = before[-1], after[0]
                    base = np.exp(lp(a) + (lp(b) - lp(a)) * (k - a) / (b - a))
                else:
                    base = np.exp(lp((before or after)[-1 if before else 0]))
                rows.append(dict(period=e.PERIOD, split=split, key=key, model=name, window=k,
                                 start=dat["days"][data["win_start"][k] - 1].date(), group=g,
                                 n_tested=int(n_tst[k, j]), n_pos=int(n_pos[k, j]), observed_pct=obs_pct[k, j],
                                 interior=bool(before and after), baseline_pct=100 * base,
                                 **score(prev[:, k, j], int(n_tst[k, j]), int(n_pos[k, j]), rng)))
        out = pd.DataFrame(rows)
        out.to_csv(ROOT / "results" / "holdout" / f"{e.PERIOD}_{split}_{key}.csv", index=False)
        print(out.round(3).to_string(), flush=True)


def summarise():
    files = sorted((ROOT / "results" / "holdout").glob("*.csv"))
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    obs = (d.n_pos + 0.5) / d.n_tested * 100
    d["abs_log_err_model"] = np.abs(np.log(d.pred_median_pct.clip(lower=1e-3) / obs))
    d["abs_log_err_baseline"] = np.abs(np.log(d.baseline_pct / obs))
    tab = (d.groupby(["period", "model"])
           .agg(windows=("covered", "size"), coverage_95=("covered", "mean"), mean_pit=("pit", "mean"),
                mae_log_model=("abs_log_err_model", "mean"), mae_log_baseline=("abs_log_err_baseline", "mean"),
                model_better=("abs_log_err_model", lambda x: np.mean(x < d.loc[x.index, "abs_log_err_baseline"])))
           .round(2).reset_index())
    tab.to_csv(ROOT / "results" / "holdout_table.csv", index=False)
    d.to_csv(ROOT / "results" / "holdout_all.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(tab)
    return d


if __name__ == "__main__":
    (ROOT / "results" / "holdout").mkdir(parents=True, exist_ok=True)
    if "--summarise" in sys.argv:
        summarise()
    else:
        keys = sys.argv[sys.argv.index("--models") + 1].split(",") if "--models" in sys.argv else DEFAULT_MODELS
        fit_split(int(sys.argv[2]), keys)
