"""Reporting that changes over time, and whether an infection survey can identify it.

Two groups: group 1 ("children") and group 2 ("adults"). Contacts within group 1 rise from
2 to 8 on day 100 and every fit is told this (exact contact matrices, known date), so the only
difficulty is reporting. The true R_t = 1.1 throughout. Adults are reported with probability
0.6 throughout; children's reporting rises from 0.2 to 0.5 (as when mass testing started in
schools):
  R1: step on day 100, the same day as the contact change;
  R2: ramp over days 100-128;
  R3: step on day 130, a month after the contact change.
Fits (all survey shape x multipliers, random walk on R_t):
  - constant reporting (the model in the note);
  - weekly reporting estimated from cases only;
  - weekly reporting estimated from cases plus a random-sample infection survey
    (2000 people per group per fortnight, positive while infection is detectable);
  - single group fitted to reported totals.
Usage: python run_reporting.py [--plot-only]
"""
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gamma

from common import (ROOT, T, S, NB, BLOCK, BLOCK_START, R_constant, gen_time, simulate_general, model,
                    _diagnostics, fit_single, summarise_R, COLOURS)

p = np.array([1.0, 0.5])
C1 = np.array([[2, 4.8], [1.2, 3.84]])
C2 = np.array([[8, 4.8], [1.2, 3.84]])
w = np.vstack([gen_time(), gen_time()])
NPOP = np.array([5e5, 1.5e6])                   # population of each group
DAYS = np.arange(1, T + 1)
QWEEK = (DAYS - 1) // 7 + 1
D = 40                                          # days an infection can stay detectable
DET = 1 - gamma.cdf(np.arange(D) + 0.5, a=(12 / 5) ** 2, scale=25 / 12)   # P(positive u days after infection): mean 12, sd 5
WIN_START = np.arange(61, T - 12, 14)           # 14-day survey windows from day 61
N_TESTED = 2000

datasets = {
    "R1": dict(q1=lambda d: 0.2 if d < 100 else 0.5, change=100, title="R1: reporting step on day 100"),
    "R2": dict(q1=lambda d: 0.2 + 0.3 * np.clip((d - 100) / 28, 0, 1), change=100, title="R2: reporting ramp, days 100-128"),
    "R3": dict(q1=lambda d: 0.2 if d < 130 else 0.5, change=130, title="R3: reporting step on day 130"),
}
FITS = {"Constant reporting (note's model)": dict(est_q=0, ons=0),
        "Weekly reporting, cases only": dict(est_q=1, ons=0),
        "Weekly reporting, cases + infection survey": dict(est_q=1, ons=1)}
COL = {"Single group, reported totals": COLOURS["Single group, reported totals"],
       "Constant reporting (note's model)": "#3b75af",
       "Weekly reporting, cases only": "#a0711c",
       "Weekly reporting, cases + infection survey": "#1b3f6b"}


def survey(X, rng):
    """Positives among N_TESTED random people per group in each 14-day window."""
    prev = np.zeros((T, 2))
    for t in range(D - 1, T):
        prev[t] = DET @ X[t::-1][:D] / NPOP
    pos = np.array([[rng.binomial(N_TESTED, prev[s - 1:s + 13, j].mean()) for j in range(2)] for s in WIN_START])
    return prev, pos


def fit(Y, pos, est_q, ons):
    data = dict(T=T, N=2, S=S, Y=Y.astype(int).tolist(), w=w, B=NB, block=BLOCK.tolist(), nP=2,
                period=[2 if d >= 100 else 1 for d in DAYS], C=np.stack([C1, C2]), est_q=est_q,
                W=int(QWEEK.max()), qweek=QWEEK.tolist(), ons=ons, Npop=NPOP, D=D, d=DET,
                K=len(WIN_START), win_start=WIN_START.tolist(), win_end=(WIN_START + 13).tolist(),
                n_tested=[[N_TESTED] * 2] * len(WIN_START), n_pos=pos.tolist())
    inits = {"inv_phi": 0.05, "tau": 0.05}
    if est_q:
        inits["tau_q"] = [0.1]
    f = model("route_b_reporting").sample(data=data, chains=4, parallel_chains=4, seed=1, adapt_delta=0.95,
                                          inits=inits, show_progress=False, show_console=False)
    rhat, div = _diagnostics(f, ["R"])
    q = f.stan_variable("q")                    # draws x T x N
    ratio = q[:, :, 0] / q[:, :, 1]
    ratio = ratio / ratio[:, :1]                # children's reporting relative to adults', day 1 = 1
    return dict(R=f.stan_variable("R"), ratio=np.quantile(ratio, [0.5, 0.025, 0.975], axis=0),
                max_rhat=rhat, divergences=div)


def errors(res):
    out = []
    for (ds, m), x in res.groupby(["dataset", "model"], sort=False):
        ch = datasets[ds]["change"]
        for lab, lo in [("5 weeks after contact change", 100), ("5 weeks after reporting change", ch)]:
            y = x[(x.start >= lo) & (x.start < lo + 35)]
            err = ((y["median"] - y.truth).abs() / y.truth).mean()
            cov = ((y.lo <= y.truth) & (y.hi >= y.truth)).mean()
            out.append(dict(dataset=ds, model=m, window=lab, value=f"{err:.2f} ({100 * cov:.0f}%)"))
    return pd.DataFrame(out).pivot(index=["dataset", "model"], columns="window", values="value").reset_index()


def plot(res, qres, path):
    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.2), sharex=True)
    for c, (ds, spec) in enumerate(datasets.items()):
        ax = axes[0, c]
        d = res[(res.dataset == ds) & (res.start >= 61)]
        models = [m for m in COL if m in set(d.model)]
        tr = d[d.model == models[0]].sort_values("start")
        ax.step(tr.start + 3, tr.truth, where="mid", color="black", lw=1.2)
        for m, off in zip(models, np.linspace(-2, 2, len(models))):
            x = d[d.model == m]
            ax.errorbar(x.start + 3 + off, x["median"], yerr=[x["median"] - x.lo, x.hi - x["median"]],
                        fmt="o", ms=2.5, lw=1, color=COL[m], label=m)
        ax.set_ylim(0.4, 1.8)
        ax.set_title(spec["title"], fontsize=9)
        ax = axes[1, c]
        truth = np.array([spec["q1"](dd) for dd in DAYS]) / 0.2
        ax.plot(DAYS, truth, color="black", lw=1.2)
        for m in [m for m in COL if m in set(qres.model)]:
            x = qres[(qres.dataset == ds) & (qres.model == m)]
            ax.plot(x.day, x["median"], color=COL[m], lw=1)
            ax.fill_between(x.day, x.lo, x.hi, color=COL[m], alpha=0.2, lw=0)
        ax.set_ylim(0, 5)
        ax.set_xlabel("Day")
        for a in axes[:, c]:
            a.axvline(100, ls="--", color="grey", lw=0.8)
            a.set_xlim(60, 205)
    axes[0, 0].set_ylabel("$R_t$")
    axes[1, 0].set_ylabel("Children's reporting relative\nto adults' (day 1 = 1)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1), pad=0.3)
    fig.savefig(path, dpi=200)


def run():
    res, qres, diag = [], [], []
    for k, (ds, spec) in enumerate(datasets.items()):
        rng = np.random.default_rng(500 + k)
        X, Y = simulate_general(lambda d: C2 if d >= 100 else C1, lambda d: p,
                                lambda d: np.array([spec["q1"](d), 0.6]), R_constant, w, n0=20000, rng=rng)
        prev, pos = survey(X, np.random.default_rng(600 + k))
        print(f"{ds}: peak prevalence by group {prev.max(axis=0).round(4)}, survey positives per window "
              f"{pos.min()}-{pos.max()}", flush=True)
        for m, opt in FITS.items():
            print(f"{ds}: {m}", flush=True)
            f = fit(Y, pos, **opt)
            res.append(summarise_R(f["R"], R_constant, dataset=ds, model=m))
            if opt["est_q"]:
                qres.append(pd.DataFrame(dict(dataset=ds, model=m, day=DAYS, median=f["ratio"][0],
                                              lo=f["ratio"][1], hi=f["ratio"][2])))
            diag.append(dict(dataset=ds, model=m, max_rhat=f["max_rhat"], divergences=f["divergences"]))
        f = fit_single(Y, w)
        res.append(summarise_R(f["R"], R_constant, dataset=ds, model="Single group, reported totals"))
        diag.append(dict(dataset=ds, model="Single group, reported totals", max_rhat=f["max_rhat"],
                         divergences=f["divergences"]))
    res, qres = pd.concat(res), pd.concat(qres)
    res.to_csv(ROOT / "results" / "reporting_R.csv", index=False)
    qres.to_csv(ROOT / "results" / "reporting_q.csv", index=False)
    tab = errors(res)
    tab.to_csv(ROOT / "results" / "reporting_table.csv", index=False)
    pd.DataFrame(diag).to_csv(ROOT / "results" / "reporting_diagnostics.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\nRelative error of weekly R_t medians (95% coverage)\n", tab)
        print("\nDiagnostics\n", pd.DataFrame(diag))
    plot(res, qres, ROOT / "figures" / "reporting.png")


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot(pd.read_csv(ROOT / "results" / "reporting_R.csv"), pd.read_csv(ROOT / "results" / "reporting_q.csv"),
             ROOT / "figures" / "reporting.png")
    else:
        run()
