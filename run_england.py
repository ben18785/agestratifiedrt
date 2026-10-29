"""England in two periods, five age groups (0-9, 10-19, 20-49, 50-69, 70+):
  spring2021  4 January to 16 May 2021: schools reopened on 8 March, with lateral-flow testing of
              secondary pupils from the same day, so contacts and reporting changed together (as in R1).
  autumn2020  3 August to 29 November 2020: schools reopened in early September without routine
              testing of pupils; universities returned in late September; November lockdown.
Inputs from data/processed/england*/ (see data/prepare_england.py). Fits:
  - single group, total cases (the usual approach);
  - single group, total cases, weekly reporting + ONS Infection Survey;
  - CoMix waves x multipliers, constant reporting;
  - CoMix waves x multipliers, weekly reporting from cases only;
  - CoMix waves x multipliers, weekly reporting + ONS Infection Survey.
All with a random walk on weekly R_t, group-specific day-of-week effects and (multi-group fits)
vaccination as a known reduction in susceptibility.
Usage: python run_england.py [spring2021|autumn2020] [--plot-only]
"""
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import gamma

from common import ROOT, model, _diagnostics

PERIODS = {
    "spring2021": dict(
        folder="england", tag="england",
        # CoMix waves: two weeks, aligned on 8 March
        waves=["2021-01-04", "2021-01-25", "2021-02-08", "2021-02-22", "2021-03-08", "2021-03-22",
               "2021-04-05", "2021-04-19", "2021-05-03"],
        events={"2021-03-08": "Schools reopen", "2021-04-02": "Easter", "2021-04-09": "", "2021-04-12": ""},
        main="2021-03-08", plot_from="2021-01-25", ons_from="2021-01-30", ylim=(0.5, 1.3)),
    "autumn2020": dict(
        folder="england_autumn2020", tag="england_autumn2020",
        # CoMix waves: two weeks where possible; one-week waves for the week schools reopened
        # (31 Aug) and half-term (26 Oct)
        waves=["2020-08-03", "2020-08-17", "2020-08-31", "2020-09-07", "2020-09-21", "2020-10-05",
               "2020-10-19", "2020-10-26", "2020-11-02", "2020-11-16"],
        events={"2020-09-03": "Schools reopen", "2020-09-21": "Universities", "2020-10-26": "Half-term",
                "2020-11-05": "Lockdown"},
        main="2020-09-03", plot_from="2020-08-10", ons_from="2020-08-29", ylim=(0.5, 3.0)),
}
PERIOD = next((a for a in sys.argv[1:] if a in PERIODS), "spring2021")
CFG = PERIODS[PERIOD]
IN = ROOT / "data" / "processed" / CFG["folder"]
GROUPS = ["0-9", "10-19", "20-49", "50-69", "70+"]
S = 21                                   # generation time support (days)
GT_MEAN, GT_SD = 5.5, 2.1                # generation time (days)
DELTA = 6                                # infection to specimen date (days)
D = 30                                   # days an infection can test PCR positive
VE1, VE2 = 0.6, 0.85                     # protection against infection: one dose (from 21 days), two (from 14 days)
WAVE_STARTS = pd.to_datetime(CFG["waves"])
EVENTS = CFG["events"]
FITS = {
    "Single group, totals": dict(N=1, est_q=0, ons=0),
    "Single group, totals + ONS, weekly reporting": dict(N=1, est_q=1, ons=1),
    "CoMix x multipliers, constant reporting": dict(N=5, est_q=0, ons=0),
    "CoMix x multipliers, weekly reporting": dict(N=5, est_q=1, ons=0),
    "CoMix x multipliers, weekly reporting + ONS": dict(N=5, est_q=1, ons=1),
}
COL = {"Single group, totals": "#d1495b", "Single group, totals + ONS, weekly reporting": "#e89aa5",
       "CoMix x multipliers, constant reporting": "#3b75af", "CoMix x multipliers, weekly reporting": "#a0711c",
       "CoMix x multipliers, weekly reporting + ONS": "#1b3f6b"}


def gen_time():
    F = gamma.cdf(np.arange(S + 1), a=(GT_MEAN / GT_SD) ** 2, scale=GT_SD ** 2 / GT_MEAN)
    w = np.diff(F)
    return w / w.sum()


def detection():
    """P(PCR positive u days after infection): becomes positive after ~2.5 days, clears after ~14."""
    u = np.arange(D) + 0.5
    on = gamma.cdf(u, a=(2.5 / 1.5) ** 2, scale=1.5 ** 2 / 2.5)
    off = 1 - gamma.cdf(u, a=(14 / 5) ** 2, scale=5 ** 2 / 14)
    return on * off


def load():
    cases = pd.read_csv(IN / "cases.csv", index_col=0, parse_dates=True)[GROUPS]
    days = cases.index
    pop = pd.read_csv(IN / "population.csv").set_index("group").population.reindex(GROUPS).values.astype(float)
    con = pd.read_csv(IN / "contacts.csv", parse_dates=["week_start"])
    vac = pd.read_csv(IN / "vaccination.csv", index_col=0, parse_dates=True).clip(upper=1)
    vac = vac.reindex(pd.date_range(days[0] - pd.Timedelta(days=30), days[-1])).fillna(0.0)
    ons = pd.read_csv(IN / "ons.csv", parse_dates=["start", "end"])
    # contact matrix per wave: participant-weighted mean over the wave's weeks, then made reciprocal
    Cw = []
    ends = list(WAVE_STARTS[1:]) + [days[-1] + pd.Timedelta(days=1)]
    for a, b in zip(WAVE_STARTS, ends):
        x = con[(con.week_start >= a) & (con.week_start < b)]
        M = np.zeros((5, 5))
        for i, g in enumerate(GROUPS):
            y = x[x.part_group == g]
            M[i] = (y[GROUPS].values * y.n_participants.values[:, None]).sum(0) / y.n_participants.sum()
        M = (M * pop[:, None] + (M * pop[:, None]).T) / 2 / pop[:, None]
        Cw.append(M)
    wave = np.searchsorted(WAVE_STARTS, days, side="right") - 1
    # susceptibility factor from vaccination
    c1 = vac[[f"cov1_{g}" for g in GROUPS]].shift(21).reindex(days).values
    c2 = vac[[f"cov2_{g}" for g in GROUPS]].shift(14).reindex(days).values
    sv = 1 - VE1 * np.clip(c1 - c2, 0, 1) - VE2 * c2
    return dict(cases=cases, days=days, pop=pop, Cw=np.array(Cw), wave=wave, sv=sv, ons=ons)


def stan_data(dat, N, est_q, ons):
    days, T = dat["days"], len(dat["days"])
    week = (np.arange(T) // 7) + 1                          # Monday-based weeks (both periods start on a Monday)
    nP = week.max()
    if N == 1:
        Y = dat["cases"].sum(axis=1).values[:, None]
        C = np.ones((nP, 1, 1))
        v = np.ones((nP, 1))
        Npop = np.array([dat["pop"].sum()])
    else:
        Y = dat["cases"].values
        C = np.array([dat["Cw"][dat["wave"][week == k][0]] for k in range(1, nP + 1)])
        v = np.array([dat["sv"][week == k].mean(0) for k in range(1, nP + 1)])
        Npop = dat["pop"]
    o = dat["ons"]
    starts = sorted(o.start.unique())
    k_ok = [s for s in starts
            if (days.get_loc(s) + 1) + DELTA - (D - 1) >= 1 and (days.get_loc(o[o.start == s].end.iloc[0]) + 1) + DELTA <= T]
    n_pos = np.array([[o[(o.start == s) & (o.group == g)].n_positive.iloc[0] for g in GROUPS] for s in k_ok])
    n_tst = np.array([[o[(o.start == s) & (o.group == g)].n_sample.iloc[0] for g in GROUPS] for s in k_ok])
    if N == 1:   # population-weighted positivity over groups with data, with the pooled number tested
        has = n_tst > 0
        wts = np.where(has, dat["pop"], 0)
        prev = (np.where(has, n_pos / np.maximum(n_tst, 1), 0) * wts).sum(1) / wts.sum(1)
        n_tst = n_tst.sum(1, keepdims=True)
        n_pos = np.round(prev[:, None] * n_tst).astype(int)
    win_start = [days.get_loc(s) + 1 for s in k_ok]
    win_end = [days.get_loc(o[o.start == s].end.iloc[0]) + 1 for s in k_ok]
    return dict(T=T, N=N, S=S, t0=S + 1, Y=Y.astype(int).tolist(), w=np.tile(gen_time(), (N, 1)),
                B=int(week.max()), block=week.tolist(), nP=int(nP), period=week.tolist(), C=C, v=v,
                dow=(days.day_of_week + 1).tolist(), est_q=est_q, W=int(week.max()), qweek=week.tolist(),
                ons=ons, Npop=Npop, D=D, d=detection(), delta=DELTA, K=len(k_ok), win_start=win_start,
                win_end=win_end, n_tested=n_tst.astype(int).tolist(), n_pos=n_pos.astype(int).tolist())


def fit(dat, N, est_q, ons):
    data = stan_data(dat, N, est_q, ons)
    inits = {"inv_phi": 0.05, "tau": 0.05, "logR1": -0.2}
    if est_q:
        inits["tau_q"] = [0.1]
    if ons:
        inits["lq0"] = [-1.0] * N
    f = model("england").sample(data=data, chains=4, parallel_chains=4, seed=1, adapt_delta=0.95,
                                inits=inits, show_progress=False, show_console=False)
    return f, data


def summarise(f, data, dat, name):
    days = dat["days"]
    R = f.stan_variable("R")
    week_start = days[::7][:R.shape[1]]
    r = pd.DataFrame(dict(model=name, week_start=week_start, median=np.median(R, 0),
                          lo=np.quantile(R, 0.025, 0), hi=np.quantile(R, 0.975, 0)))
    q = f.stan_variable("q")[:, ::7, :]                    # draws x weeks x groups
    qs = []
    for j in range(data["N"]):
        g = "all" if data["N"] == 1 else GROUPS[j]
        qs.append(pd.DataFrame(dict(model=name, group=g, week_start=week_start, median=np.median(q[:, :, j], 0),
                                    lo=np.quantile(q[:, :, j], 0.025, 0), hi=np.quantile(q[:, :, j], 0.975, 0))))
    p = f.stan_variable("prev_pct")                         # draws x K x N
    ps = []
    for j in range(data["N"]):
        g = "all" if data["N"] == 1 else GROUPS[j]
        obs = np.array(data["n_pos"])[:, j] / np.maximum(np.array(data["n_tested"])[:, j], 1) * 100
        ps.append(pd.DataFrame(dict(model=name, group=g, start=days[np.array(data["win_start"]) - 1],
                                    observed=obs, median=np.median(p[:, :, j], 0),
                                    lo=np.quantile(p[:, :, j], 0.025, 0), hi=np.quantile(p[:, :, j], 0.975, 0))))
    rhat, div = _diagnostics(f, ["R"])
    ab = {}
    if data["N"] > 1:
        for nm in ["a_raw", "b_raw"]:
            x = f.stan_variable(nm)
            ab[nm] = "; ".join(f"{np.median(x[:, i]):.2f}" for i in range(x.shape[1]))
    return r, pd.concat(qs), pd.concat(ps), dict(model=name, max_rhat=rhat, divergences=div, **ab)


def plot(res, qres, pres, path):
    from scipy.stats import beta
    plt.rcParams.update({"font.size": 9})
    fig = plt.figure(figsize=(10, 6.6))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.25, 1], hspace=0.75, wspace=0.45)
    ax = fig.add_subplot(gs[0, :3])
    for m, off in zip(COL, np.linspace(-1.6, 1.6, len(COL))):
        x = res[(res.model == m) & (res.week_start >= CFG["plot_from"])]
        t = x.week_start + pd.Timedelta(days=3 + off)
        ax.errorbar(t, x["median"], yerr=[x["median"] - x.lo, x.hi - x["median"]], fmt="o", ms=2.5, lw=1,
                    color=COL[m], label=m)
    ax.axhline(1, color="black", lw=0.6)
    ax.set_ylabel("$R_t$ (long-run)")
    ax.set_ylim(*CFG["ylim"])
    ax.set_title("A. Reproduction number, by fit", fontsize=9, loc="left")
    ax.legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    top = [ax]
    ax = fig.add_subplot(gs[0, 3:])
    x = qres[qres.model == "CoMix x multipliers, weekly reporting + ONS"]
    cols = plt.cm.viridis(np.linspace(0, 0.9, 5))
    for g, c in zip(GROUPS, cols):
        y = x[x.group == g]
        t = y.week_start + pd.Timedelta(days=3)
        ax.plot(t, y["median"], color=c, lw=1.2, label=g)
        ax.fill_between(t, y.lo, y.hi, color=c, alpha=0.2, lw=0)
    ax.set_ylabel("P(infection reported)")
    ax.set_ylim(0, 0.7)
    ax.legend(title="Age", fontsize=7, title_fontsize=7.5, frameon=False, loc="upper left", ncol=2)
    ax.set_title("B. Reporting by age (CoMix + ONS fit)", fontsize=9, loc="left")
    top.append(ax)
    for a in top:
        for d in EVENTS:
            a.axvline(pd.Timestamp(d), ls="--", color="grey", lw=0.7)
        a.xaxis.set_major_locator(mdates.MonthLocator())
        a.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        a.set_xlim(pd.Timestamp(CFG["plot_from"]) - pd.Timedelta(days=3), res.week_start.max() + pd.Timedelta(days=10))
    for d, lab in EVENTS.items():
        top[0].text(pd.Timestamp(d) + pd.Timedelta(days=1), CFG["ylim"][0] + 0.02, lab, fontsize=7, color="grey",
                    rotation=90, va="bottom")
    p = pres[pres.model == "CoMix x multipliers, weekly reporting + ONS"]
    for j, g in enumerate(GROUPS):
        ax = fig.add_subplot(gs[1, j])
        y = p[p.group == g]
        t = y.start + pd.Timedelta(days=7)
        ax.plot(t, y["median"], color=COL["CoMix x multipliers, weekly reporting + ONS"], lw=1.2,
                label="Model (reconstructed\ninfections)")
        ax.fill_between(t, y.lo, y.hi, color=COL["CoMix x multipliers, weekly reporting + ONS"], alpha=0.2, lw=0)
        ons = dat_ons(g)
        lo = 100 * beta.ppf(0.025, ons.n_positive, ons.n_sample - ons.n_positive + 1)
        hi = 100 * beta.ppf(0.975, ons.n_positive + 1, ons.n_sample - ons.n_positive)
        ob = 100 * ons.n_positive / ons.n_sample
        ax.errorbar(ons.start + pd.Timedelta(days=7), ob, yerr=[ob - lo, hi - ob], fmt="o", ms=3, color="black",
                    lw=0.8, label="ONS survey")
        ax.set_yscale("log")
        ax.set_ylim(0.01, 3)
        ax.set_title(g, fontsize=9)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.set_xlim(pd.Timestamp(CFG["ons_from"]) - pd.Timedelta(days=3), y.start.max() + pd.Timedelta(days=14))
        ax.tick_params(axis="x", labelsize=7)
        ax.axvline(pd.Timestamp(CFG["main"]), ls="--", color="grey", lw=0.7)
        if j == 0:
            ax.set_ylabel("Positivity (%)")
            ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    fig.text(0.01, 0.40, "C. ONS positivity: survey (fitted) and model, by age", fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.95, bottom=0.07)
    fig.savefig(path, dpi=200)


def dat_ons(g):
    o = pd.read_csv(IN / "ons.csv", parse_dates=["start", "end"])
    return o[(o.group == g) & (o.start >= CFG["ons_from"]) & (o.n_sample > 0)]


def run():
    dat = load()
    res, qres, pres, diag = [], [], [], []
    for name, opt in FITS.items():
        print(name, flush=True)
        f, data = fit(dat, **opt)
        r, q, p, dg = summarise(f, data, dat, name)
        res.append(r), qres.append(q), pres.append(p), diag.append(dg)
        print(dg, flush=True)
    res, qres, pres, diag = pd.concat(res), pd.concat(qres), pd.concat(pres), pd.DataFrame(diag)
    res.to_csv(ROOT / "results" / f"{CFG['tag']}_R.csv", index=False)
    qres.to_csv(ROOT / "results" / f"{CFG['tag']}_q.csv", index=False)
    pres.to_csv(ROOT / "results" / f"{CFG['tag']}_ons_fit.csv", index=False)
    diag.to_csv(ROOT / "results" / f"{CFG['tag']}_diagnostics.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(res.pivot(index="week_start", columns="model", values="median").round(2))
        print(diag)
    plot(res, qres, pres, ROOT / "figures" / f"{CFG['tag']}.png")


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot(pd.read_csv(ROOT / "results" / f"{CFG['tag']}_R.csv", parse_dates=["week_start"]),
             pd.read_csv(ROOT / "results" / f"{CFG['tag']}_q.csv", parse_dates=["week_start"]),
             pd.read_csv(ROOT / "results" / f"{CFG['tag']}_ons_fit.csv", parse_dates=["start"]), ROOT / "figures" / f"{CFG['tag']}.png")
    else:
        run()
