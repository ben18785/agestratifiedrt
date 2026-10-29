"""England in five periods, five age groups (0-9, 10-19, 20-49, 50-69, 70+):
  spring2021  4 January to 16 May 2021: schools reopened on 8 March, with lateral-flow testing of
              secondary pupils from the same day, so contacts and reporting changed together (as in R1).
  autumn2020  3 August to 29 November 2020: schools reopened in early September without routine
              testing of pupils; universities returned in late September; November lockdown.
  autumn2021  16 August to 5 December 2021 (Delta): schools reopened in early September with
              lateral-flow testing that tailed off; 12-15s vaccinated from late September; in the
              fits that use the survey, susceptibility also falls with infections in the window.
  summer2021  7 June to 12 September 2021 (Delta): Euro 2020, restrictions ended on 19 July,
              schools closed around 22 July (ending pupils' routine lateral-flow testing).
  omicron     29 November 2021 to 20 February 2022 (Omicron BA.1, then BA.2): Christmas, schools
              reopened on 4 January, positive lateral-flow tests no longer needed a confirmatory
              PCR from 11 January, reinfections counted (the archive data are by episode).
Inputs from data/processed/england*/ (see data/prepare_england.py). Fits (keys in FITS):
  totals               single group, total cases (the usual approach)
  totals_ons           single group, total cases, weekly reporting + ONS Infection Survey
  comix_const          CoMix waves x multipliers, constant reporting
  comix_weekly         CoMix waves x multipliers, weekly reporting from cases only
  comix_weekly_ons     CoMix waves x multipliers, weekly reporting + ONS
  comix_weekly_ons_gq  as comix_weekly_ons, with a separate sd of weekly reporting steps per group
  latent_ons_gq        hidden-infections model (stan/england_latent.stan), CoMix, reporting sd
                       per group, + ONS
  comix_adapt_ons      as comix_weekly_ons_gq, but no vaccine coverage or assumed depletion: each
                       group's susceptibility and infectiousness follow a weekly random walk
                       (estimated), which picks up vaccination, waning and natural immunity
  latent_adapt_ons     the same, in the hidden-infections model
All with a random walk on weekly R_t, group-specific day-of-week effects and (multi-group fits)
vaccination as a known reduction in susceptibility.
Usage:
  python run_england.py PERIOD                 # the first five fits, then combine
  python run_england.py PERIOD --fit KEY[,KEY] # run these fits; outputs to results/parts/
  python run_england.py PERIOD --combine       # combine results/parts/ into tables and the figure
  python run_england.py PERIOD --plot-only     # redraw the figure from the combined tables
PERIOD is spring2021, autumn2020, autumn2021, summer2021 or omicron.
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
        main="2021-03-08", plot_from="2021-01-25", ons_from="2021-01-30", ylim=(0.5, 1.3),
        gt=(5.5, 2.1), ve=(0.6, 0.85, 0.85), deplete=False),
    "autumn2020": dict(
        folder="england_autumn2020", tag="england_autumn2020",
        # CoMix waves: two weeks where possible; one-week waves for the week schools reopened
        # (31 Aug) and half-term (26 Oct)
        waves=["2020-08-03", "2020-08-17", "2020-08-31", "2020-09-07", "2020-09-21", "2020-10-05",
               "2020-10-19", "2020-10-26", "2020-11-02", "2020-11-16"],
        events={"2020-09-03": "Schools reopen", "2020-09-21": "Universities", "2020-10-26": "Half-term",
                "2020-11-05": "Lockdown"},
        main="2020-09-03", plot_from="2020-08-10", ons_from="2020-08-29", ylim=(0.5, 3.0),
        gt=(5.5, 2.1), ve=(0.6, 0.85, 0.85), deplete=False),
    "autumn2021": dict(
        folder="england_autumn2021", tag="england_autumn2021",
        # CoMix waves: two weeks where possible; one-week waves for the week schools reopened
        # (30 Aug), the week before half-term (18 Oct), half-term (25 Oct) and the last week
        waves=["2021-08-16", "2021-08-30", "2021-09-06", "2021-09-20", "2021-10-04", "2021-10-18",
               "2021-10-25", "2021-11-01", "2021-11-15", "2021-11-29"],
        events={"2021-09-02": "Schools reopen", "2021-09-20": "12-15s vaccinated", "2021-10-25": "Half-term"},
        main="2021-09-02", plot_from="2021-08-23", ons_from="2021-08-28", ylim=(0.5, 2.0),
        # Delta: shorter generation time (Hart et al. 2022) and lower protection against infection
        gt=(4.7, 2.6), ve=(0.45, 0.65, 0.85), deplete=True, delay=4.5),
    "summer2021": dict(
        folder="england_summer2021", tag="england_summer2021",
        # Delta from early June (over 90% of sequences). CoMix waves: two weeks, aligned on the
        # end of restrictions (19 July) and school reopening (30 August / early September)
        waves=["2021-06-07", "2021-06-21", "2021-07-05", "2021-07-19", "2021-08-02", "2021-08-16",
               "2021-08-30", "2021-09-06"],
        events={"2021-06-11": "Euro 2020", "2021-07-11": "Final", "2021-07-19": "Restrictions end",
                "2021-07-22": "Schools close", "2021-09-02": "Schools reopen"},
        main="2021-07-19", plot_from="2021-06-21", ons_from="2021-06-19", ylim=(0.5, 2.0),
        gt=(4.7, 2.6), ve=(0.45, 0.65, 0.85), deplete=True, delay=4.5),
    "omicron": dict(
        folder="england_omicron", tag="england_omicron",
        # Omicron BA.1 from mid-December (over 90% of cases by about 20 December), BA.2 rising in
        # February. The fit starts on 29 November so that the cases-based models have a month of
        # cases before the first survey window; R before about 20 December mixes Delta and Omicron
        # and is not shown. CoMix waves: one wave for the school holidays (20 December to 2 January)
        waves=["2021-11-29", "2021-12-13", "2021-12-20", "2022-01-03", "2022-01-17", "2022-01-31", "2022-02-14"],
        events={"2021-12-25": "Christmas", "2022-01-04": "Schools reopen", "2022-01-11": "No confirmatory PCR",
                "2022-01-27": "Plan B ends"},
        main="2022-01-04", plot_from="2021-12-20", ons_from="2021-12-12", ylim=(0.4, 2.0),
        # Omicron: shorter generation time and incubation period; little protection against
        # infection from two doses, more from a recent booster; more people susceptible because
        # of immune escape (all assumptions; the random-walk susceptibility fits do not use them)
        gt=(3.5, 2.4), ve=(0.1, 0.2, 0.6), deplete=True, delay=4.0, s0=0.8),
}
PERIOD = next((a for a in sys.argv[1:] if a in PERIODS), "spring2021")
CFG = PERIODS[PERIOD]
IN = ROOT / "data" / "processed" / CFG["folder"]
GROUPS = ["0-9", "10-19", "20-49", "50-69", "70+"]
S = 21                                   # generation time support (days)
GT_MEAN, GT_SD = CFG["gt"]               # generation time (days)
DELTA = 6                                # infection to specimen date (days), cases-based models
L = 21                                   # hidden-infections model: seeding days before day 1
DD = 21                                  # hidden-infections model: delays 0..20 days to specimen date
D = 30                                   # days an infection can test PCR positive
VE1, VE2, VE3 = CFG["ve"]                # protection against infection: one dose (from 21 days), two and three (from 14 days)
S0 = CFG.get("s0", 0.7)                  # susceptible fraction at the start (for depletion): ~30% infected before
WAVE_STARTS = pd.to_datetime(CFG["waves"])
EVENTS = CFG["events"]
FITS = {
    "totals": dict(name="Single group, totals", col="#d1495b", opt=dict(N=1, est_q=0, ons=0)),
    "totals_ons": dict(name="Single group, totals + ONS, weekly reporting", col="#e89aa5",
                       opt=dict(N=1, est_q=1, ons=1)),
    "comix_const": dict(name="CoMix x multipliers, constant reporting", col="#3b75af", opt=dict(N=5, est_q=0, ons=0)),
    "comix_weekly": dict(name="CoMix x multipliers, weekly reporting", col="#a0711c", opt=dict(N=5, est_q=1, ons=0)),
    "comix_weekly_ons": dict(name="CoMix x multipliers, weekly reporting + ONS", col="#1b3f6b",
                             opt=dict(N=5, est_q=1, ons=1)),
    "comix_weekly_ons_gq": dict(name="CoMix x multipliers, reporting sd by age + ONS", col="#2a9d8f",
                                opt=dict(N=5, est_q=1, ons=1, tau_by_group=1)),
    "latent_ons_gq": dict(name="Hidden infections, CoMix, reporting sd by age + ONS", col="#7b2d8b",
                          opt=dict(N=5, est_q=1, ons=1, tau_by_group=1, latent=1)),
    "comix_adapt_ons": dict(name="CoMix x random-walk susceptibility by age + ONS", col="#e9a23b",
                            opt=dict(N=5, est_q=1, ons=1, tau_by_group=1, adapt=1, vacc=0)),
    "latent_adapt_ons": dict(name="Hidden infections, random-walk susceptibility by age + ONS", col="#5c3c92",
                             opt=dict(N=5, est_q=1, ons=1, tau_by_group=1, latent=1, adapt=1, vacc=0)),
}
DEFAULT_FITS = ["totals", "totals_ons", "comix_const", "comix_weekly", "comix_weekly_ons"]
MAIN_PREFERENCE = ["latent_adapt_ons", "comix_adapt_ons", "latent_ons_gq", "comix_weekly_ons_gq",
                   "comix_weekly_ons"]                   # the first available is shown in panels B and C
NAME = {k: v["name"] for k, v in FITS.items()}
COL = {v["name"]: v["col"] for v in FITS.values()}
PARTS = ROOT / "results" / ("parts_smoke" if "--smoke" in sys.argv else "parts")   # --smoke: quick test run


def gen_time():
    F = gamma.cdf(np.arange(S + 1), a=(GT_MEAN / GT_SD) ** 2, scale=GT_SD ** 2 / GT_MEAN)
    w = np.diff(F)
    return w / w.sum()


def delay():
    """P(delay from infection to specimen date = d days), d = 0..DD-1: shorter for Delta and Omicron."""
    mean = CFG.get("delay", 6.0)
    F = gamma.cdf(np.arange(DD + 1), a=(mean / 3) ** 2, scale=9 / mean)
    f = np.diff(F)
    return f / f.sum()


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
    c3 = vac[[f"cov3_{g}" for g in GROUPS]].shift(14).reindex(days).values
    sv = 1 - VE1 * np.clip(c1 - c2, 0, 1) - VE2 * np.clip(c2 - c3, 0, 1) - VE3 * c3
    return dict(cases=cases, days=days, pop=pop, Cw=np.array(Cw), wave=wave, sv=sv, ons=ons)


def stan_data(dat, N, est_q, ons, tau_by_group=0, latent=0, adapt=0, vacc=1):
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
        v = np.array([dat["sv"][week == k].mean(0) for k in range(1, nP + 1)]) if vacc else np.ones((nP, N))
        Npop = dat["pop"]
    o = dat["ons"]
    starts = sorted(o.start.unique())
    first = lambda s: days.get_loc(s) + 1                  # day number of a window's start
    last = lambda s: days.get_loc(o[o.start == s].end.iloc[0]) + 1
    if latent:   # needs infections back to D - 1 days before the window; no shift to specimen date
        k_ok = [s for s in starts if first(s) + L - (D - 1) >= 1 and last(s) <= T]
    else:
        k_ok = [s for s in starts if first(s) + DELTA - (D - 1) >= 1 and last(s) + DELTA <= T]
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
    first_week = Y[:7].mean(0)
    return dict(T=T, N=N, S=S, t0=1 if latent else S + 1, Y=Y.astype(int).tolist(), w=np.tile(gen_time(), (N, 1)),
                L=L, Dd=DD, f=np.tile(delay(), (N, 1)), lx0_mean=np.log(np.maximum(first_week, 1) / 0.3),
                tau_by_group=tau_by_group,
                B=int(week.max()), block=week.tolist(), nP=int(nP), period=week.tolist(), C=C, v=v,
                pstart=[int(np.argmax(week == k)) + 1 for k in range(1, nP + 1)],
                deplete=int(CFG["deplete"] and ons and N > 1 and vacc), s0=np.full(N, S0), adapt=adapt,
                dow=(days.day_of_week + 1).tolist(), est_q=est_q, W=int(week.max()), qweek=week.tolist(),
                ons=ons, Npop=Npop, D=D, d=detection(), delta=DELTA, K=len(k_ok), win_start=win_start,
                win_end=win_end, n_tested=n_tst.astype(int).tolist(), n_pos=n_pos.astype(int).tolist())


def inits_for(data, est_q, ons, tau_by_group=0, latent=0, adapt=0, vacc=1):
    N = data["N"]
    inits = {"inv_phi": 0.05, "tau": 0.05, "logR1": -0.2}
    if est_q:
        inits["tau_q"] = [[0.1] * (N if tau_by_group else 1)]
    if ons:
        inits["lq0"] = [-1.0] * N
    if latent:
        inits.update(sigma_eta=0.05, r0=0.0, lx0=list(data["lx0_mean"]), eta=np.zeros((N, data["B"])))
    if adapt:
        inits.update(sigA=[0.05], rho=[0.5], Az=np.zeros((N - 1, data["nP"] - 1)))
    return inits


def fit(dat, N, est_q, ons, tau_by_group=0, latent=0, adapt=0, vacc=1, data=None):
    if data is None:
        data = stan_data(dat, N, est_q, ons, tau_by_group, latent, adapt, vacc)
    iters = dict(iter_warmup=30, iter_sampling=30) if "--smoke" in sys.argv else {}
    f = model("england_latent" if latent else "england").sample(
        data=data, chains=4, parallel_chains=4, seed=1, adapt_delta=0.95,
        inits=inits_for(data, est_q, ons, tau_by_group, latent, adapt, vacc), show_progress=False, show_console=False,
        **iters)
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
    A = []
    if data.get("adapt"):   # susceptibility and infectiousness factors by group and week (group 1 = 1)
        lA, rho = f.stan_variable("logA"), f.stan_variable("rho")[:, 0]
        for kind, w in [("susceptibility", rho), ("infectiousness", 1 - rho)]:
            x = np.exp(w[:, None, None] * lA)              # draws x groups x weeks
            for j, g in enumerate(GROUPS):
                A.append(pd.DataFrame(dict(model=name, kind=kind, group=g, week_start=week_start,
                                           median=np.median(x[:, j], 0), lo=np.quantile(x[:, j], 0.025, 0),
                                           hi=np.quantile(x[:, j], 0.975, 0))))
    rhat, div = _diagnostics(f, ["R"])
    summ = f.summary()
    ess_col = [c for c in summ.columns if c.lower().replace("_", "") in ("essbulk", "nessbulk")][0]
    ess = float(summ.loc[[i for i in summ.index if i.startswith("R[")], ess_col].min())
    treedepth = float(np.mean(f.method_variables()["treedepth__"] >= f.metadata.cmdstan_config.get("max_depth", 10)))
    ab = {}
    if data["N"] > 1:
        for nm in ["a_raw", "b_raw"]:
            x = f.stan_variable(nm)
            ab[nm] = "; ".join(f"{np.median(x[:, i]):.2f}" for i in range(x.shape[1]))
    if data.get("adapt"):
        ab["rho"] = f"{np.median(f.stan_variable('rho')):.2f}"
        ab["sigA"] = f"{np.median(f.stan_variable('sigA')):.3f}"
    dg = dict(model=name, max_rhat=rhat, min_ess_R=round(ess), divergences=div, share_max_treedepth=round(treedepth, 3), **ab)
    return r, pd.concat(qs), pd.concat(ps), dg, (pd.concat(A) if A else None)


def plot(res, qres, pres, path):
    from scipy.stats import beta
    plt.rcParams.update({"font.size": 9})
    fig = plt.figure(figsize=(10, 6.6))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.25, 1], hspace=0.75, wspace=0.45)
    ax = fig.add_subplot(gs[0, :3])
    shown = [m for m in COL if m in set(res.model)]
    for m, off in zip(shown, np.linspace(-1.6, 1.6, len(shown))):
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
    main = next(NAME[k] for k in MAIN_PREFERENCE if NAME[k] in set(qres.model))
    x = qres[qres.model == main]
    cols = plt.cm.viridis(np.linspace(0, 0.9, 5))
    for g, c in zip(GROUPS, cols):
        y = x[x.group == g]
        t = y.week_start + pd.Timedelta(days=3)
        ax.plot(t, y["median"], color=c, lw=1.2, label=g)
        ax.fill_between(t, y.lo, y.hi, color=c, alpha=0.2, lw=0)
    ax.set_ylabel("P(infection reported)")
    ax.set_ylim(0, 1)
    ax.legend(title="Age", fontsize=7, title_fontsize=7.5, frameon=False, loc="upper left", ncol=2)
    ax.set_title("B. Reporting by age (" + main.split(",")[0] + ")", fontsize=9, loc="left")
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
    p = pres[pres.model == main]
    for j, g in enumerate(GROUPS):
        ax = fig.add_subplot(gs[1, j])
        y = p[p.group == g]
        t = y.start + pd.Timedelta(days=7)
        ax.plot(t, y["median"], color=COL[main], lw=1.2, label="Model")
        ax.fill_between(t, y.lo, y.hi, color=COL[main], alpha=0.2, lw=0)
        ons = dat_ons(g)
        lo = 100 * beta.ppf(0.025, ons.n_positive, ons.n_sample - ons.n_positive + 1)
        hi = 100 * beta.ppf(0.975, ons.n_positive + 1, ons.n_sample - ons.n_positive)
        ob = 100 * ons.n_positive / ons.n_sample
        ax.errorbar(ons.start + pd.Timedelta(days=7), ob, yerr=[ob - lo, hi - ob], fmt="o", ms=3, color="black",
                    lw=0.8, label="ONS survey")
        ax.set_yscale("log")
        ax.set_ylim(0.01, 12)
        ax.set_title(g, fontsize=9)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.set_xlim(pd.Timestamp(CFG["ons_from"]) - pd.Timedelta(days=3), y.start.max() + pd.Timedelta(days=14))
        ax.tick_params(axis="x", labelsize=7)
        ax.axvline(pd.Timestamp(CFG["main"]), ls="--", color="grey", lw=0.7)
        if j == 0:
            ax.set_ylabel("Positivity (%)")
            ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    fig.text(0.01, 0.40, "C. ONS positivity: survey (fitted) and model (" + main.split(",")[0] + "), by age", fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.95, bottom=0.07)
    fig.savefig(path, dpi=200)


def dat_ons(g):
    o = pd.read_csv(IN / "ons.csv", parse_dates=["start", "end"])
    return o[(o.group == g) & (o.start >= CFG["ons_from"]) & (o.n_sample > 0)]


def run_fits(keys):
    dat = load()
    PARTS.mkdir(parents=True, exist_ok=True)
    for key in keys:
        name = NAME[key]
        print(PERIOD, key, "-", name, flush=True)
        f, data = fit(dat, **FITS[key]["opt"])
        r, q, p, dg, A = summarise(f, data, dat, name)
        print(dg, flush=True)
        for nm, x in [("R", r), ("q", q), ("ons_fit", p), ("diagnostics", pd.DataFrame([dg])), ("A", A)]:
            if x is not None:
                x.to_csv(PARTS / f"{CFG['tag']}__{key}__{nm}.csv", index=False)


def combine():
    out = {}
    for nm in ["R", "q", "ons_fit", "diagnostics", "A"]:
        xs = [pd.read_csv(PARTS / f"{CFG['tag']}__{k}__{nm}.csv") for k in FITS
              if (PARTS / f"{CFG['tag']}__{k}__{nm}.csv").exists()]
        if not xs:
            continue
        out[nm] = pd.concat(xs, ignore_index=True)
        out[nm].to_csv(ROOT / "results" / f"{CFG['tag']}_{nm}.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(out["R"].pivot(index="week_start", columns="model", values="median").round(2))
        print(out["diagnostics"])
    replot()


def replot():
    plot(pd.read_csv(ROOT / "results" / f"{CFG['tag']}_R.csv", parse_dates=["week_start"]),
         pd.read_csv(ROOT / "results" / f"{CFG['tag']}_q.csv", parse_dates=["week_start"]),
         pd.read_csv(ROOT / "results" / f"{CFG['tag']}_ons_fit.csv", parse_dates=["start"]),
         ROOT / "figures" / f"{CFG['tag']}.png")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--plot-only" in args:
        replot()
    elif "--combine" in args:
        combine()
    elif "--fit" in args:
        run_fits(args[args.index("--fit") + 1].split(","))
    else:
        run_fits(DEFAULT_FITS)
        combine()
