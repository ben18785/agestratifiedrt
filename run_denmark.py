"""Denmark, January to April 2021, five age groups (0-11, 12-19, 20-39, 40-64, 65+).

Testing was high but far from constant: the share of each group tested per week changed several
fold (school pupils were tested heavily once schools reopened, from 8 February for grades 0-4).
Unlike England, the number of people tested in each group and week is published, so reporting
can follow measured testing (log q = beta x log tests per head) instead of being a free random
walk. There is no infection survey, so reporting is only relative (R_t does not need its level).
Alpha replaced the earlier variants over these months; its generation time is taken to be the
same, so its higher transmissibility shows up in R_t.

CoMix surveyed children (0-11) only in the first two rounds (29 December, 7 January), when
schools were closed; later rounds give children's contacts only through adults' reports, and
contacts among children after schools reopened are not observed (the last observed value is
carried forward). The random-walk susceptibility and infectiousness fits can absorb this.

Fits (keys in FITS):
  totals              single group, total cases, constant reporting (the usual approach)
  totals_tests        single group, reporting follows total tests
  comix_const         CoMix x multipliers, constant reporting by group
  comix_weekly        CoMix x multipliers, reporting a random walk by group (cases only)
  comix_tests         CoMix x multipliers, reporting follows tests by group
  comix_tests_rw      reporting follows tests, plus a random walk
  comix_adapt_tests   reporting follows tests; susceptibility and infectiousness by group follow
                      weekly random walks
Usage:
  python run_denmark.py                  # all fits, then combine
  python run_denmark.py --fit KEY[,KEY]  # outputs to results/parts/
  python run_denmark.py --combine        # tables and figure
"""
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import gamma

from common import ROOT, model, _diagnostics

IN = ROOT / "data" / "processed" / "denmark2021"
TAG = "denmark2021"
GROUPS = ["0-11", "12-19", "20-39", "40-64", "65+"]
S = 21
GT_MEAN, GT_SD = 5.5, 2.1
PLOT_FROM = "2021-01-04"
EVENTS = {"2021-02-08": "Grades 0-4 back", "2021-03-01": "Shops (some)", "2021-03-22": "Older pupils (part)",
          "2021-04-06": "Grades 5-8, corona pass"}
FITS = {
    "totals": dict(name="Single group, totals", col="#d1495b", opt=dict(N=1)),
    "totals_tests": dict(name="Single group, reporting follows tests", col="#e89aa5", opt=dict(N=1, test_cov=1)),
    "comix_const": dict(name="CoMix x multipliers, constant reporting", col="#3b75af", opt=dict(N=5)),
    "comix_weekly": dict(name="CoMix x multipliers, weekly reporting (cases only)", col="#a0711c",
                         opt=dict(N=5, est_q=1, tau_by_group=1)),
    "comix_tests": dict(name="CoMix x multipliers, reporting follows tests", col="#1b3f6b", opt=dict(N=5, test_cov=1)),
    "comix_tests_rw": dict(name="CoMix x multipliers, reporting follows tests + random walk", col="#2a9d8f",
                           opt=dict(N=5, test_cov=1, est_q=1, tau_by_group=1)),
    "comix_adapt_tests": dict(name="CoMix x random-walk susceptibility, reporting follows tests", col="#e9a23b",
                              opt=dict(N=5, test_cov=1, adapt=1)),
}
MAIN_PREFERENCE = ["comix_adapt_tests", "comix_tests_rw", "comix_tests"]
NAME = {k: v["name"] for k, v in FITS.items()}
COL = {v["name"]: v["col"] for v in FITS.values()}
PARTS = ROOT / "results" / ("parts_smoke" if "--smoke" in sys.argv else "parts")


def gen_time():
    F = gamma.cdf(np.arange(S + 1), a=(GT_MEAN / GT_SD) ** 2, scale=GT_SD ** 2 / GT_MEAN)
    w = np.diff(F)
    return w / w.sum()


def load():
    Xd = pd.read_csv(IN / "cases_daily.csv", index_col=0, parse_dates=True)[GROUPS]
    Yw = pd.read_csv(IN / "cases_weekly.csv", index_col=0, parse_dates=True)[GROUPS]
    tests = pd.read_csv(IN / "tests_weekly.csv", index_col=0, parse_dates=True)[GROUPS]
    pop = pd.read_csv(IN / "population.csv").set_index("group").population.reindex(GROUPS).values.astype(float)
    con = pd.read_csv(IN / "contacts.csv", parse_dates=["round"])
    rounds = sorted(con["round"].unique())
    # one matrix per round; a group not surveyed in a round gets its row from the other groups'
    # reports (reciprocity) and its within-group contacts from the last round that surveyed it
    Ms, last = [], {}
    for r in rounds:
        x = con[con["round"] == r].set_index("part_group").reindex(GROUPS)
        M = x[GROUPS].values.astype(float)
        for i, g in enumerate(GROUPS):
            if np.isnan(M[i]).any():
                M[i] = M[:, i] * pop / pop[i]
                M[i, i] = last[g]
            else:
                last[g] = M[i, i]
        Ms.append((M * pop[:, None] + (M * pop[:, None]).T) / 2 / pop[:, None])
    weeks = Yw.index
    mid = weeks + pd.Timedelta(days=3)
    nearest = [int(np.argmin(np.abs(pd.DatetimeIndex(rounds) - m))) for m in mid]
    C = np.array([Ms[i] for i in nearest])
    return dict(Xd=Xd, Yw=Yw, tests=tests, pop=pop, C=C, weeks=weeks, days=Xd.index)


def stan_data(dat, N, est_q=0, test_cov=0, adapt=0, tau_by_group=0):
    days, weeks = dat["days"], dat["weeks"]
    wk = ((days - weeks[0]).days // 7 + 1).tolist()
    if N == 1:
        Xd = dat["Xd"].sum(1).values[:, None]
        Yw = dat["Yw"].sum(1).values[:, None]
        lt = np.log(dat["tests"].sum(1).values / dat["pop"].sum())[None, :]
        C = np.ones((len(weeks), 1, 1))
    else:
        Xd, Yw, C = dat["Xd"].values, dat["Yw"].values, dat["C"]
        lt = np.log(dat["tests"].values / dat["pop"]).T
    lt = lt - lt[:, :1]
    return dict(T=len(days), N=N, S=S, NW=len(weeks), wk=wk, k0=int(np.ceil(S / 7)) + 1, Xd=Xd,
                Yw=Yw.astype(int).tolist(), w=np.tile(gen_time(), (N, 1)), C=C, adapt=adapt, est_q=est_q,
                tau_by_group=tau_by_group, test_cov=test_cov, lt=lt)


def fit(dat, **opt):
    data = stan_data(dat, **opt)
    N, NW = data["N"], data["NW"]
    inits = {"inv_phi": 0.05, "tau": 0.05, "logR1": 0.0}
    if opt.get("est_q"):
        inits["tau_q"] = [[0.1] * (N if opt.get("tau_by_group") else 1)]
    if opt.get("test_cov"):
        inits["beta"] = [0.3]
    if opt.get("adapt"):
        inits.update(sigA=[0.05], rho=[0.5], Az=np.zeros((N - 1, NW - 1)))
    iters = dict(iter_warmup=30, iter_sampling=30) if "--smoke" in sys.argv else {}
    f = model("denmark").sample(data=data, chains=4, parallel_chains=4, seed=1, adapt_delta=0.95, inits=inits,
                                show_progress=False, show_console=False, **iters)
    return f, data


def summarise(f, data, dat, name):
    weeks = dat["weeks"]
    R = f.stan_variable("R")
    r = pd.DataFrame(dict(model=name, week_start=weeks, median=np.median(R, 0),
                          lo=np.quantile(R, 0.025, 0), hi=np.quantile(R, 0.975, 0)))
    lq = f.stan_variable("lq")                                  # draws x N x weeks
    qs = []
    for j in range(data["N"]):
        g = "all" if data["N"] == 1 else GROUPS[j]
        x = np.exp(lq[:, j, :])
        qs.append(pd.DataFrame(dict(model=name, group=g, week_start=weeks, median=np.median(x, 0),
                                    lo=np.quantile(x, 0.025, 0), hi=np.quantile(x, 0.975, 0))))
    A = []
    if data["adapt"]:
        lA, rho = f.stan_variable("logA"), f.stan_variable("rho")[:, 0]
        for kind, wt in [("susceptibility", rho), ("infectiousness", 1 - rho)]:
            x = np.exp(wt[:, None, None] * lA)
            for j, g in enumerate(GROUPS):
                A.append(pd.DataFrame(dict(model=name, kind=kind, group=g, week_start=weeks,
                                           median=np.median(x[:, j], 0), lo=np.quantile(x[:, j], 0.025, 0),
                                           hi=np.quantile(x[:, j], 0.975, 0))))
    rhat, div = _diagnostics(f, ["R"])
    treedepth = float(np.mean(f.method_variables()["treedepth__"] >= f.metadata.cmdstan_config.get("max_depth", 10)))
    dg = dict(model=name, max_rhat=rhat, divergences=div, share_max_treedepth=round(treedepth, 3))
    if data["test_cov"]:
        b = f.stan_variable("beta")[:, 0]
        dg["beta"] = f"{np.median(b):.2f} ({np.quantile(b, 0.025):.2f} to {np.quantile(b, 0.975):.2f})"
    if data["N"] > 1:
        for nm in ["a_raw", "b_raw"]:
            x = f.stan_variable(nm)
            dg[nm] = "; ".join(f"{np.median(x[:, i]):.2f}" for i in range(x.shape[1]))
    if data["adapt"]:
        dg["rho"] = f"{np.median(f.stan_variable('rho')):.2f}"
        dg["sigA"] = f"{np.median(f.stan_variable('sigA')):.3f}"
    return r, pd.concat(qs), dg, (pd.concat(A) if A else None)


def run_fits(keys):
    dat = load()
    PARTS.mkdir(parents=True, exist_ok=True)
    for key in keys:
        print(TAG, key, "-", NAME[key], flush=True)
        f, data = fit(dat, **FITS[key]["opt"])
        r, q, dg, A = summarise(f, data, dat, NAME[key])
        print(dg, flush=True)
        for nm, x in [("R", r), ("q", q), ("diagnostics", pd.DataFrame([dg])), ("A", A)]:
            if x is not None:
                x.to_csv(PARTS / f"{TAG}__{key}__{nm}.csv", index=False)


def combine():
    out = {}
    for nm in ["R", "q", "diagnostics", "A"]:
        xs = [pd.read_csv(PARTS / f"{TAG}__{k}__{nm}.csv") for k in FITS if (PARTS / f"{TAG}__{k}__{nm}.csv").exists()]
        if xs:
            out[nm] = pd.concat(xs, ignore_index=True)
            out[nm].to_csv(ROOT / "results" / f"{TAG}_{nm}.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(out["R"].pivot(index="week_start", columns="model", values="median").round(2))
        print(out["diagnostics"])
    plot(out)


def plot(out):
    dat = load()
    res = out["R"].assign(week_start=lambda d: pd.to_datetime(d.week_start))
    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), gridspec_kw=dict(width_ratios=[1.5, 1]))
    ax = axes[0, 0]
    shown = [m for m in COL if m in set(res.model)]
    for m, off in zip(shown, np.linspace(-1.8, 1.8, len(shown))):
        x = res[(res.model == m) & (res.week_start >= PLOT_FROM)]
        ax.errorbar(x.week_start + pd.Timedelta(days=3 + off), x["median"], yerr=[x["median"] - x.lo, x.hi - x["median"]],
                    fmt="o", ms=2.5, lw=1, color=COL[m], label=m)
    ax.axhline(1, color="black", lw=0.6)
    ax.set_ylim(0.4, 1.8)
    ax.set_ylabel("$R_t$ (long-run)")
    ax.set_title("A. Reproduction number, by fit", fontsize=9, loc="left")
    ax.legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    for d, lab in EVENTS.items():
        ax.axvline(pd.Timestamp(d), ls="--", color="grey", lw=0.7)
        ax.text(pd.Timestamp(d) + pd.Timedelta(days=1), 0.42, lab, fontsize=6.5, color="grey", rotation=90, va="bottom")
    cols = plt.cm.viridis(np.linspace(0, 0.9, 5))
    ax = axes[0, 1]
    for g, c in zip(GROUPS, cols):
        ax.plot(dat["weeks"] + pd.Timedelta(days=3), 100 * dat["tests"][g] / dat["pop"][GROUPS.index(g)], color=c, label=g)
    ax.set_ylabel("% of group tested per week")
    ax.set_title("B. Testing by age (PCR, SSI)", fontsize=9, loc="left")
    ax.legend(title="Age", fontsize=7, frameon=False, ncol=2)
    q = out["q"].assign(week_start=lambda d: pd.to_datetime(d.week_start))
    main = next(NAME[k] for k in MAIN_PREFERENCE if NAME[k] in set(q.model))
    ax = axes[1, 1]
    for g, c in zip(GROUPS, cols):
        y = q[(q.model == main) & (q.group == g)]
        ax.plot(y.week_start + pd.Timedelta(days=3), y["median"], color=c, label=g)
        ax.fill_between(y.week_start + pd.Timedelta(days=3), y.lo, y.hi, color=c, alpha=0.2, lw=0)
    ax.set_yscale("log")
    ax.set_ylabel("Reporting relative to 14 Dec 2020")
    ax.set_title("C. Reporting by age (main fit)", fontsize=9, loc="left")
    ax = axes[1, 0]
    if "A" in out:
        A = out["A"].assign(week_start=lambda d: pd.to_datetime(d.week_start))
        A = A[(A.kind == "susceptibility") & (A.model == NAME["comix_adapt_tests"])]
        for g, c in zip(GROUPS, cols):
            y = A[A.group == g]
            ax.plot(y.week_start + pd.Timedelta(days=3), y["median"], color=c, label=g)
            ax.fill_between(y.week_start + pd.Timedelta(days=3), y.lo, y.hi, color=c, alpha=0.2, lw=0)
        ax.set_ylabel("Susceptibility factor (week 1 = 1)")
    ax.set_title("D. Susceptibility by age, random-walk fit (0-11 = 1)", fontsize=9, loc="left")
    for a in axes.flat:
        a.xaxis.set_major_locator(mdates.MonthLocator())
        a.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        a.set_xlim(pd.Timestamp(PLOT_FROM) - pd.Timedelta(days=3), dat["weeks"].max() + pd.Timedelta(days=10))
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / f"{TAG}.png", dpi=200)


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--combine" in args:
        combine()
    elif "--fit" in args:
        run_fits(args[args.index("--fit") + 1].split(","))
    else:
        run_fits(list(FITS))
        combine()
