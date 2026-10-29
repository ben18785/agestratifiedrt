"""Simulation test: susceptibility that changes over time by group.

Three groups (children, adults, older people), contacts known and constant, a known path for
the long-run R_t, weekly reporting by group, and a random-sample infection survey (2000 people
per group per fortnight). Susceptibility changes over time in three scenarios:
  S1  vaccination of older people (coverage 0 -> 90% over days 40-70), with protection against
      infection of 70% that wanes (half-life of about 2 months); no depletion
  S2  depletion: small populations and a large epidemic in children, so their susceptibility falls
  S3  S1 and S2 together, plus children's reporting stepping from 0.2 to 0.4 on day 70
The fits use the same Stan model as the England analysis (stan/england.stan):
  totals    single group, total cases (reference, no survey)
  ignore    age groups + survey, susceptibility constant (no coverage, no depletion)
  truth     age groups + survey, with vaccine coverage x the true protection over time and depletion
  guessed   age groups + survey, coverage x an assumed protection of 85% with no waning, and depletion
  adapt     age groups + survey, susceptibility and infectiousness by group as weekly random walks
            (no coverage, no depletion)
Usage:
  python run_susceptibility.py --check                 # describe the simulated data (no fitting)
  python run_susceptibility.py --fit SCENARIO KEY      # one fit; outputs to results/parts_sus/
  python run_susceptibility.py --combine               # table and figure
"""
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gamma

from common import ROOT, model, _diagnostics, spectral_radius

PARTS = ROOT / "results" / "parts_sus"
T, S, D = 150, 21, 30
GROUPS = ["Children", "Adults", "Older"]
C = np.array([[6.0, 3.0, 0.5],                 # contacts per day (rows: infectors)
              [1.5, 4.0, 1.0],
              [0.5, 2.0, 2.0]])
P_INF = np.array([1.0, 1.0, 1.0])              # relative infectiousness
DAYS = np.arange(1, T + 1)
WEEK = (DAYS - 1) // 7 + 1
WIN_START = np.arange(31, T - 12, 14)          # 14-day survey windows
N_TESTED = 2000
Q_ADULT_OLDER = (0.5, 0.6)


def R_true(day):
    return 1.3 if day < 50 else 0.9 if day < 100 else 1.1


def gen_time():
    F = gamma.cdf(np.arange(S + 1), a=(5.5 / 2.1) ** 2, scale=2.1 ** 2 / 5.5)
    w = np.diff(F)
    return w / w.sum()


def detection():
    u = np.arange(D) + 0.5
    return gamma.cdf(u, a=(2.5 / 1.5) ** 2, scale=1.5 ** 2 / 2.5) * (1 - gamma.cdf(u, a=(14 / 5) ** 2, scale=5 ** 2 / 14))


def coverage(day):
    """Vaccine coverage of older people."""
    return 0.9 * np.clip((day - 40) / 30, 0, 1)


def protection_true(day):
    """Protection against infection among the vaccinated: 70%, waning with a half-life of 60 days
    from the middle of the campaign."""
    return 0.7 * 0.5 ** (np.maximum(day - 55, 0) / 60)


SCEN = {
    "S1": dict(title="S1: vaccination of older people, waning", pop=np.array([4e6, 8e6, 4e6]), vacc=True,
               deplete=False, n0=3000, q_child=lambda d: 0.2),
    "S2": dict(title="S2: depletion (large epidemic in children)", pop=np.array([3e5, 6e5, 3e5]), vacc=False,
               deplete=True, n0=800, q_child=lambda d: 0.2),
    "S3": dict(title="S3: vaccination, waning, depletion, reporting step", pop=np.array([3e5, 6e5, 3e5]),
               vacc=True, deplete=True, n0=800, q_child=lambda d: 0.2 if d < 70 else 0.4),
}
FITS = {
    "totals": "Single group, totals",
    "ignore": "Susceptibility constant",
    "truth": "Coverage x true protection, depletion",
    "guessed": "Coverage x assumed protection (85%, no waning), depletion",
    "adapt": "Random-walk susceptibility by group",
}
COL = {"totals": "#d1495b", "ignore": "#a0711c", "truth": "#2a9d8f", "guessed": "#e9a23b", "adapt": "#1b3f6b"}


def vacc_factor(day, sc, protection):
    v = np.ones(3)
    if sc["vacc"]:
        v[2] = 1 - coverage(day) * protection(day)
    return v


def simulate(key):
    sc = SCEN[key]
    rng = np.random.default_rng({"S1": 101, "S2": 202, "S3": 303}[key])
    w = gen_time()
    X = np.zeros((T, 3))
    cum = np.zeros(3)
    sus_true = np.zeros((T, 3))
    for t in range(T):
        day = t + 1
        dep = np.clip(1 - cum / sc["pop"], 0, 1) if sc["deplete"] else np.ones(3)
        sus = vacc_factor(day, sc, protection_true) * dep
        sus_true[t] = sus
        M = P_INF[:, None] * C * sus[None, :]
        if t == 0:
            vals, vecs = np.linalg.eig(M.T)
            mix = np.abs(np.real(vecs[:, np.argmax(np.real(vals))]))
            mix = mix / mix.sum()
        if t < S:                          # seeding: a steady inflow for the first S days
            X[t] = rng.poisson(sc["n0"] * mix)
        else:
            s = min(S, t)
            Lam = np.array([w[:s] @ X[t - 1::-1, i][:s] for i in range(3)])
            X[t] = rng.poisson(R_true(day) / spectral_radius(M) * (M.T @ Lam))
        cum += X[t]
    q = np.array([[sc["q_child"](d), *Q_ADULT_OLDER] for d in DAYS])
    Y = rng.binomial(X.astype(np.int64), q)
    det = detection()
    prev = np.zeros((T, 3))
    for t in range(D - 1, T):
        prev[t] = det @ X[t::-1][:D] / sc["pop"]
    srng = np.random.default_rng({"S1": 111, "S2": 222, "S3": 333}[key])
    pos = np.array([[srng.binomial(N_TESTED, prev[s - 1:s + 13, j].mean()) for j in range(3)] for s in WIN_START])
    return dict(X=X, Y=Y, q=q, prev=prev, pos=pos, sus=sus_true, cum=cum)


def stan_data(key, sim, fit_key):
    sc = SCEN[key]
    N = 1 if fit_key == "totals" else 3
    nP = int(WEEK.max())
    weekmid = np.array([DAYS[WEEK == k].mean() for k in range(1, nP + 1)])
    if fit_key == "truth":
        v = np.array([vacc_factor(d, sc, protection_true) for d in weekmid])
    elif fit_key == "guessed":
        v = np.array([vacc_factor(d, sc, lambda d: 0.85) for d in weekmid])
    else:
        v = np.ones((nP, N))
    Y = sim["Y"].sum(1, keepdims=True) if N == 1 else sim["Y"]
    pos = sim["pos"]
    n_tst = np.full_like(pos, N_TESTED)
    if N == 1:
        n_tst, pos = n_tst.sum(1, keepdims=True), pos.sum(1, keepdims=True)
    ons = 0 if fit_key == "totals" else 1
    return dict(T=T, N=N, S=S, t0=S + 1, Y=Y.astype(int).tolist(), w=np.tile(gen_time(), (N, 1)),
                B=nP, block=WEEK.tolist(), nP=nP, period=WEEK.tolist(),
                C=np.ones((nP, 1, 1)) if N == 1 else np.tile(P_INF[:, None] * C, (nP, 1, 1)), v=v,
                pstart=[int(np.argmax(WEEK == k)) + 1 for k in range(1, nP + 1)],
                deplete=int(fit_key in ("truth", "guessed") and sc["deplete"]), s0=np.ones(N), adapt=int(fit_key == "adapt"),
                dow=((DAYS - 1) % 7 + 1).tolist(), est_q=ons, tau_by_group=int(N > 1), W=nP, qweek=WEEK.tolist(),
                ons=ons, Npop=np.array([sc["pop"].sum()]) if N == 1 else sc["pop"], D=D, d=detection(), delta=0,
                K=len(WIN_START), win_start=WIN_START.tolist(), win_end=(WIN_START + 13).tolist(),
                n_tested=n_tst.tolist(), n_pos=pos.tolist())


def fit(key, fit_key):
    sim = simulate(key)
    data = stan_data(key, sim, fit_key)
    N = data["N"]
    inits = {"inv_phi": 0.05, "tau": 0.05, "logR1": 0.2}
    if data["est_q"]:
        inits["tau_q"] = [[0.1] * N]
        inits["lq0"] = [-1.0] * N
    if data["adapt"]:
        inits.update(sigA=[0.05], rho=[0.5], Az=np.zeros((N - 1, data["nP"] - 1)))
    iters = dict(iter_warmup=30, iter_sampling=30) if "--smoke" in sys.argv else {}
    f = model("england").sample(data=data, chains=4, parallel_chains=4, seed=1, adapt_delta=0.95, inits=inits,
                                show_progress=False, show_console=False, **iters)
    rhat, div = _diagnostics(f, ["R"])
    R = f.stan_variable("R")
    qs = np.quantile(R, [0.5, 0.025, 0.975], axis=0)
    starts = np.array([DAYS[WEEK == k][0] for k in range(1, R.shape[1] + 1)])
    res = pd.DataFrame(dict(scenario=key, fit=fit_key, block=np.arange(1, R.shape[1] + 1), start=starts,
                            median=qs[0], lo=qs[1], hi=qs[2], truth=[R_true(d + 3) for d in starts]))
    PARTS.mkdir(parents=True, exist_ok=True)
    out = PARTS if "--smoke" not in sys.argv else PARTS.with_name("parts_sus_smoke")
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / f"{key}__{fit_key}__R.csv", index=False)
    pd.DataFrame([dict(scenario=key, fit=fit_key, max_rhat=rhat, divergences=div)]).to_csv(
        out / f"{key}__{fit_key}__diagnostics.csv", index=False)
    if data["adapt"]:
        lA, rho = f.stan_variable("logA"), f.stan_variable("rho")[:, 0]
        sus = np.exp(rho[:, None, None] * lA)
        pd.DataFrame([dict(scenario=key, group=g, week=k + 1, median=np.median(sus[:, j, k]),
                           lo=np.quantile(sus[:, j, k], 0.025), hi=np.quantile(sus[:, j, k], 0.975))
                      for j, g in enumerate(GROUPS) for k in range(sus.shape[2])]).to_csv(
            out / f"{key}__{fit_key}__A.csv", index=False)
    print(key, fit_key, "max R-hat", round(rhat, 3), "divergences", div, flush=True)


def check():
    for key, sc in SCEN.items():
        sim = simulate(key)
        print(sc["title"])
        print("  attack rate by group:", (sim["cum"] / sc["pop"]).round(3),
              " peak prevalence:", sim["prev"].max(0).round(4),
              " survey positives per window:", sim["pos"].min(), "-", sim["pos"].max())
        print("  susceptibility on days 1, 50, 100, 150:", sim["sus"][[0, 49, 99, 149]].round(2).tolist())
        print("  cases per day (min, max):", sim["Y"].sum(1).min(), sim["Y"].sum(1).max())


def combine():
    res = pd.concat([pd.read_csv(f) for f in sorted(PARTS.glob("*__R.csv"))], ignore_index=True)
    diag = pd.concat([pd.read_csv(f) for f in sorted(PARTS.glob("*__diagnostics.csv"))], ignore_index=True)
    res = res[res.start >= 29]
    err = lambda x: ((x["median"] - x.truth).abs() / x.truth).mean()
    cov = lambda x: ((x.lo <= x.truth) & (x.hi >= x.truth)).mean()
    tab = (res.groupby(["scenario", "fit"]).apply(lambda x: f"{err(x):.2f} ({100 * cov(x):.0f}%)", include_groups=False)
           .unstack("scenario").reindex(list(FITS)))
    tab.index = [FITS[k] for k in tab.index]
    tab.to_csv(ROOT / "results" / "susceptibility_table.csv")
    diag.to_csv(ROOT / "results" / "susceptibility_diagnostics.csv", index=False)
    print("Relative error of weekly R_t medians from day 29 (95% coverage)\n", tab, "\n", diag)
    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.4), gridspec_kw=dict(height_ratios=[1.3, 1]))
    for c, key in enumerate(SCEN):
        ax = axes[0, c]
        d = res[res.scenario == key]
        if d.empty:
            continue
        tr = d[d.fit == d.fit.iloc[0]].sort_values("start")
        ax.step(tr.start + 3, tr.truth, where="mid", color="black", lw=1.2)
        fits = [k for k in FITS if k in set(d.fit)]
        for k, off in zip(fits, np.linspace(-2, 2, len(fits))):
            x = d[d.fit == k]
            ax.errorbar(x.start + 3 + off, x["median"], yerr=[x["median"] - x.lo, x.hi - x["median"]], fmt="o",
                        ms=2.2, lw=0.9, color=COL[k], label=FITS[k])
        ax.set_ylim(0.5, 1.9)
        ax.set_title(SCEN[key]["title"], fontsize=8.5)
        ax = axes[1, c]
        sim = simulate(key)
        rel = sim["sus"] / sim["sus"][:, :1]               # relative to children, as the model estimates
        for j, (g, col) in enumerate(zip(GROUPS, ["#440154", "#21918c", "#bddf26"])):
            ax.plot(DAYS, rel[:, j] / rel[28, j], color=col, lw=1.2, label=f"{g} (truth)")
        fa = PARTS / f"{key}__adapt__A.csv"
        if fa.exists():
            a = pd.read_csv(fa)
            for g, col in zip(GROUPS, ["#440154", "#21918c", "#bddf26"]):
                y = a[a.group == g].sort_values("week")
                base = y[y.week == 5]["median"].iloc[0]
                ax.plot(y.week * 7 - 3, y["median"] / base, color=col, lw=1, ls="--")
        ax.set_ylim(0, 2)
        ax.set_xlabel("Day")
    axes[0, 0].set_ylabel("$R_t$")
    axes[1, 0].set_ylabel("Susceptibility relative to\nchildren (day 29 = 1)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, frameon=False)
    axes[1, 0].legend(fontsize=6.5, frameon=False, loc="lower left")
    fig.tight_layout(rect=(0, 0.07, 1, 1), pad=0.3)
    fig.savefig(ROOT / "figures" / "susceptibility.png", dpi=200)


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--check" in a:
        check()
    elif "--fit" in a:
        i = a.index("--fit")
        fit(a[i + 1], a[i + 2])
    elif "--combine" in a:
        combine()
