"""Shared code: simulate the multi-group renewal epidemic, fit the Stan models, summarise and plot."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel
from scipy.stats import gamma

ROOT = Path(__file__).resolve().parent
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "figures").mkdir(exist_ok=True)

T, S, T_CHANGE = 200, 60, 100          # days, generation-time support, day contacts change
# Weekly blocks for R_t: day 1 and days 2-8 form block 1; blocks then start on days 9, 16, ...
# (so on days 72, 100 and 128, aligned with the changes).
BLOCK = np.r_[1, np.arange(T - 1) // 7 + 1]
NB = BLOCK.max()
BLOCK_START = np.array([np.argmax(BLOCK == b) + 1 for b in range(1, NB + 1)])


def R_constant(day):
    return 1.1


def R_steps(day):
    return 1.3 if day < 70 else (0.9 if day < 130 else 1.2)


def gen_time(mean=15.3, sd=9.3):
    """Gamma generation time discretised to days 1..S."""
    shape, rate = (mean / sd) ** 2, mean / sd ** 2
    F = gamma.cdf(np.arange(S + 1), a=shape, scale=1 / rate)
    w = np.diff(F)
    return w / w.sum()


def spectral_radius(M):
    return np.max(np.abs(np.linalg.eigvals(M)))


def settled_mix(M):
    """Left Perron vector of M (rows = infectors), normalised to sum to 1."""
    vals, vecs = np.linalg.eig(M.T)
    v = np.abs(np.real(vecs[:, np.argmax(np.abs(vals))]))
    return v / v.sum()


def simulate(C1, C2, p, q, R_of_day, w, n0, rng):
    """Infections X follow the multi-group renewal equation with M = diag(p) C, where C
    switches from C1 to C2 on day T_CHANGE, and transmission scaled so that the long-run
    R_t (spectral radius of the next-generation matrix) equals R_of_day(day).
    Reported cases Y are binomial thinnings of X with group-specific probabilities q."""
    return simulate_general(lambda day: C2 if day >= T_CHANGE else C1, lambda day: p, q, R_of_day, w, n0, rng)


def simulate_general(C_of_day, p_of_day, q, R_of_day, w, n0, rng):
    """As simulate(), but contacts C_of_day(day) and relative infectiousness p_of_day(day)
    can be any functions of the day, and q can be a function of the day too."""
    N = len(p_of_day(1))
    X = np.zeros((T, N), dtype=np.int64)
    X[0] = rng.multinomial(n0, settled_mix(p_of_day(1)[:, None] * C_of_day(1)))
    for t in range(1, T):
        day = t + 1
        M = p_of_day(day)[:, None] * C_of_day(day)
        s = min(S, t)
        Lam = np.array([w[i, :s] @ X[t - 1::-1, i][:s] for i in range(N)])
        X[t] = rng.poisson(R_of_day(day) / spectral_radius(M) * (M.T @ Lam))
    if callable(q):
        Y = rng.binomial(X, np.array([q(d) for d in range(1, T + 1)]))
    else:
        Y = rng.binomial(X, q[None, :])
    return X, Y


_models = {}


def model(name):
    if name not in _models:
        _models[name] = CmdStanModel(stan_file=str(ROOT / "stan" / f"{name}.stan"))
    return _models[name]


def _diagnostics(fit, names):
    summ = fit.summary()
    rhat_col = [c for c in summ.columns if c.lower().replace("_", "") == "rhat"][0]
    rows = [r for r in summ.index if any(r.startswith(n + "[") for n in names)]
    return float(summ.loc[rows, rhat_col].max()), int(np.sum(fit.divergences))


def fit_route_b(Y, w, Cs, structured, rw, seed=1, period=None, link=False):
    """Multi-group renewal model fitted to reported cases by group (stan/route_b.stan).
    Cs: one contact matrix per period. period: period (1, 2, ...) of each day; by default,
    with two matrices, period 2 starts on day T_CHANGE. link: free matrices follow a random
    walk across periods instead of being independent."""
    Tn, N = Y.shape
    if period is None:
        period = [2 if (len(Cs) == 2 and d >= T_CHANGE) else 1 for d in range(1, Tn + 1)]
    data = dict(T=Tn, N=N, S=S, Y=Y.astype(int).tolist(), w=w, B=NB, block=BLOCK.tolist(),
                nP=len(Cs), period=list(period), structured=int(structured), C=np.stack(Cs), rw=int(rw),
                link=int(link))
    # Start the overdispersion and random-walk scales away from zero: a chain started near
    # zero for both can get stuck there (8 groups, S1).
    inits = {"inv_phi": 0.05, "tau": 0.05}
    if link and not structured:
        inits["sigG"] = [0.1]
    fit = model("route_b").sample(data=data, chains=4, parallel_chains=4, seed=seed, adapt_delta=0.95,
                                  inits=inits, show_progress=False, show_console=False)
    rhat, div = _diagnostics(fit, ["R", "a_raw", "b_raw"] if structured else ["R"])
    out = dict(R=fit.stan_variable("R"), max_rhat=rhat, divergences=div)
    if structured:
        out["a"] = fit.stan_variable("a_raw")
        out["b"] = fit.stan_variable("b_raw")
    return out


def fit_single(Y, w, seed=1):
    """Single-group negative-binomial renewal model fitted to reported totals."""
    data = dict(T=Y.shape[0], S=S, I=Y.sum(axis=1).astype(int).tolist(), w=w[0], B=NB,
                block=BLOCK.tolist(), nb=1)
    fit = model("single_group").sample(data=data, chains=4, parallel_chains=4, seed=seed,
                                       show_progress=False, show_console=False)
    rhat, div = _diagnostics(fit, ["R"])
    return dict(R=fit.stan_variable("R"), max_rhat=rhat, divergences=div)


def summarise_R(draws, R_of_day, **labels):
    """Weekly posterior median and 95% interval for R_t, with the truth mid-block."""
    q = np.quantile(draws, [0.5, 0.025, 0.975], axis=0)
    return pd.DataFrame(dict(**labels, block=np.arange(1, NB + 1), start=BLOCK_START,
                             median=q[0], lo=q[1], hi=q[2],
                             truth=[R_of_day(d + 3) for d in BLOCK_START]))


def error_table(res, by):
    """Relative error of weekly medians and 95% coverage, from day 61, split into the five
    weeks after the contact change and all other weeks."""
    d = res[res.start >= 61].copy()
    d["window"] = np.where((d.start >= T_CHANGE) & (d.start < T_CHANGE + 35), "5 weeks after change", "other weeks")
    d["err"] = (d["median"] - d.truth).abs() / d.truth
    d["cov"] = (d.lo <= d.truth) & (d.hi >= d.truth)
    g = d.groupby(by + ["window"], sort=False).agg(err=("err", "mean"), cov=("cov", "mean")).reset_index()
    g["value"] = [f"{e:.2f} ({100 * c:.0f}%)" for e, c in zip(g.err, g["cov"])]
    out = g.pivot(index=by, columns="window", values="value").reset_index()
    out["model"] = pd.Categorical(out["model"], [m for m in COLOURS if m in set(out["model"])], ordered=True)
    return out.sort_values(by).reset_index(drop=True)


COLOURS = {
    "Single group, reported totals": "#d1495b",
    "Free matrices, weekly blocks": "#e0a100",
    "Free matrices, random walk": "#a0711c",
    "Survey shape x multipliers, weekly blocks": "#3b75af",
    "Survey shape x multipliers, random walk": "#1b3f6b",
    "Free matrices per wave, random walk": "#e0a100",
    "Free matrices linked across waves, random walk": "#a0711c",
    "Survey waves x multipliers, weekly blocks": "#3b75af",
    "Survey waves x multipliers, random walk": "#1b3f6b",
}


def plot_panel(ax, d, title):
    models = [m for m in COLOURS if m in set(d.model)]
    offsets = np.linspace(-2, 2, len(models))
    tr = d[d.model == models[0]].sort_values("start")
    ax.step(tr.start + 3, tr.truth, where="mid", color="black", lw=1.2)
    for m, off in zip(models, offsets):
        x = d[d.model == m]
        ax.errorbar(x.start + 3 + off, x["median"], yerr=[x["median"] - x.lo, x.hi - x["median"]],
                    fmt="o", ms=2.5, lw=1, color=COLOURS[m], label=m)
    ax.axhline(1, ls=":", color="black", lw=0.8)
    ax.axvline(T_CHANGE, ls="--", color="grey", lw=0.8)
    ax.set_ylim(0.4, 1.6)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Day")


def plot_two_groups(res, path):
    """Figure 1 of the note. Redraw from saved results with:
    python -c "import pandas as pd, common; common.plot_two_groups(pd.read_csv('results/two_groups_R.csv'), 'figures/two_groups.png')"
    """
    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.7), sharey=True)
    for ax, sc, title in zip(axes, ["S1", "S2"], ["S1: R constant", "S2: R steps"]):
        plot_panel(ax, res[(res.scenario == sc) & (res.start >= 61)], title)
    axes[0].set_ylabel("$R_t$")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0.15, 1, 1), pad=0.3)
    fig.savefig(path, dpi=200)


def log_r_axis(ax, lo, hi):
    """Log-scale R_t axis with plain-number ticks."""
    from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ticks = [t for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1, 1.2, 1.5, 2, 2.5, 3, 4) if lo <= t <= hi]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
