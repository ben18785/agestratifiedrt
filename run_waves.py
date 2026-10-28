"""Two groups, with contact periods given by survey waves rather than by the (unknown) day
on which contacts change.

Contact surveys run in two-week waves, with boundaries on days 8, 22, ..., 92, 106, ...,
so the change in contacts on day 100 falls inside a wave and no fit is told when it happens.
Each wave's survey matrix is the true average contact matrix over the wave, measured with
noise (300 participants per group, negative-binomial contact counts). Reporting
q = (0.2, 0.6), unknown to every fit. Datasets:
  A: contacts in group 1 step from 2 to 8 on day 100; R_t = 1.1.
  B: same step; R_t = 1.3, then 0.9 from day 70 and 1.2 from day 130.
  C: contacts ramp from 2 to 8 over days 100-121; R_t as in B.
  D: as A, but group 2's infectiousness drifts from 0.5 to 0.75 over the 200 days,
     while the survey-shape model assumes constant multipliers.
"""
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import (ROOT, T, R_constant, R_steps, gen_time, simulate_general, fit_route_b, fit_single,
                    summarise_R, error_table, plot_panel)

q = np.array([0.2, 0.6])
C1 = np.array([[2, 4.8], [1.2, 3.84]])
C2 = np.array([[8, 4.8], [1.2, 3.84]])
w = np.vstack([gen_time(), gen_time()])
DAYS = np.arange(1, T + 1)
WAVE = (DAYS + 6) // 14 + 1                 # survey wave of each day (boundaries on days 8, 22, ...)
NW = WAVE.max()
N_PARTICIPANTS, CONTACT_SIZE = 300, 1.0     # per group per wave; negative-binomial size per participant


def step(day):
    return C2 if day >= 100 else C1


def ramp(day):
    return C1 + (C2 - C1) * np.clip((day - 100) / 21, 0, 1)


def p_fixed(day):
    return np.array([1.0, 0.5])


def p_drift(day):
    return np.array([1.0, 0.5 + 0.25 * (day - 1) / (T - 1)])


datasets = {
    "A": dict(C=step, p=p_fixed, R=R_constant, title="A: step, R constant"),
    "B": dict(C=step, p=p_fixed, R=R_steps, title="B: step, R steps"),
    "C": dict(C=ramp, p=p_fixed, R=R_steps, title="C: ramp, R steps"),
    "D": dict(C=step, p=p_drift, R=R_constant, title="D: step, drifting infectiousness"),
}
fits = {"Free matrices per wave, random walk": dict(structured=0, rw=1, link=False),
        "Free matrices linked across waves, random walk": dict(structured=0, rw=1, link=True),
        "Survey waves x multipliers, weekly blocks": dict(structured=1, rw=0, link=False),
        "Survey waves x multipliers, random walk": dict(structured=1, rw=1, link=False)}


def survey_matrices(C_of_day, rng):
    """Noisy survey estimate of each wave's average contact matrix."""
    out = []
    for k in range(1, NW + 1):
        Cbar = np.mean([C_of_day(d) for d in DAYS[WAVE == k]], axis=0)
        r = N_PARTICIPANTS * CONTACT_SIZE
        m = N_PARTICIPANTS * Cbar
        out.append(rng.negative_binomial(r, r / (r + m)) / N_PARTICIPANTS)
    return out


def plot_waves(res, path):
    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.5), sharey=True)
    for ax, (name, ds) in zip(axes.flat, datasets.items()):
        plot_panel(ax, res[(res.dataset == name) & (res.start >= 61)], ds["title"])
        for b in np.unique(DAYS[np.r_[True, np.diff(WAVE) > 0]])[1:]:
            ax.axvline(b, color="grey", lw=0.3, alpha=0.5)
        ax.set_xlim(60, 205)
    for ax in axes[:, 0]:
        ax.set_ylabel("$R_t$")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1), pad=0.3)
    fig.savefig(path, dpi=200)


def run():
    res, diag, mult = [], [], []
    for k, (name, ds) in enumerate(datasets.items()):
        rng = np.random.default_rng(300 + k)
        X, Y = simulate_general(ds["C"], ds["p"], q, ds["R"], w, n0=1000, rng=rng)
        Cs = survey_matrices(ds["C"], np.random.default_rng(400 + k))
        for fit_name, opt in fits.items():
            print(f"{name}: {fit_name}", flush=True)
            f = fit_route_b(Y, w, Cs, opt["structured"], opt["rw"], period=WAVE, link=opt["link"])
            res.append(summarise_R(f["R"], ds["R"], dataset=name, model=fit_name))
            diag.append(dict(dataset=name, model=fit_name, max_rhat=f["max_rhat"], divergences=f["divergences"]))
            if opt["structured"]:
                p = ds["p"](100)
                for par, dr, truth in [("a2 = (p2/q2)/(p1/q1)", f["a"][:, 0], (p[1] / q[1]) / (p[0] / q[0])),
                                       ("b2 = q2/q1", f["b"][:, 0], q[1] / q[0])]:
                    mult.append(dict(dataset=name, model=fit_name, parameter=par, truth_day100=truth,
                                     median=np.median(dr), lo=np.quantile(dr, 0.025), hi=np.quantile(dr, 0.975)))
        print(f"{name}: single group", flush=True)
        f = fit_single(Y, w)
        res.append(summarise_R(f["R"], ds["R"], dataset=name, model="Single group, reported totals"))
        diag.append(dict(dataset=name, model="Single group, reported totals", max_rhat=f["max_rhat"],
                         divergences=f["divergences"]))

    res = pd.concat(res)
    res.to_csv(ROOT / "results" / "waves_R.csv", index=False)
    tab = error_table(res, ["dataset", "model"])
    tab.to_csv(ROOT / "results" / "waves_table.csv", index=False)
    pd.DataFrame(mult).to_csv(ROOT / "results" / "waves_multipliers.csv", index=False)
    pd.DataFrame(diag).to_csv(ROOT / "results" / "waves_diagnostics.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 3):
        print("\nRelative error of weekly R_t medians (95% coverage), from day 61\n", tab)
        print("\nMultipliers (survey-shape fits)\n", pd.DataFrame(mult))
        print("\nDiagnostics\n", pd.DataFrame(diag))
    plot_waves(res, ROOT / "figures" / "waves.png")


if __name__ == "__main__":
    # python run_waves.py --plot-only redraws the figure from results/waves_R.csv
    if "--plot-only" in sys.argv:
        plot_waves(pd.read_csv(ROOT / "results" / "waves_R.csv"), ROOT / "figures" / "waves.png")
    else:
        run()
