"""One figure comparing R_t across the England periods for three fits: totals, totals + ONS,
and the multi-group model with the survey (the best available version: hidden infections if
run, else reporting sd by age, else the shared sd).
Usage: python plot_england_summary.py   (after run_england.py ... --combine for each period)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from common import ROOT, log_r_axis

PERIODS = [("england_autumn2020", "Autumn 2020", {"2020-09-03": "Schools", "2020-09-21": "Universities",
                                                   "2020-10-26": "Half-term", "2020-11-05": "Lockdown"}),
           ("england", "Spring 2021", {"2021-03-08": "Schools + LFD", "2021-04-02": "Easter"}),
           ("england_summer2021", "Summer 2021 (Delta)", {"2021-07-19": "Restrictions end", "2021-07-22": "Schools close"}),
           ("england_autumn2021", "Autumn 2021 (Delta)", {"2021-09-02": "Schools", "2021-10-25": "Half-term"}),
           ("england_omicron", "Winter 2021-22 (Omicron)", {"2022-01-04": "Schools", "2022-01-11": "No PCR"})]
PERIODS = [p for p in PERIODS if (ROOT / "results" / f"{p[0]}_R.csv").exists()]   # periods fitted so far
MAIN = ["Hidden infections, random-walk susceptibility by age + ONS", "CoMix x random-walk susceptibility by age + ONS",
        "Hidden infections, CoMix, reporting sd by age + ONS", "CoMix x multipliers, reporting sd by age + ONS",
        "CoMix x multipliers, weekly reporting + ONS"]
SHOW = {"Single group, totals": ("Totals", "#d1495b"),
        "Single group, totals + ONS, weekly reporting": ("Totals + ONS", "#e89aa5")}

plt.rcParams.update({"font.size": 9})
fig, axes = plt.subplots(len(PERIODS), 1, figsize=(7.5, 2.35 * len(PERIODS)), squeeze=False)
axes = axes[:, 0]
for ax, (tag, title, events) in zip(axes, PERIODS):
    res = pd.read_csv(ROOT / "results" / f"{tag}_R.csv", parse_dates=["week_start"])
    main = next(m for m in MAIN if m in set(res.model))
    show = dict(SHOW, **{main: ("Age groups + CoMix + ONS", "#1b3f6b")})
    start = res.week_start.min() + pd.Timedelta(days=14)
    for (m, (lab, col)), off in zip(show.items(), [-1.5, 0, 1.5]):
        x = res[(res.model == m) & (res.week_start >= start)]
        ax.errorbar(x.week_start + pd.Timedelta(days=3 + off), x["median"],
                    yerr=[x["median"] - x.lo, x.hi - x["median"]], fmt="o", ms=2.5, lw=1, color=col, label=lab)
    ax.axhline(1, color="black", lw=0.6)
    lo, hi = 0.5, min(2.2, res[res.week_start >= start].hi.max() + 0.05)
    log_r_axis(ax, lo, hi)
    prev, level = None, 0
    for d, lab in events.items():
        d = pd.Timestamp(d)
        level = level + 1 if prev is not None and (d - prev).days < 10 else 0   # stack labels of close events
        prev = d
        ax.axvline(d, ls="--", color="grey", lw=0.7)
        ax.text(d + pd.Timedelta(days=1), hi * 0.97 * 0.9 ** level, lab, fontsize=7, color="grey", va="top")
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_ylabel("$R_t$")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
axes[0].legend(fontsize=7.5, frameon=False, loc="lower left", ncol=3)
fig.tight_layout(pad=0.4)
fig.savefig(ROOT / "figures" / "england_summary.png", dpi=200)
