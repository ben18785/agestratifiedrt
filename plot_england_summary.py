"""One figure comparing R_t across the England periods for three fits: totals, totals + ONS,
and the multi-group model with the survey (the best available version: hidden infections if
run, else reporting sd by age, else the shared sd), with outside estimates for comparison:
  - the government's published range for England (grey band; 90% intervals combining many models,
    each placed 2-3 weeks before its publication date, the period it is said to describe),
  - REACT-1 random-sample swab survey, R within each round (box: round dates and 95% interval; line:
    estimate), from data/processed/external/react_R_england.csv (source of each row given there).
Usage: python plot_england_summary.py   (after run_england.py ... --combine for each period)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

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

EXT = ROOT / "data" / "processed" / "external"
OFFICIAL = pd.read_csv(EXT / "official_R_england.csv", parse_dates=["ref_start", "ref_end"])
REACT = (pd.read_csv(EXT / "react_R_england.csv", parse_dates=["start", "end"])
         .query("plot == 1") if (EXT / "react_R_england.csv").exists() else None)
GOV_COL, REACT_COL = "#9a9a9a", "#2a9d5c"


def outside(ax, t0, t1):
    """Draw the government range and REACT-1 rounds that overlap [t0, t1]."""
    o = OFFICIAL[(OFFICIAL.ref_end >= t0) & (OFFICIAL.ref_start <= t1)].sort_values("ref_start")
    for _, r in o.iterrows():
        ax.fill_between([r.ref_start, r.ref_end], r.lo, r.hi, color=GOV_COL, alpha=0.28, lw=0, zorder=0)
    if REACT is not None:
        for _, r in REACT[(REACT.end >= t0) & (REACT.start <= t1)].iterrows():
            w = r.end - r.start + pd.Timedelta(days=1)
            ax.add_patch(Rectangle((r.start, r.lo), w, r.hi - r.lo, facecolor=REACT_COL, alpha=0.18,
                                   edgecolor=REACT_COL, lw=0.6, zorder=0.5))
            ax.plot([r.start, r.start + w], [r.R, r.R], color=REACT_COL, lw=1.6, zorder=0.6)


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
    t0 = start - pd.Timedelta(days=3)
    t1 = res.week_start.max() + pd.Timedelta(days=10)
    outside(ax, t0, t1)
    ax.set_xlim(t0, t1)
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
handles, labels = axes[0].get_legend_handles_labels()
handles += [Patch(color=GOV_COL, alpha=0.4, lw=0)]
labels += ["Government range (90%), 2-3 weeks before publication"]
if REACT is not None:
    handles += [(Patch(facecolor=REACT_COL, alpha=0.18, edgecolor=REACT_COL), Line2D([], [], color=REACT_COL, lw=1.6))]
    labels += ["REACT-1 survey round (95%)"]
fig.legend(handles, labels, fontsize=7.5, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0))
fig.tight_layout(pad=0.4, rect=(0, 0.035, 1, 1))
fig.savefig(ROOT / "figures" / "england_summary.png", dpi=200)
