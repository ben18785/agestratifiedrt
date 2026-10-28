"""4 and 8 groups (the last sentence of the note's Results).

Age-like groups with assortative contacts; contacts within the youngest quarter of groups
rise fourfold on day 100. Reporting rises with age from 0.2 to 0.6 and relative
infectiousness falls from 1 to 0.6. Truths S1 and S2 as in run_two_groups.py.
Usage: python run_more_groups.py [N ...]   (default: 4 8)
"""
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import (ROOT, T_CHANGE, R_constant, R_steps, gen_time, simulate, fit_route_b, fit_single,
                    summarise_R, error_table, plot_panel)

Ns = [int(n) for n in sys.argv[1:]] or [4, 8]
truths = {"S1": R_constant, "S2": R_steps}
fits = {"Free matrices, random walk": (0, 1), "Survey shape x multipliers, weekly blocks": (1, 0),
        "Survey shape x multipliers, random walk": (1, 1)}

res, diag = [], []
for N in Ns:
    i = np.arange(1, N + 1)
    C1 = (6 * np.exp(-np.abs(i[:, None] - i[None, :]) / (N / 4)) + 1) * 8 / N
    young = slice(0, max(1, N // 4))
    C2 = C1.copy()
    C2[young, young] *= 4
    p = np.linspace(1, 0.6, N)
    q = np.linspace(0.2, 0.6, N)
    w = np.tile(gen_time(), (N, 1))
    for k, (sc, Rf) in enumerate(truths.items()):
        rng = np.random.default_rng(200 + 10 * N + k)
        X, Y = simulate(C1, C2, p, q, Rf, w, n0=500 * N, rng=rng)
        for name, (structured, rw) in fits.items():
            print(f"N = {N}, {sc}: {name}", flush=True)
            f = fit_route_b(Y, w, [C1, C2], structured, rw)
            res.append(summarise_R(f["R"], Rf, N=N, scenario=sc, model=name))
            diag.append(dict(N=N, scenario=sc, model=name, max_rhat=f["max_rhat"], divergences=f["divergences"]))
        f = fit_single(Y, w)
        res.append(summarise_R(f["R"], Rf, N=N, scenario=sc, model="Single group, reported totals"))
        diag.append(dict(N=N, scenario=sc, model="Single group, reported totals", max_rhat=f["max_rhat"],
                         divergences=f["divergences"]))

res = pd.concat(res)
res.to_csv(ROOT / "results" / "more_groups_R.csv", index=False)
tab = error_table(res, ["N", "scenario", "model"])
tab.to_csv(ROOT / "results" / "more_groups_table.csv", index=False)
pd.DataFrame(diag).to_csv(ROOT / "results" / "more_groups_diagnostics.csv", index=False)
with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 3):
    print("\nRelative error of weekly R_t medians (95% coverage), from day 61\n", tab)
    print("\nDiagnostics\n", pd.DataFrame(diag))

fig, axes = plt.subplots(2, len(Ns), figsize=(5 * len(Ns), 6), sharey=True, squeeze=False)
for c, N in enumerate(Ns):
    for r, sc in enumerate(truths):
        plot_panel(axes[r, c], res[(res.N == N) & (res.scenario == sc) & (res.start >= 61)], f"{N} groups, {sc}")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(ROOT / "figures" / "more_groups.png", dpi=200)
