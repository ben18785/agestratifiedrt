"""Two groups (Table 1 and Figure 1 of the note).

Contacts within group 1 rise from 2 to 8 on day 100. Group 1 is reported with probability
0.2 and group 2 with 0.6; no fit is told this. Two truths:
  S1: R_t = 1.1 throughout.
  S2: R_t = 1.3, then 0.9 from day 70 and 1.2 from day 130.
Fits to reported cases by group (free matrices or survey shape x multipliers, each with
weekly blocks or a random walk for R_t), and a single-group model fitted to reported totals.
"""
import numpy as np
import pandas as pd

from common import (ROOT, R_constant, R_steps, gen_time, simulate, fit_route_b, fit_single,
                    summarise_R, error_table, plot_two_groups)

p = np.array([1.0, 0.5])                 # relative infectiousness
q = np.array([0.2, 0.6])                 # reporting probabilities
C1 = np.array([[2, 4.8], [1.2, 3.84]])   # contacts before day 100 (rows: infectors)
C2 = np.array([[8, 4.8], [1.2, 3.84]])   # after day 100
w = np.vstack([gen_time(), gen_time()])
truths = {"S1": R_constant, "S2": R_steps}
fits = {"Free matrices, weekly blocks": (0, 0), "Free matrices, random walk": (0, 1),
        "Survey shape x multipliers, weekly blocks": (1, 0), "Survey shape x multipliers, random walk": (1, 1)}

res, diag, mult = [], [], []
for k, (sc, Rf) in enumerate(truths.items()):
    rng = np.random.default_rng(100 + k)
    X, Y = simulate(C1, C2, p, q, Rf, w, n0=1000, rng=rng)
    for name, (structured, rw) in fits.items():
        print(f"{sc}: {name}", flush=True)
        f = fit_route_b(Y, w, [C1, C2], structured, rw)
        res.append(summarise_R(f["R"], Rf, scenario=sc, model=name))
        diag.append(dict(scenario=sc, model=name, max_rhat=f["max_rhat"], divergences=f["divergences"]))
        if structured:
            for par, dr, truth in [("a2 = (p2/q2)/(p1/q1)", f["a"][:, 0], (p[1] / q[1]) / (p[0] / q[0])),
                                   ("b2 = q2/q1", f["b"][:, 0], q[1] / q[0])]:
                mult.append(dict(scenario=sc, model=name, parameter=par, truth=truth,
                                 median=np.median(dr), lo=np.quantile(dr, 0.025), hi=np.quantile(dr, 0.975)))
    print(f"{sc}: single group", flush=True)
    f = fit_single(Y, w)
    res.append(summarise_R(f["R"], Rf, scenario=sc, model="Single group, reported totals"))
    diag.append(dict(scenario=sc, model="Single group, reported totals", max_rhat=f["max_rhat"],
                     divergences=f["divergences"]))

res = pd.concat(res)
res.to_csv(ROOT / "results" / "two_groups_R.csv", index=False)
tab = error_table(res, ["scenario", "model"])
tab.to_csv(ROOT / "results" / "two_groups_table.csv", index=False)
pd.DataFrame(mult).to_csv(ROOT / "results" / "two_groups_multipliers.csv", index=False)
pd.DataFrame(diag).to_csv(ROOT / "results" / "two_groups_diagnostics.csv", index=False)
with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 3):
    print("\nRelative error of weekly R_t medians (95% coverage), from day 61\n", tab)
    print("\nMultipliers (survey-shape fits)\n", pd.DataFrame(mult))
    print("\nDiagnostics\n", pd.DataFrame(diag))

plot_two_groups(res, ROOT / "figures" / "two_groups.png")
