"""Model inputs for Denmark, 14 December 2020 to 18 April 2021, in five age groups:
  0-11   (SSI groups 0-2, 3-5, 6-11)  schools reopened for grades 0-4 on 8 February 2021
  12-19  (12-15, 16-19)               older pupils back in stages from March
  20-39, 40-64, 65+ (65-79, 80+)
Outputs (committed) in data/processed/denmark2021/:
  population.csv     Statistics Denmark, 1 January 2021, by group
  cases_weekly.csv   people testing positive by week (ISO weeks, Monday start) and group
  tests_weekly.csv   people tested by week and group (PCR, as published by SSI)
  cases_daily.csv    a smooth daily version of cases_weekly (used only to build infection
                     pressure in the model): each week's cases are spread over its days in
                     proportion to the national daily cases, with the day-of-week pattern removed
  contacts.csv       CoMix mean contacts per participant per day, by survey round, participant
                     group (rows) and contact group (columns), population-weighted within groups
Usage: python -I data/prepare_denmark.py   (after python data/download_denmark.py)
"""
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW, OUT = HERE / "raw" / "denmark", HERE / "processed" / "denmark2021"
OUT.mkdir(parents=True, exist_ok=True)
START, END = pd.Timestamp("2020-12-14"), pd.Timestamp("2021-04-18")
GROUPS = ["0-11", "12-19", "20-39", "40-64", "65+"]
LO = np.array([0, 12, 20, 40, 65])
HI = np.array([11, 19, 39, 64, 120])
SSI = {"00-02": "0-11", "03-05": "0-11", "06-11": "0-11", "12-15": "12-19", "16-19": "12-19",
       "20-39": "20-39", "40-64": "40-64", "65-79": "65+", "80+": "65+"}

# ---- population --------------------------------------------------------------------------------
pop = pd.read_csv(RAW / "population_2021K1.csv", sep=";", encoding="utf-8-sig")
pop = pop[pop.ALDER.str.match(r"^\d+ år$")]
pop_age = pd.Series(pop.INDHOLD.values, index=pop.ALDER.str.extract(r"(\d+)")[0].astype(int).values).sort_index()


def pop_range(lo, hi):
    return pop_age[(pop_age.index >= lo) & (pop_age.index <= hi)].sum()


pop_g = pd.Series([pop_range(l, h) for l, h in zip(LO, HI)], index=GROUPS)
pop_g.rename("population").rename_axis("group").reset_index().to_csv(OUT / "population.csv", index=False)

# ---- weekly tests and positives by age ------------------------------------------------------
z = zipfile.ZipFile(RAW / "ssi_dashboard_25102023.zip")
x = pd.read_csv(io.BytesIO(z.read("Regionalt_DB/18_fnkt_alder_uge_testede_positive_nye_indlaeggelser.csv")),
                sep=";", encoding="latin-1")
x.columns = ["week", "age", "tested_100k", "pos_100k", "hosp_100k", "tested", "positive", "pct"]
x["week_start"] = pd.to_datetime(x.week.str.replace("U", "W") + "-1", format="%G-W%V-%u")
x = x[(x.week_start >= START) & (x.week_start <= END)]
x["group"] = x.age.map(SSI)
wk = x.groupby(["week_start", "group"])[["tested", "positive"]].sum()
cases_w = wk.positive.unstack()[GROUPS]
tests_w = wk.tested.unstack()[GROUPS]
cases_w.to_csv(OUT / "cases_weekly.csv")
tests_w.to_csv(OUT / "tests_weekly.csv")

# ---- smooth daily cases ----------------------------------------------------------------------
d = pd.read_csv(io.BytesIO(z.read("Regionalt_DB/03_bekraeftede_tilfaelde_doede_indlagte_pr_dag_pr_koen.csv")),
                sep=";", encoding="latin-1")
daily = d.groupby(pd.to_datetime(d["Prøvetagningsdato"]))["Bekræftede tilfælde i alt"].sum()
daily = daily.reindex(pd.date_range(START - pd.Timedelta(days=60), END + pd.Timedelta(days=7)), fill_value=0)
ratio = daily / daily.rolling(7, center=True).mean()
dowf = ratio.groupby(ratio.index.day_of_week).mean()           # day-of-week factors
adj = (daily / dowf.reindex(daily.index.day_of_week).values).loc[START:END]
days = pd.date_range(START, END)
wkstart = days - pd.to_timedelta(days.day_of_week, unit="D")
share = adj.values / adj.groupby(wkstart).transform("sum").values
cases_d = pd.DataFrame(cases_w.reindex(wkstart).values * share[:, None], index=days, columns=GROUPS)
cases_d.rename_axis("date").round(3).to_csv(OUT / "cases_daily.csv")
check = pd.concat([cases_w.sum(1), daily.loc[START:END].groupby(wkstart).sum()], axis=1)
print("weekly positives by age vs daily confirmed cases (sum):\n", check.round(0).to_string())

# ---- CoMix ----------------------------------------------------------------------------------
# Participant bands are averaged within each group with weights equal to the population of the
# part of the band that lies in the group (e.g. 18-29 counts towards 12-19 with the population
# aged 18-19 and towards 20-39 with those aged 20-29). Contact ages are split across groups in
# proportion to overlap; contacts per participant-day are capped at 50.
BAND = {"Under 1": (0, 0), "1-4": (1, 4), "5-11": (5, 11), "12-15": (12, 15), "16-17": (16, 17),
        "18-29": (18, 29), "30-39": (30, 39), "40-49": (40, 49), "50-59": (50, 59), "60-69": (60, 69),
        "70-120": (70, 120)}
p = pd.read_csv(RAW / "CoMix_dk_participant_common.csv")
s = pd.read_csv(RAW / "CoMix_dk_sday.csv")
s["survey_date"] = pd.to_datetime(dict(year=s.year, month=s.month, day=s.day))
s = s.merge(p[["part_id", "part_age"]], on="part_id")
s = s[s.part_age.isin(BAND)]
k = pd.read_csv(RAW / "CoMix_dk_contact_common.csv", usecols=["part_id", "cnt_age_exact", "cnt_age_est_min", "cnt_age_est_max"])
k = k[k.part_id.isin(s.part_id)]
amin = k.cnt_age_exact.fillna(k.cnt_age_est_min)
amax = k.cnt_age_exact.fillna(k.cnt_age_est_max).clip(upper=120)
ok = amin.notna() & amax.notna()
k, amin, amax = k[ok].copy(), amin[ok].values, amax[ok].values
for g, l, h in zip(GROUPS, LO, HI):
    k[g] = np.clip(np.minimum(amax, h) - np.maximum(amin, l) + 1, 0, None) / (amax - amin + 1)
per_part = k.groupby("part_id")[GROUPS].sum()
tot = per_part.sum(axis=1)
per_part = per_part.mul(np.minimum(1, 50 / tot), axis=0)
s = s.join(per_part, on="part_id").fillna({g: 0.0 for g in GROUPS})
# survey rounds: CoMix waves, dated by their median survey day
s["round"] = s.groupby("wave").survey_date.transform("median").dt.normalize()
band = s.groupby(["round", "part_age"]).agg(n=("part_id", "size"), **{g: (g, "mean") for g in GROUPS}).reset_index()
rows = []
for r, xr in band.groupby("round"):
    for g, l, h in zip(GROUPS, LO, HI):
        w = np.array([pop_range(max(l, BAND[b][0]), min(h, BAND[b][1])) for b in xr.part_age], float)
        if w.sum() == 0:
            continue
        rows.append(dict(round=r.date(), part_group=g, n_participants=int(xr.n[w > 0].sum()),
                         **{c: float((w * xr[c]).sum() / w.sum()) for c in GROUPS}))
contacts = pd.DataFrame(rows)
contacts.to_csv(OUT / "contacts.csv", index=False)

print("population:", pop_g.to_dict())
print("cases:", cases_w.sum().to_dict())
print("tested (share of population per week, mean):", (tests_w.mean() / pop_g).round(3).to_dict())
print("CoMix rounds:", contacts.groupby("round").n_participants.sum().to_dict())
