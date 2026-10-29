"""Model inputs for England in five age groups, for one of two periods:
  spring2021  4 January to 16 May 2021  -> data/processed/england/
  autumn2020  3 August to 29 November 2020 -> data/processed/england_autumn2020/
  autumn2021  16 August to 5 December 2021 -> data/processed/england_autumn2021/

Groups (chosen so that cases, CoMix, the ONS Infection Survey and vaccination line up):
  0-9    primary school age
  10-19  secondary school and college age (lateral-flow testing in schools from 8 March 2021)
  20-49
  50-69
  70+
Outputs (committed):
  population.csv    ONS mid-2020 population by group
  cases.csv         cases by specimen date and group
  contacts.csv      CoMix mean contacts per participant per day, by survey week, participant
                    group (rows) and contact group (columns), population-weighted within groups
  ons.csv           ONS Infection Survey positives and people tested, by 14-day window and group
  vaccination.csv   first- and second-dose coverage by date and group
Usage: python -I data/prepare_england.py [spring2021|autumn2020|autumn2021]   (after python data/download_data.py)
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PERIODS = {"spring2021": ("2021-01-04", "2021-05-16", "england"),
           "autumn2020": ("2020-08-03", "2020-11-29", "england_autumn2020"),
           "autumn2021": ("2021-08-16", "2021-12-05", "england_autumn2021")}
START, END, folder = PERIODS[sys.argv[1] if len(sys.argv) > 1 else "spring2021"]
RAW, OUT = HERE / "raw", HERE / "processed" / folder
OUT.mkdir(parents=True, exist_ok=True)
GROUPS = ["0-9", "10-19", "20-49", "50-69", "70+"]
LO = np.array([0, 10, 20, 50, 70])          # lower age of each group
HI = np.array([9, 19, 49, 69, 120])         # upper age of each group


def group_of_age(a):
    return GROUPS[int(np.searchsorted(LO, a, side="right") - 1)]


# ---- population (ONS mid-2020, single year of age) ---------------------------------------
pop = pd.read_csv(RAW / "ons_population/england_mid2020_single_year.csv")
pop["age"] = pop.C_AGE_NAME.str.extract(r"(\d+)").astype(int)
pop_age = pop.set_index("age").OBS_VALUE                # 90 means 90+
pop_g = pop_age.groupby([group_of_age(a) for a in pop_age.index]).sum().reindex(GROUPS)
pop_g.rename("population").rename_axis("group").reset_index().to_csv(OUT / "population.csv", index=False)


def pop_range(lo, hi):
    return pop_age[(pop_age.index >= lo) & (pop_age.index <= hi)].sum()


# ---- cases ------------------------------------------------------------------------------------
c = pd.read_csv(HERE / "processed/england_cases_by_age.csv")
c = c[(c.date >= START) & (c.date <= END)].copy()
lo = c.age_band.str[:2].astype(int)
c["group"] = [group_of_age(a) for a in lo]
cases = c.groupby(["date", "group"]).cases.sum().unstack()[GROUPS]
cases.to_csv(OUT / "cases.csv")

# ---- CoMix ------------------------------------------------------------------------------------
# Participants are recorded in bands; each band is mapped to one group and the bands within a
# group are averaged with population weights (e.g. 0-9 = 0-4 and 5-11 participants, weighted by
# the population aged 0-4 and 5-9). Contact ages are usually ranges (e.g. 45-54): each contact is
# split across groups in proportion to the overlap of its range with each group. Contacts
# recorded only in aggregate (not in contact_common) are not included, and each participant's
# contacts are capped at 50 (scaled down proportionally).
PART = {"0-4": ("0-9", 0, 4), "5-11": ("0-9", 5, 9), "12-17": ("10-19", 10, 19),
        "18-29": ("20-49", 20, 29), "30-39": ("20-49", 30, 39), "40-49": ("20-49", 40, 49),
        "50-59": ("50-69", 50, 59), "60-69": ("50-69", 60, 69), "70-120": ("70+", 70, 120)}
p = pd.read_csv(RAW / "comix/CoMix_uk_participant_common.csv")
s = pd.read_csv(RAW / "comix/CoMix_uk_sday.csv")
s["survey_date"] = pd.to_datetime(dict(year=2000 + s.year, month=s.month, day=s.day))
s = s[(s.survey_date >= pd.Timestamp(START) - pd.Timedelta(days=7)) & (s.survey_date <= END)]
s["week_start"] = (s.survey_date - pd.to_timedelta(s.survey_date.dt.day_of_week, unit="D")).dt.date
s = s.merge(p[["part_id", "part_age"]], on="part_id")
s = s[s.part_age.isin(PART)]
k = pd.read_csv(RAW / "comix/CoMix_uk_contact_common.csv",
                usecols=["part_id", "cnt_age_exact", "cnt_age_est_min", "cnt_age_est_max"])
k = k[k.part_id.isin(s.part_id)]
amin = k.cnt_age_exact.fillna(k.cnt_age_est_min)
amax = k.cnt_age_exact.fillna(k.cnt_age_est_max).clip(upper=120)
ok = amin.notna() & amax.notna()
k, amin, amax = k[ok], amin[ok].values, amax[ok].values
width = amax - amin + 1
for g, l, h in zip(GROUPS, LO, HI):
    k[g] = np.clip(np.minimum(amax, h) - np.maximum(amin, l) + 1, 0, None) / width
per_part = k.groupby("part_id")[GROUPS].sum()
CAP = 50                                     # at most 50 contacts per participant-day (one reported 4002)
tot = per_part.sum(axis=1)
per_part = per_part.mul(np.minimum(1, CAP / tot), axis=0)
s = s.join(per_part, on="part_id").fillna({g: 0.0 for g in GROUPS})
band = s.groupby(["week_start", "part_age"]).agg(n=("part_id", "size"), **{g: (g, "mean") for g in GROUPS})
band = band.reset_index()
band["group"] = band.part_age.map(lambda b: PART[b][0])
band["w"] = band.part_age.map(lambda b: pop_range(PART[b][1], PART[b][2]))
rows = []
for (wk, g), x in band.groupby(["week_start", "group"]):
    wts = x.w / x.w.sum()
    rows.append(dict(week_start=wk, part_group=g, n_participants=int(x.n.sum()),
                     **{h: float((wts * x[h]).sum()) for h in GROUPS}))
contacts = pd.DataFrame(rows)
contacts["part_group"] = pd.Categorical(contacts.part_group, GROUPS)
contacts = contacts.sort_values(["week_start", "part_group"])
contacts.to_csv(OUT / "contacts.csv", index=False)

# ---- ONS Infection Survey (14-day windows, raw counts) ------------------------------------
# 16/17-24 straddles two groups and is left out; 20-49 is 25-34 plus 35-49.
ONSG = {"Age 2 - Age 10/11 (Age 2 - School year 6)": "0-9",
        "Age 11/12 - Age 15/16 (School Year 7 - School Year 11)": "10-19",
        "Age 25 - Age 34": "20-49", "Age 35 - Age 49": "20-49", "Age 50 - Age 69": "50-69", "Age 70+": "70+"}
o = pd.read_csv(HERE / "processed/ons_cis_14day_by_age.csv")
o = o[o.age_group.isin(ONSG) & (o.start >= START) & (o.end <= END)].copy()
o["group"] = o.age_group.map(ONSG)
for col in ["n_positive", "n_sample"]:
    o[col] = pd.to_numeric(o[col], errors="coerce")
ons = o.groupby(["start", "end", "group"], as_index=False).agg(
    n_positive=("n_positive", lambda x: x.sum(min_count=len(x))), n_sample=("n_sample", "sum"))
# a suppressed count ("*", small numbers) in any part of a group: leave that window out (0 tested)
sup = ons.n_positive.isna()
ons.loc[sup, ["n_positive", "n_sample"]] = 0
ons[["n_positive", "n_sample"]] = ons[["n_positive", "n_sample"]].astype(int)
ons["group"] = pd.Categorical(ons.group, GROUPS)
ons.sort_values(["start", "group"]).to_csv(OUT / "ons.csv", index=False)

# ---- vaccination --------------------------------------------------------------------------
# Upper-tier local authorities summed over England. The archive lacks most of February 2021,
# so cumulative doses are interpolated linearly in time within each authority and age band.
VBAND = {"12_15": "10-19", "16_17": "10-19", "18_24": "20-49", "25_29": "20-49", "30_34": "20-49", "35_39": "20-49", "40_44": "20-49",
         "45_49": "20-49", "50_54": "50-69", "55_59": "50-69", "60_64": "50-69", "65_69": "50-69",
         "70_74": "70+", "75_79": "70+", "80_84": "70+", "85_89": "70+", "90+": "70+"}
cols = ["date", "area_code", "ageCategory", "cumPeopleVaccinatedFirstDoseByVaccinationDate",
        "cumPeopleVaccinatedSecondDoseByVaccinationDate", "cumPeopleVaccinatedThirdDoseByVaccinationDate"]
v = pd.concat([pd.read_csv(f, usecols=cols) for f in sorted(glob.glob(str(RAW / "ukhsa_vaccinations/utla_*.csv")))])
v = v[v.area_code.str.startswith("E") & v.ageCategory.isin(VBAND) & (v.date >= "2020-12-01") & (v.date <= max(END, "2020-12-31"))]
v = v.drop_duplicates(["date", "area_code", "ageCategory"])
days = pd.date_range("2020-12-08", max(END, "2020-12-31"))
out = []
for (a, b), x in v.groupby(["area_code", "ageCategory"]):
    x = x.assign(date=pd.to_datetime(x.date)).set_index("date").sort_index()
    x = x[cols[3:]].reindex(days).interpolate(limit_area="inside").fillna(0)
    out.append(x.assign(group=VBAND[b]))
v = pd.concat(out).rename_axis("date").reset_index().groupby(["date", "group"])[cols[3:]].sum()
v.columns = ["dose1", "dose2", "dose3"]
v = v.reset_index()
v = v.merge(pop_g.rename("population"), left_on="group", right_index=True)
for k in (1, 2, 3):
    v[f"cov{k}"] = v[f"dose{k}"] / v.population
v = v.pivot(index="date", columns="group", values=["cov1", "cov2", "cov3"])
v.columns = [f"{m}_{g}" for m, g in v.columns]
for m in ["cov1", "cov2", "cov3"]:
    v[f"{m}_0-9"] = 0.0                     # no vaccination of under-12s in these periods
v = v[[f"{m}_{g}" for m in ["cov1", "cov2", "cov3"] for g in GROUPS]]
v = v.reindex(pd.date_range(min(START, "2020-12-08"), END)).fillna(0.0).rename_axis("date")   # zero before roll-out
v.round(5).to_csv(OUT / "vaccination.csv")

print("population:", pop_g.to_dict())
print("cases:", cases.sum().to_dict())
print("CoMix weeks:", contacts.week_start.nunique(), "participants:", contacts.n_participants.sum())
print("ONS windows:", ons.start.nunique())
for m in ["cov1", "cov2", "cov3"]:
    print(m, "on", END, v.loc[pd.Timestamp(END), [f"{m}_{g}" for g in GROUPS]].round(2).to_dict())
