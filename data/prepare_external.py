"""Outside estimates of R for England, used as benchmarks in the summary figure.

  official_R_england.csv  government (SPI-M-O, later UKHSA) published ranges for England (90% intervals,
                          a combination of many models), from the final time series of December 2022.
                          Each range is said to describe transmission "2 to 3 weeks ago", so it is
                          given the reference window [published - 21 days, published - 14 days].
  react_R_england.csv     REACT-1 random-sample swab survey, R within each round or half-round,
                          entered by hand from the round reports and papers (source URL in each row;
                          plot = 1 marks the one estimate per period shown in the figure). Intervals
                          are 95% credible intervals where stated. The generation time assumed by
                          REACT changes over time (mean 6.3 d in round 15, 4.6 d in round 16, 3.3 d
                          for Omicron in rounds 17-19). Not produced by this script.
Usage: python -I data/prepare_external.py   (after python data/download_data.py)
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RAW, OUT = HERE / "raw" / "external", HERE / "processed" / "external"
OUT.mkdir(parents=True, exist_ok=True)

x = pd.read_excel(RAW / "r_growth_timeseries_20221221.ods", sheet_name="Table1_-_R", engine="odf", header=None)
head = x.index[x[1].astype(str).str.strip() == "Date"][0]
cols = x.loc[head].ffill()
eng = [j for j in x.columns if cols[j] == "England"]          # lower, upper
d = x.loc[head + 2:, [1] + eng].dropna(subset=[1])
d.columns = ["published", "lo", "hi"]
d = d[pd.to_numeric(d.lo, errors="coerce").notna()].astype({"lo": float, "hi": float})
d["published"] = pd.to_datetime(d.published.astype(str).str.rstrip("*").str.strip(), format="mixed")
d["ref_start"] = d.published - pd.Timedelta(days=21)
d["ref_end"] = d.published - pd.Timedelta(days=14)
d.to_csv(OUT / "official_R_england.csv", index=False, date_format="%Y-%m-%d")
print(len(d), "official estimates,", d.published.min().date(), "to", d.published.max().date())
