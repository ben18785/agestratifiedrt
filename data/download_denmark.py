"""Download the raw data for the Denmark example into data/raw/denmark/ (not committed).

Sources (all open):
  Statens Serum Institut, final COVID-19 dashboard file (25 October 2023): weekly people tested
    and positive by age group (Regionalt_DB/18_...), 2020 to 2023
    https://www.ssi.dk/sygdomme-beredskab-og-forskning/sygdomsovervaagning/c/historiske-covid-19-opgoerelser
  CoMix social contact survey, Denmark (CC-BY-4.0), Gimma et al., doi:10.5281/zenodo.6362899
    https://zenodo.org/records/6362899  (21 December 2020 to 15 April 2021)
  Statistics Denmark, population on 1 January 2021 by single year of age (table FOLK1A)
    https://api.statbank.dk
Usage: python data/download_denmark.py
"""
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parent / "raw" / "denmark"
COMIX = "https://zenodo.org/api/records/6362899/files"
FILES = {
    "ssi_dashboard_25102023.zip": "https://files.ssi.dk/covid19/overvagning/dashboard/"
                                  "overvaagningsdata-dashboard-covid19-25102023-b5fc15f50b854829829b8c1060b89398",
    "CoMix_dk_sday.csv": f"{COMIX}/CoMix_dk_sday.csv/content",
    "CoMix_dk_participant_common.csv": f"{COMIX}/CoMix_dk_participant_common.csv/content",
    "CoMix_dk_contact_common.csv": f"{COMIX}/CoMix_dk_contact_common.csv/content",
    "population_2021K1.csv": "https://api.statbank.dk/v1/data/FOLK1A/CSV?OMR%C3%85DE=000&ALDER=*&Tid=2021K1",
}
RAW.mkdir(parents=True, exist_ok=True)
for name, url in FILES.items():
    dest = RAW / name
    if dest.exists():
        print(f"have {name}")
        continue
    print(f"downloading {name}", flush=True)
    urllib.request.urlretrieve(url, dest)
print("done")
