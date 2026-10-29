"""Download the raw data for the England example into data/raw/ (not committed: about 600 MB).

Sources (all open licences):
  UKHSA coronavirus dashboard archive, cases (Open Government Licence v3.0)
    https://ukhsa-dashboard.data.gov.uk/covid-19-archive-data-download
  CoMix social contact survey, UK, v4 (CC-BY-4.0), Gimma et al., doi:10.5281/zenodo.4905745
    https://zenodo.org/records/13684044
  ONS Coronavirus (COVID-19) Infection Survey datasets, England (Open Government Licence v3.0)
    https://www.ons.gov.uk/peoplepopulationandcommunity/healthandsocialcare/conditionsanddiseases/datasets/coronaviruscovid19infectionsurveydata
Usage: python data/download_data.py
"""
import urllib.request
import zipfile
from pathlib import Path

RAW = Path(__file__).resolve().parent / "raw"
ONS = ("https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/healthandsocialcare/conditionsanddiseases/"
       "datasets/coronaviruscovid19infectionsurveydata/2021")
COMIX = "https://zenodo.org/api/records/13684044/files"
FILES = {
    "ukhsa_cases/cases.zip": "https://archive.ukhsa-dashboard.data.gov.uk/coronavirus-dashboard/cases.zip",
    "comix/CoMix_Uk_dictionary.xlsx": f"{COMIX}/CoMix_Uk_dictionary.xlsx/content",
    "comix/CoMix_uk_participant_common.csv": f"{COMIX}/CoMix_uk_participant_common.csv/content",
    "comix/CoMix_uk_contact_common.csv": f"{COMIX}/CoMix_uk_contact_common.csv/content",
    "comix/CoMix_uk_sday.csv": f"{COMIX}/CoMix_uk_sday.csv/content",
    # full-year edition: fortnightly positivity by age/school year, May 2020 to Dec 2021 (Table 1j)
    "ons_cis/ons_cis_2021_edition.xlsx": f"{ONS}/20220107covid19infectionsurveydatasetsengland.xlsx",
    # releases of 16 and 30 April 2021: daily modelled positivity by age/school year (28 Feb to 24 Apr 2021)
    "ons_cis/ons_cis_20210416.xlsx": f"{ONS}/previous/v19/covid19infectionsurveydatasets202104161.xlsx",
    "ons_cis/ons_cis_20210430.xlsx": f"{ONS}/previous/v22/covid19infectionsurveydatasets20210430eng1.xlsx",
}
UKHSA_MEMBERS = ["Cases/nation_newCasesBySpecimenDateAgeDemographics_areas_letter_E.csv",
                 "Cases/nation_newCasesBySpecimenDate.csv"]

for rel, url in FILES.items():
    dest = RAW / rel
    if dest.exists():
        print(f"have {rel}")
        continue
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {rel}", flush=True)
    urllib.request.urlretrieve(url, dest)

with zipfile.ZipFile(RAW / "ukhsa_cases/cases.zip") as z:
    for m in UKHSA_MEMBERS:
        out = RAW / "ukhsa_cases" / Path(m).name
        if not out.exists():
            out.write_bytes(z.read(m))
print("done")
