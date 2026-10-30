"""Download the raw data for the England example into data/raw/ (not committed: about 600 MB).

Sources (all open licences):
  UKHSA coronavirus dashboard archive, cases (Open Government Licence v3.0)
    https://ukhsa-dashboard.data.gov.uk/covid-19-archive-data-download
  CoMix social contact survey, UK, v4 (CC-BY-4.0), Gimma et al., doi:10.5281/zenodo.4905745
    https://zenodo.org/records/13684044
  ONS Coronavirus (COVID-19) Infection Survey datasets, England (Open Government Licence v3.0)
    https://www.ons.gov.uk/peoplepopulationandcommunity/healthandsocialcare/conditionsanddiseases/datasets/coronaviruscovid19infectionsurveydata
  UKHSA coronavirus dashboard archive, vaccinations by age and upper-tier local authority
    (Open Government Licence v3.0); only the needed files are read from the 760 MB archive
  UKHSA, R value and growth rate time series (Open Government Licence v3.0)
    https://www.gov.uk/guidance/the-r-value-and-growth-rate
  ONS mid-2020 population estimates for England by single year of age, via Nomis
    (Open Government Licence v3.0) https://www.nomisweb.co.uk
Usage: python data/download_data.py
"""
import io
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
    # release of 25 February 2022: the same table to 19 February 2022 (for the Omicron period)
    "ons_cis/ons_cis_20220225.xlsx": f"{ONS[:-5]}/2022/previous/v7/20220225covid19infectionsurveydatasetsengland.xlsx",
    # releases of 16 and 30 April 2021: daily modelled positivity by age/school year (28 Feb to 24 Apr 2021)
    "ons_cis/ons_cis_20210416.xlsx": f"{ONS}/previous/v19/covid19infectionsurveydatasets202104161.xlsx",
    "ons_cis/ons_cis_20210430.xlsx": f"{ONS}/previous/v22/covid19infectionsurveydatasets20210430eng1.xlsx",
    # government (SPI-M-O / UKHSA) published R ranges for England, May 2020 to Dec 2022
    "external/r_growth_timeseries_20221221.ods": "https://assets.publishing.service.gov.uk/media/63a46e3ed3bf7f375c7d833d/"
                                                 "221221_R_and_growth_rate_time_series_for_publication_v1.0.ods",
}
VACC_ZIP = "https://archive.ukhsa-dashboard.data.gov.uk/coronavirus-dashboard/vaccinations.zip"
POP_URL = ("https://www.nomisweb.co.uk/api/v01/dataset/NM_2002_1.data.csv?geography=2092957699&date=2020"
           "&gender=0&c_age=101...191&measures=20100&select=date_name,geography_name,c_age_name,obs_value")
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


class HTTPRangeFile(io.RawIOBase):
    """Read-only file over HTTP range requests, so single members of a large zip can be fetched."""
    def __init__(self, url):
        self.url, self.pos = url, 0
        head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"))
        self.size = int(head.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size or len(b) == 0:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        data = urllib.request.urlopen(req).read()
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


vdir = RAW / "ukhsa_vaccinations"
if not any(vdir.glob("utla_vaccinationsAgeDemographics_areas_letter_*.csv")):
    vdir.mkdir(parents=True, exist_ok=True)
    print("downloading vaccinations by age (UTLA files only)", flush=True)
    with zipfile.ZipFile(io.BufferedReader(HTTPRangeFile(VACC_ZIP), buffer_size=1 << 20)) as z:
        for m in z.namelist():
            if "utla_vaccinationsAgeDemographics_areas_letter_" in m:
                (vdir / Path(m).name).write_bytes(z.read(m))

pop = RAW / "ons_population" / "england_mid2020_single_year.csv"
if not pop.exists():
    pop.parent.mkdir(parents=True, exist_ok=True)
    print("downloading population", flush=True)
    urllib.request.urlretrieve(POP_URL, pop)
print("done")
