# agestratifiedrt

Simulations for the note *Estimating the long-run reproduction number from cases reported by group*.

A multi-group renewal model is fitted to reported cases by group, with the transmission matrix
normalised to spectral radius 1, so its scale factor is the long-run $R_t$. Constant
differences in reporting between groups do not bias $R_t$; the simulations check this when
the mix of cases shifts after a change in contacts.

## Setup (once)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m cmdstanpy.install_cmdstan    # about 10 minutes; skip if CmdStan is installed
```

## Reproduce

```bash
python worked_example.py     # Section 3 numbers (seconds)
python run_two_groups.py     # Table 1 and Figure 1 (about 5 minutes)
python run_more_groups.py    # 4 and 8 groups (about 25 minutes)
python run_waves.py          # survey waves, change date unknown (about 15 minutes)
python run_reporting.py      # reporting that changes over time, with an infection survey (about 1 hour)
python run_england.py spring2021   # England, January to May 2021 (about 40 minutes)
python run_england.py autumn2020   # England, August to November 2020 (about 50 minutes)
python run_england.py autumn2021   # England, August to December 2021 (Delta)
```

Tables and posterior summaries are written to `results/` and figures to `figures/`.
Seeds are fixed. Results can differ slightly between machines and CmdStan versions.

## Running fits on GitHub Actions

The England fits take 30 minutes to several hours each. To run them on GitHub's runners instead
of a laptop, list the commands in `jobs.txt` and push it to `main`:

```text
run_england.py autumn2021 --fit latent_ons_gq      # one job per line, run in parallel
run_holdout.py autumn2021 0 --models latent_ons_gq
after: run_england.py autumn2021 --combine         # runs once all jobs have finished
```

The workflow (`.github/workflows/fits.yml`) installs CmdStan (cached after the first run), runs each
line as its own job on a 4-core runner (6-hour limit per job), then commits the new results,
figures and job logs (`results/logs/`) back to `main`. Pull to get them. It can also be started
from the Actions tab.

Per-fit outputs go to `results/parts/` and hold-out outputs to `results/holdout/`; `--combine` and
`run_holdout.py --summarise` turn them into the tables in `results/` and the figures.

## Real data (England, 2020-21)

```bash
python data/download_data.py      # about 1.2 GB into data/raw/ (not committed)
python data/prepare_data.py       # small CSVs into data/processed/ (committed)
python data/prepare_england.py spring2021   # model inputs into data/processed/england/ (committed)
python data/prepare_england.py autumn2020   # into data/processed/england_autumn2020/ (committed)
python data/prepare_england.py autumn2021   # into data/processed/england_autumn2021/ (committed)
```

`run_england.py` needs only the committed files in `data/processed/england*/`, so the download
step can be skipped.

| File in `data/processed/` | Contents | Source |
|---|---|---|
| `england_cases_by_age.csv` | Daily cases by specimen date, 5-year age bands | UKHSA dashboard archive (OGL v3) |
| `england_cases_total.csv` | Daily cases, all ages (check: age bands sum to this) | UKHSA dashboard archive |
| `comix_participants_by_week.csv`, `comix_contacts_by_week.csv` | Contacts by survey week, participant age, contact age and setting (unweighted) | CoMix UK v4, Gimma et al., [doi:10.5281/zenodo.4905745](https://doi.org/10.5281/zenodo.4905745) (CC-BY-4.0) |
| `ons_cis_daily_by_age.csv` | Modelled daily positivity by age/school year, 28 Feb to 24 Apr 2021 | ONS Infection Survey, releases of 16 and 30 April 2021 (OGL v3) |
| `ons_cis_14day_by_age.csv` | Fortnightly positivity by age/school year, May 2020 to Dec 2021 | ONS Infection Survey, 2021 edition (OGL v3) |
| `england/`, `england_autumn2020/`, `england_autumn2021/` | Model inputs, 4 Jan to 16 May 2021, 3 Aug to 29 Nov 2020 and 16 Aug to 5 Dec 2021, groups 0-9, 10-19, 20-49, 50-69, 70+: cases, CoMix contact matrices by week (population-weighted, contacts capped at 50 a day), ONS positives and people tested by fortnight, vaccine coverage, population | As above, plus UKHSA vaccinations by age and local authority, and ONS mid-2020 population estimates via Nomis (OGL v3) |
| `key_dates_england.csv` | School and policy dates (entered by hand; check before use) | |

Notes. The UKHSA age file holds many snapshots of each day's count without release dates; we
take the largest, as the ukc19 package does, which makes the age bands sum to the final total.
CoMix contacts are individually reported contacts only (large group contacts excluded), with
contact age taken as the midpoint of the reported range. From 8 March 2021 schoolchildren were
tested twice weekly with lateral flow tests, so reporting in children changed exactly when
schools reopened: the constant-reporting assumption fails there. The vaccination archive lacks
most of February 2021 for local authorities; cumulative doses are interpolated linearly between
31 January and 1 March.

Contains public sector information licensed under the Open Government Licence v3.0.

## Files

| File | What it does |
|---|---|
| `stan/route_b.stan` | Multi-group model for cases by group: free matrices or survey shape × multipliers; weekly or random-walk $R_t$ |
| `stan/single_group.stan` | Single-group renewal model for total cases (the reference) |
| `common.py` | Simulation, fitting, summaries and plotting |
| `worked_example.py` | Reporting changes the matrix but not its spectral radius |
| `run_two_groups.py` | Two groups, scenarios S1 and S2 |
| `run_more_groups.py` | 4 and 8 age-like groups |
| `run_reporting.py` | Children's reporting steps or ramps up (as with school testing); fits with constant reporting, weekly reporting from cases only, and weekly reporting plus a random-sample infection survey |
| `stan/route_b_reporting.stan` | Multi-group model with weekly reporting rates and an optional infection-survey likelihood |
| `comix_adapt_ons`, `latent_adapt_ons` (fits in `run_england.py`) | Susceptibility and infectiousness by age as weekly random walks (after Hinch et al. 2025), instead of assumed vaccine effectiveness and depletion |
| `stan/england_latent.stan` | Hidden-infections version: infections are unknowns generated by the renewal equation, with cases and the ONS survey as noisy observations |
| `run_holdout.py` | Hold-out check: leaves out alternate ONS windows and scores the predictions |
| `jobs.txt`, `.github/workflows/fits.yml` | Fits to run on GitHub Actions |
| `run_england.py`, `stan/england.stan` | England, spring 2021, autumn 2020 and autumn 2021: CoMix waves × multipliers, vaccination as known loss of susceptibility, day-of-week effects, weekly reporting and the ONS Infection Survey; compared with single-group fits to totals |
| `data/prepare_england.py` | Builds the England model inputs |
| `run_susceptibility.py` | Simulation: susceptibility changing over time by group (vaccination with waning, depletion); compares ignoring it, coverage with true or assumed protection, and random-walk susceptibility |
| `run_waves.py` | Two-week survey waves instead of a known change date: noisy survey matrices per wave, free matrices linked across waves, a gradual change, drifting infectiousness |
