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
```

Tables and posterior summaries are written to `results/` and figures to `figures/`.
Seeds are fixed. Results can differ slightly between machines and CmdStan versions.

## Files

| File | What it does |
|---|---|
| `stan/route_b.stan` | Multi-group model for cases by group: free matrices or survey shape × multipliers; weekly or random-walk $R_t$ |
| `stan/single_group.stan` | Single-group renewal model for total cases (the reference) |
| `common.py` | Simulation, fitting, summaries and plotting |
| `worked_example.py` | Reporting changes the matrix but not its spectral radius |
| `run_two_groups.py` | Two groups, scenarios S1 and S2 |
| `run_more_groups.py` | 4 and 8 age-like groups |
