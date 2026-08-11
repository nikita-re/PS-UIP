# PS-UIP: A Propensity-Score-Calibrated Unit Information Prior

This repository contains the implementation and reproducibility code for PS-UIP, a
Bayesian dynamic-borrowing method for augmenting the control arm of a current
randomized trial with multiple historical control sources.

PS-UIP constructs a unit information prior for the current-control parameter in two
stages. The design stage uses baseline covariates to transport each historical source
toward the current-control covariate distribution and quantifies its effective sample
size and overlap. The conflict stage compares transported historical and current
control outcomes and reduces borrowing when they disagree. Historical summaries are
treated as candidate information rather than assumed to be exactly exchangeable with
the current controls.

For source `k`, the borrowed information is

```text
M_k = m_k * r_k * rho_k_star,
M   = sum_k M_k,
```

where `m_k` is the effective sample size after weighting, `r_k` measures covariate
overlap, and `rho_k_star` is the final conflict discount. The bound
`M <= sum_k n_k` limits the amount borrowed but does not provide uniform type-I error
control. Operating characteristics should therefore be evaluated for the intended
trial design.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Repository structure

```text
config.py                 Simulation coefficients and method settings
psuipc/
  dgm.py                  Data-generating mechanisms
  stage1.py               Membership-score transport and overlap summaries
  methods.py              PS-UIP and benchmark methods
  fast_sampler.py         Numba Gibbs and Polya-Gamma sampler
  c_backend.py            Interface to the optional compiled C sampler
  csrc/sampler.c          C implementation of the Gibbs sampler
  run.py                  Main operating-characteristic simulations
  sweep.py                Conflict-magnitude sweep
  sensitivity.py          Discount sensitivity analysis
  application.py          ACTG175 illustration
  outputs/                Tracked summary CSV files
scripts/run_server.sh     Full simulation driver
```

The repository tracks compact summary CSV files. Replicate-level `psuipc_raw*.csv`
files are generated locally and excluded from version control.

## Methods included

| Method | Description |
|---|---|
| `no_borrowing` | Current randomized trial only |
| `pooling` | Current and historical controls pooled without dynamic discounting |
| `standard_uip` | Unit information prior using all available historical information |
| `ps_power_prior` | Propensity-score-adjusted power-prior comparator |
| `ps_uip_c` | Proposed PS-UIP |
| `ps_uip_psonly` | PS-UIP ablation without the aggregate conflict factor |
| `ps_sam` | Matched self-adapting mixture comparator |

## Reproducing the numerical studies

Run commands from the repository root. Fixed seeds are defined by each driver, and
single-threaded BLAS is recommended for reproducible parallel execution.

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

Main operating-characteristic simulations:

```bash
python -m psuipc.run --reps 1000 --n-jobs -1
```

Conflict-magnitude sweep:

```bash
python -m psuipc.sweep --reps 1000 --n-jobs -1
```

Second sample-size configuration and aggregate-discount ablation:

```bash
python -m psuipc.run --reps 1000 --nC 60 --nT 120 --rh 5 --tag _nc60rh5
python -m psuipc.run --methods ablation --reps 1000 --tag _ablation
```

Discount sensitivity analysis:

```bash
python -m psuipc.sensitivity --reps 1000
```

A short smoke run is:

```bash
python -m psuipc.run --reps 20 --draws 200 --tune 200 --n-jobs 1
```

The full simulation driver can be run from Bash, Git Bash, or WSL. `PY` may be set
to a specific Python executable.

```bash
PY=python REPS=20 JOBS=4 bash scripts/run_server.sh
```

## ACTG175 application

The ACTG175 data file is not distributed in this repository. The application script
downloads the public dataset from the CRAN mirror of the `speff2trial` package, checks
the download, and stores a local copy as `ACTG175.txt`. The file is excluded from
version control.

```bash
python -m psuipc.application
```

The application figure is written to
`psuipc/outputs/psuipc_application.png`. Generated figures are ignored by Git.

## Sampling backends

`PSUIPC_BACKEND` selects the sampler.

- `c` uses the optional compiled sampler in `psuipc/csrc/sampler.c`. If the shared
  library is unavailable, the code falls back to the `fast` backend.
- `fast` uses the Numba Gibbs and Polya-Gamma implementation and requires no C
  compiler.
- `pymc` provides an optional cross-check and requires PyMC, nutpie, and ArviZ.

For example:

```bash
PSUIPC_BACKEND=fast python -m psuipc.run --reps 20 --n-jobs 1
```

The C backend can be built on a system with a compatible C compiler:

```bash
bash psuipc/csrc/build.sh
```

Compiled libraries are platform-specific and are not tracked.

## Reproducibility notes

- The design-stage and conflict-stage quantities are estimated before posterior
  sampling, so the current implementation has an empirical Bayes form.
- Propensity-score transport addresses differences in measured covariates and
  depends on adequate covariate overlap.
- Weighted binary-endpoint standard errors are approximate.
- The simulations include negative-drift stress settings, including scenario S3.


## License 

The code is released under the MIT License. 
