# synfit

Core numerical functions for drug response curve fitting.

## Overview

`synfit` is a pure Python package (no Django) that provides:

- 4-parameter Hill equation fitting for single-drug dose-response data
- Drug-drug interaction matrix fitting with Bliss independence comparison
- Configurable `NoiseSpec` models: constant/heteroscedastic Gaussian, Lognormal, and compound additive+multiplicative noise
- Synthetic data generation for testing and validation
- Static matplotlib plotting (dose-response curves, heatmaps)

## Installation

```bash
# Install with development dependencies
uv sync --extra dev
```

## Quality Gates

```bash
# Full quality gate (seed data check + all tests)
just check

# Individual commands
just test                # Run all tests
just test-coverage       # Run tests with coverage report
just generate-seed-data  # Regenerate synthetic seed CSVs
just check-seed-data     # Verify seed CSVs are up-to-date (CI check)

# Show all available commands
just help
```

## Project Structure

```
synfit/
├── src/synfit/
│   ├── hill.py          # Hill equation, log_wall prior, concentration series
│   ├── bliss.py         # Bliss independence model, deviation calculation
│   ├── fitting.py       # FitBase: scipy minimize, log-prior, bounded optimization
│   ├── single.py        # SingleDrugFit, SingleDrugFitWithError
│   ├── matrix.py        # MatrixFit: drug-drug interaction fitting
│   ├── noise.py         # NoiseSpec union, serde, profiled/weighted likelihood
│   ├── data.py          # Dataclasses: FitConfig, FitBounds, FitResult
│   ├── plotting.py      # Static matplotlib: dose-response, heatmaps
│   ├── synthetic.py     # Synthetic data generation from config dicts
│   └── __init__.py      # Public API exports
├── tests/
│   ├── test_hill.py     # Hill equation, concentration series
│   ├── test_fitting.py  # scipy minimize on synthetic data
│   ├── test_bliss.py    # Bliss independence
│   ├── test_noise.py    # Error models, profiled variance, masking
│   ├── test_matrix.py   # Matrix fit parameter recovery
│   └── test_plotting.py # Plot smoke tests (PNG output)
├── data/                # Synthetic seed datasets (committed, CI-verified)
│   └── */config.json    # Generation configs with ground truth
├── scripts/
│   └── generate_seed_datasets.py  # Seed data generator (supports --check)
├── pyproject.toml
└── justfile
```

## Key Concepts

### Fitting

All fitters inherit from `FitBase`, which provides:
- `scipy.optimize.minimize` with multiple restarts
- Bounded parameter search via `FitBounds`
- `log_wall` soft boundary prior (smooth penalty near bounds)
- Profiled variance log-likelihood (MLE variance computed analytically)

### Noise models

Configured via `FitConfig.noise`, a tagged `NoiseSpec` union serialized as JSON (for example `{"kind": "gaussian_linear", "a_init": 0.001, "b_init": 0.0}`). Variants are `GaussianConstant`, `GaussianLinear`, `GaussianQuadratic`, `Lognormal`, and `CompoundAddMult`. Constant Gaussian and Lognormal profile one variance term analytically; Gaussian linear/quadratic fit response-dependent variance coefficients; per-point `y_err` weights remain available for constant Gaussian/Lognormal fits.

### Synthetic Data

`synfit.synthetic` generates reproducible test data from config dicts:

```python
from synfit.synthetic import generate_single_drug, generate_matrix

df = generate_single_drug({
    "seed": 42,
    "hill_params": {"ic50": 5.0, "hill": 1.8, "top": 1.0, "bottom": 0.02},
    "concentration_series": {"initial_conc": 100.0, "fold_dilutions": 3.0, "length": 8, "has_zero": True},
    "n_replicates": 3,
    "noise_sigma_log": 0.05,
    "outliers": [],
})
```

## License

Licensed under the MIT License — see [LICENSE](LICENSE).
