# synfit

Dose–response curve fitting and drug-combination synergy scoring for
pharmacology: Hill models (4- and 5-parameter), explicit noise models, and
Bliss, HSA, Loewe and ZIP references for inhibition and activation assays.

## Quick start

```bash
pip install synfit
```

Published on [PyPI](https://pypi.org/project/synfit/) from 0.7.2 onward. (The
`0.0.0a0` placeholder predates the first real release — do not use it for
fitting.) To work from a checkout instead:

```bash
pip install "synfit @ git+https://github.com/szarma/synfit.git"
```

Minimal single-drug fit (inhibition). Half-max concentration (IC₅₀ / EC₅₀) is `result.half_max`; `ic50` appears only in `result.to_dict()`, not as an attribute:

```python
import pandas as pd
from synfit import SingleDrugFit

data = pd.DataFrame({
    "concentration": [0, 0.1, 1, 10, 100, 100],
    "y": [1.0, 0.95, 0.75, 0.35, 0.08, 0.05],
    "replicate": [0, 0, 0, 0, 0, 0],
})
result = SingleDrugFit(data).fit()
print(f"c50 (κ): {result.c50:.3g}")
print(f"IC50:    {result.half_max:.3g}")
print(f"Hill:    {result.hill:.3g}")
cis = result.param_ci()
if cis and "c50" in cis:
    lo, hi = cis["c50"]
    print(f"c50 95% CI: [{lo:.3g}, {hi:.3g}]")
```

## Overview

`synfit` is a pure Python package (no Django) that provides:

- 4-parameter Hill equation fitting for single-drug dose-response data
- Drug-drug interaction matrix fitting with Bliss independence comparison
- Synergy scores: Bliss, HSA, Loewe and ZIP, the latter implementing the
  published Zero Interaction Potency model (Yadav et al. 2015), unit- and
  direction-neutral — see [References](#references)
- Configurable `NoiseSpec` models: constant and heteroscedastic Gaussian and Lognormal
- Synthetic data generation for testing and validation
- Static matplotlib plotting (dose-response curves, heatmaps)

## Installation

**End users (installed package):** see [Quick start](#quick-start) above.

**Development checkout:**

```bash
uv sync --extra dev
```

## Quality Gates

```bash
# Full quality gate (README examples + all tests)
just check

# Individual commands
just test                # Run all tests
just test-coverage       # Run tests with coverage report
just check-readme-examples  # Run README ```python blocks
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
│   ├── generate_seed_datasets.py  # Seed data generator (supports --check)
│   └── check_readme_examples.py   # Execute README python blocks (CI)
├── pyproject.toml
└── justfile
```

## Key Concepts

### Fitting

All fitters inherit from `FitBase`, which provides:

- `scipy.optimize.minimize` (L-BFGS-B) in scale-relative coordinates so fits are stable across response magnitudes
- A single optional retry with refreshed variance initials when heteroscedastic Gaussian noise stalls at the first iteration
- Bounded parameter search via `FitBounds`
- `log_wall` soft boundary prior (smooth penalty near bounds)
- Profiled variance log-likelihood (MLE variance computed analytically)

### Noise models

Configured via `FitConfig.noise`, a tagged `NoiseSpec` union serialized as JSON (for example `{"kind": "gaussian_linear", "a_init": 0.001, "b_init": 0.0}`). Fittable variants are `GaussianConstant`, `GaussianLinear`, `GaussianQuadratic`, and `Lognormal`. Constant Gaussian and Lognormal profile one variance term analytically; Gaussian linear/quadratic fit response-dependent variance coefficients; per-point `y_err` weights remain available for constant Gaussian/Lognormal fits.

### Synthetic Data

`synfit.synthetic` generates reproducible test data from config dicts:

```python
from synfit.synthetic import generate_single_drug, generate_matrix

df = generate_single_drug({
    "seed": 42,
    "hill_params": {"c50": 5.0, "hill": 1.8, "effect_0": 1.0, "effect_inf": 0.02},
    "concentration_series": {"initial_conc": 100.0, "fold_dilutions": 3.0, "length": 8, "has_zero": True},
    "n_replicates": 3,
    "noise_sigma_log": 0.05,
    "outliers": [],
})
```

## References

Synergy scores follow their published definitions:

- **ZIP (Zero Interaction Potency)** — Yadav B, Wennerberg K, Aittokallio T, Tang J.
  *Searching for drug synergy in complex dose–response landscapes using an interaction
  potency model.* Computational and Structural Biotechnology Journal 13:504–513 (2015).
  [doi:10.1016/j.csbj.2015.09.001](https://doi.org/10.1016/j.csbj.2015.09.001) —
  the best-known implementation of this method is SynergyFinder. `synfit` implements
  the published formulation for symmetric (4-parameter) curves; for 5-parameter drugs
  it keeps the moving drug's asymmetry in the slice fits, which is an extension of the
  published method and reduces to it when the asymmetry equals 1. See `src/synfit/zip.py`.

  Tested scope: the δ-scores are checked against fixtures generated by the
  independent [`synergy`](https://pypi.org/project/synergy/) package for two
  dose–response landscapes (one synergistic, one potentiated), agreeing to better
  than `1e-3` in δ units. That is evidence for these cases, not a general proof of
  equivalence to any other implementation.
- **Bliss independence** — Bliss CI. *The toxicity of poisons applied jointly.*
  Annals of Applied Biology 26:585–615 (1939).
- **HSA (Highest Single Agent)** — the combination is compared against whichever
  single agent acts more strongly; see Berenbaum MC. *What is synergy?*
  Pharmacological Reviews 41:93–141 (1989) for the reference frames and how they
  differ. HSA is undefined for an inhibitor combined with an activator.
- **Loewe additivity** — Loewe S, Muischnek H. *Über Kombinationswirkungen.*
  Naunyn-Schmiedeberg's Archiv für experimentelle Pathologie und Pharmakologie
  114:313–326 (1926).

### What each function returns

The synergy helpers do not share one sign convention — they return three
different kinds of quantity, so read the units before reading the sign:

| Function | Returns | Synergy is |
|---|---|---|
| `bliss_independence`, `bliss_reference`, `hsa_reference`, `loewe_reference`, `zip_reference`, `zip_fitted_surface` | the **expected response** under that null model, in response units | not a sign — compare against the observed response |
| `zip_delta`, `zip_scores` | a **signed δ-score**, `-(f_observed - f_zip)` | **negative** |
| `loewe_ci` | a **combination index**, a dimensionless ratio | **below 1** (1 is additive, above 1 antagonistic) |

A deviation score for Bliss, HSA or Loewe is formed by the caller, by comparing
an observed response surface against the reference surface; `synfit` exports the
reference, not the deviation.

## License

Licensed under the MIT License — see [LICENSE](LICENSE).
