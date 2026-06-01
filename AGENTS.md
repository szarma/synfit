# AGENTS.md — synfit (numerical core)

## What this is

`synfit` is a **pure-Python** numerical library (numpy / scipy / pandas; no
Django, no web, no I/O beyond reading bundled seed CSVs) for dose-response
curve fitting and drug-combination synergy. It is developed here as a git
submodule and consumed by the private `synfit_app` as a path dependency
(`../core`); it is being prepared for **public release**, so treat the public
API as a contract (see Versioning).

## Layout

| Module | Purpose |
|--------|---------|
| `hill.py` | Hill curve, `log_wall` soft-boundary prior, concentration series |
| `single.py` | `SingleDrugFit` (and `SingleDrugFitWithError`) — 4p/5p single-drug fit |
| `matrix.py` | `MatrixFit` — full-surface Bliss-independence null model |
| `joint_marginal.py` | `JointMarginalFit` — shared-asymptote two-drug joint fit (the **default** matrix path in the app) |
| `bliss.py` / `loewe.py` / `zip.py` | Synergy reference models |
| `noise.py` | `NoiseSpec` variants + log-likelihood (gaussian const/linear/quadratic, lognormal, additive+multiplicative) |
| `fitting.py` | `FitBase` — shared optimiser + covariance machinery |
| `data.py` | `FitConfig`, `FitBounds`, `FitResult` |
| `plotting.py` | matplotlib/Bokeh plots |
| `synthetic.py` | synthetic data generators |
| `data/synthetic/` | committed seed configs + CSVs |

## Commands (uv + just)

- `just check` — full gate: `uv sync --extra dev` + all tests. Source of truth.
- `just test` / `just test-coverage` — `uv run pytest` (pytest `pythonpath` is
  preconfigured to `["src", "."]`, so no `PYTHONPATH=` needed, and
  `from tests.helpers import ...` works).
- `just generate-seed-data` / `just check-seed-data` — regenerate / verify the
  committed seed CSVs.

## Versioning

- `pyproject.toml` `version` is the source of truth, exported as
  `synfit.__version__` (via `importlib.metadata`). The app stamps it onto every
  fit as `Analysis.core_version`.
- **SemVer, with a domain rule:** a change that *moves fitted values*
  (optimiser, objective, priors, bounds, noise) is **≥ MINOR**, never PATCH.
- Update `CHANGELOG.md` `[Unreleased]` in the same commit as the change; a
  release promotes it to a dated, tagged `[X.Y.Z]`. Tag on the **default
  branch after merge**, not on a feature-branch tip.

## Architecture: the `FitBase` contract

Fitters subclass `FitBase` and implement `_log_prob_data(x, **kwargs)`.
`FitBase` provides the shared `_run_minimize` and `_estimate_covariance`.

- **Optimisation and the Hessian run in scale-relative coordinates**
  (`z = x / scale`, `scale = max(|x0|, 1.0)`), and results are back-transformed
  (`Σ_x = D Σ_z D`). This is load-bearing — it makes fits scale-invariant and
  keeps covariances valid for large-magnitude (ELISA/RFU) data. **Do not
  reintroduce a raw `scipy.optimize.minimize` call or an absolute-step
  Hessian** in a fitter; that was the exact bug fixed in 0.2.0.
- Multi-parameter fitters (`JointMarginalFit`, `MatrixFit`) don't fit the
  single-drug `FitConfig`/`FitBounds` attribute schema, so — by design — they
  pass explicit `self._x0` / `self._bounds` to `self._run_minimize(...)` and
  **override `_log_prior_prob`**, bypassing the config-driven
  `_get_x0_and_bounds` / `_x_to_kwargs`. Mirror that pattern for new fitters.
- The objective is `-(log_prob_data + log_prior_prob)`; the prior is a soft
  `log_wall` barrier scaled by `PRIOR_PENALTY_WEIGHT`.

## Gotchas

1. **`param_cov` can be `None`.** `_estimate_covariance` returns `None` for a
   singular / ill-conditioned / non-finite Hessian, and zeroes rows/cols for
   bound-active parameters (degenerate CI). Callers must handle missing or
   partial covariance — never assume a CI exists.
2. **`MatrixFit` is not yet scale-invariant.** The 6-parameter Bliss surface is
   ill-conditioned; it drifts by ~10× and collapses by ~100× in magnitude
   (#14). It's a *diagnostic null model*, not the default matrix path
   (`JointMarginalFit` is, and that one *is* scale-invariant). Don't wire
   `MatrixFit` into a default flow without addressing #14.
3. **Optimiser preconditioning uses `x0`.** `scale` comes from the initial
   guess. The built-in configs seed asymptote `x0` from the data min/max, so
   this is fine — but a hand-crafted `FitConfig` with an unrepresentative `x0`
   is not preconditioned (#14).
4. **Lognormal noise needs strictly-positive predictions.** The lower asymptote
   bound is floored at `1e-6`; only *constant* variance is supported under
   lognormal (heteroscedastic σ²(μ) doesn't compose with log-space residuals).
5. **Parameter names are direction-neutral.** `effect_0` = low-concentration
   asymptote, `effect_inf` = high-concentration asymptote; `log_c50` is fitted
   in log space (`c50` is linear, external). The `direction`
   (`inhibition`/`activation`) field swaps which asymptote is the displayed
   IC₅₀/EC₅₀ top/bottom — the same `hill_curve` handles both.
6. **Seed CSVs are committed.** Edit `data/synthetic/*/config.json`, then
   `just generate-seed-data`; CI runs `just check-seed-data` to catch drift.
7. **Tests are mandatory** for any behaviour change; a fix needs a test that
   reproduces the bug. Scale/numerical claims belong in
   `tests/test_covariance_scaling.py`; read neighbouring tests for fixtures
   (`tests/helpers.py`).
