# Changelog

All notable changes to **synfit** (the numerical core). Format follows
[Keep a Changelog](https://keepachangelog.com/); versioning is
[SemVer](https://semver.org/), with one domain-specific rule:

> A change that **moves fitted values** (optimiser, objective, priors, bounds,
> noise model) is at least a **MINOR** bump, never a PATCH — downstream results
> shift, even if the public API is unchanged.

The version string lives in `pyproject.toml` and is exported as
`synfit.__version__`. The consuming app records it per analysis
(`Analysis.core_version`). A release promotes `[Unreleased]` to a dated,
tagged `[X.Y.Z]` heading.

## [Unreleased]

## [0.2.0] — 2026-06-01

First version intended for public release. **Fitted values change** vs 0.1.0
for ill-scaled data (see below), hence the MINOR bump.

### Changed
- **Scale-invariant fitting.** The optimiser runs in scale-relative
  coordinates (`z = x / scale`, `scale = max(|x0|, 1)`), so fitted parameters
  no longer depend on the absolute magnitude of the response. Large-magnitude
  responses previously under-converged (tiny gradients tripped L-BFGS-B's
  projected-gradient stop early). Same data scaled by a constant now yields the
  same EC₅₀/Hill and proportionally-scaled asymptotes.

### Fixed
- **Confidence intervals for large-magnitude data.** The Hessian is probed in
  scale-relative coordinates, so an asymptote of order 1e4 no longer gets a
  ~1e-9 *relative* finite-difference step that produced spurious negative
  variances and discarded the whole covariance (`param_cov is None` → "no CI").
- **Non-finite covariances rejected.** `_estimate_covariance` now returns
  `None` on a non-finite optimum or covariance instead of letting `NaN` slip
  past the `diag < 0` sign check.

### Added
- `synfit.__version__`, read from package metadata.
- `MatrixFit` now subclasses `FitBase`: it routes through the shared optimiser
  and reports a parameter covariance (`param_cov` / `param_names`) it never
  produced before. (Full scale-invariance of the 6-parameter Bliss surface is
  not yet delivered — tracked in
  [#14](https://github.com/szarma/synfit/issues/14).)

## [0.1.0]

Initial release (internal). Single-drug Hill fitting (4-/5-parameter),
drug-combination matrix fitting (joint-marginal shared-asymptote fit plus a
Bliss-independence null model), Bliss / Loewe / ZIP synergy references,
configurable noise models (Gaussian constant / linear / quadratic, lognormal,
additive+multiplicative), synthetic data generation, and matplotlib/Bokeh
plotting.
