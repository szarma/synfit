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

### Added
- **`default_fit_config` / `default_bounds`** — public entry points to the
  data-driven default derivation (initials + asymptote/log-c50 bounds, noise-
  aware) that `SingleDrugFit` already used internally. Lets consumers that
  surface *editable* bounds (e.g. the app's analysis modals) source their
  defaults from one place instead of re-deriving them. ``noise`` accepts a
  ``NoiseSpec``, a tagged dict, or a kind string (a ``"lognormal"`` string /
  dict now applies the positive-floor like a ``Lognormal()`` instance).

### Changed
- **Bounds are honoured literally; absent bounds are derived from data.**
  `SingleDrugFit` no longer uses the all-or-nothing equality override
  (`config.bounds == FitBounds()` → swap in data-derived). Instead,
  `FitConfig.bounds` defaults to ``None`` meaning "derive from data at fit
  time"; a config whose bounds were *not supplied* gets data-driven bounds (and
  its still-at-default initials seeded from data), while *explicitly-supplied*
  bounds are used verbatim. Storing ``None`` (rather than a derived-then-flagged
  value) keeps the semantic correct across `dataclasses.replace` / `asdict`
  round-trips. **Fitted values change** for the previously-unhandled case of a
  bare `FitConfig()` against non-normalised data (e.g. `direction="inhibition"`
  with default `[0, 2]` asymptote bounds no longer clamps large-magnitude
  responses) — hence a MINOR bump. Explicit-bounds and `config=None` paths are
  unchanged.

## [0.3.0] — 2026-06-01

**Fitted values change** for `MatrixFit` (edge pre-fit bounds now data-derived),
hence the MINOR bump. Single-drug / joint-marginal fits with data-appropriate
configs are unaffected.

### Changed
- **Full optimiser scale-invariance ([#14](https://github.com/szarma/synfit/issues/14)).**
  The per-parameter preconditioning scale is ``max(|x0|, 1)``, with a
  bound-magnitude fallback (``max(|x0|, bound magnitude, 1)``) applied **only to
  the magnitude-bearing asymptotes** (``effect_0`` / ``effect_inf`` /
  ``top`` / ``bottom``). A hand-crafted ``FitConfig`` with an unrepresentative
  ``x0`` but data-appropriate bounds then fits the same at any response
  magnitude. The fallback deliberately never touches log-domain location
  parameters (``log_c50``…) or dimensionless shape exponents (``hill``…,
  ``asymmetry``): their bounds describe a log / shape range, so scaling them by
  the bound magnitude would wreck the optimiser conditioning. ``MatrixFit`` edge
  pre-fits now derive initials and bounds from data (as the joint-marginal path
  already did), and the 6-parameter Bliss surface is scale-equivariant across
  0.01×–1000× rescaling (single-drug fits hold to 0.001×).

### Fixed
- **Small EC₅₀ no longer over-preconditioned.** The first cut of #14 applied the
  bound-magnitude fallback to *every* parameter, including ``log_c50``. With the
  default seed (``log_c50 = 0``) against wide ``(-5, 5)`` log-decade bounds, that
  inflated the ``log_c50`` scale to the bound magnitude, so one optimiser step
  spanned ~5 decades of EC₅₀ — sending the heteroscedastic ``gaussian_linear``
  fit to a spurious optimum (EC₅₀ ≈ 15 vs a true 5) with a non-invertible Hessian
  and a discarded covariance. Restricting the fallback to the asymptotes restores
  both the EC₅₀ recovery and the covariance. Guarded by
  ``test_small_c50_with_wide_log_bounds_keeps_covariance``.

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
