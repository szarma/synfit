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

### Fixed
- **Pinned Hill-curve parameters now enter the single-drug likelihood.**
  `FitConfig.fitting_parameters` lists the parameters to estimate; any Hill
  parameter not in that list is meant to be held at its configured value.
  `FitBase._x_to_kwargs` previously built curve kwargs only from the free
  vector, so a pinned `hill` / `effect_0` / `effect_inf` / `asymmetry` was
  silently replaced by `hill_curve`'s own default, while `SingleDrugFit.fit`
  still echoed the configured value on `FitResult`. Pinning `log_c50` crashed
  (`c50` has no default) because `fit` also assumed it was always free.
  Successful fits that pinned a curve parameter to a value other than
  `hill_curve`'s own default (`hill=1.0`, `effect_0=1.0`, `effect_inf=0.0`,
  `asymmetry=1.0`) used the library default rather than the pin and must
  be re-run.

## [0.4.0] — 2026-09-17

### Added
- **`default_fit_config` / `default_bounds`** — public entry points to the
  data-driven default derivation (initials + asymptote/log-c50 bounds, noise-
  aware) that `SingleDrugFit` already used internally. Lets consumers that
  surface *editable* bounds (e.g. the app's analysis modals) source their
  defaults from one place instead of re-deriving them. ``noise`` accepts a
  ``NoiseSpec``, a tagged dict, or a kind string (a ``"lognormal"`` string /
  dict now applies the positive-floor like a ``Lognormal()`` instance).
- **`default_joint_marginal_config`** — the joint-marginal counterpart of
  `default_fit_config`: data-derived initials + bounds for the shared
  ``top``/``bottom`` asymptotes and each drug's ``log_c50``/``hill`` (and
  ``asymmetry`` for 5p), matching what `JointMarginalFit` derives when no
  ``param_config`` overrides are supplied. Lets the app's matrix modal surface
  editable bounds from the same single source of truth. Noise-aware; asymmetry
  keys present only for 5p drugs.
- **`default_joint_marginal_config` includes variance coefficients for heteroscedastic Gaussian noise.** Linear and quadratic Gaussian noise now return `var_*` entries (`init` / `lo` / `hi`) matching `JointMarginalFit` when no `param_config` is supplied. Constant Gaussian and lognormal are unchanged (no `var_*` keys).

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
  responses) — hence a MINOR bump. Explicit bounds are otherwise unchanged, with one
  compatibility note: an explicit `FitBounds()` equal to the old defaults was
  previously replaced by data-derived bounds and is now used literally
  (`[0, 2]` asymptotes). `config=None` paths are unchanged.
- **Parameter-role taxonomy centralised** (`synfit/param_roles.py`). The
  classification of fitting parameters — log-domain locations, variance
  coefficients, magnitude-bearing asymptotes — was previously re-derived in
  several places (`fitting.parameter_scale`, `FitBase._x_to_kwargs`,
  `MatrixFit._unpack_x`, plus per-class name tuples). It now lives behind
  `is_log_param` / `is_variance_param` / `is_asymptote_param` so the role tests
  cannot drift apart. No change to fitted values. `FitResult.param_ci()` now
  recognises exactly `var_a` / `var_b` / `var_c` as variance coefficients (it
  previously matched any five-character `var_*` name; no fitter emits others).
- **Heteroscedastic Gaussian variance is anchored at the lower asymptote.** Linear and quadratic models now use σ² = a + b·d + c·d² with d = μ − m and m = min(effect_0, effect_inf) of the current trial (joint-marginal: m = min(top, bottom)). Hill predictions sit between the asymptotes, so σ² ≥ a > 0 by construction and a baseline shift no longer drives the polynomial negative. Constant Gaussian, lognormal, compound, and user-supplied `y_err` paths are unchanged. Synthetic data generation still uses the true-μ polynomial. *Moves heteroscedastic fitted values relative to the un-anchored polynomial; results stay close to 0.3.0 for data whose predictions are ≥ 0.*
- **`var_a` / `var_b` / `var_c` domain is `a > 0`, `b ≥ 0`, `c ≥ 0`.** Default `FitBounds` lower bounds match that. Explicit negative lower bounds (legacy stored configs sent `var_b` (−1e4, 1e4)) and negative initials are clamped to the domain floor at fitter construction — `1e-12` for `var_a`, or `1e-12·s²` when a response scale exists, and 0 for `var_b` / `var_c`. After clamping, an empty interval (`lo > hi`) raises `ValueError` naming the parameter, the given bounds, and the domain rule, in both single-drug and joint-marginal paths.
- **Omitted heteroscedastic initials are `None`, not class-default numbers.** `GaussianLinear` / `GaussianQuadratic` `a_init` / `b_init` / `c_init` default to `None` meaning data-derived `1e-3·s²` / `0` / `0`. An explicit value, including `a_init=0.001`, is kept after the domain clamp. `to_dict` omits unset fields; `from_dict` without the key yields `None`. Callers without a data scale (`FitConfig` with no data) fall back to the historical `1e-3` / `0` / `0`.
- **Response-scaled variance bounds and default initials.** Data-derived fits set `var_a ∈ [1e-12·s², 10·s²]`, `var_b ∈ [0, 10·s]`, `var_c ∈ [0, 10]` and default initials `a = 1e-3·s²`, `b = 0`, `c = 0`, where `s` is the same robust response range used for asymptote bounds. Explicit `var_*` bounds are honoured literally apart from the domain clamp. This is what makes heteroscedastic fits scale-equivariant (`y → s·y` ⇒ `a → s²a`, `b → s·b`, `c → c`).
- **Reported retry when L-BFGS-B stalls at iteration 0** on a linear/quadratic Gaussian fit with a finite starting objective: one retry from default variance initials (`b = 0`, `c = 0`, `a = 1e-3·s²`), with a note on the result message. Failures after iteration 0, non-finite starts, and invalid input are not retried.
- **Per-asymptote default bounds that meet at the response midpoint.** The two
  asymptotes no longer share one wide symmetric bracket. Each gets its own
  `(lo, hi)`: the bottom is bounded `[…, m]` and the top `[m, …]` at the
  midpoint `m = ½(y_min + y_max)`, so `bottom.lo < bottom.hi == top.lo <
  top.hi`. The gaussian top gets extra headroom (`y_max + 2·Δy`, vs. `y_min −
  ½·Δy` below the bottom) because the response is far less constrained from
  above. *Moves fitted values* — a near-zero bottom asymptote can land on the
  opposite side of zero from the old symmetric scheme.
- **Data-derived asymptote bounds are rounded outward without collapsing narrow
  high-offset ranges.** The bottom ceiling and top floor retain one shared split,
  preserving `bottom.lo < bottom.hi == top.lo < top.hi`; rounding precision also
  follows the observed dynamic range so small positive lognormal floors remain
  positive.
- **Adaptive extrema for small samples instead of a blanket minimum.** The
  robust extremes (second-smallest / second-largest response, which discard a
  lone outlier at each end) are used whenever trimming leaves a positive
  dynamic range; otherwise the true min/max are used. This naturally covers
  two- or three-point inputs and sparse plateaus where trimming would collapse
  the usable range. Only `n < 2` is rejected. The previous hard five-point
  minimum 500'd the matrix-defaults endpoint on 2×2 plates and refused
  legitimate four-point single-drug curves.
- **`MatrixFit` (Bliss) shares the single-drug asymptote-bound derivation.** Its
  shared `effect_0` / `effect_inf` bounds are now the `(min lo, max hi)` merge of
  the two edge configs' data-derived bounds — same as `JointMarginalFit` — rather
  than a fresh `(0.5–2)×` envelope around the pre-fit results. This carries the
  per-asymptote midpoint scheme and outward rounding onto the combination fit and
  floors the lognormal bottom bound strictly positive (it was hardcoded to `0.0`,
  which let the Bliss surface bottom out at zero and break the likelihood).
  *Moves fitted values* for Bliss matrix fits.

### Fixed
- **Optimiser `response_scale` is only applied for linear/quadratic Gaussian fits.** Constant Gaussian, lognormal, and compound fits keep the historical `parameter_scale(x0, bounds, names)` path, so normalized data no longer changes their asymptote preconditioning.
- **A one-row (or empty) response no longer IndexErrors in `_robust_response_range`.** Construction falls through to the existing fit-size `ValueError`.
- **Joint fits and `default_joint_marginal_config` reject compound noise with `ValueError`.** Compound additive-multiplicative noise is not supported for joint-marginal fitting; both the fitter and the public helper now raise instead of an `AttributeError` on missing `sigma_log` bounds.
- **`FitBase` rejects an unresolved `bounds=None` at fit time with an actionable
  error.** `FitConfig.bounds=None` means "derive from data" — a contract the
  data-bearing subclasses resolve before fitting. The base optimiser readers
  (`_get_x0_and_bounds` / `_log_prior_prob`) now route through a `_require_bounds`
  guard, so a future subclass that forgets to resolve gets a clear message
  ("bounds is None at fit time… derive from data") instead of an opaque
  `AttributeError: 'NoneType' object has no attribute 'effect_0'`. No behavioural
  change for the existing fits (all resolve bounds first); the guard deliberately
  does not invent a neutral `FitBounds()` default, which would silently fit
  against the wrong bracket the data-derived scheme replaced.
- **Lognormal feasibility is checked against the true minimum response, not the
  trimmed robust extreme.** A single non-positive `y` breaks the lognormal
  likelihood, so a dataset containing one is now rejected up front (previously a
  negative point hidden behind the second-smallest value slipped through). The
  lognormal asymptote lower bound floors at `y_min/100`, strictly positive.
- **`MatrixFit` rejects mixed-direction plates up front.** The Bliss surface
  applies one shared `(effect_0, effect_inf)` pair to both marginals, so it
  cannot represent one drug activating while the other inhibits — the shared-
  bound merge would otherwise combine one drug's top with the other's bottom.
  A plate with `direction_horizontal != direction_vertical` now raises with a
  pointer to `JointMarginalFit` (whose per-drug top/bottom remapping handles
  mixed directions). Same-direction plates are unaffected.
- **Invalid curve directions raise `ValueError` instead of silently using
  inhibition math.** `direction` (and the joint / matrix equivalents) must be
  exactly `"inhibition"` or `"activation"`; any other value is rejected at the
  public helpers and fit constructors, naming the bad value and the allowed
  ones.

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
