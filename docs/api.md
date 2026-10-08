# API reference

[README](../README.md) · [Runnable tutorials](tutorials.md)

This reference describes synfit 0.8.2. Record `synfit.__version__` with results.
Signatures below show the preferred interface; source links include legacy
compatibility arguments and implementation details. Concentrations use the unit
you supply, consistently within each drug. Responses retain their assay units.

- [Hill curves and concentration series](#hill-curves-and-concentration-series)
- [Configuration and bounds](#configuration-and-bounds)
- [Single-drug fitting](#single-drug-fitting)
- [Fit results and uncertainty](#fit-results-and-uncertainty)
- [Joint marginal fitting](#joint-marginal-fitting)
- [Diagnostic matrix fitting](#diagnostic-matrix-fitting)
- [Noise models](#noise-models)
- [Synergy references and scores](#synergy-references-and-scores)
- [Datasets and simulation](#datasets-and-simulation)
- [Plotting](#plotting)

## Hill curves and concentration series

Import these functions from `synfit` or `synfit.hill`.
[Source](../src/synfit/hill.py)

```text
hill_curve(conc, c50, hill=1.0, effect_0=1.0, effect_inf=0.0, asymmetry=1.0)
calculate_concentration_series(initial_conc, fold_dilutions, length,
                               has_zero=True, ascending=True)
log_wall(x, boundaries, steep=1)
```

The curve is
`effect_inf + (effect_0 - effect_inf) / (1 + (conc / c50)**hill)**asymmetry`.
Use nonnegative concentrations, positive `c50`, slope and asymmetry.
`effect_0` is the zero-concentration asymptote and `effect_inf` the saturating
asymptote, regardless of direction. `asymmetry=None` is treated as 1 (4p).
For 5p, `c50` is the concentration parameter κ, not generally the concentration
at half the response range; see `FitResult.half_max` below.

The concentration-series helper returns a NumPy array. `initial_conc > 0` and
`fold_dilutions > 1` are required. `length` includes the zero dose: `has_zero=True`
replaces the lowest dilution with zero, rather than appending an extra point.
`log_wall` is zero inside its bounds and a linear penalty outside them; it is
not an interior barrier. Fitters also enforce hard optimizer bounds.

## Configuration and bounds

Import `FitConfig`, `FitBounds`, `default_fit_config` and `default_bounds` from
`synfit`. [Configuration source](../src/synfit/data.py) ·
[Data-derived defaults](../src/synfit/single.py)

```text
FitConfig(log_c50=0.0, hill=1.0, effect_0=1.0, effect_inf=0.0,
          fitting_parameters=None, bounds=None, noise=None,
          asymmetry=1.0, direction="inhibition")
default_fit_config(data, direction="inhibition", noise=None) -> FitConfig
default_bounds(data, direction="inhibition", noise=None) -> FitBounds
```

`log_c50` is **log10** concentration. The default free parameter list is
`["log_c50", "hill", "effect_0", "effect_inf"]`; add `"asymmetry"` for a 5p fit.
Omitted Hill parameters are pinned at their configured values. Bounds apply
only to estimated parameters: a pin is not clipped to its corresponding bounds.
At least one free parameter is required. Gaussian variance coefficients are
added automatically when the selected noise model requires them.

`direction` is exactly `"inhibition"` or `"activation"`. It controls default
asymptote placement and IC50/EC50 reporting. `noise=None` selects constant
Gaussian noise. Pass a noise object, tagged dictionary, or kind string; see
[noise models](#noise-models).

`bounds=None` means derive bounds from data. `SingleDrugFit` also refreshes
initial values still at their defaults in this path. Explicit `FitBounds` are
used literally apart from variance-domain clamping. Use `default_fit_config`
to obtain editable data-derived initials and bounds before fitting. Fitters
copy a supplied config; you may adjust `fitter.config` before calling `fit()`.

Bare `FitBounds()` has the following defaults; these are **not** data-derived:

| Parameter | Bounds |
|---|---|
| `log_c50` | `(-5, 5)` |
| `hill` | `(0.1, 4)` |
| `effect_0`, `effect_inf` | `(0, 2)` each |
| `asymmetry` | `(0.1, 10)` |
| `var_a` | `(1e-12, 1e4)` |
| `var_b`, `var_c` | `(0, 1e4)` each |

For RFU or other unnormalized responses, derive bounds from data or explicitly
set asymptote bounds in the right units. Scale-relative optimization helps
conditioning; it cannot compensate for bounds excluding the appropriate scale.

## Single-drug fitting

Import both classes from `synfit`. [Source](../src/synfit/single.py)

```text
SingleDrugFit(data, config=None)
    .fit(valids=None) -> FitResult
    .predict(result) -> ndarray
SingleDrugFitWithError(data, config=None)
    .fit(valids=None) -> FitResult
```

`SingleDrugFit` requires a pandas DataFrame with `concentration`, `y` and
`replicate` columns. Keep each replicate observation as a row. `replicate`
identifies observations for plotting; it does not introduce a random-effects
model. `.predict(result)` evaluates the curve at the fitter's input doses.
For a new concentration grid, call `hill_curve` with the result's parameters.

`SingleDrugFitWithError` requires `concentration`, `y` and `y_err` (`replicate`
is not required by this fitter). `y_err` is a standard deviation on the response
scale, not a variance or CI width. It must be positive and finite on included
rows. Constant Gaussian and lognormal noise are supported; lognormal errors
are transformed approximately as `y_err / y`. This fitter does not populate
the goodness-of-fit metrics or estimate an additional noise scale.

`valids` is a Boolean array aligned with input rows. Non-finite concentration
or response is excluded regardless of that mask. A fit requires at least
`number of free parameters + 1` valid observations. The mask passed to
`.fit()` does not retroactively exclude points from constructor-derived
initials and bounds; prefilter the DataFrame if they must not influence those.
Use strictly positive responses and predictions for lognormal fits.

## Fit results and uncertainty

`FitResult` is returned by the single-drug fitters and is also available from
`synfit`. [Source](../src/synfit/data.py)

| Field or method | Meaning |
|---|---|
| `success`, `message` | Optimizer termination status; check before interpretation |
| `c50`, `kappa`, `log_c50`, `hill`, `effect_0`, `effect_inf`, `asymmetry` | Estimated or pinned curve parameters; `kappa == c50` |
| `half_max` | `c50 * (2**(1/S) - 1)**(1/hill)`, with `S=1` if asymmetry is absent |
| `n_valid`, `n_total` | Finite, mask-included observations and total input rows |
| `rss`, `r2`, `log_likelihood`, `aic`, `bic`, `n_params` | Metrics populated by `SingleDrugFit`; otherwise may be `None` |
| `param_cov`, `param_names` | Covariance and its row/column parameter order; covariance can be `None` |
| `sigma` | Profiled constant-noise standard deviation: response units for Gaussian, natural-log response units for lognormal |
| `variance_params` | Fitted heteroscedastic coefficients under keys `a`, `b`, optionally `c`; otherwise `None` |
| `param_ci(alpha=0.05)` | Dictionary of parameter intervals, or `None` without covariance |
| `predict_ci(conc, alpha=0.05)` | `(lower, upper)` arrays for the fitted curve, or `None` |
| `predict_variance(mu)` | Observation variance at predicted responses `mu`, in response units squared, or `None` |
| `to_dict()` | Parameters, status, metrics and available intervals/covariance using serializable containers |

`half_max` equals κ for a 4p curve. The properties `ic50` and `ec50` do not
exist: use `half_max`; `to_dict()` uses the direction-appropriate `ic50` or
`ec50` key. Its serialized intervals are rounded to six decimal places; use
`param_ci()` for unrounded parameter intervals.

Parameter intervals are local Wald approximations. The `c50` interval is
obtained by exponentiating the `log_c50` interval. Curve bands use covariance
propagation (the delta method); they are pointwise confidence bands, **not**
prediction intervals for individual observations. Singular, ill-conditioned
or non-finite curvature can leave covariance unavailable even after optimizer
success. Bound-active parameters have zero covariance rows/columns; a collapsed
interval at a bound does not demonstrate certainty.

For lognormal noise the curve predicts the response median. If `sigma=s`,
`predict_variance(mu)` returns `mu**2 * exp(s**2) * (exp(s**2) - 1)` on the
original response scale. User-supplied-error fits have no estimated `sigma`.

## Joint marginal fitting

Import from `synfit.joint_marginal`, not the top-level namespace.
[Source](../src/synfit/joint_marginal.py)

```text
JointMarginalFit(data_a, data_b, *, model_a="4p", model_b="4p",
                 direction_a="inhibition", direction_b="inhibition",
                 noise=None, param_config=None, valids_a=None, valids_b=None)
    .fit() -> JointMarginalResult
default_joint_marginal_config(data_a, data_b, *, model_a="4p", model_b="4p",
                              direction_a="inhibition", direction_b="inhibition",
                              noise=None) -> dict
```

Both inputs use the raw replicate DataFrame schema above. The fitter estimates
two dose-response curves with shared physical high/low asymptotes `top` and
`bottom`; it does not fit combination cells. This is the recommended starting
point for marginal fits used in synergy analysis. See the [matrix tutorial](tutorials.md#from-a-shipped-matrix-to-interaction-results)
for extracting edges and avoiding double-counting the shared untreated control.

Choose `"4p"` or `"5p"` independently per drug. `param_config` overrides have
the form `{name: {"init": value, "lo": lower, "hi": upper, "fit": True}}`.
Missing entries use data-derived defaults; `fit=False` pins at `init`.
Names are `top`, `bottom`, `log_c50_a`, `hill_a`, `log_c50_b`, `hill_b`,
`asymmetry_a`/`asymmetry_b` for 5p drugs, and the selected noise's `var_*`
coefficients. `default_joint_marginal_config` returns editable `init`/`lo`/`hi`
entries for the selected models and noise. Constructor masks also exclude data
from its default derivation.

`JointMarginalResult` exposes `drug_a`, `drug_b` (each a `FitResult`), shared
asymptotes, joint `success`/`message`, covariance/names, `aic`, `log_likelihood`,
`n_params`, `n_data`, and noise estimates. `.predict_variance(mu)` evaluates
the shared noise model. Per-drug results do not carry their own covariance;
`.to_dict()` derives intervals from the joint covariance and places them in
`horizontal["ci"]`, `vertical["ci"]` and the shared `ci` dictionary.

Directions remap each drug's asymptotes. For activation, use the drug result's
`effect_0`/`effect_inf` when calling synergy helpers: the serialized top-level
`effect_0`/`effect_inf` remain `top`/`bottom`, following the inhibition convention.
Mixed-direction marginal fits are supported, but that does not make the
same-direction synergy references valid for an inhibitor/activator pair.

`fit_joint_marginal_auto` accepts the model/direction settings above,
`variance_model="constant"` and `param_config`. It compares Gaussian and
lognormal AIC (Gaussian alone for nonconstant variance). In 0.8.2 it skips
exceptions but does **not** filter returned fits by `success`; always inspect
the selected result. AIC comparison also requires the same observations.

## Diagnostic matrix fitting

Import `MatrixFit` from `synfit`; its result type lives in `synfit.matrix`.
[Source](../src/synfit/matrix.py)

```text
MatrixFit(replicates, conc_horizontal, conc_vertical, valids=None, noise=None,
          direction_horizontal="inhibition", direction_vertical="inhibition")
    .fit() -> MatrixFitResult
```

`replicates` is a nonempty list of 2D arrays with shape
`(len(conc_vertical), len(conc_horizontal))`. `valids` is an optional list of
Boolean masks with the same shapes. Supply zero first on each concentration
axis: initialization uses the first row/column as single-agent edges. A missing
leading zero generates a warning and treats the first row/column as the edge.

This is a six-parameter **Bliss null model fitted to the entire response
surface**, not a flexible interaction model. It can absorb interaction into
its parameters; use joint marginal fitting for the usual synergy workflow.
Both drugs must have the same direction. Only constant Gaussian or lognormal
noise is supported.

The result provides `horizontal`, `vertical` (`FitResult`), shared asymptotes,
`success`, `message`, `warnings`, `sigma`, and the joint `param_cov`/`param_names`.
Per-drug results do not carry separate covariance. `.to_dict()` serializes
these fields; a joint covariance is not a ready-made CI on synergy scores.

## Noise models

Import concrete models and serializers from `synfit.noise`.
`NoiseSpec` and `noise_log_prob` (an alias of `log_prob`) are also top-level.
[Source](../src/synfit/noise.py)

| Object / kind string | Model and support |
|---|---|
| `GaussianConstant()` / `"gaussian_constant"` | Profiled constant Gaussian variance; all fitters |
| `GaussianLinear(a_init=None, b_init=None)` / `"gaussian_linear"` | `a + b*d`; single and joint marginal fits |
| `GaussianQuadratic(a_init=None, b_init=None, c_init=None)` / `"gaussian_quadratic"` | `a + b*d + c*d**2`; single and joint marginal fits |
| `Lognormal()` / `"lognormal"` | Profiled constant variance of natural-log responses; all fitters |
| `CompoundAddMult(sigma_log_init=0.1)` / `"compound_add_mult"` | Low-level likelihood with supplied scales; unsupported by the fitters above |

For fitted Gaussian variance, `d = mu - min(effect_0, effect_inf)` (joint:
`mu - min(top, bottom)`). Domains are `a > 0`, `b >= 0`, `c >= 0`. Data-derived
initials/bounds scale with the response range. Unset initials are derived from
data; explicit initials are retained subject to the domain clamp. Do not mix
the preferred `noise` argument with legacy `error_model`/`variance_model` fields.

`from_dict(value)` accepts a noise object, tagged dictionary or string;
`to_dict(noise)` returns a tagged dictionary such as
`{"kind": "gaussian_linear", "a_init": 0.001, "b_init": 0.0}`.

```text
log_prob(y, y_pred, noise=None, *, mask=None, y_err=None,
         variance_params=None, variance_anchor=None, compound_params=None) -> float
variance_at(mu, a, b=0.0, c=0.0, anchor=0.0) -> ndarray
```

`variance_params` is `(a, b, c)`; `compound_params` is `(sigma_add, sigma_log)`.
The low-level default anchor is zero; pass the fitted lower asymptote when
reproducing a heteroscedastic fit's likelihood. Lognormal likelihoods include
the change-of-variables Jacobian. Use strictly positive input: data-derived
lognormal bounds reject nonpositive responses, while the lower-level likelihood
can drop them with a warning. Finite-row counts are not a guarantee that a
lognormal likelihood used every row if you bypass those checks.

## Synergy references and scores

All matrix outputs use rows = vertical drug and columns = horizontal drug.
Use shared, direction-appropriate asymptotes in the same units as the responses.
The references below assume compatible single-agent effects in the same direction.
See the [README references](../README.md#synergy-references) for the publications
and the scope of numerical cross-validation.

### Bliss and HSA

Reference helpers are top-level; deviation helpers live in `synfit.bliss`.
[Source](../src/synfit/bliss.py)

```text
bliss_reference(resp_horizontal, resp_vertical, effect_0, effect_inf)
hsa_reference(resp_horizontal, resp_vertical, *, effect_0=None, effect_inf=None)
bliss_independence(conc_hor, conc_ver, c50_hor, c50_ver, hill_hor=1.0,
                   hill_ver=1.0, effect_0=1.0, effect_inf=0.0,
                   asymmetry_hor=1.0, asymmetry_ver=1.0)
bliss_deviation(data, bliss, effect_0, effect_inf, conc_hor=None, conc_ver=None)
hsa_deviation(mean_matrix, resp_horizontal, resp_vertical, effect_0,
              effect_inf, conc_hor=None, conc_ver=None)
```

The first two use observed 1D single-agent response arrays, typically matrix
edges. `bliss_independence` evaluates fitted Hill marginals instead. They return
expected **responses**, not scores. Bliss clips normalized marginal responses
to `[0, 1]` before multiplication. HSA selects the lower response for inhibition
or higher for activation; supply both asymptotes for direction-aware behavior.
Omitting both retains the inhibition convention.

Deviation helpers compute `(observed - reference) / (effect_0 - effect_inf)`:
**negative = synergy** in either direction. With both concentration arrays
provided, zero-dose edges become NaN. These fractional scores are not percentages.

### Loewe

Both main functions are top-level. [Source](../src/synfit/loewe.py)

```text
loewe_reference(conc_hor, conc_ver, c50_hor, c50_ver, hill_hor, hill_ver,
                effect_0, effect_inf, asymmetry_hor=1.0, asymmetry_ver=1.0)
loewe_ci(conc_hor, conc_ver, c50_hor, c50_ver, hill_hor, hill_ver,
         mean_matrix, effect_0, effect_inf, asymmetry_hor=1.0, asymmetry_ver=1.0)
```

`loewe_reference` returns expected responses. `loewe_ci` returns the dimensionless
combination index `dose_A / A(response) + dose_B / B(response)`:
**below 1 = synergy**, 1 = additive, above 1 = antagonistic. Responses outside
the open interval between asymptotes cannot be inverted and yield NaN. Do not
replace these with an additive score. Zero-dose cells are not automatically all
masked by this function; exclude controls when summarizing interaction scores.
`synfit.loewe.slope_mismatch_warning(hill_hor, hill_ver)` returns a warning
dictionary or `None`; it is a diagnostic, not a correction to the index.

### ZIP

Import `ZipResult`, `zip_scores`, `zip_reference`, `zip_fitted_surface` and
`zip_delta` from `synfit`. All four functions share this signature:
[Source](../src/synfit/zip.py)

```text
zip_scores(mean_matrix, conc_hor, conc_ver, c50_hor, c50_ver, hill_hor, hill_ver,
           effect_0, effect_inf, *, asymmetry_hor=None, asymmetry_ver=None)
```

| Function | Return |
|---|---|
| `zip_reference` | Zero-interaction expected response surface |
| `zip_fitted_surface` | Combination response surface fitted from row/column slices |
| `zip_delta` | Array of `-(f_c - f_zip)`; **negative = synergy** |
| `zip_scores` | `ZipResult` containing all three surfaces and failure metadata |

Here `f(y) = (effect_0 - y) / (effect_0 - effect_inf)`. `f_c` is the fitted
combination fraction affected, not the raw observed response. Delta is fractional
and dimensionless. `ZipResult` fields are `reference`, `fitted`, `delta`,
`failed_rows`, `failed_cols` and `n_unscored`. Row indices identify vertical
doses, column indices horizontal doses. If one slice direction fails, the other
is used; both failing leaves the interior cell unscored. Delta is also NaN on
zero-dose edges. Inspect failure metadata before taking summaries.

The 5p slice-asymmetry treatment is an extension of the published 4p method.
These helpers do not return confidence intervals on the interaction scores.

## Datasets and simulation

Import from `synfit.synthetic`. [Source](../src/synfit/synthetic.py)

| Function | Return |
|---|---|
| `list_scenarios()` | Sorted names of bundled scenarios |
| `scenario_dir(name)` | Installed scenario directory as a `Path`; unknown name raises `KeyError` |
| `scenario_config(name)` | Generating configuration dictionary |
| `generate_single_drug(config, rng=None)` | Tidy `concentration`, `y`, `replicate` DataFrame |
| `generate_matrix(config, rng=None)` | `(replicate_arrays, conc_horizontal, conc_vertical)` |

Single-drug scenarios contain `<name>.csv`; matrices contain headerless,
tab-separated `rep*.csv` files whose axes come from the config. `SCENARIOS_DIR`
is the installed data root. Neither generator writes files.

For single-drug generation, supply `hill_params` (`c50`, `hill`, `effect_0`,
`effect_inf`) and `concentration_series` (`initial_conc`, `fold_dilutions`,
`length`, optional `has_zero`). Matrix configs use `horizontal_drug` and
`vertical_drug`, each containing potency, slope and concentration series, plus
shared `effect_0`/`effect_inf`. These generators construct 4p curves; use
`hill_curve(..., asymmetry=...)` directly for a 5p simulation.

Common options are `seed` (default 42 if `rng` is absent), `n_replicates`
(default 3), and `noise_model` (**default `"gaussian"`**). Gaussian simulation
accepts `noise_sigma` (default 0.05), or `noise_var_a`/`noise_var_b`/`noise_var_c`
for variance `a + b*mu + c*mu**2` at the true generating response. This is an
unanchored polynomial, unlike the fitted heteroscedastic parameterization.
Lognormal simulation uses `noise_model="lognormal"` and `noise_sigma_log`
(default 0.05). Supplying a NumPy `Generator` uses its state instead of `seed`.

Matrix `reference_model` is `"bliss"` by default or `"loewe"`.
`synergy_factor=0` leaves the reference unchanged before noise; positive values
add inhibition. The injection parameter is not a known ZIP delta or Loewe CI.
Start from a bundled config for a complete, editable schema.

## Plotting

Import from `synfit.plotting`. All helpers return **PNG bytes**, not a Figure or
Axes, and close the figure internally. Importing the module selects Matplotlib's
noninteractive Agg backend and sets plotting defaults. [Source](../src/synfit/plotting.py)

```text
raw_scatter_plot(data)
raw_replicate_heatmap(data)
dose_response_plot(data, result, valids=None, *, reference=None,
                   reference_label="Ground truth", x_label=None, title=None)
matrix_heatmap(matrix, conc_horizontal, conc_vertical, *, vmin=None, vmax=None,
               cmap="viridis", norm=None, cbar_label=None, title=None)
raw_matrix_heatmap(matrix, conc_horizontal, conc_vertical, label="")
synergy_heatmaps(mean_matrix, conc_horizontal, conc_vertical, *,
                 c50_hor, c50_ver, hill_hor, hill_ver, effect_0, effect_inf,
                 asymmetry_hor=None, asymmetry_ver=None,
                 x_label=None, y_label=None, title=None)
```

Single-drug plots use `concentration`, `y`, `replicate` and need at least one
positive dose. The dose-response plot includes a pointwise 95% curve band when
covariance is available. It infers a symmetric-log x-axis when zero-dose data
is present and a log axis otherwise. A lognormal fit uses a log y-axis only if
every drawn response and confidence limit is strictly positive. Its optional
`reference` mapping takes `c50`, `hill`, `effect_0`, `effect_inf`, and optionally
`asymmetry`; it draws a dashed curve, which is useful for known synthetic truth
or a specified benchmark. Matrix inputs follow the same row/column orientation
as synergy functions. `matrix_heatmap` accepts a custom Matplotlib normalization;
use it to plot an already computed score array.

`deviation_heatmap`, `hsa_heatmap`, `zip_heatmap` and `loewe_heatmap` are
convenience wrappers computing their respective scores before plotting.
The model-based wrappers do not accept 5p asymmetry arguments: for 5p, compute
scores with explicit asymmetries and pass them to `matrix_heatmap`.
`synergy_heatmaps` is the compact four-panel alternative: it computes Bliss,
HSA, Loewe and ZIP from a mean response matrix and fitted marginal parameters.
It requires one zero-dose row and column to obtain observed single-agent edges,
then excludes every zero-dose row and column from the four score panels.
Bliss, HSA and ZIP use a fixed linear color range of −1 to +1; Loewe CI uses
a logarithmic color range of 0.25 to 4. White denotes the null (0 for fractional
scores, CI = 1 for Loewe), blue synergy, and red antagonism. The
[tutorials](tutorials.md) use this helper with a bundled matrix.
