import math
import warnings

import numpy as np
import pandas as pd

from .data import (
    FitConfig, FitBounds, FitResult, validate_direction,
    apply_variance_coefficient_domain,
    variance_anchor_from_asymptotes,
)
from .fitting import FitBase, effective_observation_mask, validate_included_y_err
from .hill import hill_curve
from .noise import (
    Lognormal,
    NoiseSpec,
    error_model_name,
    fit_noise_scale,
    free_coefficients,
    from_dict as noise_from_dict,
    is_heteroscedastic_gaussian,
    log_prob as noise_log_prob,
    noise_spec_with_variance_initials,
    require_supported_single_drug_fitting_noise,
    require_supported_single_drug_with_error_fitting_noise,
    scaled_variance_coefficient_defaults,
    variance_initials_for_noise,
    variance_model_name,
)


def _sig_quantum(value: float, sig: int) -> float:
    """Place value of the last retained significant digit."""
    if value == 0.0:
        return math.inf
    return 10.0 ** (math.floor(math.log10(abs(value))) - sig + 1)


def _clean_float(value: float) -> float:
    """Strip arithmetic noise; hierarchy checks retain raw values if needed."""
    return float(f"{value:.15g}")


def _round_bound(value: float, scale: float, *, lower: bool, sig: int = 2) -> float:
    """Round one bound outward without losing resolution relative to the data.

    Plain two-significant-figure rounding is too coarse for a narrow response
    range on a large offset (e.g. 1000.0..1000.5): both endpoints become 1000.
    Use the finer of the value's and the dynamic range's significant-digit
    quanta, then floor lower bounds and ceil upper bounds.
    """
    quantum = min(_sig_quantum(value, sig), _sig_quantum(scale, sig))
    scaled = value / quantum
    # Tolerate division noise when the value is already on the quantum grid.
    units = math.floor(scaled + 1e-12) if lower else math.ceil(scaled - 1e-12)
    rounded = _clean_float(units * quantum)
    if lower and rounded > value:
        return value
    if not lower and rounded < value:
        return value
    return rounded


def _round_asymptote_bounds(
    bottom: tuple[float, float],
    top: tuple[float, float],
    scale: float,
    sig: int = 2,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Round bottom/top bounds while preserving their shared hierarchy.

    The raw intervals meet at one midpoint. Their inner edges cannot both be
    rounded outward independently without crossing, so round that split once
    and reuse it as ``bottom.hi == top.lo``. Increase its precision as needed
    until ``bottom.lo < split < top.hi`` remains strict.
    """
    bottom_lo = _round_bound(bottom[0], scale, lower=True, sig=sig)
    top_hi = _round_bound(top[1], scale, lower=False, sig=sig)
    split = 0.5 * (bottom[1] + top[0])
    quantum = min(_sig_quantum(split, sig), _sig_quantum(scale, sig))

    rounded_split = split
    for _ in range(14):
        candidate = _clean_float(round(split / quantum) * quantum)
        if bottom_lo < candidate < top_hi:
            rounded_split = candidate
            break
        quantum /= 10.0

    if not bottom_lo < rounded_split < top_hi:
        raise ValueError("Could not derive ordered asymptote bounds.")

    return (bottom_lo, rounded_split), (rounded_split, top_hi)


def _robust_response_extrema(y: np.ndarray) -> tuple[float, float]:
    """Robust (ymin, ymax) used for asymptote bounds and the variance scale.

    Prefer the second-smallest / second-largest when trimming still leaves a
    range; otherwise use the true min/max. Same rule ``_init_config_from_data``
    uses for the response dynamic range. Fewer than two finite values return
    a degenerate (v, v) pair; ``_robust_response_range`` turns that into a
    safe positive scale so construction does not IndexError.
    """
    y_sorted = np.sort(np.asarray(y, dtype=float))
    if y_sorted.size == 0:
        return 0.0, 0.0
    true_ymin = float(y_sorted[0])
    true_ymax = float(y_sorted[-1])
    if y_sorted.size < 2:
        return true_ymin, true_ymax
    trimmed_ymin = float(y_sorted[1])
    trimmed_ymax = float(y_sorted[-2])
    if trimmed_ymin < trimmed_ymax:
        return trimmed_ymin, trimmed_ymax
    return true_ymin, true_ymax


def _robust_response_range(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 2:
        if y.size == 1:
            return max(abs(float(y[0])), 1.0)
        return 1.0
    ymin, ymax = _robust_response_extrema(y)
    return ymax - ymin


def _init_config_from_data(
    data: pd.DataFrame,
    direction: str = "inhibition",
    noise: NoiseSpec | None = None,
) -> FitConfig:
    """Derive sensible initial parameters and bounds from data."""
    validate_direction(direction)
    finite = effective_observation_mask(
        None,
        len(data),
        data["concentration"].values,
        data["y"].values,
    )
    if not finite.all():
        warnings.warn(
            "Rows with non-finite concentration or response are excluded when "
            "deriving fit defaults.",
            UserWarning,
            stacklevel=2,
        )
    data = data.loc[finite]
    if data.empty:
        raise ValueError(
            "Need at least 2 finite (concentration, response) pairs to derive fit defaults; "
            "got 0."
        )

    y = data["y"]
    conc_nonzero = data.query("concentration > 0")["concentration"]

    n = len(y)
    if n < 2:
        raise ValueError(
            f"Need at least 2 non-NaN response values to derive fit defaults; got {n}."
        )

    # Prefer robust extremes (second-smallest / second-largest) when trimming
    # still leaves a range. Otherwise use the true min/max; this naturally covers
    # two/three-point inputs and sparse plateaus where trimming would collapse
    # the usable dynamic range. Cast to float — numpy scalars otherwise leak
    # into the config and break JSON-serialisability.
    ymin, ymax = _robust_response_extrema(y.to_numpy())
    dy = ymax - ymin
    if dy == 0.0:
        raise ValueError(
            "All response values too close together; fitting is not possible."
        )

    if isinstance(noise, Lognormal):
        # Feasibility is about the *actual* responses, not the robust extreme:
        # a single non-positive y breaks the lognormal likelihood, so check the
        # true minimum (not the second-smallest ``ymin`` used for the bounds).
        if float(y.min()) <= 0:
            raise ValueError("Cannot fit lognormal with non-positive responses.")
        bottom_lo = ymin / 100
        top_hi = ymax * 10
    else:  # gaussian
        bottom_lo = ymin - dy / 2
        # The top is far less constrained from above than the bottom is from
        # below — the response space is more unbounded upward — so give the top
        # asymptote extra headroom (2·dy above the observed top vs. dy/2 below
        # the bottom).
        top_hi = ymax + 2 * dy
    bottom_hi = ymin + dy / 2
    top_lo = ymax - dy / 2
    bottom_bounds, top_bounds = _round_asymptote_bounds(
        (bottom_lo, bottom_hi),
        (top_lo, top_hi),
        dy,
    )

    ym = ymin + dy / 2

    # Concentration closest to the midpoint response (finite rows only).
    idx = (data["y"] - ym).abs().sort_values().index[0]
    ic50_init = float(data.loc[idx, "concentration"])
    if ic50_init <= 0:
        ic50_init = float(conc_nonzero.median())

    log_c50_bounds = (
        float(np.floor(np.log10(conc_nonzero.min()))),
        float(np.ceil(np.log10(conc_nonzero.max()))) + 1.0,
    )
    log_c50_init = np.log10(ic50_init)

    def _make_config(
        effect_0_init, effect_inf_init, effect_0_bounds, effect_inf_bounds
    ) -> FitConfig:
        bounds_kwargs: dict = {
            "log_c50": log_c50_bounds,
            "hill": (0.1, 4.0),
            "effect_0": effect_0_bounds,
            "effect_inf": effect_inf_bounds,
        }
        noise_out = noise
        if is_heteroscedastic_gaussian(noise):
            var_inits = variance_initials_for_noise(noise, dy)
            _, var_bounds = scaled_variance_coefficient_defaults(dy)
            bounds_kwargs["var_a"] = var_bounds["var_a"]
            bounds_kwargs["var_b"] = var_bounds["var_b"]
            bounds_kwargs["var_c"] = var_bounds["var_c"]
            noise_out = noise_spec_with_variance_initials(noise, var_inits)
        bounds = FitBounds(**bounds_kwargs)
        # Rounding a bound inward can leave the data-seeded initial just outside
        # its [lo, hi]; clamp the asymptote initials back into the rounded bounds.
        e0 = float(np.clip(effect_0_init, *bounds.effect_0))
        einf = float(np.clip(effect_inf_init, *bounds.effect_inf))
        kwargs: dict = {
            "log_c50": log_c50_init,
            "hill": 1.0,
            "effect_0": e0,
            "effect_inf": einf,
            "bounds": bounds,
            "direction": direction,
        }
        if noise_out is not None:
            kwargs["noise"] = noise_out
        return FitConfig(**kwargs)

    if direction == "activation":
        # effect_0 = bottom (low-conc), effect_inf = top (high-conc)
        return _make_config(
            effect_0_init=ymin,
            effect_inf_init=ymax,
            effect_0_bounds=bottom_bounds,
            effect_inf_bounds=top_bounds,
        )
    else:  # inhibition
        # effect_0 = top (low-conc), effect_inf = bottom (high-conc)
        return _make_config(
            effect_0_init=ymax,
            effect_inf_init=ymin,
            effect_0_bounds=top_bounds,
            effect_inf_bounds=bottom_bounds,
        )


def _coerce_noise(noise: NoiseSpec | dict | str | None) -> NoiseSpec | None:
    """Accept a NoiseSpec, a tagged dict, or a kind string at the public API.

    Without this, the ``isinstance(noise, Lognormal)`` floor in
    ``_init_config_from_data`` only fires for a ``Lognormal`` *instance* — a
    ``"lognormal"`` string or ``{"kind": "lognormal"}`` dict would silently
    skip the positive lower-bound floor.
    """
    if noise is None or isinstance(noise, NoiseSpec):
        return noise
    return noise_from_dict(noise)


def default_fit_config(
    data: pd.DataFrame,
    direction: str = "inhibition",
    noise: NoiseSpec | dict | str | None = None,
) -> FitConfig:
    """Data-driven default fit configuration (initials + bounds) for ``data``.

    Public entry point to the same derivation ``SingleDrugFit`` applies when no
    config — or no bounds — is supplied: initials seeded from the response and
    concentration ranges, with per-asymptote bounds that meet at the response
    midpoint. Gaussian top bounds get extra upward headroom; lognormal lower
    bounds are floored strictly positive. ``noise`` accepts a
    :data:`NoiseSpec`, a tagged dict, or a kind string. Consumers that surface
    *editable* bounds in a UI should source their defaults here rather than
    re-deriving them, so the derivation has a single source of truth.
    """
    coerced = _coerce_noise(noise)
    require_supported_single_drug_fitting_noise(coerced)
    return _init_config_from_data(data, direction=direction, noise=coerced)


def default_bounds(
    data: pd.DataFrame,
    direction: str = "inhibition",
    noise: NoiseSpec | dict | str | None = None,
) -> FitBounds:
    """Data-driven default :class:`FitBounds` for ``data`` (see :func:`default_fit_config`)."""
    coerced = _coerce_noise(noise)
    require_supported_single_drug_fitting_noise(coerced)
    return _init_config_from_data(
        data, direction=direction, noise=coerced,
    ).bounds


def _predict(conc: np.ndarray, result: FitResult) -> np.ndarray:
    """Compute model predictions from a FitResult."""
    return hill_curve(
        conc,
        c50=result.c50, hill=result.hill,
        effect_0=result.effect_0, effect_inf=result.effect_inf,
        asymmetry=result.asymmetry,
    )


class SingleDrugFit(FitBase):
    """
    Fit a single drug dose-response curve to raw replicate data.

    Expected DataFrame columns: concentration, y, replicate.
    """

    def __init__(self, data: pd.DataFrame, config: FitConfig | None = None):
        if not {"concentration", "y", "replicate"}.issubset(data.columns):
            raise ValueError("data must have columns: concentration, y, replicate")
        self.data = data
        if config is None:
            config = _init_config_from_data(data)
        else:
            require_supported_single_drug_fitting_noise(config.noise)
            config = config.copy()
            # Re-derive data-driven defaults for anything the caller left
            # unspecified, using the config's direction heuristics and noise
            # model (so lognormal forces positive bounds). We derive when the
            # direction is non-default (initials need the activation heuristics)
            # or when bounds were not supplied at all. In either case the
            # still-at-default initials are seeded from data alongside the
            # derived bounds — deriving bounds while leaving the historical
            # zero-point initials can otherwise strand the optimiser in a poor
            # basin. Explicitly-supplied bounds are always honoured literally
            # (no equality-based override).
            if config.direction != "inhibition" or config.bounds is None:
                auto = _init_config_from_data(
                    data, direction=config.direction, noise=config.noise,
                )
                # Only seed still-at-default *free* parameters. A name omitted
                # from fitting_parameters is pinned; overwriting it would
                # silently replace a pin at the type default (log_c50=0,
                # effect_inf=0) with the data-derived initial.
                if (
                    config.log_c50 == 0.0
                    and "log_c50" in config.fitting_parameters
                ):
                    config.log_c50 = auto.log_c50
                if config.effect_0 == 1.0 and config.effect_inf == 0.0:
                    if "effect_0" in config.fitting_parameters:
                        config.effect_0 = auto.effect_0
                    if "effect_inf" in config.fitting_parameters:
                        config.effect_inf = auto.effect_inf
                if config.bounds is None:
                    config.bounds = auto.bounds
                    # Data-derived variance bounds/initials (user-given
                    # a_init/b_init/c_init are already overlaid in auto.noise).
                    config.noise = auto.noise
        validate_direction(config.direction)
        if is_heteroscedastic_gaussian(config.noise):
            finite = effective_observation_mask(
                None,
                len(data),
                data["concentration"].values,
                data["y"].values,
            )
            self._response_scale = _robust_response_range(
                data["y"].values[finite]
            )
            apply_variance_coefficient_domain(
                config, response_scale=self._response_scale,
            )
        else:
            apply_variance_coefficient_domain(config)
        if not config.fitting_parameters:
            raise ValueError(
                "At least one parameter must be listed in fitting_parameters — "
                "cannot run an optimiser with no free parameters."
            )
        super().__init__(config)

    def _curve(self, conc: np.ndarray, kwargs: dict) -> np.ndarray:
        """Evaluate the Hill curve."""
        return hill_curve(conc, **kwargs)

    def _log_prob_data(self, x: np.ndarray, valids: np.ndarray | None = None) -> float:
        kwargs = self._x_to_kwargs(x)
        conc = self.data["concentration"].values
        y = self.data["y"].values
        y_pred = self._curve(conc, kwargs)
        var_params = self._x_to_variance_params(x)
        return noise_log_prob(
            y, y_pred,
            self.config.noise,
            mask=valids,
            variance_params=var_params,
            variance_anchor=self._variance_anchor(x),
        )

    def predict(self, result: FitResult) -> np.ndarray:
        """Compute fitted curve predictions for all data points."""
        return _predict(self.data["concentration"].values, result)

    def _compute_gof_metrics(self, result: FitResult, valids: np.ndarray) -> None:
        conc = self.data["concentration"].values
        y = self.data["y"].values
        y_pred = _predict(conc, result)

        y_v = y[valids]
        y_pred_v = y_pred[valids]
        rss = float(np.sum((y_v - y_pred_v) ** 2))
        ss_tot = float(np.sum((y_v - y_v.mean()) ** 2))
        r2 = float(1.0 - rss / ss_tot) if ss_tot > 0 else None

        # If the variance model fitted (a, b, c) jointly, the log-likelihood
        # for AIC/BIC must use those — not the constant profiled σ̂. Read
        # them off the result we just built.
        if result.variance_params is not None:
            vp = result.variance_params
            var_tuple = (
                float(vp.get("a", 0.0)),
                float(vp.get("b", 0.0)),
                float(vp.get("c", 0.0)),
            )
        else:
            var_tuple = None

        log_like = noise_log_prob(
            y, y_pred,
            self.config.noise,
            mask=valids,
            variance_params=var_tuple,
            variance_anchor=variance_anchor_from_asymptotes(
                result.effect_0, result.effect_inf,
            ),
        )
        n = int(valids.sum())
        k = len(self.config.fitting_parameters)
        # Constant variance is profiled out — its single coefficient still
        # counts as a free parameter in AIC/BIC. Non-constant models already
        # have their coefficients in ``fitting_parameters`` (so ``k`` covers
        # them), but the constant case adds 1 implicitly.
        if result.variance_params is None:
            k = k + 1
        aic = float(-2 * log_like + 2 * k)
        bic = float(-2 * log_like + k * np.log(n)) if n > 0 else None

        result.r2 = r2
        result.rss = rss
        result.log_likelihood = float(log_like)
        result.aic = aic
        result.bic = bic
        result.n_params = k
        # Constant model: report the profiled σ̂ as a single scalar.
        # Non-constant: σ varies with μ; ``variance_params`` holds the answer
        # and ``predict_variance`` evaluates it.
        if result.variance_params is None:
            result.sigma = fit_noise_scale(
                y, y_pred, self.config.noise, mask=valids,
            )

    def fit(self, valids: np.ndarray | None = None) -> FitResult:
        """
        Fit the Hill curve. Optionally pass a boolean array to exclude points.
        """
        mask = effective_observation_mask(
            valids,
            len(self.data),
            self.data["concentration"].values,
            self.data["y"].values,
        )
        n_valid = int(mask.sum())
        n_params = len(self.config.fitting_parameters)
        if n_valid < n_params + 1:
            raise ValueError(
                f"Need at least {n_params + 1} valid data points for {n_params} free parameters; "
                f"got {n_valid}."
            )

        x0, bounds = self._get_x0_and_bounds()
        x_opt, success, message, pcov = self._run_minimize(x0, bounds, valids=mask)
        kwargs = self._x_to_kwargs(x_opt)
        var_tuple = self._x_to_variance_params(x_opt)

        log_c50 = self._parameter_value("log_c50", x_opt)
        if "asymmetry" in self.config.fitting_parameters:
            asymmetry = self._parameter_value("asymmetry", x_opt)
        else:
            asymmetry = self.config.asymmetry
        # pop asymmetry out so kwargs only has hill_curve params
        kwargs.pop("asymmetry", None)

        vm_str = variance_model_name(self.config.noise)
        variance_params: dict | None = None
        if var_tuple is not None:
            a, b, c = var_tuple
            free = set(free_coefficients(self.config.noise))
            # Only emit coefficients the model actually fits — keeps the
            # serialised dict honest about what was estimated vs. held at 0.
            entries: dict[str, float] = {}
            if "var_a" in free:
                entries["a"] = float(a)
            if "var_b" in free:
                entries["b"] = float(b)
            if "var_c" in free:
                entries["c"] = float(c)
            variance_params = entries

        result = FitResult(
            c50=kwargs["c50"],
            log_c50=float(log_c50),
            hill=kwargs["hill"],
            effect_0=kwargs["effect_0"],
            effect_inf=kwargs["effect_inf"],
            success=success,
            n_valid=n_valid,
            n_total=len(self.data),
            message=message,
            asymmetry=asymmetry,
            direction=self.config.direction,
            error_model=error_model_name(self.config.noise),
            variance_model=vm_str,
            variance_params=variance_params,
            param_cov=pcov,
            param_names=list(self.config.fitting_parameters),
        )

        self._compute_gof_metrics(result, mask)

        return result


class SingleDrugFitWithError(FitBase):
    """
    Fit a single drug dose-response curve when explicit error bars are provided.

    Expected DataFrame columns: concentration, y, y_err.
    """

    def __init__(self, data: pd.DataFrame, config: FitConfig | None = None):
        if not {"concentration", "y", "y_err"}.issubset(data.columns):
            raise ValueError("data must have columns: concentration, y, y_err")
        self.data = data
        if config is None:
            cfg = _init_config_from_data(data)
        else:
            cfg = config.copy()
        require_supported_single_drug_with_error_fitting_noise(cfg.noise)
        validate_direction(cfg.direction)
        # Honour the bounds=None "derive from data" signal (FitBase readers
        # dereference config.bounds.<param>, so it must be concrete by fit time).
        if cfg.bounds is None:
            cfg.bounds = _init_config_from_data(
                data, direction=cfg.direction, noise=cfg.noise,
            ).bounds
        if not cfg.fitting_parameters:
            raise ValueError(
                "At least one parameter must be listed in fitting_parameters — "
                "cannot run an optimiser with no free parameters."
            )
        super().__init__(cfg)

    def _log_prob_data(self, x: np.ndarray, valids: np.ndarray | None = None) -> float:
        kwargs = self._x_to_kwargs(x)
        conc = self.data["concentration"].values
        y = self.data["y"].values
        y_err = self.data["y_err"].values
        y_pred = hill_curve(conc, **kwargs)
        mask = valids
        if mask is None:
            mask = getattr(self, "_likelihood_mask", None)
        return noise_log_prob(
            y, y_pred,
            self.config.noise,
            y_err=y_err,
            mask=mask,
        )

    def fit(self, valids: np.ndarray | None = None) -> FitResult:
        mask = effective_observation_mask(
            valids,
            len(self.data),
            self.data["concentration"].values,
            self.data["y"].values,
        )
        validate_included_y_err(self.data["y_err"].values, mask)
        self._likelihood_mask = mask
        n_valid = int(mask.sum())
        n_params = len(self.config.fitting_parameters)
        if n_valid < n_params + 1:
            raise ValueError(
                f"Need at least {n_params + 1} valid data points for {n_params} free parameters; "
                f"got {n_valid}."
            )

        x0, bounds = self._get_x0_and_bounds()
        x_opt, success, message, pcov = self._run_minimize(x0, bounds, valids=mask)
        kwargs = self._x_to_kwargs(x_opt)
        log_c50 = self._parameter_value("log_c50", x_opt)
        if "asymmetry" in self.config.fitting_parameters:
            asymmetry = self._parameter_value("asymmetry", x_opt)
        else:
            asymmetry = self.config.asymmetry
        kwargs.pop("asymmetry", None)

        vm_str = variance_model_name(self.config.noise)

        return FitResult(
            c50=kwargs["c50"],
            log_c50=float(log_c50),
            hill=kwargs["hill"],
            effect_0=kwargs["effect_0"],
            effect_inf=kwargs["effect_inf"],
            success=success,
            n_valid=n_valid,
            n_total=len(self.data),
            message=message,
            asymmetry=asymmetry,
            direction=self.config.direction,
            error_model=error_model_name(self.config.noise),
            variance_model=vm_str,
            param_cov=pcov,
            param_names=list(self.config.fitting_parameters),
        )
