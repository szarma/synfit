import warnings

import numpy as np
import pandas as pd

from .data import FitConfig, FitBounds, FitResult
from .fitting import FitBase
from .hill import hill_curve
from .noise import (
    GaussianConstant,
    Lognormal,
    NoiseSpec,
    error_model_name,
    fit_noise_scale,
    free_coefficients,
    from_dict as noise_from_dict,
    log_prob as noise_log_prob,
    variance_model_name,
)

# Minimum number of non-NaN response points required to derive data-driven
# defaults. The robust extremes below discard the single most-extreme point at
# each end, so fewer than a handful of points leaves too little to characterise
# the dynamic range — and np.partition(y, 1) is undefined below two points.
# Bump this if you want a stricter floor.
_MIN_POINTS_FOR_DEFAULTS = 5


def _round_sig(values: tuple[float, float], sig: int = 2) -> tuple[float, float]:
    """Round a (lo, hi) bound pair to ``sig`` significant figures.

    The data-derived bounds carry float noise (``0.21593148226201883``); the UI
    surfaces them editable, so present clean numbers (``0.22``). ``float(format)``
    rather than ``eval("%.2g" % x)`` — no parser invocation, and it handles the
    exponent / inf forms ``%g`` can emit.
    """
    lo, hi = values
    return (float(f"{lo:.{sig}g}"), float(f"{hi:.{sig}g}"))


def _init_config_from_data(
    data: pd.DataFrame,
    direction: str = "inhibition",
    noise: NoiseSpec | None = None,
) -> FitConfig:
    """Derive sensible initial parameters and bounds from data."""
    if data["y"].isna().any():
        warnings.warn(
            "Response column contains NaN values; those rows will be dropped before fitting.",
            UserWarning,
            stacklevel=2,
        )

    y = data["y"].dropna()
    conc_nonzero = data.query("concentration > 0")["concentration"]

    if len(y) < _MIN_POINTS_FOR_DEFAULTS:
        raise ValueError(
            f"Need at least {_MIN_POINTS_FOR_DEFAULTS} non-NaN response values to derive "
            f"fit defaults; got {len(y)}."
        )

    # Robust extremes: the second-smallest / second-largest response, so a lone
    # outlier at either end doesn't set an asymptote bound. Cast to float —
    # np.partition returns a numpy scalar — so the derived config stays plain
    # float / JSON-serialisable.
    ymin = float(np.partition(y, 1)[1])
    ymax = float(np.partition(y, -2)[-2])
    dy = ymax - ymin
    if dy == 0.0:
        raise ValueError("All response values too close together; fitting is not possible.")

    if isinstance(noise, Lognormal):
        # Feasibility is about the *actual* responses, not the robust extreme:
        # a single non-positive y breaks the lognormal likelihood, so check the
        # true minimum (not the second-smallest ``ymin`` used for the bounds).
        if float(y.min()) <= 0:
            raise ValueError("Cannot fit lognormal with non-positive responses.")
        bottom_lo = ymin/100
        top_hi = ymax*10
    else: # gaussian
        bottom_lo = ymin-dy/2
        # The top is far less constrained from above than the bottom is from
        # below — the response space is more unbounded upward — so give the top
        # asymptote extra headroom (2·dy above the observed top vs. dy/2 below
        # the bottom).
        top_hi = ymax+2*dy
    bottom_hi = ymin + dy/2
    top_lo = ymax - dy/2

    ym = ymin + dy/2

    # find concentration closest to midpoint response
    idx = (data["y"] - ym).abs().sort_values().index[0]
    ic50_init = float(data.loc[idx, "concentration"])
    if ic50_init <= 0:
        ic50_init = float(conc_nonzero.median())

    log_c50_bounds = (
        float(np.floor(np.log10(conc_nonzero.min()))),
        float(np.ceil(np.log10(conc_nonzero.max()))) + 1.0,
    )
    log_c50_init = np.log10(ic50_init)

    def _make_config(effect_0_init, effect_inf_init, effect_0_bounds, effect_inf_bounds) -> FitConfig:
        bounds = FitBounds(
            # log_c50 bounds are already whole decades (floor/ceil); only the
            # asymptote bounds carry float noise worth rounding away.
            log_c50=log_c50_bounds,
            hill=(0.1, 4.0),
            effect_0=_round_sig(effect_0_bounds),
            effect_inf=_round_sig(effect_inf_bounds),
        )
        # Rounding a bound inward can leave the data-seeded initial just outside
        # its [lo, hi]; clamp the asymptote initials back into the rounded bounds.
        e0 = min(max(effect_0_init, bounds.effect_0[0]), bounds.effect_0[1])
        einf = min(max(effect_inf_init, bounds.effect_inf[0]), bounds.effect_inf[1])
        kwargs: dict = {
            "log_c50": log_c50_init,
            "hill": 1.0,
            "effect_0": e0,
            "effect_inf": einf,
            "bounds": bounds,
            "direction": direction,
        }
        if noise is not None:
            kwargs["noise"] = noise
        return FitConfig(**kwargs)

    if direction == "activation":
        # effect_0 = bottom (low-conc), effect_inf = top (high-conc)
        return _make_config(
            effect_0_init=ymin, effect_inf_init=ymax,
            effect_0_bounds=(bottom_lo, bottom_hi), effect_inf_bounds=(top_lo, top_hi),
        )
    else: # inhibition
        # effect_0 = top (low-conc), effect_inf = bottom (high-conc)
        return _make_config(
            effect_0_init=ymax, effect_inf_init=ymin,
            effect_0_bounds=(top_lo, top_hi), effect_inf_bounds=(bottom_lo, bottom_hi),
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
    concentration ranges, and asymptote bounds that extend half a dynamic range
    past the observed extremes (lognormal floors the lower bound strictly
    positive). ``noise`` accepts a :data:`NoiseSpec`, a tagged dict, or a kind
    string. Consumers that surface *editable* bounds in a UI should source their
    defaults here rather than re-deriving them, so the derivation has a single
    source of truth.
    """
    return _init_config_from_data(data, direction=direction, noise=_coerce_noise(noise))


def default_bounds(
    data: pd.DataFrame,
    direction: str = "inhibition",
    noise: NoiseSpec | dict | str | None = None,
) -> FitBounds:
    """Data-driven default :class:`FitBounds` for ``data`` (see :func:`default_fit_config`)."""
    return _init_config_from_data(
        data, direction=direction, noise=_coerce_noise(noise)
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
                if config.log_c50 == 0.0:
                    config.log_c50 = auto.log_c50
                if config.effect_0 == 1.0 and config.effect_inf == 0.0:
                    config.effect_0 = auto.effect_0
                    config.effect_inf = auto.effect_inf
                if config.bounds is None:
                    config.bounds = auto.bounds
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
        if valids is None:
            valids = np.ones(len(self.data), dtype=bool)

        n_valid = int(valids.sum())
        n_params = len(self.config.fitting_parameters)
        if n_valid < n_params + 1:
            raise ValueError(
                f"Need at least {n_params + 1} valid data points for {n_params} free parameters; "
                f"got {n_valid}."
            )

        x0, bounds = self._get_x0_and_bounds()
        x_opt, success, message, pcov = self._run_minimize(x0, bounds, valids=valids)
        kwargs = self._x_to_kwargs(x_opt)
        var_tuple = self._x_to_variance_params(x_opt)

        log_c50 = x_opt[self.config.fitting_parameters.index("log_c50")]
        asymmetry = None
        if "asymmetry" in self.config.fitting_parameters:
            asymmetry = float(x_opt[self.config.fitting_parameters.index("asymmetry")])
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
            hill=kwargs.get("hill", self.config.hill),
            effect_0=kwargs.get("effect_0", self.config.effect_0),
            effect_inf=kwargs.get("effect_inf", self.config.effect_inf),
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

        self._compute_gof_metrics(result, valids)

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
        cfg = config or _init_config_from_data(data)
        # Honour the bounds=None "derive from data" signal (FitBase readers
        # dereference config.bounds.<param>, so it must be concrete by fit time).
        if cfg.bounds is None:
            cfg.bounds = _init_config_from_data(
                data, direction=cfg.direction, noise=cfg.noise,
            ).bounds
        # Per-point user errors and a fitted σ²(μ) polynomial are mutually
        # exclusive — log_prob enforces it, but rejecting at construction time
        # gives a clearer message.
        if not isinstance(cfg.noise, (GaussianConstant, Lognormal)):
            raise ValueError(
                "SingleDrugFitWithError uses per-point y_err — pair it with "
                "noise.kind='gaussian_constant' or 'lognormal'. To fit a heteroscedastic σ²(μ) "
                "polynomial, drop y_err and use SingleDrugFit instead."
            )
        super().__init__(cfg)

    def _log_prob_data(self, x: np.ndarray, valids: np.ndarray | None = None) -> float:
        kwargs = self._x_to_kwargs(x)
        conc = self.data["concentration"].values
        y = self.data["y"].values
        y_err = self.data["y_err"].values
        y_pred = hill_curve(conc, **kwargs)
        return noise_log_prob(
            y, y_pred,
            self.config.noise,
            y_err=y_err if np.isfinite(y_err).all() else None,
            mask=valids,
        )

    def fit(self, valids: np.ndarray | None = None) -> FitResult:
        if valids is None:
            valids = np.ones(len(self.data), dtype=bool)

        n_valid = int(valids.sum())
        n_params = len(self.config.fitting_parameters)
        if n_valid < n_params + 1:
            raise ValueError(
                f"Need at least {n_params + 1} valid data points for {n_params} free parameters; "
                f"got {n_valid}."
            )

        x0, bounds = self._get_x0_and_bounds()
        x_opt, success, message, pcov = self._run_minimize(x0, bounds, valids=valids)
        kwargs = self._x_to_kwargs(x_opt)
        log_c50 = x_opt[self.config.fitting_parameters.index("log_c50")]

        return FitResult(
            c50=kwargs["c50"],
            log_c50=float(log_c50),
            hill=kwargs.get("hill", self.config.hill),
            effect_0=kwargs.get("effect_0", self.config.effect_0),
            effect_inf=kwargs.get("effect_inf", self.config.effect_inf),
            success=success,
            n_valid=n_valid,
            n_total=len(self.data),
            message=message,
            direction=self.config.direction,
            param_cov=pcov,
            param_names=list(self.config.fitting_parameters),
        )
