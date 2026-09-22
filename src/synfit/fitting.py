import numpy as np
from scipy.optimize import minimize, approx_fprime

from .data import FitConfig
from .hill import log_wall
from .param_roles import VARIANCE_PARAM_NAMES, is_asymptote_param, is_log_param, is_variance_param
from .noise import (
    _DEFAULT_VAR_A_INIT,
    is_heteroscedastic_gaussian,
)

# Large multiplier so soft-boundary violations dominate the objective
PRIOR_PENALTY_WEIGHT = 1000

# Step size for the numerical Hessian. The Hessian is built from forward
# finite differences of a forward-difference gradient (``approx_fprime`` twice),
# taken in scale-relative coordinates (see ``_estimate_covariance``).
_HESS_EPS = 1e-5


def effective_observation_mask(
    valids: np.ndarray | None,
    n_rows: int,
    concentration: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Boolean mask of rows that enter the likelihood (caller mask and finite data)."""
    if valids is None:
        mask = np.ones(n_rows, dtype=bool)
    else:
        valids_arr = np.asarray(valids)
        if valids_arr.ndim != 1 or valids_arr.shape[0] != n_rows:
            raise ValueError(
                f"valids must be a 1-D boolean array of length {n_rows}; "
                f"got shape {valids_arr.shape}."
            )
        mask = valids_arr.astype(bool, copy=False)
    conc = np.asarray(concentration, dtype=float)
    resp = np.asarray(y, dtype=float)
    return mask & np.isfinite(conc) & np.isfinite(resp)


def matrix_observation_masks(
    replicates: list[np.ndarray],
    valids: list[np.ndarray] | None,
) -> list[np.ndarray]:
    """Per-replicate masks: caller valids (if any) and finite matrix responses."""
    if valids is not None and len(valids) != len(replicates):
        raise ValueError(
            f"valids must have one mask per replicate; got {len(valids)} "
            f"masks for {len(replicates)} replicates."
        )
    masks: list[np.ndarray] = []
    for i, rep in enumerate(replicates):
        rep_arr = np.asarray(rep)
        if valids is None:
            base = np.ones(rep_arr.shape, dtype=bool)
        else:
            v = np.asarray(valids[i])
            if v.shape != rep_arr.shape:
                raise ValueError(
                    f"valids[{i}] has shape {v.shape}, expected {rep_arr.shape} "
                    f"(len(conc_vertical) x len(conc_horizontal))"
                )
            base = v.astype(bool, copy=False)
        masks.append(base & np.isfinite(rep_arr))
    return masks


def parameter_scale(
    x0: np.ndarray | list[float],
    bounds: list[tuple[float, float]] | None = None,
    names: list[str] | None = None,
    *,
    bound_scale_threshold: float = 0.25,
    response_scale: float | None = None,
) -> np.ndarray:
    """Per-parameter optimiser / Hessian scale from x0 and bounds.

    The base scale is ``max(|x0|, 1)``, which leaves already-O(1) parameters
    unchanged. For the magnitude-bearing asymptotes *only* (``effect_0`` /
    ``effect_inf`` for single-drug & matrix fits, ``top`` / ``bottom`` for the
    joint-marginal fit — see :func:`param_roles.is_asymptote_param`), when
    ``|x0|`` is small relative to the feasible range the bound magnitude is used
    instead. This preconditions a
    hand-crafted ``x0`` that does not reflect the response magnitude — e.g. a
    fixed ``effect_inf=1`` fitted against ELISA-scale data, where the true
    asymptote is order 1e4 and the data-derived bounds are correspondingly
    wide.

    The fallback deliberately never touches log-domain location parameters
    (``log_c50``…) or dimensionless shape exponents (``hill``…,
    ``asymmetry``): their bounds describe a log / shape range, not the data
    range, so inflating their scale by the bound magnitude destroys the
    optimiser conditioning — a single step in ``log_c50`` would span the whole
    feasible decade range, sending the fit to a spurious optimum with a
    non-invertible Hessian (the regression that #14's first cut introduced).

    Variance polynomial coefficients (``var_a`` / ``var_b`` / ``var_c``) use
    ``max(|x0|, typical)`` *without* flooring at 1, where ``typical`` comes
    from the fitter's response scale ``s`` (``a ~ s²``, ``b ~ s``, ``c ~ 1``).
    Data-derived ``var_*`` bounds already track ``s``; using ``s`` here keeps
    the optimiser equivariant at tiny and ELISA magnitudes even when
    ``b_init = c_init = 0``. When ``s`` is unknown the ``max(|x0|, 1)`` floor
    is left in place (explicit-bound / non-data fitters).

    ``response_scale`` is only passed for linear/quadratic Gaussian fits.
    Constant Gaussian, lognormal, and compound noise must see the historical
    ``parameter_scale(x0, bounds, names)`` path (no ``s``). For heteroscedastic
    fits with ``0 < s < 1``, asymptote scales use ``s`` instead of the global
    floor at 1 — otherwise tiny ``effect_0`` / ``effect_inf`` freeze in z-space
    and scale equivariance at ``s = 1e-3`` fails.

    ``names`` aligns with ``x0`` / ``bounds`` and selects which parameters are
    asymptotes. When omitted (no name information available) the fallback
    applies to every parameter, preserving the original behaviour.
    """
    x0 = np.asarray(x0, dtype=float)
    scale = np.maximum(np.abs(x0), 1.0)
    if bounds is None and response_scale is None:
        return scale
    for i in range(len(x0)):
        if names is not None and is_variance_param(names[i]):
            s = response_scale
            if s is not None and np.isfinite(s) and s > 0:
                if names[i] == "var_a":
                    typical = _DEFAULT_VAR_A_INIT * float(s) ** 2
                elif names[i] == "var_b":
                    typical = float(s)
                else:
                    typical = 1.0
                scale[i] = max(abs(x0[i]), typical, 1e-12)
            continue
        if bounds is None:
            continue
        lo, hi = bounds[i]
        if names is not None and not is_asymptote_param(names[i]):
            continue
        lo_f, hi_f = float(lo), float(hi)
        width = max(hi_f - lo_f, 0.0)
        bound_mag = max(abs(lo_f), abs(hi_f), width * 0.5)
        s = response_scale
        if s is not None and np.isfinite(s) and 0 < s < 1.0:
            # Heteroscedastic only (callers omit ``s`` otherwise). Sub-unit
            # responses: the global floor at 1 freezes tiny asymptotes in
            # z-space (z ~ 1e-3) so L-BFGS-B stops early. Use s itself as
            # the scale so z matches the native-magnitude fit.
            scale[i] = max(abs(x0[i]), float(s), 1e-12)
            if abs(x0[i]) < bound_scale_threshold * bound_mag:
                scale[i] = max(scale[i], bound_mag, 1e-12)
            continue
        if abs(x0[i]) < bound_scale_threshold * bound_mag:
            scale[i] = max(scale[i], bound_mag, 1.0)
    return scale


class FitBase:
    """
    Base class for scipy-minimize fitting.

    Child classes implement _log_prob_data(x, **kwargs).
    """

    # Backwards compatibility: exposed callers may reference this.
    _VARIANCE_PARAM_NAMES = tuple(VARIANCE_PARAM_NAMES)

    def __init__(self, config: FitConfig | None = None):
        self.config = config or FitConfig()

    def _parameter_value(self, name: str, x: np.ndarray) -> float:
        """Return ``name`` from ``x`` if it is being estimated, else FitConfig.

        ``fitting_parameters`` is the estimated set; anything else is held
        at its configured value. Used by single-drug ``fit`` so a pinned
        ``log_c50`` does not assume it is in the optimiser vector.
        """
        try:
            idx = self.config.fitting_parameters.index(name)
        except ValueError:
            return float(getattr(self.config, name))
        return float(x[idx])

    def _x_to_kwargs(self, x: np.ndarray) -> dict:
        """Convert parameter array to hill_curve kwargs (un-logs log_ params).

        Every Hill-curve parameter is present: values in ``fitting_parameters``
        come from ``x``; any others fall back to the FitConfig so a pinned
        hill / baseline / log_c50 actually enters the likelihood. Log-space
        names are un-logged in both cases (``log_c50`` → ``c50 = 10**v``).
        Variance polynomial coefficients (``var_a`` etc.) are filtered out —
        retrieve them via ``_x_to_variance_params`` instead.

        ``JointMarginalFit`` / ``MatrixFit`` do not use this helper: they
        unpack via ``_get`` / ``_unpack_x`` and already restore fixed values
        themselves. Changing the fallback here does not double-handle them.
        """
        values = {
            "log_c50": float(self.config.log_c50),
            "hill": float(self.config.hill),
            "effect_0": float(self.config.effect_0),
            "effect_inf": float(self.config.effect_inf),
            "asymmetry": float(self.config.asymmetry),
        }
        for par, v in zip(self.config.fitting_parameters, x):
            if is_variance_param(par):
                continue
            values[par] = float(v)
        out = {}
        for par, v in values.items():
            if is_log_param(par):
                out[par[4:]] = 10.0 ** v
            else:
                out[par] = v
        return out

    def _x_to_variance_params(self, x: np.ndarray) -> tuple[float, float, float] | None:
        """Return ``(a, b, c)`` for the heteroscedastic σ² polynomial.

        Falls back to the FitConfig's static initial values for any coefficient
        not in ``fitting_parameters`` (i.e. a fixed coefficient such as ``c=0``
        for the linear model). Returns ``None`` for the constant model — that
        path is handled by the profile-likelihood branch of ``log_prob``.
        """
        from .noise import GaussianLinear, GaussianQuadratic  # avoid cycle at import time
        if not isinstance(self.config.noise, (GaussianLinear, GaussianQuadratic)):
            return None
        defaults = {
            "var_a": float(self.config.var_a),
            "var_b": float(self.config.var_b),
            "var_c": float(self.config.var_c),
        }
        for par, v in zip(self.config.fitting_parameters, x):
            if is_variance_param(par):
                defaults[par] = float(v)
        return defaults["var_a"], defaults["var_b"], defaults["var_c"]

    def _variance_anchor(self, x: np.ndarray) -> float:
        """Lower-asymptote anchor for σ² at this trial point.

        Default: ``min(effect_0, effect_inf)`` from the unpacked curve kwargs.
        ``JointMarginalFit`` overrides with ``min(top, bottom)``.
        """
        kwargs = self._x_to_kwargs(x)
        e0 = kwargs.get("effect_0")
        einf = kwargs.get("effect_inf")
        if e0 is None or einf is None:
            return 0.0
        return min(float(e0), float(einf))

    def _require_bounds(self):
        """Return concrete ``config.bounds``, or raise if still unresolved.

        ``FitConfig.bounds`` defaults to ``None``, meaning "derive from data at
        fit time" — a signal only a *data-bearing* subclass can honour (it sets
        ``config.bounds`` to a concrete ``FitBounds`` before fitting, as
        ``SingleDrugFit`` / ``SingleDrugFitWithError`` do; ``JointMarginalFit`` /
        ``MatrixFit`` instead override the readers below and supply their own
        x0/bounds). The base optimiser path dereferences ``bounds.<param>``, so
        a ``None`` here is a programming error in a subclass that forgot to
        resolve it. Raise an actionable message instead of an opaque
        ``AttributeError: 'NoneType' object has no attribute 'effect_0'``. The
        base class deliberately does not invent a default: with no data it could
        only fall back to the neutral ``FitBounds()`` the data-derived scheme
        replaced, silently fitting against the wrong bracket.
        """
        bounds = self.config.bounds
        if bounds is None:
            raise ValueError(
                "FitConfig.bounds is None at fit time. bounds=None means "
                "'derive from data', which only a data-bearing FitBase subclass "
                "can do — it must resolve config.bounds to a concrete FitBounds "
                "before fitting (see SingleDrugFit), or the subclass must supply "
                "its own x0/bounds and override _log_prior_prob (see "
                "JointMarginalFit / MatrixFit). Otherwise pass an explicit "
                "FitBounds in the FitConfig."
            )
        return bounds

    def _log_prior_prob(self, x: np.ndarray) -> float:
        bounds_obj = self._require_bounds()
        penalty = 0.0
        for par, v in zip(self.config.fitting_parameters, x):
            bounds = getattr(bounds_obj, par)
            penalty += log_wall(np.array([v]), bounds).sum()
        return -penalty * PRIOR_PENALTY_WEIGHT

    def _log_prob_data(self, x: np.ndarray, **kwargs) -> float:
        raise NotImplementedError

    def _objective(self, x: np.ndarray, **kwargs) -> float:
        return -(self._log_prob_data(x, **kwargs) + self._log_prior_prob(x))

    def _get_x0_and_bounds(self):
        """Build initial values and bounds arrays from config."""
        bounds_obj = self._require_bounds()
        x0 = [getattr(self.config, par) for par in self.config.fitting_parameters]
        bounds = [getattr(bounds_obj, par) for par in self.config.fitting_parameters]
        return x0, bounds

    def _scale_param_names(self) -> list[str] | None:
        """Ordered fitting-parameter names aligned with x0/bounds, if known.

        Lets ``parameter_scale`` tell shape/log parameters from
        magnitude-bearing asymptotes. Subclasses that pass their own x0/bounds
        to the optimiser expose the aligned names in one of two ways:
        ``MatrixFit`` via a ``_PARAM_NAMES`` class attribute, ``JointMarginalFit``
        via a ``_param_names`` instance list set in ``fit()``. Everything else
        uses the config's ``fitting_parameters``. Returns ``None`` when no names
        are available (the scale fallback then applies to every parameter).
        """
        names = getattr(self, "_PARAM_NAMES", None)
        if names is None:
            existing = getattr(self, "_param_names", None)
            if isinstance(existing, (list, tuple)):
                names = existing
        if names is None:
            config = getattr(self, "config", None)
            names = getattr(config, "fitting_parameters", None)
        return list(names) if names is not None else None

    def _optimizer_response_scale(self) -> float | None:
        """Response scale for ``parameter_scale``, or ``None`` for non-hetero fits.

        Constant Gaussian, lognormal, and compound noise must not see
        ``response_scale`` — that would change asymptote preconditioning on
        sub-unit data relative to the historical ``parameter_scale(x0, bounds,
        names)`` path.
        """
        noise = getattr(self, "noise", None)
        if noise is None:
            noise = getattr(getattr(self, "config", None), "noise", None)
        if not is_heteroscedastic_gaussian(noise):
            return None
        s = getattr(self, "_response_scale", None)
        if s is None or not np.isfinite(s) or s <= 0:
            return None
        return float(s)

    def _run_minimize(self, x0, bounds, **kwargs) -> tuple[np.ndarray, bool, str, np.ndarray | None]:
        # Optimise in scale-relative coordinates z = x / scale so L-BFGS-B sees
        # O(1) variables regardless of the response magnitude. With raw
        # parameters, a large-magnitude asymptote (e.g. effect_inf ~ 1e5) carries
        # a correspondingly tiny gradient, so the projected-gradient stopping
        # criterion (pgtol) is met prematurely and the fit under-converges —
        # which makes the result depend on the absolute scale of the data.
        # Flooring the scale at 1.0 leaves already-O(1) parameters unchanged.
        x0 = np.asarray(x0, dtype=float)
        start_f = self._objective(x0, **kwargs)

        def _once(x_start):
            scale = parameter_scale(
                x_start, bounds, self._scale_param_names(),
                response_scale=self._optimizer_response_scale(),
            )
            z0 = x_start / scale
            z_bounds = (
                [(lo / sc, hi / sc) for (lo, hi), sc in zip(bounds, scale)]
                if bounds is not None
                else None
            )

            def objective_z(z):
                return self._objective(z * scale, **kwargs)

            res = minimize(objective_z, z0, bounds=z_bounds, method="L-BFGS-B")
            x_opt = res.x * scale
            return res, x_opt

        res, x_opt = _once(x0)
        message = res.message
        if self._should_retry_variance_init(res, start_f):
            x0_retry = self._variance_retry_x0(x0, bounds)
            if not np.allclose(x0_retry, x0):
                res, x_opt = _once(x0_retry)
                message = (
                    f"{res.message}; retried from default variance initials "
                    "after the optimizer stalled at the start"
                )
        pcov = self._estimate_covariance(x_opt, bounds, **kwargs)
        return x_opt, res.success, message, pcov

    def _should_retry_variance_init(self, res, start_f: float) -> bool:
        """Retry once if L-BFGS-B stalled at iteration 0 with a finite start.

        Only for linear/quadratic Gaussian noise. Invalid input, a non-finite
        starting objective, and failures after iteration 0 are not retried.
        """
        if getattr(res, "success", False):
            return False
        if getattr(res, "nit", -1) != 0:
            return False
        if not np.isfinite(start_f):
            return False
        noise = getattr(getattr(self, "config", None), "noise", None)
        return noise is not None and is_heteroscedastic_gaussian(noise)

    def _variance_retry_x0(self, x0: np.ndarray, bounds) -> np.ndarray:
        """Reset variance initials to b = 0, c = 0, a = 1e-3·s² (or 1e-3)."""
        x0 = np.array(x0, dtype=float, copy=True)
        names = self._scale_param_names() or []
        s = getattr(self, "_response_scale", None)
        default_a = (
            _DEFAULT_VAR_A_INIT * float(s) ** 2
            if s is not None and np.isfinite(s)
            else _DEFAULT_VAR_A_INIT
        )
        for i, name in enumerate(names):
            if i >= len(x0):
                break
            if name == "var_a":
                x0[i] = default_a
            elif name in ("var_b", "var_c"):
                x0[i] = 0.0
            else:
                continue
            if bounds is not None and i < len(bounds):
                lo, hi = bounds[i]
                x0[i] = float(np.clip(x0[i], lo, hi))
        return x0

    def _estimate_covariance(
        self,
        x_opt: np.ndarray,
        bounds: list[tuple[float, float]] | None = None,
        **kwargs,
    ) -> np.ndarray | None:
        """Estimate parameter covariance from the numerical Hessian at the optimum.

        When a parameter is pinned at one of its bounds, the unconstrained
        Hessian is not positive-definite at the optimum (the gradient in that
        direction doesn't vanish — the bound holds it back). Naively inverting
        the full Hessian then produces negative diagonals, masking the genuine
        information present in the *interior* parameters. Strategy:

        1. Identify active-bound parameters (within tolerance scaled to bound width).
        2. Take the principal sub-block of the Hessian over interior parameters.
        3. Invert that sub-block; place it back into a full-size matrix.
        4. Rows/cols for bound-active parameters stay zero — they're effectively
           fixed at the bound, so their CI is degenerate but interior CIs survive.

        If the interior sub-block is singular or has negative diagonals, return None.
        """
        n = len(x_opt)

        # Probe the Hessian in a *scaled* parameter space so the finite-difference
        # step is relative to each parameter's magnitude rather than a fixed
        # absolute ``_HESS_EPS``. Large-magnitude parameters — e.g. ELISA/RFU
        # asymptotes of order 1e4 — otherwise receive a ~1e-9 *relative* step,
        # deep in floating-point cancellation noise; the resulting second
        # derivative comes out with the wrong sign, the inverse Hessian has a
        # negative diagonal, and the whole covariance is rejected. Flooring the
        # scale at 1.0 leaves O(1) parameters (log_c50, hill, variance
        # coefficients) on their original, already-adequate step, so this only
        # changes behaviour for the ill-scaled case.
        x_opt = np.asarray(x_opt, dtype=float)
        if not np.all(np.isfinite(x_opt)):
            return None
        scale = parameter_scale(
            x_opt, bounds, self._scale_param_names(),
            response_scale=self._optimizer_response_scale(),
        )
        z_opt = x_opt / scale

        def neg_ll_scaled(z):
            return -self._log_prob_data(z * scale, **kwargs)

        # Hessian in scaled coordinates: H_z[i, j] = scale_i * scale_j * H_x[i, j].
        H = np.zeros((n, n))
        eps = _HESS_EPS
        for i in range(n):
            def grad_i(z):
                return approx_fprime(z, neg_ll_scaled, eps)[i]
            H[i, :] = approx_fprime(z_opt, grad_i, eps)

        # Symmetrise
        H = 0.5 * (H + H.T)

        # Back-transform the scaled covariance to original units:
        # Σ_x = D Σ_z D with D = diag(scale), i.e. Σ_x[i, j] = scale_i scale_j Σ_z[i, j].
        # Positive scales preserve diagonal signs, so the negative-variance
        # rejection below is equivalent whether checked before or after.
        outer_scale = np.outer(scale, scale)

        # Identify bound-active parameters. Tolerance scales with bound width so
        # very narrow bounds don't get false positives from finite-difference noise.
        active = np.zeros(n, dtype=bool)
        if bounds is not None:
            for i, (lo, hi) in enumerate(bounds):
                width = float(hi) - float(lo)
                tol = max(1e-8, 1e-6 * width)
                if abs(x_opt[i] - lo) <= tol or abs(x_opt[i] - hi) <= tol:
                    active[i] = True

        try:
            if not np.any(active):
                pcov = np.linalg.inv(H) * outer_scale
                # NaN/inf must be rejected explicitly: ``NaN < 0`` is False, so a
                # non-finite covariance would otherwise slip past the sign check.
                if not np.all(np.isfinite(pcov)) or np.any(np.diag(pcov) < 0):
                    return None
                return pcov

            # Invert only the interior sub-block; zero-fill bound-active rows/cols.
            interior = ~active
            if not np.any(interior):
                return None
            H_int = H[np.ix_(interior, interior)]
            pcov_int = np.linalg.inv(H_int)
            if not np.all(np.isfinite(pcov_int)) or np.any(np.diag(pcov_int) < 0):
                return None
            pcov = np.zeros((n, n))
            pcov[np.ix_(interior, interior)] = pcov_int
            pcov = pcov * outer_scale
            if not np.all(np.isfinite(pcov)):
                return None
            return pcov
        except np.linalg.LinAlgError:
            return None
