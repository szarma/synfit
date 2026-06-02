from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm as _norm

from .noise import (
    CompoundAddMult,
    GaussianLinear,
    GaussianQuadratic,
    NoiseSpec,
    default_for_kind,
    error_model_name,
    free_coefficients,
    from_dict as noise_from_dict,
    legacy_to_noise_spec,
    variance_at,
    variance_model_name,
)


@dataclass
class FitBounds:
    log_c50: tuple = (-5.0, 5.0)
    hill: tuple = (0.1, 4.0)
    effect_0: tuple = (0.0, 2.0)
    effect_inf: tuple = (0.0, 2.0)
    asymmetry: tuple = (0.1, 10.0)
    # Variance polynomial coefficients σ²(μ) = a + b·μ + c·μ².
    # ``a`` (intercept) must stay strictly positive; floor pinned away from 0
    # since log(0) blows up the likelihood. Wide upper bounds — the optimiser
    # will land far inside them in practice.
    var_a: tuple = (1e-12, 1e4)
    var_b: tuple = (-1e4, 1e4)
    var_c: tuple = (0.0, 1e4)


@dataclass
class FitConfig:
    """Initial parameter values and bounds for a single-drug fit."""
    log_c50: float = 0.0
    hill: float = 1.0
    effect_0: float = 1.0
    effect_inf: float = 0.0
    fitting_parameters: list = field(
        default_factory=lambda: ["log_c50", "hill", "effect_0", "effect_inf"]
    )
    # ``None`` means "derive from data at fit time" (see ``SingleDrugFit``);
    # a concrete ``FitBounds`` is honoured literally.
    bounds: FitBounds | None = None
    noise: NoiseSpec | dict | str = field(default_factory=lambda: default_for_kind("gaussian_constant"))
    # Initial value for the asymmetry parameter (5p model only)
    asymmetry: float = 1.0
    # "inhibition" (default, IC₅₀, response falls with dose) or
    # "activation" (EC₅₀, response rises with dose)
    direction: str = "inhibition"

    def __init__(
        self,
        log_c50: float = 0.0,
        hill: float = 1.0,
        effect_0: float = 1.0,
        effect_inf: float = 0.0,
        fitting_parameters: list | None = None,
        bounds: FitBounds | None = None,
        noise: NoiseSpec | dict | str | None = None,
        asymmetry: float = 1.0,
        direction: str = "inhibition",
        *,
        error_model: str | None = None,
        variance_model: str | None = None,
        var_a: float = 1e-3,
        var_b: float = 0.0,
        var_c: float = 0.0,
        sigma_log_init: float = 0.1,
    ) -> None:
        if noise is not None and (error_model is not None or variance_model is not None):
            raise ValueError("Pass either noise or legacy error_model/variance_model, not both.")
        self.log_c50 = log_c50
        self.hill = hill
        self.effect_0 = effect_0
        self.effect_inf = effect_inf
        self.fitting_parameters = (
            list(fitting_parameters)
            if fitting_parameters is not None
            else ["log_c50", "hill", "effect_0", "effect_inf"]
        )
        # ``bounds is None`` is the persistent signal for "derive from data":
        # ``SingleDrugFit`` fills data-driven bounds when none were supplied (so
        # a bare ``FitConfig()`` fits any-magnitude data), and honours supplied
        # bounds literally — no equality-based override. Storing ``None`` (rather
        # than a flag) keeps the semantic correct across ``dataclasses.replace``
        # / ``asdict`` round-trips, which would otherwise re-pass resolved bounds
        # and mark them explicit.
        self.bounds = bounds
        self.noise = (
            noise
            if noise is not None
            else legacy_to_noise_spec(
                error_model or "gaussian",
                variance_model or "constant",
                var_a=var_a,
                var_b=var_b,
                var_c=var_c,
                sigma_log_init=sigma_log_init,
            )
        )
        self.asymmetry = asymmetry
        self.direction = direction
        self.__post_init__()

    def __post_init__(self) -> None:
        object.__setattr__(self, "noise", noise_from_dict(self.noise))

        # Auto-extend the optimiser parameter list with the variance
        # coefficients this model needs. Idempotent — only adds names that
        # aren't already there, so a user who hand-crafts ``fitting_parameters``
        # to fix a coefficient still wins.
        for name in free_coefficients(self.noise):
            if name not in self.fitting_parameters:
                self.fitting_parameters.append(name)

    @classmethod
    def from_legacy(
        cls,
        *,
        error_model: str = "gaussian",
        variance_model: str = "constant",
        var_a: float = 1e-3,
        var_b: float = 0.0,
        var_c: float = 0.0,
        sigma_log_init: float = 0.1,
        **kwargs,
    ) -> "FitConfig":
        """Build from the old parallel error/variance fields."""
        return cls(
            noise=legacy_to_noise_spec(
                error_model,
                variance_model,
                var_a=var_a,
                var_b=var_b,
                var_c=var_c,
                sigma_log_init=sigma_log_init,
            ),
            **kwargs,
        )

    @property
    def error_model(self) -> str:
        return error_model_name(self.noise)

    @property
    def variance_model(self) -> str:
        return variance_model_name(self.noise)

    @property
    def var_a(self) -> float:
        if isinstance(self.noise, (GaussianLinear, GaussianQuadratic)):
            return self.noise.a_init
        return 1e-3

    @property
    def var_b(self) -> float:
        if isinstance(self.noise, (GaussianLinear, GaussianQuadratic)):
            return self.noise.b_init
        return 0.0

    @property
    def var_c(self) -> float:
        if isinstance(self.noise, GaussianQuadratic):
            return self.noise.c_init
        return 0.0

    @property
    def sigma_log(self) -> float:
        if isinstance(self.noise, CompoundAddMult):
            return self.noise.sigma_log_init
        return 0.1


@dataclass
class FitResult:
    """Result of a scipy-minimize fit."""
    c50: float
    log_c50: float
    hill: float
    effect_0: float
    effect_inf: float
    success: bool
    n_valid: int
    n_total: int
    message: str = ""
    # Goodness-of-fit metrics (populated by SingleDrugFit)
    r2: float | None = None
    rss: float | None = None
    log_likelihood: float | None = None
    aic: float | None = None
    bic: float | None = None
    n_params: int | None = None
    # 5-parameter Hill asymmetry factor (None for 4p model)
    asymmetry: float | None = None
    # Curve direction — matches FitConfig.direction
    direction: str = "inhibition"
    # Which error model was used for this fit
    error_model: str = "gaussian"
    # MLE noise scale σ̂ for the *constant* variance case, in the error
    # model's natural likelihood space (response units for gaussian, log-
    # response units for lognormal). None for non-constant variance models
    # or when per-point errors were supplied by the user.
    sigma: float | None = None
    # Variance model used for this fit (constant / linear / quadratic).
    variance_model: str = "constant"
    # Fitted variance polynomial coefficients σ²(μ) = a + b·μ + c·μ².
    # Populated when ``variance_model`` is non-constant. Only the
    # coefficients that the model actually fits are present.
    variance_params: dict | None = None
    # Parameter covariance matrix (rows/cols ordered as fitting_parameters)
    param_cov: np.ndarray | None = field(default=None, repr=False)
    # Names of the fitted parameters (index into param_cov)
    param_names: list | None = field(default=None, repr=False)

    def predict_variance(self, mu: np.ndarray | float) -> np.ndarray | None:
        """σ²(μ) for this fit given a predicted response.

        Returns the per-point variance using ``variance_params`` when the fit
        used a non-constant variance model, falling back to the profiled
        ``sigma**2`` for the constant case. Returns ``None`` if neither is
        available (e.g. a user-supplied-y_err fit).

        Always evaluated on the original response scale, irrespective of
        ``error_model`` — that's the space the σ² polynomial is defined in.
        """
        mu = np.asarray(mu, dtype=float)
        if self.variance_params:
            a = float(self.variance_params.get("a", 0.0))
            b = float(self.variance_params.get("b", 0.0))
            c = float(self.variance_params.get("c", 0.0))
            return variance_at(mu, a, b, c)
        if self.sigma is not None:
            return np.full_like(mu, float(self.sigma) ** 2)
        return None

    def param_ci(self, alpha: float = 0.05) -> dict[str, tuple[float, float]] | None:
        """Return ``{name: (lo, hi)}`` for each fitted parameter at 1-alpha level.

        For ``log_c50`` an additional ``c50`` key is added with the CI
        propagated to linear space via the delta method (10**log_c50).
        """
        if self.param_cov is None or self.param_names is None:
            return None
        z = _norm.ppf(1 - alpha / 2)
        cis: dict[str, tuple[float, float]] = {}
        var_short = (self.variance_params or {})
        for i, name in enumerate(self.param_names):
            se = float(np.sqrt(self.param_cov[i, i]))
            # Variance polynomial coefficients live on ``variance_params``,
            # not as direct attributes — pull them from there so their CIs
            # show up alongside the curve params.
            if name.startswith("var_") and len(name) == 5:
                val = var_short.get(name[4:])
            else:
                val = getattr(self, name, None)
            if val is None:
                continue
            cis[name] = (val - z * se, val + z * se)

        # Propagate log_c50 CI to c50 (monotonic transform)
        if "log_c50" in cis:
            lo_log, hi_log = cis["log_c50"]
            cis["c50"] = (10**lo_log, 10**hi_log)
        return cis

    def predict_ci(
        self,
        conc: np.ndarray,
        alpha: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Confidence band on the fitted curve via the delta method.

        Returns ``(lo, hi)`` arrays or ``None`` if covariance is unavailable.
        Uses the analytic Jacobian of the 4p Hill model.
        """
        from .hill import hill_curve

        if self.param_cov is None or self.param_names is None:
            return None

        n = len(conc)
        p = len(self.param_names)
        J = np.zeros((n, p))
        name_to_ix = {n: i for i, n in enumerate(self.param_names)}

        log_c50 = self.log_c50
        hill = self.hill
        top = self.effect_0
        bottom = self.effect_inf
        s = self.asymmetry if self.asymmetry is not None else 1.0

        log10_x = np.log10(np.maximum(conc, 1e-300))
        H = np.exp(hill * np.log(10) * (log10_x - log_c50))
        frac = np.where(np.isfinite(H), 1.0 / (1.0 + H) ** s, 0.0)
        Hfrac_over_opH = np.where(np.isfinite(H), H * frac / (1.0 + H), 0.0)

        if "effect_0" in name_to_ix:
            J[:, name_to_ix["effect_0"]] = frac
        if "effect_inf" in name_to_ix:
            J[:, name_to_ix["effect_inf"]] = 1.0 - frac
        if "log_c50" in name_to_ix:
            J[:, name_to_ix["log_c50"]] = (top - bottom) * s * hill * np.log(10) * Hfrac_over_opH
        if "hill" in name_to_ix:
            # ∂y/∂hill = −(top − bottom) · S · ln(10) · H·frac/(1+H) · (log10_x − log_c50)
            # Earlier versions had ``log_c50 − log10_x`` here, sign-flipping the column
            # and corrupting any var_y contribution from Cov[hill, ·] off-diagonals.
            # Test: ``test_predict_ci_jacobian_matches_numerical_finite_differences``.
            J[:, name_to_ix["hill"]] = -(top - bottom) * s * np.log(10) * Hfrac_over_opH * (log10_x - log_c50)
        if "asymmetry" in name_to_ix:
            log_one_plus_H = np.where(np.isfinite(H), np.log(1.0 + H), np.log(np.maximum(H, 1e-300)))
            J[:, name_to_ix["asymmetry"]] = -(top - bottom) * frac * log_one_plus_H

        var_y = np.einsum("ij,jk,ik->i", J, self.param_cov, J)
        std_y = np.sqrt(np.maximum(var_y, 0.0))

        y_hat = hill_curve(
            conc, c50=self.c50, hill=hill,
            effect_0=top, effect_inf=bottom, asymmetry=self.asymmetry,
        )
        z = _norm.ppf(1 - alpha / 2)
        return y_hat - z * std_y, y_hat + z * std_y

    @property
    def kappa(self) -> float:
        """The Hill concentration parameter κ.

        Always equals ``c50`` (the optimised parameter). Named ``kappa``
        because that is what the parameter is in the asymmetric (5p) Hill
        equation; for the symmetric (4p) case it additionally equals the
        half-maximal IC₅₀/EC₅₀, but the parameter itself is the same in
        both cases.
        """
        return self.c50

    @property
    def half_max(self) -> float:
        """True half-maximal concentration (IC₅₀ for inhibition, EC₅₀ for
        activation).

        For 4p (or 5p with S=1) this equals ``kappa``. For 5p with S≠1 the
        half-max sits below or above κ depending on S:

            half_max = κ · (2^(1/S) − 1)^(1/h)

        At S=2, n_H=1 the half-max is ~0.41·κ; at S=0.5, n_H=1 it is ~1.73·κ.
        Reporting ``ic50``/``ec50`` as κ for asymmetric fits would
        misrepresent the actual half-max by tens of percent.
        """
        s = self.asymmetry if self.asymmetry is not None else 1.0
        if s == 1.0:
            return self.c50
        factor = (2.0 ** (1.0 / s) - 1.0) ** (1.0 / self.hill)
        return self.c50 * factor

    def _half_max_ci(self) -> tuple[float, float] | None:
        """Delta-method 95% CI on the half-max, propagating through the
        ``half_max = κ · (2^(1/S) − 1)^(1/h)`` transform.

        For 4p (no asymmetry param), the half-max equals c50 so we just
        return the c50 CI verbatim. For 5p, build the gradient of
        ``log(half_max)`` w.r.t. ``(log_c50, hill, asymmetry)`` and
        propagate variance through the cov matrix.
        """
        if self.param_cov is None or self.param_names is None:
            return None

        s = self.asymmetry
        if s is None or s == 1.0:
            cis = self.param_ci()
            return cis.get("c50") if cis else None

        # log(half_max) = ln(10)·log_c50 + (1/h)·ln(2^(1/s) - 1)
        # ∂/∂log_c50 = ln(10)
        # ∂/∂hill    = -(1/h²)·ln(2^(1/s) - 1)
        # ∂/∂s       = (1/h) · [(-ln(2)/s²) · 2^(1/s)] / (2^(1/s) - 1)
        h = self.hill
        two_over_s = 2.0 ** (1.0 / s)
        denom = two_over_s - 1.0
        ln_denom = np.log(denom)
        grad = {
            "log_c50": np.log(10.0),
            "hill": -(1.0 / (h * h)) * ln_denom,
            "asymmetry": (1.0 / h) * ((-np.log(2.0) / (s * s)) * two_over_s) / denom,
        }
        ix = {n: i for i, n in enumerate(self.param_names)}
        g = np.zeros(len(self.param_names))
        for name, val in grad.items():
            if name in ix:
                g[ix[name]] = val
        var_log_hm = float(g @ self.param_cov @ g)
        if not np.isfinite(var_log_hm) or var_log_hm < 0:
            return None
        z = _norm.ppf(0.975)
        sd = np.sqrt(var_log_hm)
        hm = self.half_max
        return (hm * np.exp(-z * sd), hm * np.exp(z * sd))

    def to_dict(self) -> dict:
        d = {
            "c50": self.c50,
            "kappa": self.c50,
            "log_c50": self.log_c50,
            "hill": self.hill,
            "effect_0": self.effect_0,
            "effect_inf": self.effect_inf,
            "success": self.success,
            "n_valid": self.n_valid,
            "n_total": self.n_total,
            "message": self.message,
            "r2": self.r2,
            "rss": self.rss,
            "log_likelihood": self.log_likelihood,
            "aic": self.aic,
            "bic": self.bic,
            "n_params": self.n_params,
            "asymmetry": self.asymmetry,
            "direction": self.direction,
            "error_model": self.error_model,
            "sigma": self.sigma,
            "variance_model": self.variance_model,
            "variance_params": self.variance_params,
        }
        # ``ic50``/``ec50`` is the *true* half-max, not the κ parameter:
        #   4p / 5p with S=1 → equals κ (the existing behaviour, unchanged)
        #   5p with S≠1     → κ · (2^(1/S)−1)^(1/h)  (was wrongly = κ before)
        # Consumers reading these fields now see a number that always
        # matches the pharmacological IC₅₀/EC₅₀ definition.
        half = self.half_max
        if self.direction == "inhibition":
            d["ic50"] = half
        else:
            d["ec50"] = half

        cis = self.param_ci()
        if cis:
            d["ci"] = {k: [round(float(lo), 6), round(float(hi), 6)] for k, (lo, hi) in cis.items()}
            # Mirror c50 CI under the kappa name so consumers who read by
            # the canonical parameter name find it.
            if "c50" in cis:
                lo, hi = cis["c50"]
                d["ci"]["kappa"] = [round(float(lo), 6), round(float(hi), 6)]
            half_ci = self._half_max_ci()
            if half_ci is not None:
                lo, hi = half_ci
                key = "ic50" if self.direction == "inhibition" else "ec50"
                d["ci"][key] = [round(float(lo), 6), round(float(hi), 6)]
        if self.param_cov is not None:
            d["param_cov"] = self.param_cov.tolist()
        if self.param_names:
            d["param_names"] = list(self.param_names)
        return d
