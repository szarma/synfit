"""
Heteroscedastic noise / error models for dose-response fitting.

Noise models are represented by ``NoiseSpec`` variants:

``gaussian``
    Residuals on the original response scale: ``y = f(x) + ε``,
    ``ε ~ N(0, σ²(μ))``. The variance can be a function of the predicted
    response μ (``GaussianLinear`` / ``GaussianQuadratic``) or a single
    constant (``GaussianConstant``).

``lognormal``
    Residuals on the log response scale: ``y = f(x) · exp(ε)``,
    ``ε ~ N(0, σ²)``. Constant CV. Variance models other than ``constant``
    are not currently supported under lognormal — power-law heteroscedasticity
    on the original scale doesn't compose cleanly with log-space residuals.

``add_mult`` (additive + multiplicative, a.k.a. Rocke–Lorenzato)
    Compound model: ``y = μ · exp(η) + ε``, ``η ~ N(0, σ_log²)``,
    ``ε ~ N(0, σ_add²)``. Multiplicative lognormal noise on the signal *plus*
    additive Gaussian noise on top — captures assays with a detector floor
    (constant noise at low signal, constant CV at high signal). The marginal
    likelihood has no closed form and is evaluated by Gauss–Hermite quadrature
    over η (pass ``compound_params`` to :func:`log_prob`). Public dose-response
    fitters do not yet estimate these scales — use the likelihood primitive or
    synthetic data generation with fixed noise parameters.

Gaussian variance specs
-----------------------
``GaussianConstant``
    σ² = a, profiled out by default. Today's behaviour.

``GaussianLinear`` / ``GaussianQuadratic``
    Anchored at the lower asymptote of the *current* trial parameters:

        σ²(μ) = a + b·d + c·d²,  d = μ − m,  m = min(effect_0, effect_inf)

    (joint-marginal: ``m = min(top, bottom)``). Hill predictions lie between
    the two asymptotes, so ``d ≥ 0`` and σ² ≥ a > 0. Coefficients satisfy
    ``a > 0``, ``b ≥ 0``, ``c ≥ 0``. Linear is the ``c = 0`` special case.

    ``variance_at(..., anchor=0)`` recovers the historical polynomial
    ``a + b·μ + c·μ²`` for callers that are not a fitter.

Log-likelihood
--------------
When per-point errors are **not** known and the variance model is constant,
σ² is *profiled out* analytically (MLE integrated out):

    σ̂² = mean(residuals²)
    log p = -(n/2) · (1 + log(2π σ̂²))

When per-point errors **are** known (``y_err`` argument), or when a richer
variance model produces σ²(μᵢ) per point, the weighted Gaussian log-
likelihood is used:

    log p = Σ -0.5 · (log(2π σᵢ²) + rᵢ² / σᵢ²)

The lognormal model uses the same machinery in log-space, with a Jacobian
correction so AIC / BIC stay comparable to gaussian.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum

import numpy as np


@dataclass(frozen=True)
class GaussianConstant:
    """Gaussian residuals with one profiled constant variance."""

    kind: str = "gaussian_constant"


@dataclass(frozen=True)
class GaussianLinear:
    """Gaussian residuals with σ² = a + b·d, d = μ − m (see module docstring).

    ``a_init`` / ``b_init`` default to ``None`` (data-derived
    ``1e-3·s²`` / ``0``). An explicit value, including ``0.001``, is kept.
    """

    a_init: float | None = None
    b_init: float | None = None
    kind: str = "gaussian_linear"


@dataclass(frozen=True)
class GaussianQuadratic:
    """Gaussian residuals with σ² = a + b·d + c·d², d = μ − m (see module docstring).

    ``a_init`` / ``b_init`` / ``c_init`` default to ``None`` (data-derived
    ``1e-3·s²`` / ``0`` / ``0``). An explicit value, including ``0.001``, is kept.
    """

    a_init: float | None = None
    b_init: float | None = None
    c_init: float | None = None
    kind: str = "gaussian_quadratic"


@dataclass(frozen=True)
class Lognormal:
    """Lognormal residuals with one profiled log-space variance."""

    kind: str = "lognormal"


@dataclass(frozen=True)
class CompoundAddMult:
    """Compound additive + multiplicative residual model."""

    sigma_log_init: float = 0.1
    kind: str = "compound_add_mult"


NoiseSpec = GaussianConstant | GaussianLinear | GaussianQuadratic | Lognormal | CompoundAddMult

_FITTABLE_GAUSSIAN_LOGNORMAL_NOISE = (
    GaussianConstant,
    GaussianLinear,
    GaussianQuadratic,
    Lognormal,
)


def require_supported_single_drug_fitting_noise(noise: NoiseSpec | None) -> None:
    """Reject noise kinds public single-drug fitters cannot estimate.

    Compound additive-multiplicative noise has a likelihood primitive but no
    bound/initial scheme on :class:`~synfit.data.FitBounds`. Raise after noise
    coercion and before any bound derivation.
    """
    if noise is not None and not isinstance(noise, _FITTABLE_GAUSSIAN_LOGNORMAL_NOISE):
        raise ValueError(
            "Single-drug fitting currently supports constant/linear/quadratic "
            "gaussian and lognormal noise only."
        )


_SINGLE_DRUG_WITH_ERROR_NOISE = (GaussianConstant, Lognormal)


def require_supported_single_drug_with_error_fitting_noise(
    noise: NoiseSpec | None,
) -> None:
    """Reject noise kinds :class:`~synfit.single.SingleDrugFitWithError` cannot use."""
    if noise is None or isinstance(noise, _SINGLE_DRUG_WITH_ERROR_NOISE):
        return
    if isinstance(noise, CompoundAddMult):
        raise ValueError(
            "SingleDrugFitWithError currently supports gaussian_constant and "
            "lognormal noise only."
        )
    if isinstance(noise, (GaussianLinear, GaussianQuadratic)):
        raise ValueError(
            "SingleDrugFitWithError uses per-point y_err — pair it with "
            "noise.kind='gaussian_constant' or 'lognormal'. To fit a heteroscedastic σ²(μ) "
            "polynomial, drop y_err and use SingleDrugFit instead."
        )
    raise ValueError(
        "SingleDrugFitWithError currently supports gaussian_constant and "
        "lognormal noise only."
    )


def require_supported_joint_marginal_fitting_noise(noise: NoiseSpec | None) -> None:
    """Reject noise kinds :class:`~synfit.joint_marginal.JointMarginalFit` cannot fit."""
    if noise is not None and not isinstance(noise, _FITTABLE_GAUSSIAN_LOGNORMAL_NOISE):
        raise ValueError(
            "JointMarginalFit currently supports constant/linear/quadratic "
            "gaussian and lognormal noise only."
        )


class ErrorModel(str, Enum):
    """Legacy residual-family names kept for compatibility during issue #58."""

    GAUSSIAN = "gaussian"
    LOGNORMAL = "lognormal"
    ADD_MULT = "add_mult"

    @classmethod
    def coerce(cls, value: "ErrorModel | str") -> "ErrorModel":
        if isinstance(value, cls):
            return value
        return cls(value)


class VarianceModel(str, Enum):
    """Legacy Gaussian variance names kept for compatibility during issue #58."""

    CONSTANT = "constant"
    LINEAR = "linear"
    QUADRATIC = "quadratic"

    @classmethod
    def coerce(cls, value: "VarianceModel | str") -> "VarianceModel":
        if isinstance(value, cls):
            return value
        return cls(value)

    @property
    def free_coefficients(self) -> tuple[str, ...]:
        if self is VarianceModel.CONSTANT:
            return ()
        if self is VarianceModel.LINEAR:
            return ("var_a", "var_b")
        return ("var_a", "var_b", "var_c")


SUPPORTED_NOISE_KINDS = (
    "gaussian_constant",
    "gaussian_linear",
    "gaussian_quadratic",
    "lognormal",
    "compound_add_mult",
)

# Legacy constants kept while callers migrate their display strings.
SUPPORTED_ERROR_MODELS = ("gaussian", "lognormal", "add_mult")
SUPPORTED_VARIANCE_MODELS = ("constant", "linear", "quadratic")


def from_dict(value: NoiseSpec | dict | str | None) -> NoiseSpec:
    """Coerce a tagged JSON dict (or legacy string) to a ``NoiseSpec``."""
    if value is None:
        return GaussianConstant()
    if isinstance(value, (GaussianConstant, GaussianLinear, GaussianQuadratic, Lognormal, CompoundAddMult)):
        return value
    if isinstance(value, str):
        if value == 'auto':
            return GaussianConstant()
        return default_for_kind(value)
    if not isinstance(value, dict):
        raise TypeError(f"noise must be a NoiseSpec, dict, str, or None, not {type(value).__name__}")

    kind = value.get("kind")
    if kind == "gaussian_constant":
        return GaussianConstant()
    if kind == "gaussian_linear":
        return GaussianLinear(
            a_init=_optional_init(value, "a_init", "a"),
            b_init=_optional_init(value, "b_init", "b"),
        )
    if kind == "gaussian_quadratic":
        return GaussianQuadratic(
            a_init=_optional_init(value, "a_init", "a"),
            b_init=_optional_init(value, "b_init", "b"),
            c_init=_optional_init(value, "c_init", "c"),
        )
    if kind == "lognormal":
        return Lognormal()
    if kind in ("compound_add_mult", "add_mult"):
        return CompoundAddMult(sigma_log_init=float(value.get("sigma_log_init", 0.1)))
    raise ValueError(f"Unknown noise kind: {kind!r}. Choose from {SUPPORTED_NOISE_KINDS}")


def to_dict(noise: NoiseSpec) -> dict:
    """Serialise a ``NoiseSpec`` as a tagged JSON dict.

    Unset heteroscedastic initials (``None``) are omitted so a round-trip
    without those keys yields ``None`` again, not the historical 1e-3 / 0 / 0.
    """
    match noise:
        case GaussianConstant():
            return {"kind": "gaussian_constant"}
        case GaussianLinear(a_init=a, b_init=b):
            return _tagged("gaussian_linear", a_init=a, b_init=b)
        case GaussianQuadratic(a_init=a, b_init=b, c_init=c):
            return _tagged("gaussian_quadratic", a_init=a, b_init=b, c_init=c)
        case Lognormal():
            return {"kind": "lognormal"}
        case CompoundAddMult(sigma_log_init=sigma_log):
            return {"kind": "compound_add_mult", "sigma_log_init": sigma_log}
    raise TypeError(f"Unknown NoiseSpec variant: {noise!r}")


def _optional_init(value: dict, *keys: str) -> float | None:
    for key in keys:
        if key in value:
            raw = value[key]
            return None if raw is None else float(raw)
    return None


def _tagged(kind: str, **fields) -> dict:
    out = {"kind": kind}
    for name, val in fields.items():
        if val is not None:
            out[name] = val
    return out


def default_for_kind(kind: str) -> NoiseSpec:
    """Return the default ``NoiseSpec`` for a discriminator or legacy name."""
    if kind in ("gaussian", "constant", "gaussian_constant"):
        return GaussianConstant()
    if kind in ("linear", "gaussian_linear"):
        return GaussianLinear()
    if kind in ("quadratic", "gaussian_quadratic"):
        return GaussianQuadratic()
    if kind == "lognormal":
        return Lognormal()
    if kind in ("add_mult", "compound_add_mult"):
        return CompoundAddMult()
    raise ValueError(f"Unknown noise kind: {kind!r}. Choose from {SUPPORTED_NOISE_KINDS}")


def legacy_to_noise_spec(
    error_model: str = "gaussian",
    variance_model: str = "constant",
    *,
    var_a: float | None = None,
    var_b: float | None = None,
    var_c: float | None = None,
    sigma_log_init: float = 0.1,
) -> NoiseSpec:
    """Convert the old ``error_model`` + ``variance_model`` pair to ``NoiseSpec``.

    TODO(issue #58): remove this compatibility seam after persisted analyses
    have had one release cycle to be rewritten by the new writer.
    """
    if error_model == "auto":
        error_model = "gaussian"
    if error_model == "gaussian":
        if variance_model == "constant":
            return GaussianConstant()
        if variance_model == "linear":
            return GaussianLinear(
                a_init=None if var_a is None else float(var_a),
                b_init=None if var_b is None else float(var_b),
            )
        if variance_model == "quadratic":
            return GaussianQuadratic(
                a_init=None if var_a is None else float(var_a),
                b_init=None if var_b is None else float(var_b),
                c_init=None if var_c is None else float(var_c),
            )
        raise ValueError(f"Unknown variance_model: {variance_model!r}")
    if error_model == "lognormal":
        if variance_model != "constant":
            raise ValueError(
                f"variance_model={variance_model!r} is only supported under "
                "gaussian noise. Lognormal uses constant log-space variance."
            )
        return Lognormal()
    if error_model in ("add_mult", "compound_add_mult"):
        return CompoundAddMult(sigma_log_init=float(sigma_log_init))
    raise ValueError(f"Unknown error_model: {error_model!r}. Choose from {SUPPORTED_ERROR_MODELS}")


def free_coefficients(noise: NoiseSpec) -> tuple[str, ...]:
    """Names of noise coefficients optimized under this spec."""
    match noise:
        case GaussianConstant() | Lognormal():
            return ()
        case GaussianLinear():
            return ("var_a", "var_b")
        case GaussianQuadratic():
            return ("var_a", "var_b", "var_c")
        case CompoundAddMult():
            return ("sigma_log",)
    raise TypeError(f"Unknown NoiseSpec variant: {noise!r}")


def error_model_name(noise: NoiseSpec) -> str:
    """Legacy display family for fit result payloads."""
    match noise:
        case GaussianConstant() | GaussianLinear() | GaussianQuadratic():
            return "gaussian"
        case Lognormal():
            return "lognormal"
        case CompoundAddMult():
            return "add_mult"
    raise TypeError(f"Unknown NoiseSpec variant: {noise!r}")


def variance_model_name(noise: NoiseSpec) -> str:
    """Legacy Gaussian variance display name for fit result payloads."""
    match noise:
        case GaussianConstant() | Lognormal() | CompoundAddMult():
            return "constant"
        case GaussianLinear():
            return "linear"
        case GaussianQuadratic():
            return "quadratic"
    raise TypeError(f"Unknown NoiseSpec variant: {noise!r}")

# Floor for σ² inside the likelihood — a last-resort guard so log(σ²) stays
# finite. Heteroscedastic Gaussian models keep variance positive by construction
# (a > 0, b ≥ 0, c ≥ 0, d ≥ 0); this floor must not be the mechanism that
# enforces that.
_VARIANCE_FLOOR = float(np.finfo(float).eps)

# Unset (``None``) initials on GaussianLinear / GaussianQuadratic mean
# "data-derived default" (a = 1e-3·s², b = 0, c = 0). An explicit value,
# including the historical class default 1e-3, is kept after domain clamping.
_DEFAULT_VAR_A_INIT = 1e-3
_DEFAULT_VAR_B_INIT = 0.0
_DEFAULT_VAR_C_INIT = 0.0
_VAR_A_DOMAIN_FLOOR = 1e-12
_VAR_BC_DOMAIN_FLOOR = 0.0
_VAR_DOMAIN_RULE = {
    "var_a": "a > 0",
    "var_b": "b >= 0",
    "var_c": "c >= 0",
}

# Number of Gauss–Hermite nodes for the compound (add_mult) likelihood.
# 24 is overkill for σ_log ≲ 0.5 (relative quadrature error ~1e-12) and still
# cheap (one np.einsum per likelihood eval).
_COMPOUND_QUAD_NODES = 24
_COMPOUND_NODES, _COMPOUND_WEIGHTS = np.polynomial.hermite.hermgauss(_COMPOUND_QUAD_NODES)
_COMPOUND_LOG_WEIGHTS = np.log(_COMPOUND_WEIGHTS)


def variance_at(
    mu: np.ndarray | float,
    a: float,
    b: float = 0.0,
    c: float = 0.0,
    anchor: float = 0.0,
) -> np.ndarray:
    """σ² = a + b·d + c·d² with d = μ − anchor, floored to keep log(σ²) finite.

    Default ``anchor=0`` recovers the historical polynomial ``a + b·μ + c·μ²``.
    Fitters pass the current lower asymptote so ``d ≥ 0`` for Hill predictions.

    Independent of error model; the caller is responsible for passing μ in
    the right space (original for gaussian, log for lognormal — though only
    gaussian is supported with non-trivial b/c at the moment).
    """
    mu = np.asarray(mu, dtype=float)
    d = mu - float(anchor)
    return np.maximum(a + b * d + c * d * d, _VARIANCE_FLOOR)


def scaled_variance_coefficient_defaults(
    s: float,
) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    """Response-scaled default initials and bounds for ``var_a`` / ``var_b`` / ``var_c``.

    ``s`` is the robust response range (same dynamic range the asymptote
    derivation uses). Under ``y → s·y`` the coefficients transform as
    ``a → s²a``, ``b → s·b``, ``c → c``.
    """
    s = abs(float(s))
    s2 = s * s
    initials = {
        "var_a": _DEFAULT_VAR_A_INIT * s2,
        "var_b": _DEFAULT_VAR_B_INIT,
        "var_c": _DEFAULT_VAR_C_INIT,
    }
    bounds = {
        "var_a": (_VAR_A_DOMAIN_FLOOR * s2, 10.0 * s2),
        "var_b": (_VAR_BC_DOMAIN_FLOOR, 10.0 * s),
        "var_c": (_VAR_BC_DOMAIN_FLOOR, 10.0),
    }
    return initials, bounds


def variance_coefficient_floor(name: str, response_scale: float | None = None) -> float:
    """Domain floor for ``var_a`` / ``var_b`` / ``var_c``.

    ``var_a`` is ``1e-12``, or ``1e-12·s²`` when a response scale is given, so
    explicit bounds stay consistent with the data-derived lower bound.
    ``var_b`` / ``var_c`` floor at 0.
    """
    if name == "var_a":
        if response_scale is not None and np.isfinite(response_scale) and response_scale > 0:
            return _VAR_A_DOMAIN_FLOOR * float(response_scale) ** 2
        return _VAR_A_DOMAIN_FLOOR
    if name in ("var_b", "var_c"):
        return _VAR_BC_DOMAIN_FLOOR
    raise ValueError(f"Unknown variance coefficient {name!r}")


def apply_variance_param_domain(
    name: str,
    lo: float,
    hi: float,
    init: float | None = None,
    *,
    response_scale: float | None = None,
) -> tuple[float, float, float | None]:
    """Clamp a ``var_*`` bound pair (and optional initial) onto the domain.

    Raises ``ValueError`` naming the parameter, the given bounds, and the
    domain rule when the clamped interval is empty (``lo > hi``).
    """
    if name not in _VAR_DOMAIN_RULE:
        raise ValueError(f"Unknown variance coefficient {name!r}")
    floor = variance_coefficient_floor(name, response_scale)
    given_lo, given_hi = float(lo), float(hi)
    clamped_lo = max(given_lo, floor)
    clamped_hi = given_hi
    if clamped_lo > clamped_hi:
        raise ValueError(
            f"{name} bounds ({given_lo}, {given_hi}) are empty after applying "
            f"the domain {_VAR_DOMAIN_RULE[name]} "
            f"(lower bound clamped to {floor:g})"
        )
    clamped_init = None if init is None else max(float(init), floor)
    return clamped_lo, clamped_hi, clamped_init


def historical_variance_init(name: str) -> float:
    """Concrete fallback when no data scale exists (``FitConfig`` without data)."""
    if name == "var_a":
        return _DEFAULT_VAR_A_INIT
    if name == "var_b":
        return _DEFAULT_VAR_B_INIT
    if name == "var_c":
        return _DEFAULT_VAR_C_INIT
    raise ValueError(f"Unknown variance coefficient {name!r}")


def variance_initials_for_noise(noise: NoiseSpec, s: float) -> dict[str, float]:
    """Scaled default initials, overlaying any user-given ``a_init`` / ``b_init`` / ``c_init``.

    ``None`` initials are replaced by the response-scaled defaults. Any
    explicit value is kept, then clamped to the coefficient domain
    (``a > 0``, ``b ≥ 0``, ``c ≥ 0``).
    """
    scaled, _ = scaled_variance_coefficient_defaults(s)
    if not isinstance(noise, (GaussianLinear, GaussianQuadratic)):
        return {}
    a = scaled["var_a"] if noise.a_init is None else float(noise.a_init)
    b = scaled["var_b"] if noise.b_init is None else float(noise.b_init)
    c_raw = getattr(noise, "c_init", None)
    c = scaled["var_c"] if c_raw is None else float(c_raw)
    floor_a = variance_coefficient_floor("var_a", s)
    out = {
        "var_a": max(a, floor_a),
        "var_b": max(b, _VAR_BC_DOMAIN_FLOOR),
    }
    if isinstance(noise, GaussianQuadratic):
        out["var_c"] = max(c, _VAR_BC_DOMAIN_FLOOR)
    return out


def noise_spec_with_variance_initials(
    noise: NoiseSpec,
    initials: dict[str, float],
) -> NoiseSpec:
    """Return a copy of ``noise`` with the given ``var_*`` initials."""
    if isinstance(noise, GaussianLinear):
        return GaussianLinear(
            a_init=float(initials["var_a"]),
            b_init=float(initials["var_b"]),
        )
    if isinstance(noise, GaussianQuadratic):
        return GaussianQuadratic(
            a_init=float(initials["var_a"]),
            b_init=float(initials["var_b"]),
            c_init=float(initials.get("var_c", _DEFAULT_VAR_C_INIT)),
        )
    return noise


def clamp_noise_spec_initials(
    noise: NoiseSpec,
    response_scale: float | None = None,
) -> NoiseSpec:
    """Clamp heteroscedastic initials onto ``a > 0``, ``b ≥ 0``, ``c ≥ 0``.

    ``None`` initials stay ``None`` (data-derived later). Explicit values
    are floored; ``var_a`` uses ``1e-12`` or ``1e-12·s²`` when ``s`` is given.
    """
    floor_a = variance_coefficient_floor("var_a", response_scale)
    if isinstance(noise, GaussianLinear):
        a, b = noise.a_init, noise.b_init
        return GaussianLinear(
            a_init=None if a is None else max(float(a), floor_a),
            b_init=None if b is None else max(float(b), _VAR_BC_DOMAIN_FLOOR),
        )
    if isinstance(noise, GaussianQuadratic):
        a, b, c = noise.a_init, noise.b_init, noise.c_init
        return GaussianQuadratic(
            a_init=None if a is None else max(float(a), floor_a),
            b_init=None if b is None else max(float(b), _VAR_BC_DOMAIN_FLOOR),
            c_init=None if c is None else max(float(c), _VAR_BC_DOMAIN_FLOOR),
        )
    return noise


def is_heteroscedastic_gaussian(noise: NoiseSpec) -> bool:
    return isinstance(noise, (GaussianLinear, GaussianQuadratic))


def _variance_params_from_spec(noise: GaussianLinear | GaussianQuadratic) -> tuple[float, float, float]:
    match noise:
        case GaussianLinear(a_init=a, b_init=b):
            return (
                _DEFAULT_VAR_A_INIT if a is None else float(a),
                _DEFAULT_VAR_B_INIT if b is None else float(b),
                0.0,
            )
        case GaussianQuadratic(a_init=a, b_init=b, c_init=c):
            return (
                _DEFAULT_VAR_A_INIT if a is None else float(a),
                _DEFAULT_VAR_B_INIT if b is None else float(b),
                _DEFAULT_VAR_C_INIT if c is None else float(c),
            )


def _coerce_noise_arg(
    noise: NoiseSpec | dict | str | None,
    *,
    error_model: str | None = None,
    variance_params: tuple[float, float, float] | None = None,
) -> NoiseSpec:
    if noise is not None:
        spec = from_dict(noise)
    elif error_model is not None:
        spec = legacy_to_noise_spec(error_model)
    else:
        spec = GaussianConstant()

    if variance_params is None:
        return spec
    if isinstance(spec, GaussianConstant):
        a, b, c = variance_params
        if c != 0.0:
            return GaussianQuadratic(a_init=float(a), b_init=float(b), c_init=float(c))
        return GaussianLinear(a_init=float(a), b_init=float(b))
    return spec


def _transform(y: np.ndarray, y_pred: np.ndarray, noise: NoiseSpec):
    """Apply the error-model transform to observations and predictions.

    For ``Lognormal``, non-positive values become ``NaN`` rather than being
    clamped — the downstream ``np.isfinite(residuals)`` filter then drops
    them from the likelihood and Jacobian-correction sums, so the result is
    AIC-comparable to gaussian instead of being inflated by spurious
    ``log(1e-12)`` terms. A ``UserWarning`` surfaces the dropped count once
    per call so the user can decide whether to switch noise model.

    Not used for ``add_mult`` — the compound likelihood operates on the
    original scale and dispatches separately in ``log_prob``.
    """
    if isinstance(noise, Lognormal):
        # Only warn on non-positive *observations* — that's a user-actionable
        # data condition. Non-positive *predictions* happen transiently when
        # the optimiser explores parameter regions where the model dips below
        # zero; warning every iteration would swamp the user.
        n_bad_y = int(np.sum(y <= 0))
        if n_bad_y:
            warnings.warn(
                f"Lognormal error model: dropping {n_bad_y} non-positive "
                "observation(s) from the likelihood. Switch to a Gaussian "
                "model if zeros / negatives are expected.",
                UserWarning,
                stacklevel=3,
            )
        with np.errstate(invalid="ignore", divide="ignore"):
            y = np.where(y > 0, np.log(y), np.nan)
            y_pred = np.where(y_pred > 0, np.log(y_pred), np.nan)
    elif isinstance(noise, CompoundAddMult):
        raise ValueError(
            "_transform: add_mult is a compound model with no scalar transform; "
            "use log_prob's add_mult dispatch instead."
        )
    elif not isinstance(noise, (GaussianConstant, GaussianLinear, GaussianQuadratic)):
        raise ValueError(f"Unknown noise spec: {noise!r}. Choose from {SUPPORTED_NOISE_KINDS}")
    return y, y_pred


def _compound_log_prob(
    y: np.ndarray,
    mu: np.ndarray,
    sigma_add: float,
    sigma_log: float,
) -> float:
    """Marginal log-likelihood of the additive+multiplicative compound model.

        y = μ · exp(η) + ε,   η ~ N(0, σ_log²),   ε ~ N(0, σ_add²)

    The integral over η has no closed form. Evaluated by Gauss–Hermite
    quadrature with the substitution η = √2·σ_log·t::

        p(yⱼ | μⱼ) = (1/√π) · Σᵢ wᵢ · N(yⱼ | μⱼ·exp(√2·σ_log·xᵢ), σ_add²)

    The per-observation log probability is computed via log-sum-exp over the
    quadrature nodes for numerical stability.

    Parameters
    ----------
    y, mu :       Observations and predictions (original response scale).
                  Must be the same length and contain only finite values.
    sigma_add :   Additive Gaussian noise scale (response units, ≥ 0).
    sigma_log :   Multiplicative log-space noise scale (≥ 0).

    Returns
    -------
    float — total log-likelihood Σⱼ log p(yⱼ | μⱼ).
    """
    sigma_add2 = max(float(sigma_add) ** 2, _VARIANCE_FLOOR)
    sigma_log_clip = max(float(sigma_log), 0.0)

    y = np.asarray(y, dtype=float).reshape(-1, 1)        # (n, 1)
    mu = np.asarray(mu, dtype=float).reshape(-1, 1)      # (n, 1)
    nodes = _COMPOUND_NODES.reshape(1, -1)               # (1, k)
    log_w = _COMPOUND_LOG_WEIGHTS.reshape(1, -1)         # (1, k)

    # μ·exp(√2·σ_log·xᵢ) — the conditional mean given η at quadrature node i.
    cond_mean = mu * np.exp(np.sqrt(2.0) * sigma_log_clip * nodes)
    delta = y - cond_mean                                # (n, k)

    # log of integrand at each (j, i): log w_i - 0.5·(log(2π σ_add²) + δ²/σ_add²)
    log_norm = -0.5 * (np.log(2.0 * np.pi * sigma_add2) + (delta * delta) / sigma_add2)
    log_terms = log_w + log_norm                          # (n, k)

    # logsumexp over the node axis, then -0.5·log(π) prefactor per observation.
    m = np.max(log_terms, axis=1, keepdims=True)
    lse = (m + np.log(np.sum(np.exp(log_terms - m), axis=1, keepdims=True))).ravel()
    return float(np.sum(lse) - 0.5 * np.log(np.pi) * y.shape[0])


def fit_noise_scale(
    y: np.ndarray,
    y_pred: np.ndarray,
    noise: NoiseSpec | dict | str | None = None,
    mask: np.ndarray | None = None,
    *,
    error_model: str | None = None,
) -> float | None:
    """MLE noise scale σ̂ in the model's natural likelihood space.

    Returns the same σ̂ that ``log_prob`` profiles out internally for the
    *constant* variance model. Units depend on the error family:

    - ``gaussian``  → σ̂ in the response units of ``y``
    - ``lognormal`` → σ̂ in log-response units (constant CV approx σ̂)

    Returns ``None`` when no finite residuals remain after masking.

    For non-constant variance models σ becomes a function of ``y_pred``;
    use ``variance_at`` together with the fitted ``(a, b, c)`` instead.
    """
    spec = _coerce_noise_arg(noise, error_model=error_model)
    if isinstance(spec, CompoundAddMult):
        raise ValueError(
            "fit_noise_scale: compound_add_mult has two scale parameters; "
            "use fit_noise_scale_compound() instead."
        )
    y = np.asarray(y, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    mask = np.asarray(mask, dtype=bool)

    y_t, y_pred_t = _transform(y, y_pred, spec)
    residuals = (y_t - y_pred_t)[mask]
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size == 0:
        return None
    return float(np.sqrt(np.mean(residuals ** 2)))


def fit_noise_scale_compound(
    y: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[float, float] | None:
    """Crude seed for the compound (add_mult) noise scales (σ_add, σ_log).

    Used to initialise the optimiser; the joint MLE is found by minimising
    ``-log_prob`` with both parameters free. Strategy:

    - Split observations by predicted magnitude. Low-μ residuals are dominated
      by σ_add; high-μ relative residuals are dominated by σ_log.
    - Use the low half (μ ≤ median) for σ_add: σ_add ≈ rms(yⱼ - μⱼ).
    - Use the high half (μ > median) for σ_log: σ_log ≈ rms((yⱼ - μⱼ) / μⱼ).

    Returns ``None`` when there isn't enough finite data on either side.
    Returned scales are floored at ``1e-6`` to keep the optimiser inside its
    bounds.
    """
    y = np.asarray(y, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    mask = np.asarray(mask, dtype=bool)

    valid = mask & np.isfinite(y) & np.isfinite(y_pred)
    if not np.any(valid):
        return None
    yv = y[valid]
    mv = y_pred[valid]
    if yv.size < 2:
        return None

    median_mu = float(np.median(mv))
    low = mv <= median_mu
    high = ~low
    res = yv - mv

    if np.any(low):
        sigma_add = float(np.sqrt(np.mean(res[low] ** 2)))
    else:
        sigma_add = float(np.sqrt(np.mean(res ** 2)))

    if np.any(high) and np.all(np.abs(mv[high]) > 1e-12):
        sigma_log = float(np.sqrt(np.mean((res[high] / mv[high]) ** 2)))
    else:
        # Fall back to overall CV-like estimate
        denom = np.where(np.abs(mv) > 1e-12, mv, np.nan)
        ratio = res / denom
        ratio = ratio[np.isfinite(ratio)]
        sigma_log = float(np.sqrt(np.mean(ratio ** 2))) if ratio.size else 0.1

    return max(sigma_add, 1e-6), max(sigma_log, 1e-6)


def log_prob(
    y: np.ndarray,
    y_pred: np.ndarray,
    noise: NoiseSpec | dict | str | None = None,
    *,
    mask: np.ndarray | None = None,
    y_err: np.ndarray | None = None,
    variance_params: tuple[float, float, float] | None = None,
    variance_anchor: float | None = None,
    compound_params: tuple[float, float] | None = None,
    error_model: str | None = None,
) -> float:
    """
    Compute log-likelihood of observations under the chosen error model.

    Parameters
    ----------
    y:               Observed responses.
    y_pred:          Model predictions at the same concentrations.
    error_model:     ``ErrorModel`` or ``"gaussian"`` / ``"lognormal"``.
    mask:            Boolean array; True = include point. None means include all.
    y_err:           Per-point standard errors. When provided uses weighted
                     likelihood; otherwise profiles out variance.
    variance_params: ``(a, b, c)`` for the heteroscedastic variance polynomial
                     ``σ² = a + b·d + c·d²``, ``d = μ − m``. When provided, σ²
                     is computed per point from ``y_pred`` and the weighted
                     likelihood is used. Mutually exclusive with ``y_err``.
                     Currently only supported under the ``gaussian`` error model.
    variance_anchor: Lower-asymptote anchor ``m``. ``None`` (default) is
                     treated as ``0``, recovering ``σ²(μ) = a + b·μ + c·μ²``.
                     Fitters pass ``min(effect_0, effect_inf)`` (or
                     ``min(top, bottom)`` for a joint fit).
    compound_params: ``(sigma_add, sigma_log)`` for the ``add_mult`` family.
                     Required when ``error_model='add_mult'``; rejected for
                     other families. Mutually exclusive with ``y_err`` and
                     ``variance_params``.

    Returns
    -------
    float — log p(y | y_pred, error_model, ...)
    """
    spec = _coerce_noise_arg(noise, error_model=error_model, variance_params=variance_params)
    y = np.asarray(y, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # ── Compound (add_mult) — separate code path; no transform on y/y_pred. ──
    if isinstance(spec, CompoundAddMult):
        if compound_params is None:
            raise ValueError(
                "log_prob: noise='compound_add_mult' requires compound_params=(sigma_add, sigma_log)."
            )
        if y_err is not None or variance_params is not None:
            raise ValueError(
                "log_prob: compound_params is mutually exclusive with y_err and variance_params."
            )
        if mask is None:
            mask = np.ones(len(y), dtype=bool)
        mask = np.asarray(mask, dtype=bool)
        y_m = y[mask]
        mu_m = y_pred[mask]
        finite = np.isfinite(y_m) & np.isfinite(mu_m)
        if not np.any(finite):
            return -np.inf
        sigma_add, sigma_log = compound_params
        return _compound_log_prob(y_m[finite], mu_m[finite], sigma_add, sigma_log)

    if compound_params is not None:
        raise ValueError(
            "log_prob: compound_params is only supported under noise='compound_add_mult'."
        )
    if y_err is not None and variance_params is not None:
        raise ValueError(
            "log_prob: y_err and variance_params are mutually exclusive — "
            "pick one source of σ²."
        )
    if variance_params is not None and not isinstance(spec, (GaussianLinear, GaussianQuadratic)):
        raise ValueError(
            "log_prob: variance_params is currently only supported under "
            "gaussian heteroscedastic noise (lognormal carries variance in log-space)."
        )
    if isinstance(spec, (GaussianLinear, GaussianQuadratic)) and variance_params is None:
        variance_params = _variance_params_from_spec(spec)

    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    mask = np.asarray(mask, dtype=bool)

    y_t, y_pred_t = _transform(y, y_pred, spec)

    y_t_m = y_t[mask]
    y_pred_t_m = y_pred_t[mask]
    residuals = y_t_m - y_pred_t_m

    finite = np.isfinite(residuals)
    residuals = residuals[finite]
    n = len(residuals)
    if n == 0:
        return -np.inf

    # Jacobian correction for lognormal: the change-of-variables y → log(y)
    # contributes -Σ log(yᵢ) to the log-likelihood. Without this term the
    # lognormal and gaussian likelihoods are not on the same scale and AIC / BIC
    # comparisons between the two models are invalid. ``finite`` already
    # excludes any non-positive y (those produce NaN residuals via _transform),
    # so ``y_valid`` is guaranteed strictly positive — no clamping needed.
    jacobian_correction = 0.0
    if isinstance(spec, Lognormal):
        y_valid = np.asarray(y, dtype=float)[mask][finite]
        jacobian_correction = -np.sum(np.log(y_valid))

    if variance_params is not None:
        a, b, c = variance_params
        # σ² is keyed off the *prediction* in the original space — that's
        # the meaningful "expected response" for which heteroscedasticity is
        # specified. We index by the original (un-transformed) y_pred.
        mu = np.asarray(y_pred, dtype=float)[mask][finite]
        anchor = 0.0 if variance_anchor is None else float(variance_anchor)
        sigma2 = variance_at(mu, a, b, c, anchor=anchor)
        return float(
            -0.5 * np.sum(np.log(2 * np.pi * sigma2) + residuals ** 2 / sigma2)
            + jacobian_correction
        )

    if y_err is not None:
        # Weighted likelihood with known per-point errors
        y_err_t = np.asarray(y_err, dtype=float)
        if isinstance(spec, Lognormal):
            # Error propagation: σ_log ≈ σ / y (delta method)
            y_err_t = y_err_t / np.maximum(y, 1e-12)
        y_err_t = y_err_t[mask][finite]
        sigma2 = np.maximum(y_err_t ** 2, _VARIANCE_FLOOR)
        return float(-0.5 * np.sum(np.log(2 * np.pi * sigma2) + residuals ** 2 / sigma2) + jacobian_correction)

    # Profiled variance (MLE of σ integrated out — constant model only)
    sigma2 = np.mean(residuals ** 2)
    sigma2 = np.maximum(sigma2, _VARIANCE_FLOOR)
    return float(-0.5 * n * (1.0 + np.log(2 * np.pi * sigma2)) + jacobian_correction)
