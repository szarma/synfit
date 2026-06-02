"""Single source of truth for fitting-parameter role classification.

The same Hill / variance parameters appear across single-drug, matrix, and
joint-marginal fits under slightly different names (``effect_inf`` vs ``top``,
``log_c50`` vs ``log_c50_hor``), yet each plays a fixed role in the optimiser.
Centralising "what kind of parameter is this" here keeps the scale
preconditioning (``fitting.parameter_scale``), the curve-kwarg unpacking
(``FitBase._x_to_kwargs``, ``MatrixFit._unpack_x``), and the variance handling
from each re-deriving the classification — and silently drifting apart.
"""

# Magnitude-bearing asymptotes: their true value tracks the response scale, so
# they alone are eligible for bound-magnitude preconditioning (see
# ``fitting.parameter_scale``). Single-drug / matrix fits name them
# ``effect_0`` / ``effect_inf``; the joint-marginal fit shares them as
# ``top`` / ``bottom``.
ASYMPTOTE_PARAM_NAMES: tuple[str, ...] = ("effect_0", "effect_inf", "top", "bottom")

# Variance-polynomial coefficients (σ²(μ) = a + b·μ + c·μ²) that may appear in
# the optimiser parameter list; stripped out before building hill_curve kwargs.
VARIANCE_PARAM_NAMES: tuple[str, ...] = ("var_a", "var_b", "var_c")


def is_log_param(name: str) -> bool:
    """A log-domain location parameter (``log_c50``, ``log_c50_hor``, …).

    Optimised in log10 space; un-logged to linear (drop the ``log_`` prefix)
    before it goes into ``hill_curve``.
    """
    return name.startswith("log_")


def is_variance_param(name: str) -> bool:
    """A variance-polynomial coefficient (``var_a`` / ``var_b`` / ``var_c``)."""
    return name in VARIANCE_PARAM_NAMES


def is_asymptote_param(name: str) -> bool:
    """A magnitude-bearing asymptote (``effect_0`` / ``effect_inf`` / ``top`` / ``bottom``)."""
    return name in ASYMPTOTE_PARAM_NAMES
