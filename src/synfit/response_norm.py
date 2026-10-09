"""Internal response-unit normalisation for dose-response fitting.

Fits optimise in coordinates where the response magnitude is O(1); public
data, ``fitter.config``, and :class:`~synfit.data.FitResult` values stay in
caller units. See ``FitBase._enter_response_normalization``.
"""

from __future__ import annotations

import copy
from typing import Iterable

import numpy as np

from .data import FitBounds, FitConfig
from .noise import (
    GaussianLinear,
    GaussianQuadratic,
    NoiseSpec,
    _coerce_noise_arg,
    _transform,
    is_heteroscedastic_gaussian,
    noise_spec_with_variance_initials,
)
from .param_roles import is_asymptote_param, is_variance_param


def _robust_response_range(y: np.ndarray) -> float:
    """Local copy of the single-drug range rule (avoids import cycles)."""
    from .single import _robust_response_range as _rrr

    return _rrr(y)


def response_scale_from_y(
    y: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Per-fit scale ``s``: robust range of the valid ``y`` (degenerate → 1).

    The raw range, not a rounded one, so ``y → f·y`` maps onto identical
    normalised coordinates and fits are equivariant by construction.
    """
    y = np.asarray(y, dtype=float)
    if mask is not None:
        y = y[np.asarray(mask, dtype=bool)]
    y = y[np.isfinite(y)]
    if y.size == 0:
        return 1.0
    raw = _robust_response_range(y)
    if not np.isfinite(raw) or raw <= 0.0:
        return 1.0
    return float(raw)


def magnitude_scale_factor(param_name: str, s: float) -> float:
    """Multiply a user-unit parameter value by this to get normalised units."""
    if is_asymptote_param(param_name):
        return s
    if is_variance_param(param_name):
        key = param_name[4:]
        if key == "a":
            return s * s
        if key == "b":
            return s
        return 1.0
    return 1.0


def user_from_normalized(param_name: str, norm_value: float, s: float) -> float:
    return float(norm_value) * magnitude_scale_factor(param_name, s)


def normalized_from_user(param_name: str, user_value: float, s: float) -> float:
    fac = magnitude_scale_factor(param_name, s)
    if fac == 0.0:
        return float(user_value)
    return float(user_value) / fac


def scale_bound_pair(
    param_name: str,
    bounds: tuple[float, float],
    s: float,
) -> tuple[float, float]:
    fac = magnitude_scale_factor(param_name, s)
    if fac == 1.0:
        return bounds
    lo, hi = bounds
    return (lo / fac, hi / fac)


def _scale_noise_spec(noise: NoiseSpec, s: float) -> NoiseSpec:
    if s == 1.0 or not is_heteroscedastic_gaussian(noise):
        return noise
    if isinstance(noise, GaussianLinear):
        a = noise.a_init
        b = noise.b_init
        return GaussianLinear(
            a_init=a / (s * s) if a is not None else None,
            b_init=b / s if b is not None else None,
        )
    if isinstance(noise, GaussianQuadratic):
        a, b, c = noise.a_init, noise.b_init, noise.c_init
        return GaussianQuadratic(
            a_init=a / (s * s) if a is not None else None,
            b_init=b / s if b is not None else None,
            c_init=c,
        )
    return noise


def normalized_fit_config(cfg: FitConfig, s: float) -> FitConfig:
    """Private working copy of ``cfg`` in normalised units (never mutates ``cfg``)."""
    work = cfg.copy()
    # Resolve unset initials to the user-unit values ``cfg`` exposes (and that
    # 0.9.0 used), then normalise them; never re-derive defaults here.
    resolved = {}
    if is_heteroscedastic_gaussian(cfg.noise):
        names = ("var_a", "var_b", "var_c") if isinstance(cfg.noise, GaussianQuadratic) else ("var_a", "var_b")
        resolved = {name: float(getattr(cfg, name)) for name in names}
    work.noise = _scale_noise_spec(
        noise_spec_with_variance_initials(cfg.noise, resolved),
        s,
    )
    work.effect_0 = normalized_from_user("effect_0", float(cfg.effect_0), s)
    work.effect_inf = normalized_from_user("effect_inf", float(cfg.effect_inf), s)
    b = work.bounds
    if b is not None:
        nb = copy.copy(b)
        for field in (
            "effect_0", "effect_inf", "var_a", "var_b", "var_c",
        ):
            if hasattr(nb, field):
                setattr(nb, field, scale_bound_pair(field, getattr(nb, field), s))
        work.bounds = nb
    return work


def normalize_parameter_vector(
    names: Iterable[str],
    x_user: np.ndarray,
    s: float,
) -> np.ndarray:
    names = list(names)
    out = np.asarray(x_user, dtype=float).copy()
    for i, name in enumerate(names):
        out[i] = normalized_from_user(name, out[i], s)
    return out


def denormalize_parameter_vector(
    names: Iterable[str],
    x_norm: np.ndarray,
    s: float,
) -> np.ndarray:
    names = list(names)
    out = np.asarray(x_norm, dtype=float).copy()
    for i, name in enumerate(names):
        out[i] = user_from_normalized(name, out[i], s)
    return out


def denormalize_covariance(
    cov_norm: np.ndarray,
    param_names: list[str],
    s: float,
) -> np.ndarray:
    if s == 1.0:
        return np.asarray(cov_norm, dtype=float)
    d = np.array([magnitude_scale_factor(n, s) for n in param_names], dtype=float)
    return cov_norm * np.outer(d, d)


def log_likelihood_user_units(
    log_likelihood_normalized: float,
    n_admitted: int,
    s: float,
) -> float:
    if s == 1.0 or n_admitted == 0:
        return float(log_likelihood_normalized)
    return float(log_likelihood_normalized - n_admitted * np.log(s))


def count_admitted_likelihood(
    y: np.ndarray,
    y_pred: np.ndarray,
    noise: NoiseSpec | str | None,
    *,
    mask: np.ndarray | None = None,
) -> int:
    """Rows that enter ``noise.log_prob`` (mask, finite, lognormal positivity)."""
    from .noise import CompoundAddMult

    spec = _coerce_noise_arg(noise)
    y = np.asarray(y, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if isinstance(spec, CompoundAddMult):
        if mask is None:
            mask = np.ones(len(y), dtype=bool)
        mask = np.asarray(mask, dtype=bool)
        y_m = y[mask]
        mu_m = y_pred[mask]
        return int(np.sum(np.isfinite(y_m) & np.isfinite(mu_m)))
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    y_t, y_pred_t = _transform(y, y_pred, spec)
    residuals = (y_t - y_pred_t)[mask]
    return int(np.sum(np.isfinite(residuals)))
