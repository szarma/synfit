"""Likelihood behaviour under a response unit change (y → s·y)."""

import warnings

import numpy as np
import pytest

from synfit.noise import (
    CompoundAddMult,
    GaussianConstant,
    GaussianLinear,
    GaussianQuadratic,
    Lognormal,
    _VARIANCE_FLOOR,
    log_prob,
)

from tests.response_scaling_helpers import n_likelihood_admitted


def _base_vectors(n: int = 24, seed: int = 0):
    rng = np.random.default_rng(seed)
    y_pred = np.linspace(0.5, 8.0, n)
    y = y_pred + rng.normal(0, 0.08, size=n)
    y_err = 0.05 + 0.02 * np.sin(np.linspace(0, 3, n))
    return y, y_pred, y_err


def _hand_weighted_gaussian(y, y_pred, y_err, mask):
    y = y[mask]
    y_pred = y_pred[mask]
    y_err = y_err[mask]
    resid = y - y_pred
    sigma2 = np.maximum(y_err ** 2, _VARIANCE_FLOOR)
    return float(-0.5 * np.sum(np.log(2 * np.pi * sigma2) + resid ** 2 / sigma2))


def _hand_weighted_lognormal(y, y_pred, y_err, mask):
    y = y[mask]
    y_pred = y_pred[mask]
    y_err = y_err[mask]
    y_t = np.log(y)
    y_pred_t = np.log(y_pred)
    resid = y_t - y_pred_t
    sigma2 = np.maximum((y_err / np.maximum(y, 1e-12)) ** 2, _VARIANCE_FLOOR)
    jacobian = -np.sum(np.log(y))
    return float(-0.5 * np.sum(np.log(2 * np.pi * sigma2) + resid ** 2 / sigma2) + jacobian)


@pytest.mark.parametrize("s", [1e-6, 1e3, 1e6])
@pytest.mark.parametrize(
    "noise,extra",
    [
        (GaussianConstant(), {}),
        (GaussianLinear(a_init=0.01, b_init=0.02), {"variance_params": (0.01, 0.02, 0.0), "variance_anchor": 0.5}),
        (GaussianQuadratic(a_init=0.01, b_init=0.01, c_init=0.001), {
            "variance_params": (0.01, 0.01, 0.001),
            "variance_anchor": 0.5,
        }),
        (Lognormal(), {}),
    ],
)
def test_likelihood_change_of_units(noise, extra, s):
    """Δ log-likelihood equals −n_admitted·log s when all observations are rescaled."""
    y, y_pred, y_err = _base_vectors()
    mask = np.ones_like(y, dtype=bool)
    mask[3] = False
    y[7] = np.nan

    extra_s = dict(extra)
    if extra_s.get("variance_params") is not None:
        a, b, c = extra_s["variance_params"]
        extra_s["variance_params"] = (a * s * s, b * s, c)
        if "variance_anchor" in extra_s:
            extra_s["variance_anchor"] = float(extra_s["variance_anchor"]) * s

    ll0 = log_prob(y, y_pred, noise=noise, mask=mask, y_err=None, **extra)
    ll_s = log_prob(y * s, y_pred * s, noise=noise, mask=mask, y_err=None, **extra_s)

    n = n_likelihood_admitted(y, y_pred, noise, mask=mask)
    expected = -n * np.log(s)
    if abs(expected) > 1:
        np.testing.assert_allclose(ll_s - ll0, expected, rtol=1e-9)
    else:
        np.testing.assert_allclose(ll_s - ll0, expected, atol=1e-9)

    if extra.get("variance_params") is None:
        ll_w0 = log_prob(y, y_pred, noise=noise, mask=mask, y_err=y_err)
        ll_ws = log_prob(y * s, y_pred * s, noise=noise, mask=mask, y_err=y_err * s)
        n_w = n_likelihood_admitted(y, y_pred, noise, mask=mask)
        np.testing.assert_allclose(ll_ws - ll_w0, -n_w * np.log(s), rtol=1e-9, atol=1e-9)


def test_likelihood_change_of_units_lognormal_drops_nonpositive():
    """Non-positive observations are excluded consistently under rescaling."""
    y = np.array([0.0, -0.1, 0.4, 0.8, 1.2])
    y_pred = np.array([0.2, 0.3, 0.45, 0.85, 1.1])
    s = 1e3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ll0 = log_prob(y, y_pred, noise=Lognormal())
        ll_s = log_prob(y * s, y_pred * s, noise=Lognormal())
    n = n_likelihood_admitted(y, y_pred, Lognormal())
    np.testing.assert_allclose(ll_s - ll0, -n * np.log(s), rtol=1e-9, atol=1e-9)


def test_likelihood_change_of_units_compound():
    """Compound noise: scale y, μ and σ_add together; σ_log unchanged."""
    y, y_pred, _ = _base_vectors(n=20, seed=2)
    cp = (0.08, 0.12)
    for s in (1e-6, 1e3, 1e6):
        ll0 = log_prob(y, y_pred, noise=CompoundAddMult(), compound_params=cp)
        ll_s = log_prob(
            y * s,
            y_pred * s,
            noise=CompoundAddMult(),
            compound_params=(cp[0] * s, cp[1]),
        )
        n = n_likelihood_admitted(y, y_pred, CompoundAddMult())
        np.testing.assert_allclose(ll_s - ll0, -n * np.log(s), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("s", [1e-6, 1e3])
@pytest.mark.parametrize("noise", [GaussianConstant(), Lognormal()])
def test_weighted_likelihood_uses_user_errors(noise, s):
    """Weighted likelihood follows supplied y_err and changes when errors change."""
    y, y_pred, y_err = _base_vectors()
    mask = np.ones_like(y, dtype=bool)

    ll = log_prob(y * s, y_pred * s, noise=noise, y_err=y_err * s)
    if isinstance(noise, GaussianConstant):
        hand = _hand_weighted_gaussian(y * s, y_pred * s, y_err * s, mask)
    else:
        hand = _hand_weighted_lognormal(y * s, y_pred * s, y_err * s, mask)
    np.testing.assert_allclose(ll, hand, rtol=1e-12)

    ll_loose = log_prob(y * s, y_pred * s, noise=noise, y_err=y_err * s * 5)
    assert ll_loose < ll
