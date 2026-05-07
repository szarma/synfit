import numpy as np
import pytest

from synfit.data import FitConfig
from synfit.noise import (
    CompoundAddMult,
    GaussianConstant,
    GaussianLinear,
    GaussianQuadratic,
    Lognormal,
    fit_noise_scale,
    from_dict,
    legacy_to_noise_spec,
    log_prob,
    to_dict,
)


def _manual_profiled_gaussian(y, mu):
    residuals = y - mu
    sigma2 = max(float(np.mean(residuals ** 2)), np.finfo(float).eps)
    return float(-0.5 * len(y) * (1.0 + np.log(2 * np.pi * sigma2)))


def _manual_known_variance(y, mu, a, b=0.0, c=0.0):
    residuals = y - mu
    sigma2 = np.maximum(a + b * mu + c * mu * mu, np.finfo(float).eps)
    return float(-0.5 * np.sum(np.log(2 * np.pi * sigma2) + residuals ** 2 / sigma2))


def test_noise_spec_serde_round_trip_all_variants():
    specs = [
        GaussianConstant(),
        GaussianLinear(a_init=0.2, b_init=0.3),
        GaussianQuadratic(a_init=0.2, b_init=0.3, c_init=0.4),
        Lognormal(),
        CompoundAddMult(sigma_log_init=0.25),
    ]
    for spec in specs:
        assert from_dict(to_dict(spec)) == spec


def test_legacy_to_noise_spec_coerces_old_shape():
    assert legacy_to_noise_spec("gaussian", "constant") == GaussianConstant()
    assert legacy_to_noise_spec("gaussian", "linear", var_a=0.1, var_b=0.2) == GaussianLinear(0.1, 0.2)
    assert legacy_to_noise_spec("gaussian", "quadratic", var_a=0.1, var_b=0.2, var_c=0.3) == GaussianQuadratic(0.1, 0.2, 0.3)
    assert legacy_to_noise_spec("lognormal", "constant") == Lognormal()
    assert legacy_to_noise_spec("add_mult", "constant", sigma_log_init=0.2) == CompoundAddMult(0.2)


def test_fit_config_extends_noise_parameters():
    cfg = FitConfig(noise={"kind": "gaussian_quadratic", "a_init": 0.1, "b_init": 0.2, "c_init": 0.3})
    assert cfg.noise == GaussianQuadratic(0.1, 0.2, 0.3)
    assert cfg.fitting_parameters[-3:] == ["var_a", "var_b", "var_c"]


def test_log_prob_variants_match_manual_regression_values():
    y = np.array([0.2, 0.45, 0.9, 1.1])
    mu = np.array([0.22, 0.4, 0.85, 1.0])

    assert log_prob(y, mu, GaussianConstant()) == pytest.approx(_manual_profiled_gaussian(y, mu))
    assert log_prob(y, mu, GaussianLinear(0.01, 0.03)) == pytest.approx(
        _manual_known_variance(y, mu, 0.01, 0.03, 0.0)
    )
    assert log_prob(y, mu, GaussianQuadratic(0.01, 0.0, 0.02)) == pytest.approx(
        _manual_known_variance(y, mu, 0.01, 0.0, 0.02)
    )

    log_y = np.log(y)
    log_mu = np.log(mu)
    expected_lognormal = _manual_profiled_gaussian(log_y, log_mu) - float(np.sum(np.log(y)))
    assert log_prob(y, mu, Lognormal()) == pytest.approx(expected_lognormal)


def test_fit_noise_scale_dispatches_by_noise_spec():
    y = np.array([0.2, 0.5, 1.0])
    mu = np.array([0.21, 0.48, 0.9])
    assert fit_noise_scale(y, mu, GaussianConstant()) == pytest.approx(np.sqrt(np.mean((y - mu) ** 2)))
    assert fit_noise_scale(y, mu, Lognormal()) == pytest.approx(
        np.sqrt(np.mean((np.log(y) - np.log(mu)) ** 2))
    )
    with pytest.raises(ValueError, match="compound_add_mult"):
        fit_noise_scale(y, mu, CompoundAddMult())


def test_legacy_log_prob_arguments_still_work():
    y = np.array([0.2, 0.5, 1.0])
    assert log_prob(y, y, error_model="lognormal") == pytest.approx(log_prob(y, y, Lognormal()))
    assert log_prob(y, y, error_model="gaussian") == pytest.approx(log_prob(y, y, GaussianConstant()))
