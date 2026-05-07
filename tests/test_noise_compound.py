"""Tests for the additive+multiplicative (add_mult / Rocke–Lorenzato) compound
error model in synfit.noise.
"""
import numpy as np
import pytest

from synfit.noise import (
    ErrorModel,
    SUPPORTED_ERROR_MODELS,
    _compound_log_prob,
    fit_noise_scale,
    fit_noise_scale_compound,
    log_prob,
)


class TestEnumRegistration:
    def test_add_mult_registered(self):
        assert ErrorModel.ADD_MULT.value == "add_mult"
        assert "add_mult" in SUPPORTED_ERROR_MODELS

    def test_coerce_string(self):
        assert ErrorModel.coerce("add_mult") is ErrorModel.ADD_MULT


class TestCompoundLogProb:
    def test_returns_finite_float(self):
        y = np.linspace(0.1, 1.0, 10)
        mu = y.copy()
        lp = log_prob(y, mu, error_model="add_mult", compound_params=(0.05, 0.1))
        assert isinstance(lp, float)
        assert np.isfinite(lp)

    def test_requires_compound_params(self):
        y = np.linspace(0.1, 1.0, 10)
        with pytest.raises(ValueError, match="compound_params"):
            log_prob(y, y.copy(), error_model="add_mult")

    def test_compound_params_rejected_for_other_models(self):
        y = np.linspace(0.1, 1.0, 10)
        with pytest.raises(ValueError, match="add_mult"):
            log_prob(y, y.copy(), error_model="gaussian", compound_params=(0.05, 0.1))

    def test_mutually_exclusive_with_y_err(self):
        y = np.linspace(0.1, 1.0, 10)
        with pytest.raises(ValueError, match="mutually exclusive"):
            log_prob(
                y, y.copy(),
                error_model="add_mult",
                compound_params=(0.05, 0.1),
                y_err=np.full_like(y, 0.05),
            )

    def test_mutually_exclusive_with_variance_params(self):
        y = np.linspace(0.1, 1.0, 10)
        with pytest.raises(ValueError, match="mutually exclusive"):
            log_prob(
                y, y.copy(),
                error_model="add_mult",
                compound_params=(0.05, 0.1),
                variance_params=(1e-3, 0.0, 0.0),
            )

    def test_better_prediction_has_higher_log_prob(self):
        rng = np.random.default_rng(0)
        y = np.linspace(0.1, 1.0, 30)
        y_good = y + rng.normal(0, 0.02, size=y.shape)
        y_bad = y + rng.normal(0, 0.2, size=y.shape)
        cp = (0.05, 0.1)
        assert (
            log_prob(y, y_good, error_model="add_mult", compound_params=cp)
            > log_prob(y, y_bad, error_model="add_mult", compound_params=cp)
        )

    def test_degenerates_to_gaussian_when_sigma_log_is_tiny(self):
        # σ_log → 0 collapses the compound to plain N(y; μ, σ_add²).
        rng = np.random.default_rng(1)
        sigma_add = 0.1
        mu = np.linspace(0.2, 1.0, 25)
        y = mu + rng.normal(0, sigma_add, size=mu.shape)

        lp_compound = log_prob(
            y, mu, error_model="add_mult", compound_params=(sigma_add, 1e-8),
        )
        # Closed-form Gaussian log-likelihood with known σ_add (per-point σ).
        lp_gauss = float(
            -0.5 * np.sum(
                np.log(2 * np.pi * sigma_add**2) + (y - mu) ** 2 / sigma_add**2
            )
        )
        assert lp_compound == pytest.approx(lp_gauss, rel=1e-5, abs=1e-4)

    def test_quadrature_stable_for_typical_scales(self):
        # σ_log up to 0.5 is the realistic range; the integrator should not blow up.
        y = np.linspace(0.1, 1.5, 40)
        mu = y * 0.95
        for sigma_log in [0.05, 0.2, 0.5]:
            lp = log_prob(
                y, mu, error_model="add_mult", compound_params=(0.05, sigma_log),
            )
            assert np.isfinite(lp)

    def test_mask_applied(self):
        y = np.array([0.5, 0.6, 0.7, 1e9])  # last point would dominate
        mu = np.array([0.5, 0.6, 0.7, 0.8])
        mask = np.array([True, True, True, False])
        lp_masked = log_prob(
            y, mu, error_model="add_mult",
            compound_params=(0.05, 0.1), mask=mask,
        )
        lp_clean = log_prob(
            y[:3], mu[:3], error_model="add_mult", compound_params=(0.05, 0.1),
        )
        assert lp_masked == pytest.approx(lp_clean)

    def test_handles_negative_y(self):
        # Compound model has Gaussian additive component → negative y allowed.
        y = np.array([-0.05, 0.1, 0.5, 1.0])
        mu = np.array([0.0, 0.1, 0.5, 1.0])
        lp = log_prob(y, mu, error_model="add_mult", compound_params=(0.1, 0.1))
        assert np.isfinite(lp)


class TestFitNoiseScaleCompound:
    def test_returns_two_positive_scales(self):
        rng = np.random.default_rng(2)
        mu = np.linspace(0.1, 1.0, 50)
        # σ_add = 0.05, σ_log = 0.15
        eta = rng.normal(0, 0.15, size=mu.shape)
        eps = rng.normal(0, 0.05, size=mu.shape)
        y = mu * np.exp(eta) + eps
        seed = fit_noise_scale_compound(y, mu)
        assert seed is not None
        sigma_add, sigma_log = seed
        assert sigma_add > 0
        assert sigma_log > 0
        # Seed should be in the right ballpark — within ~3× the truth
        assert 0.01 < sigma_add < 0.3
        assert 0.01 < sigma_log < 0.5

    def test_returns_none_for_empty(self):
        y = np.array([])
        mu = np.array([])
        assert fit_noise_scale_compound(y, mu) is None

    def test_scalar_fit_noise_scale_rejects_add_mult(self):
        y = np.linspace(0.1, 1.0, 5)
        with pytest.raises(ValueError, match="add_mult"):
            fit_noise_scale(y, y.copy(), error_model="add_mult")
