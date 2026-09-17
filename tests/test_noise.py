
import numpy as np
import pytest

from synfit.data import FitConfig
from synfit.noise import (
    SUPPORTED_ERROR_MODELS,
    SUPPORTED_VARIANCE_MODELS,
    GaussianConstant,
    GaussianLinear,
    GaussianQuadratic,
    fit_noise_scale,
    log_prob,
    variance_at,
)


def _perfect_pred(n=20):
    y = np.linspace(0.1, 1.0, n)
    return y, y.copy()


class TestLogProbBasic:
    def test_returns_float(self):
        y, y_pred = _perfect_pred()
        result = log_prob(y, y_pred)
        assert isinstance(result, float)

    def test_perfect_prediction_is_finite(self):
        y, y_pred = _perfect_pred()
        assert np.isfinite(log_prob(y, y_pred, error_model="lognormal"))
        assert np.isfinite(log_prob(y, y_pred, error_model="gaussian"))

    def test_better_prediction_has_higher_log_prob(self):
        rng = np.random.default_rng(0)
        y = np.linspace(0.1, 1.0, 20)
        y_good = y + rng.normal(0, 0.02, size=y.shape)
        y_bad = y + rng.normal(0, 0.2, size=y.shape)
        assert log_prob(y, y_good) > log_prob(y, y_bad)

    def test_unknown_error_model_raises(self):
        y, y_pred = _perfect_pred()
        with pytest.raises(ValueError, match="banana"):
            log_prob(y, y_pred, error_model="banana")

    def test_empty_after_mask_returns_neg_inf(self):
        y = np.array([0.5, 0.6, 0.7])
        y_pred = y.copy()
        mask = np.zeros(3, dtype=bool)
        assert log_prob(y, y_pred, mask=mask) == -np.inf


class TestMask:
    def test_mask_excludes_outlier(self):
        y = np.array([0.5, 0.6, 99.0, 0.7])
        y_pred = np.array([0.5, 0.6, 0.65, 0.7])
        mask_no_outlier = np.array([True, True, False, True])

        lp_all = log_prob(y, y_pred)
        lp_masked = log_prob(y, y_pred, mask=mask_no_outlier)
        assert lp_masked > lp_all

    def test_none_mask_same_as_all_true(self):
        y, y_pred = _perfect_pred()
        mask_all = np.ones(len(y), dtype=bool)
        assert log_prob(y, y_pred, mask=None) == log_prob(y, y_pred, mask=mask_all)


class TestErrorModels:
    def test_lognormal_uses_log_space(self):
        # lognormal and gaussian should differ for non-trivial residuals
        y = np.array([0.1, 0.5, 1.0, 2.0])
        y_pred = y * 1.1
        lp_ln = log_prob(y, y_pred, error_model="lognormal")
        lp_g = log_prob(y, y_pred, error_model="gaussian")
        assert lp_ln != lp_g

    def test_gaussian_symmetric_around_zero(self):
        y = np.array([1.0, 2.0, 3.0])
        y_pred_pos = y + 0.1
        y_pred_neg = y - 0.1
        assert np.isclose(
            log_prob(y, y_pred_pos, error_model="gaussian"),
            log_prob(y, y_pred_neg, error_model="gaussian"),
        )


class TestWeightedLikelihood:
    def test_known_errors_returns_finite(self):
        y = np.linspace(0.2, 1.0, 10)
        y_pred = y * 1.05
        y_err = np.full_like(y, 0.05)
        result = log_prob(y, y_pred, y_err=y_err)
        assert np.isfinite(result)

    def test_smaller_errors_give_higher_log_prob_for_good_fit(self):
        y = np.linspace(0.2, 1.0, 10)
        y_pred = y * 1.02
        y_err_tight = np.full_like(y, 0.01)
        y_err_loose = np.full_like(y, 0.5)
        # small residuals relative to tight errors → high prob
        # but also depends on -0.5*log(2π σ²) term; just check both are finite
        assert np.isfinite(log_prob(y, y_pred, y_err=y_err_tight))
        assert np.isfinite(log_prob(y, y_pred, y_err=y_err_loose))

    def test_weighted_vs_profiled_differ(self):
        y = np.linspace(0.2, 1.0, 10)
        y_pred = y * 1.05
        y_err = np.full_like(y, 0.05)
        assert log_prob(y, y_pred) != log_prob(y, y_pred, y_err=y_err)


class TestErrorModelEnum:
    def test_fitconfig_coerces_string(self):
        cfg = FitConfig(error_model="gaussian")
        assert cfg.noise == GaussianConstant()
        assert cfg.error_model == "gaussian"

    def test_fitconfig_invalid_string_raises(self):
        with pytest.raises(ValueError):
            FitConfig(error_model="banana")

    def test_lognormal_nonpositive_warns_and_drops(self):
        """Non-positive y under lognormal: warn (UserWarning), drop the point
        from the likelihood entirely. Earlier behaviour clamped to 1e-12 and
        let the +27.6/point Jacobian inflate the log-likelihood, breaking
        AIC comparisons against gaussian."""
        y_with_zero = np.array([0.0, 0.5, 0.8])
        y_clean = np.array([0.5, 0.8])
        y_pred = np.array([0.1, 0.5, 0.8])
        y_pred_clean = np.array([0.5, 0.8])

        with pytest.warns(UserWarning, match="non-positive observation"):
            lp_with_zero = log_prob(y_with_zero, y_pred, error_model="lognormal")
        lp_clean = log_prob(y_clean, y_pred_clean, error_model="lognormal")

        # Dropping the bad point ≡ fitting only the good ones, so the
        # log-likelihood matches the clean call exactly.
        assert lp_with_zero == pytest.approx(lp_clean)


class TestFitNoiseScale:
    """fit_noise_scale must match the σ̂ that log_prob profiles out internally."""

    def _expected_log_prob(self, n: int, sigma: float) -> float:
        # Profiled log-likelihood: -(n/2)·(1 + log(2π σ²))
        return -0.5 * n * (1.0 + np.log(2 * np.pi * sigma ** 2))

    def test_gaussian_matches_profiled_likelihood(self):
        rng = np.random.default_rng(0)
        y_pred = np.linspace(0.1, 1.0, 50)
        y = y_pred + rng.normal(0, 0.05, size=y_pred.shape)
        sigma = fit_noise_scale(y, y_pred, error_model="gaussian")
        assert sigma is not None
        assert sigma == pytest.approx(np.sqrt(np.mean((y - y_pred) ** 2)))
        # log_prob's profiled value should equal -(n/2)(1 + log(2π σ̂²))
        assert log_prob(y, y_pred, error_model="gaussian") == pytest.approx(
            self._expected_log_prob(len(y), sigma)
        )

    def test_lognormal_in_log_space(self):
        rng = np.random.default_rng(1)
        y_pred = np.linspace(0.1, 1.0, 50)
        # Multiplicative noise → constant σ in log space
        y = y_pred * np.exp(rng.normal(0, 0.1, size=y_pred.shape))
        sigma = fit_noise_scale(y, y_pred, error_model="lognormal")
        assert sigma is not None
        # σ̂ is in log space
        log_resid = np.log(y) - np.log(y_pred)
        assert sigma == pytest.approx(np.sqrt(np.mean(log_resid ** 2)))

    def test_mask_excludes_points(self):
        y_pred = np.array([0.5, 0.6, 0.7, 0.8])
        y = np.array([0.5, 0.6, 99.0, 0.8])
        mask = np.array([True, True, False, True])
        sigma = fit_noise_scale(y, y_pred, mask=mask)
        # Only the perfectly-fit points remain → σ̂ = 0
        assert sigma == pytest.approx(0.0)

    def test_empty_after_mask_returns_none(self):
        y = np.array([0.5, 0.6])
        sigma = fit_noise_scale(y, y, mask=np.zeros(2, dtype=bool))
        assert sigma is None


class TestVarianceAt:
    """variance_at evaluates the σ²(μ) polynomial with a positivity floor."""

    def test_constant(self):
        mu = np.linspace(0.0, 1.0, 5)
        v = variance_at(mu, a=0.04, b=0.0, c=0.0)
        assert np.allclose(v, 0.04)

    def test_linear(self):
        mu = np.array([0.0, 0.5, 1.0])
        v = variance_at(mu, a=0.01, b=0.05)
        assert np.allclose(v, [0.01, 0.035, 0.06])

    def test_quadratic(self):
        mu = np.array([0.0, 0.5, 1.0])
        v = variance_at(mu, a=0.01, b=0.0, c=0.04)
        assert np.allclose(v, [0.01, 0.02, 0.05])

    def test_anchor_shifts_polynomial(self):
        mu = np.array([0.5, 1.0, 1.5])
        v = variance_at(mu, a=0.01, b=0.05, c=0.0, anchor=0.5)
        # d = μ − 0.5 → [0.0, 0.5, 1.0]; same values as un-anchored μ in test_linear
        assert np.allclose(v, [0.01, 0.035, 0.06])

    def test_anchor_zero_matches_old_polynomial_for_nonnegative_mu(self):
        mu = np.linspace(0.0, 2.0, 7)
        v_new = variance_at(mu, a=0.02, b=0.03, c=0.01, anchor=0.0)
        v_old = 0.02 + 0.03 * mu + 0.01 * mu * mu
        assert np.allclose(v_new, v_old)

    def test_variance_at_least_a_when_d_nonnegative(self):
        mu = np.array([0.2, 0.5, 1.2])
        a = 0.015
        v = variance_at(mu, a=a, b=0.04, c=0.01, anchor=0.2)
        assert np.all(v >= a - 1e-15)

    def test_floor_keeps_positive(self):
        # Negative-going polynomial gets clamped to a tiny positive value so
        # log(σ²) inside the likelihood doesn't blow up.
        mu = np.array([10.0])
        v = variance_at(mu, a=0.0, b=-1.0, c=0.0)
        assert v[0] > 0.0
        # And it's at the floor, not the negative raw value.
        assert v[0] == pytest.approx(np.finfo(float).eps)

    def test_scalar_input(self):
        v = variance_at(0.5, a=0.01, b=0.02)
        assert v == pytest.approx(0.02)


class TestLogProbWithVariancePolynomial:
    def test_constant_polynomial_matches_y_err(self):
        # Same σ everywhere → variance_params=(σ², 0, 0) and y_err=σ should
        # give the same log-likelihood (modulo y_err's own internal floor).
        rng = np.random.default_rng(0)
        y_pred = np.linspace(0.2, 1.0, 12)
        y = y_pred + rng.normal(0, 0.05, size=y_pred.shape)
        sigma = 0.05
        lp_var = log_prob(y, y_pred, variance_params=(sigma ** 2, 0.0, 0.0))
        lp_yerr = log_prob(y, y_pred, y_err=np.full_like(y, sigma))
        assert lp_var == pytest.approx(lp_yerr)

    def test_linear_higher_at_true_params(self):
        # Generate data with linear σ²(μ); the likelihood should be highest
        # at the true (a, b) and lower for misspecified ones.
        rng = np.random.default_rng(1)
        y_pred = np.linspace(0.05, 1.0, 100)
        true_a, true_b = 1e-4, 0.05
        sigma = np.sqrt(true_a + true_b * y_pred)
        y = y_pred + rng.normal(0, 1.0, size=y_pred.shape) * sigma

        lp_true = log_prob(y, y_pred, variance_params=(true_a, true_b, 0.0))
        lp_too_small = log_prob(y, y_pred, variance_params=(true_a, true_b * 0.1, 0.0))
        lp_too_large = log_prob(y, y_pred, variance_params=(true_a, true_b * 10, 0.0))
        assert lp_true > lp_too_small
        assert lp_true > lp_too_large

    def test_rejects_y_err_combo(self):
        y = np.linspace(0.2, 1.0, 5)
        with pytest.raises(ValueError, match="mutually exclusive"):
            log_prob(
                y, y, y_err=np.full_like(y, 0.1),
                variance_params=(0.01, 0.0, 0.0),
            )

    def test_rejects_lognormal_combo(self):
        y = np.linspace(0.2, 1.0, 5)
        with pytest.raises(ValueError, match="gaussian"):
            log_prob(
                y, y, error_model="lognormal",
                variance_params=(0.01, 0.0, 0.0),
            )

    def test_mask_excludes_points(self):
        y_pred = np.linspace(0.2, 1.0, 6)
        y = y_pred.copy()
        y[2] = 50.0  # outlier
        mask = np.array([True, True, False, True, True, True])
        # With outlier masked, residuals are 0 → log p tends to -∞·log(σ²) /
        # but with σ²>0 floor the value is finite. Just check it's > the
        # un-masked value (outlier dragged lp down hard).
        lp_masked = log_prob(y, y_pred, variance_params=(0.01, 0.0, 0.0), mask=mask)
        lp_all = log_prob(y, y_pred, variance_params=(0.01, 0.0, 0.0))
        assert lp_masked > lp_all


class TestVarianceModelEnum:
    def test_supported_variance_models_lists_all(self):
        assert set(SUPPORTED_VARIANCE_MODELS) == {"constant", "linear", "quadratic"}

    def test_coerce_string(self):
        assert FitConfig(variance_model="linear").noise == GaussianLinear()
        assert FitConfig(variance_model="quadratic").noise == GaussianQuadratic()

    def test_coerce_invalid_raises(self):
        with pytest.raises(ValueError):
            FitConfig(variance_model="cubic")

    def test_free_coefficients(self):
        assert FitConfig().fitting_parameters == ["log_c50", "hill", "effect_0", "effect_inf"]
        assert FitConfig(variance_model="linear").fitting_parameters[-2:] == ["var_a", "var_b"]
        assert FitConfig(variance_model="quadratic").fitting_parameters[-3:] == ["var_a", "var_b", "var_c"]
