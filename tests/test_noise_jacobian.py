"""Tests for the Jacobian correction in the lognormal log_prob."""

import numpy as np
import pytest

from synfit.noise import log_prob


class TestJacobianCorrection:
    def test_gaussian_and_lognormal_differ_for_same_predictions(self):
        # With the Jacobian correction the two are on the same scale but still
        # produce different values; they should never be identical for real data.
        y = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        y_pred = y * 1.05
        lp_g = log_prob(y, y_pred, error_model="gaussian")
        lp_ln = log_prob(y, y_pred, error_model="lognormal")
        assert lp_g != lp_ln

    def test_jacobian_penalises_large_observations(self):
        # The Jacobian term is -sum(log(y_i)).  When y >> 1, log(y) > 0,
        # so the term is negative and *reduces* the lognormal log_prob
        # relative to what it would be without the correction.
        #
        # We verify this by comparing the full lognormal log_prob against
        # a manually-computed version that omits the Jacobian term.
        y = np.array([10.0, 20.0, 50.0, 100.0])
        y_pred = y * 1.01

        lp_ln_with_jacobian = log_prob(y, y_pred, error_model="lognormal")

        # Compute the base lognormal log_prob (log-space residuals, no Jacobian)
        residuals = np.log(y) - np.log(y_pred)
        sigma2 = float(np.mean(residuals ** 2))
        n = len(y)
        lp_ln_without_jacobian = -0.5 * n * (1.0 + np.log(2 * np.pi * sigma2))

        # Jacobian is -sum(log(y)); for y >> 1 this is large and negative
        jacobian = -float(np.sum(np.log(y)))
        assert jacobian < 0

        # The correction must equal the difference between the two values
        assert np.isclose(lp_ln_with_jacobian, lp_ln_without_jacobian + jacobian)

        # And the corrected value is lower (penalised)
        assert lp_ln_with_jacobian < lp_ln_without_jacobian

    def test_jacobian_boosts_small_observations(self):
        # When 0 < y < 1, log(y) < 0, so -sum(log(y)) > 0 → correction is
        # positive → lognormal log_prob is higher than gaussian for same fit.
        y_small = np.array([0.01, 0.05, 0.1, 0.2])
        y_pred = y_small * 1.01
        lp_g = log_prob(y_small, y_pred, error_model="gaussian")
        lp_ln = log_prob(y_small, y_pred, error_model="lognormal")
        assert lp_ln > lp_g

    def test_lognormal_log_prob_is_finite_for_positive_data(self):
        y = np.linspace(0.1, 2.0, 10)
        y_pred = y * 1.02
        assert np.isfinite(log_prob(y, y_pred, error_model="lognormal"))

    def test_jacobian_scales_with_number_of_points(self):
        # Doubling the number of observations (all y = 10) should double the
        # magnitude of the Jacobian correction relative to gaussian.
        rng = np.random.default_rng(0)
        y5 = np.full(5, 10.0) + rng.normal(0, 0.01, 5)
        y10 = np.full(10, 10.0) + rng.normal(0, 0.01, 10)
        y_pred5 = y5 * 1.005
        y_pred10 = y10 * 1.005

        diff5 = log_prob(y5, y_pred5, error_model="lognormal") - log_prob(y5, y_pred5, error_model="gaussian")
        diff10 = log_prob(y10, y_pred10, error_model="lognormal") - log_prob(y10, y_pred10, error_model="gaussian")

        # More points → larger magnitude difference (both are negative for y>1)
        assert abs(diff10) > abs(diff5)
