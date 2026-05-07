"""End-to-end tests for the heteroscedastic variance models (issue #20).

Covers:

- ``FitConfig`` validation: variance_model coercion, fitting_parameters
  auto-extension, the lognormal+non-constant error case.
- ``SingleDrugFit`` recovery on synthetic linear and quadratic noise.
- AIC-based model selection between constant / linear / quadratic.
- ``FitResult`` payload (variance_params dict, predict_variance, to_dict).
- ``SingleDrugFitWithError`` rejecting non-constant variance models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synfit.data import FitBounds, FitConfig, FitResult
from synfit.noise import GaussianConstant, GaussianLinear, Lognormal, variance_at
from synfit.single import SingleDrugFit, SingleDrugFitWithError
from synfit.synthetic import generate_single_drug


# ----------------------------- helpers ------------------------------------ #


def _hetero_data(
    a: float, b: float = 0.0, c: float = 0.0,
    c50: float = 1.0, hill: float = 1.5,
    effect_0: float = 0.9, effect_inf: float = 0.05,
    n_replicates: int = 6, length: int = 10, seed: int = 7,
) -> pd.DataFrame:
    """Generate single-drug data with σ²(μ) = a + b·μ + c·μ²."""
    config = {
        "seed": seed,
        "hill_params": {"c50": c50, "hill": hill, "effect_0": effect_0, "effect_inf": effect_inf},
        "concentration_series": {
            "initial_conc": 100.0, "fold_dilutions": 10 ** (2 / 7),
            "length": length, "has_zero": False,
        },
        "n_replicates": n_replicates,
        "noise_model": "gaussian",
        "noise_var_a": a,
        "noise_var_b": b,
        "noise_var_c": c,
    }
    return generate_single_drug(config)


# ----------------------------- config plumbing ---------------------------- #


class TestFitConfigVarianceModel:
    def test_default_is_constant(self):
        cfg = FitConfig()
        assert cfg.noise == GaussianConstant()
        # Default fitting params unchanged — constant variance is profiled out.
        assert cfg.fitting_parameters == ["log_c50", "hill", "effect_0", "effect_inf"]

    def test_string_coerced_to_enum(self):
        cfg = FitConfig(variance_model="linear")
        assert cfg.noise == GaussianLinear()

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            FitConfig(variance_model="cubic")

    def test_linear_extends_fitting_parameters(self):
        cfg = FitConfig(variance_model="linear")
        assert "var_a" in cfg.fitting_parameters
        assert "var_b" in cfg.fitting_parameters
        assert "var_c" not in cfg.fitting_parameters

    def test_quadratic_extends_fitting_parameters(self):
        cfg = FitConfig(variance_model="quadratic")
        assert cfg.fitting_parameters[-3:] == ["var_a", "var_b", "var_c"]

    def test_idempotent_no_duplicate_appends(self):
        # Constructing twice (mock of cfg surviving multiple __post_init__
        # calls) shouldn't double-append the variance names.
        cfg = FitConfig(variance_model="quadratic")
        original_count = sum(1 for n in cfg.fitting_parameters if n == "var_a")
        cfg.__post_init__()
        assert sum(1 for n in cfg.fitting_parameters if n == "var_a") == original_count

    def test_lognormal_with_non_constant_rejected(self):
        with pytest.raises(ValueError, match="gaussian"):
            FitConfig(error_model="lognormal", variance_model="linear")

    def test_lognormal_with_constant_ok(self):
        cfg = FitConfig(error_model="lognormal", variance_model="constant")
        assert cfg.noise == Lognormal()

    def test_bounds_have_var_coefficients(self):
        b = FitBounds()
        assert b.var_a[0] > 0  # σ² > 0 required
        assert b.var_c[0] >= 0  # quadratic coefficient stays non-negative


# ----------------------------- fitter recovery ---------------------------- #


class TestLinearVarianceRecovery:
    """Linear σ²(μ) = a + b·μ should be recovered on synthetic data."""

    @pytest.fixture(scope="class")
    def fit(self):
        true_a, true_b = 5e-4, 0.04
        data = _hetero_data(a=true_a, b=true_b, n_replicates=8, length=12, seed=11)
        cfg = FitConfig(error_model="gaussian", variance_model="linear")
        fitter = SingleDrugFit(data, cfg)
        return fitter.fit(), true_a, true_b

    def test_success(self, fit):
        result, *_ = fit
        assert result.success

    def test_variance_params_populated(self, fit):
        result, *_ = fit
        assert result.variance_params is not None
        assert set(result.variance_params.keys()) == {"a", "b"}
        # ``c`` is fixed at 0 for linear — should not appear.
        assert "c" not in result.variance_params

    def test_variance_model_string(self, fit):
        result, *_ = fit
        assert result.variance_model == "linear"

    def test_recovers_b_within_factor_2(self, fit):
        result, true_a, true_b = fit
        b_hat = result.variance_params["b"]
        # Variance estimation is noisy; rough recovery is enough.
        assert 0.5 * true_b < b_hat < 2.0 * true_b

    def test_a_is_positive(self, fit):
        result, *_ = fit
        assert result.variance_params["a"] > 0

    def test_sigma_field_not_set(self, fit):
        # σ is no longer a single scalar — it varies with μ.
        result, *_ = fit
        assert result.sigma is None

    def test_predict_variance_uses_polynomial(self, fit):
        result, true_a, true_b = fit
        mu = np.array([0.0, 0.5, 1.0])
        var = result.predict_variance(mu)
        a_hat = result.variance_params["a"]
        b_hat = result.variance_params["b"]
        expected = variance_at(mu, a_hat, b_hat, 0.0)
        assert np.allclose(var, expected)

    def test_to_dict_round_trip(self, fit):
        result, *_ = fit
        d = result.to_dict()
        assert d["variance_model"] == "linear"
        assert d["variance_params"]["b"] == result.variance_params["b"]

    def test_n_params_includes_var_coefficients(self, fit):
        # 4 hill + 2 variance params → k = 6
        result, *_ = fit
        assert result.n_params == 6


class TestQuadraticVarianceRecovery:
    @pytest.fixture(scope="class")
    def fit(self):
        true_a, true_b, true_c = 1e-4, 0.0, 0.05
        data = _hetero_data(a=true_a, b=true_b, c=true_c, n_replicates=10, length=14, seed=23)
        cfg = FitConfig(variance_model="quadratic")
        fitter = SingleDrugFit(data, cfg)
        return fitter.fit(), true_a, true_b, true_c

    def test_success(self, fit):
        result, *_ = fit
        assert result.success

    def test_all_three_coefficients(self, fit):
        result, *_ = fit
        assert set(result.variance_params.keys()) == {"a", "b", "c"}

    def test_recovers_c_within_factor_3(self, fit):
        result, _a, _b, true_c = fit
        c_hat = result.variance_params["c"]
        assert 0.3 * true_c < c_hat < 3.0 * true_c

    def test_predict_variance_grows_with_mu(self, fit):
        result, *_ = fit
        v_low = result.predict_variance(np.array([0.05]))[0]
        v_high = result.predict_variance(np.array([0.9]))[0]
        # Quadratic positive c ⇒ variance grows with μ in this regime.
        assert v_high > v_low

    def test_n_params(self, fit):
        result, *_ = fit
        assert result.n_params == 7  # 4 hill + 3 variance


# ----------------------------- model selection ---------------------------- #


class TestModelSelectionAIC:
    """AIC should prefer the variance model that matches the data."""

    @staticmethod
    def _fit(data, variance_model: str) -> FitResult:
        return SingleDrugFit(
            data, FitConfig(variance_model=variance_model),
        ).fit()

    def test_constant_data_prefers_constant(self):
        # Homoscedastic noise — extra variance params shouldn't help.
        data = _hetero_data(a=0.04 ** 2, n_replicates=8, length=12, seed=42)
        r_const = self._fit(data, "constant")
        r_lin = self._fit(data, "linear")
        # AIC of constant should not be much worse — extra param costs ~2.
        assert r_const.aic < r_lin.aic + 1.0

    def test_linear_data_prefers_linear_over_constant(self):
        # Strong heteroscedasticity — constant model should lose.
        data = _hetero_data(a=1e-4, b=0.06, n_replicates=10, length=14, seed=55)
        r_const = self._fit(data, "constant")
        r_lin = self._fit(data, "linear")
        assert r_lin.aic < r_const.aic


# ----------------------------- guard rails -------------------------------- #


class TestSingleDrugFitWithErrorRejectsVarianceModel:
    """y_err and σ²(μ) polynomial are mutually exclusive — at construction."""

    def _df(self):
        rng = np.random.default_rng(0)
        n = 8
        return pd.DataFrame({
            "concentration": np.geomspace(0.01, 100, n),
            "y": np.linspace(0.1, 0.9, n) + rng.normal(0, 0.02, n),
            "y_err": np.full(n, 0.05),
        })

    def test_linear_rejected(self):
        with pytest.raises(ValueError, match="noise.kind='gaussian_constant'"):
            SingleDrugFitWithError(
                self._df(), FitConfig(variance_model="linear"),
            )

    def test_constant_accepted(self):
        # No error — paired with the implicit default.
        SingleDrugFitWithError(self._df())


# ----------------------------- predict_variance --------------------------- #


class TestPredictVarianceFallback:
    def test_constant_fit_falls_back_to_sigma(self):
        # σ̂ scalar ⇒ predict_variance returns σ̂² for any μ.
        result = FitResult(
            c50=1.0, log_c50=0.0, hill=1.0, effect_0=1.0, effect_inf=0.0,
            success=True, n_valid=20, n_total=20,
            sigma=0.07, variance_model="constant",
        )
        v = result.predict_variance(np.array([0.1, 0.5, 0.9]))
        assert v is not None
        assert np.allclose(v, 0.07 ** 2)

    def test_no_sigma_no_variance_returns_none(self):
        # User-supplied y_err path leaves both unset → nothing to predict.
        result = FitResult(
            c50=1.0, log_c50=0.0, hill=1.0, effect_0=1.0, effect_inf=0.0,
            success=True, n_valid=20, n_total=20,
        )
        assert result.predict_variance(np.array([0.5])) is None


class TestConfidenceIntervalsRespectVarianceModel:
    def test_prediction_widths_differ_across_variance_models(self):
        data = _hetero_data(a=1e-4, b=0.06, n_replicates=10, length=14, seed=55)
        conc = np.geomspace(data["concentration"].min(), data["concentration"].max(), 200)

        const = SingleDrugFit(data, FitConfig(variance_model="constant")).fit()
        linear = SingleDrugFit(data, FitConfig(variance_model="linear")).fit()

        const_ci = const.predict_ci(conc)
        linear_ci = linear.predict_ci(conc)
        assert const_ci is not None
        assert linear_ci is not None

        const_width = _prediction_width(const, conc, const_ci)
        linear_width = _prediction_width(linear, conc, linear_ci)

        assert linear.variance_params is not None
        # Linear σ²(μ) should fan out near the high-response asymptote and
        # pinch near low response; a constant-variance fit cannot do that.
        assert linear_width[0] > const_width[0] * 1.2
        assert linear_width[-1] < const_width[-1] * 0.5


def _prediction_width(
    result: FitResult,
    conc: np.ndarray,
    ci: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Convert the parameter CI plus fitted noise into total prediction width."""
    y_hat = result.effect_inf + (
        (result.effect_0 - result.effect_inf)
        / (1 + (conc / result.c50) ** result.hill) ** (result.asymmetry or 1.0)
    )
    noise_var = result.predict_variance(y_hat)
    assert noise_var is not None
    z = 1.959963984540054
    param_var = ((ci[1] - ci[0]) / (2 * z)) ** 2
    return 2 * z * np.sqrt(param_var + noise_var)
