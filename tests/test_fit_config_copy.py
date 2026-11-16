"""Caller-supplied FitConfig must not be mutated by fitters."""
import copy

import pytest

from synfit.data import FitBounds, FitConfig
from synfit.noise import GaussianConstant, GaussianLinear
from synfit.single import SingleDrugFit, SingleDrugFitWithError, default_fit_config
from tests.test_fitting import _synthetic_data


def _fit_config_equal(a: FitConfig, b: FitConfig) -> bool:
    if (
        a.log_c50 != b.log_c50
        or a.hill != b.hill
        or a.effect_0 != b.effect_0
        or a.effect_inf != b.effect_inf
        or a.asymmetry != b.asymmetry
        or a.direction != b.direction
        or a.fitting_parameters != b.fitting_parameters
        or a.noise != b.noise
    ):
        return False
    if a.bounds is None and b.bounds is None:
        return True
    if a.bounds is None or b.bounds is None:
        return False
    return a.bounds == b.bounds


def _caller_snapshot(cfg: FitConfig) -> FitConfig:
    return copy.deepcopy(cfg)


def test_single_drug_caller_config_unchanged_after_construct_and_fit():
    data = _synthetic_data()
    caller = FitConfig(direction="inhibition")
    before = _caller_snapshot(caller)
    fitter = SingleDrugFit(data, caller)
    fitter.fit()
    assert _fit_config_equal(before, caller)
    assert fitter.config is not caller


def test_single_drug_with_error_caller_config_unchanged_after_construct_and_fit():
    data = _synthetic_data().assign(y_err=0.02)
    caller = FitConfig(direction="inhibition")
    before = _caller_snapshot(caller)
    fitter = SingleDrugFitWithError(data, caller)
    fitter.fit()
    assert _fit_config_equal(before, caller)
    assert fitter.config is not caller


def test_reused_fit_config_matches_fresh_on_rescaled_data():
    """Regression: bounds from the first dataset must not leak into the second fit."""
    data = _synthetic_data(c50=2.0)
    caller = FitConfig(direction="inhibition")
    before = _caller_snapshot(caller)

    SingleDrugFit(data, caller).fit()
    assert _fit_config_equal(before, caller)

    scaled = data.assign(y=data["y"] * 1000.0)
    r_fresh = SingleDrugFit(scaled, FitConfig(direction="inhibition")).fit()
    r_reuse = SingleDrugFit(scaled, caller).fit()

    assert r_fresh.success
    assert r_reuse.success
    assert r_fresh.effect_0 == pytest.approx(r_reuse.effect_0, rel=0.05)
    assert r_fresh.r2 == pytest.approx(r_reuse.r2, abs=0.05)
    assert r_reuse.r2 is not None and r_reuse.r2 > 0.9


_VAR_PIN_MSG = "cannot be pinned or partially selected"
_VAR_EXTRAS_MSG = "not part of the 'gaussian_linear'"
_VAR_ZERO_WIDTH_MSG = "cannot be fixed, including by setting equal"


def test_fit_config_rejects_partial_variance_selection_at_constructor():
    with pytest.raises(ValueError, match=_VAR_PIN_MSG):
        FitConfig(
            direction="inhibition",
            noise=GaussianLinear(a_init=0.001, b_init=10.0),
            fitting_parameters=["log_c50", "hill", "effect_0", "effect_inf", "var_b"],
        )


def test_fit_config_rejects_extra_variance_coefficients_for_constant_noise():
    with pytest.raises(ValueError, match="not part of the 'gaussian_constant'"):
        FitConfig(
            direction="inhibition",
            noise=GaussianConstant(),
            fitting_parameters=[
                "log_c50", "hill", "effect_0", "effect_inf", "var_a", "var_b",
            ],
        )


def test_fit_config_rejects_unknown_variance_coefficient_for_noise_model():
    with pytest.raises(ValueError, match=_VAR_EXTRAS_MSG):
        FitConfig(
            direction="inhibition",
            noise=GaussianLinear(),
            fitting_parameters=[
                "log_c50", "hill", "effect_0", "effect_inf", "var_a", "var_b", "var_c",
            ],
        )


def test_single_drug_fit_rejects_stale_variance_after_noise_switch():
    data = _synthetic_data()
    fitter = SingleDrugFit(
        data, default_fit_config(data, direction="inhibition", noise=GaussianLinear()),
    )
    fitter.config.noise = GaussianConstant()
    with pytest.raises(ValueError, match="not part of the 'gaussian_constant'"):
        fitter.fit()


def test_fit_config_rejects_equal_variance_bounds_at_constructor():
    with pytest.raises(ValueError, match=_VAR_ZERO_WIDTH_MSG):
        FitConfig(
            direction="inhibition",
            noise=GaussianLinear(a_init=0.001),
            bounds=FitBounds(var_a=(0.001, 0.001)),
        )


def test_single_drug_fit_rejects_equal_variance_bounds_after_edit():
    data = _synthetic_data()
    fitter = SingleDrugFit(
        data, FitConfig(direction="inhibition", noise=GaussianLinear(a_init=0.001)),
    )
    fitter.config.bounds.var_a = (0.001, 0.001)
    with pytest.raises(ValueError, match=_VAR_ZERO_WIDTH_MSG):
        fitter.fit()


def test_single_drug_with_error_rejects_equal_variance_bounds_at_fit():
    data = _synthetic_data().assign(y_err=0.02)
    fitter = SingleDrugFitWithError(data, FitConfig(direction="inhibition"))
    fitter.config.noise = GaussianLinear(a_init=0.001, b_init=0.05)
    fitter.config.fitting_parameters = [
        *fitter.config.fitting_parameters, "var_a", "var_b",
    ]
    fitter.config.bounds.var_a = (0.001, 0.001)
    with pytest.raises(ValueError, match=_VAR_ZERO_WIDTH_MSG):
        fitter.fit()


def test_single_drug_fit_rejects_post_construction_variance_pin():
    data = _synthetic_data()
    fitter = SingleDrugFit(
        data,
        FitConfig(direction="inhibition", noise=GaussianLinear(a_init=0.001, b_init=0.05)),
    )
    fitter.config.fitting_parameters = [
        p for p in fitter.config.fitting_parameters if p != "var_b"
    ]
    with pytest.raises(ValueError, match=_VAR_PIN_MSG):
        fitter.fit()


def test_heteroscedastic_explicit_all_variance_coefficients_ok():
    cfg = FitConfig(
        direction="inhibition",
        noise=GaussianLinear(),
        fitting_parameters=[
            "log_c50", "hill", "effect_0", "effect_inf", "var_a", "var_b",
        ],
    )
    assert cfg.fitting_parameters[-2:] == ["var_a", "var_b"]


def test_heteroscedastic_auto_extend_without_variance_names_still_works():
    data = _synthetic_data()
    cfg = FitConfig(direction="inhibition", variance_model="linear")
    assert "var_a" in cfg.fitting_parameters and "var_b" in cfg.fitting_parameters
    result = SingleDrugFit(data, cfg).fit()
    assert result.success

    copied = cfg.copy()
    assert copied.fitting_parameters == cfg.fitting_parameters


def test_heteroscedastic_caller_config_unchanged():
    data = _synthetic_data()
    caller = FitConfig(direction="inhibition", variance_model="linear")
    before = _caller_snapshot(caller)
    SingleDrugFit(data, caller).fit()
    assert _fit_config_equal(before, caller)


def test_post_construction_fitter_config_edits_apply():
    data = _synthetic_data()
    caller = FitConfig(direction="inhibition")
    before = _caller_snapshot(caller)

    fitter = SingleDrugFit(data, caller)
    assert _fit_config_equal(before, caller)

    new_hill_bounds = (1.8, 2.2)
    fitter.config.bounds.hill = new_hill_bounds
    fitter.config.hill = 2.0

    r_edited = fitter.fit()

    assert r_edited.success
    assert new_hill_bounds[0] <= r_edited.hill <= new_hill_bounds[1]


def test_post_construction_pin_on_fitter_config_applies():
    data = _synthetic_data()
    fitter = SingleDrugFit(data, FitConfig(direction="inhibition"))
    pin = 0.82
    fitter.config.fitting_parameters = [
        p for p in fitter.config.fitting_parameters if p != "effect_0"
    ]
    fitter.config.effect_0 = pin
    result = fitter.fit()
    assert result.effect_0 == pytest.approx(pin)


@pytest.mark.parametrize("noise", ["gaussian", "lognormal"])
@pytest.mark.parametrize("fitter_cls", [SingleDrugFit, SingleDrugFitWithError])
@pytest.mark.parametrize("s", [1.0, 1e3])
def test_scaled_config_edits_and_refit(fitter_cls, noise, s):
    """Explicit initials and user-unit bounds survive rescaling across consecutive fits."""
    base = _synthetic_data()
    if fitter_cls is SingleDrugFitWithError:
        base = base.assign(y_err=0.1)
    scaled = base.assign(y=base["y"] * s)
    if fitter_cls is SingleDrugFitWithError:
        scaled = scaled.assign(y_err=base["y_err"] * s)

    caller = FitConfig(direction="inhibition", 
        noise=noise,
        log_c50=0.3,
        hill=1.4,
        effect_0=0.95,
        effect_inf=0.04,
    )
    before = _caller_snapshot(caller)
    fitter = fitter_cls(scaled, caller)
    assert _fit_config_equal(before, caller)

    pin = 0.91 * s
    fitter.config.fitting_parameters = [
        p for p in fitter.config.fitting_parameters if p != "effect_0"
    ]
    fitter.config.effect_0 = pin
    tight = (fitter.config.bounds.hill[0] + 0.05, fitter.config.bounds.hill[1] - 0.05)
    fitter.config.bounds.hill = tight

    r1 = fitter.fit()
    r2 = fitter.fit()
    assert r1.success and r2.success
    assert r1.effect_0 == pytest.approx(pin)
    assert r2.effect_0 == pytest.approx(pin)
    assert tight[0] <= r2.hill <= tight[1]
    assert _fit_config_equal(before, caller)


def test_post_construction_bounds_tuple_mutation_applies():
    data = _synthetic_data()
    fitter = SingleDrugFit(data, FitConfig(direction="inhibition"))
    tight_hi = fitter.config.bounds.effect_inf[0] + 0.01
    fitter.config.bounds.effect_inf = (
        fitter.config.bounds.effect_inf[0],
        tight_hi,
    )
    lo = fitter.config.bounds.effect_inf[0]
    r_tight = fitter.fit()

    assert r_tight.success
    assert lo <= r_tight.effect_inf <= tight_hi


def test_zero_width_after_normalisation_leaves_normalisation_inactive():
    """Distinct user-unit bounds that round to equality once normalised are
    rejected before the fitter enters its normalised working state."""
    data = _synthetic_data()
    fitter = SingleDrugFit(
        data, default_fit_config(data, direction="inhibition", noise=GaussianLinear()),
    )
    fitter.config.bounds.var_a = (0.00010990990990990992, 0.00010990990990990993)
    with pytest.raises(ValueError, match=_VAR_ZERO_WIDTH_MSG):
        fitter.fit()
    assert getattr(fitter, "_work_config", None) is None
    assert fitter._response_norm_factor() == 1.0
