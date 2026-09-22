"""Caller-supplied FitConfig must not be mutated by fitters."""
import copy

import pytest

from synfit.data import FitConfig
from synfit.noise import GaussianLinear
from synfit.single import SingleDrugFit, SingleDrugFitWithError
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
    caller = FitConfig()
    before = _caller_snapshot(caller)
    fitter = SingleDrugFit(data, caller)
    fitter.fit()
    assert _fit_config_equal(before, caller)
    assert fitter.config is not caller


def test_single_drug_with_error_caller_config_unchanged_after_construct_and_fit():
    data = _synthetic_data().assign(y_err=0.02)
    caller = FitConfig()
    before = _caller_snapshot(caller)
    fitter = SingleDrugFitWithError(data, caller)
    fitter.fit()
    assert _fit_config_equal(before, caller)
    assert fitter.config is not caller


def test_reused_fit_config_matches_fresh_on_rescaled_data():
    """Regression: bounds from the first dataset must not leak into the second fit."""
    data = _synthetic_data(c50=2.0)
    caller = FitConfig()
    before = _caller_snapshot(caller)

    SingleDrugFit(data, caller).fit()
    assert _fit_config_equal(before, caller)

    scaled = data.assign(y=data["y"] * 1000.0)
    r_fresh = SingleDrugFit(scaled, FitConfig()).fit()
    r_reuse = SingleDrugFit(scaled, caller).fit()

    assert r_fresh.success
    assert r_reuse.success
    assert r_fresh.effect_0 == pytest.approx(r_reuse.effect_0, rel=0.05)
    assert r_fresh.r2 == pytest.approx(r_reuse.r2, abs=0.05)
    assert r_reuse.r2 is not None and r_reuse.r2 > 0.9


def test_fit_config_copy_preserves_removed_var_b():
    pin_b = 0.05
    cfg = FitConfig(noise=GaussianLinear(a_init=0.001, b_init=pin_b))
    cfg.fitting_parameters = [p for p in cfg.fitting_parameters if p != "var_b"]
    assert "var_b" not in cfg.fitting_parameters

    copied = cfg.copy()
    assert "var_b" not in copied.fitting_parameters

    data = _synthetic_data()
    caller = FitConfig(noise=GaussianLinear(a_init=0.001, b_init=pin_b))
    caller.fitting_parameters = [p for p in caller.fitting_parameters if p != "var_b"]
    fitter = SingleDrugFit(data, caller)
    assert "var_b" not in fitter.config.fitting_parameters
    assert "var_b" not in caller.fitting_parameters


def test_heteroscedastic_caller_config_unchanged():
    data = _synthetic_data()
    caller = FitConfig(variance_model="linear")
    before = _caller_snapshot(caller)
    SingleDrugFit(data, caller).fit()
    assert _fit_config_equal(before, caller)


def test_post_construction_fitter_config_edits_apply():
    data = _synthetic_data()
    caller = FitConfig()
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
    fitter = SingleDrugFit(data, FitConfig())
    pin = 0.82
    fitter.config.fitting_parameters = [
        p for p in fitter.config.fitting_parameters if p != "effect_0"
    ]
    fitter.config.effect_0 = pin
    result = fitter.fit()
    assert result.effect_0 == pytest.approx(pin)


def test_post_construction_bounds_tuple_mutation_applies():
    data = _synthetic_data()
    fitter = SingleDrugFit(data, FitConfig())
    tight_hi = fitter.config.bounds.effect_inf[0] + 0.01
    fitter.config.bounds.effect_inf = (
        fitter.config.bounds.effect_inf[0],
        tight_hi,
    )
    lo = fitter.config.bounds.effect_inf[0]
    r_tight = fitter.fit()

    assert r_tight.success
    assert lo <= r_tight.effect_inf <= tight_hi
