import numpy as np
import pytest
from synfit.hill import hill_curve, log_wall, calculate_concentration_series


def test_hill_curve_midpoint():
    # at c50, response should be halfway between effect_0 and effect_inf
    assert hill_curve(1.0, c50=1.0, effect_0=1.0, effect_inf=0.0) == pytest.approx(0.5)


def test_hill_curve_limits():
    conc = np.array([0.0001, 10000.0])
    result = hill_curve(conc, c50=1.0, effect_0=1.0, effect_inf=0.0)
    assert result[0] == pytest.approx(1.0, abs=0.01)
    assert result[1] == pytest.approx(0.0, abs=0.01)


def test_hill_curve_custom_effect_0_inf():
    val = hill_curve(2.0, c50=2.0, effect_0=0.8, effect_inf=0.1, hill=2.0)
    assert val == pytest.approx(0.45, abs=0.01)


def test_log_wall_inside_bounds():
    penalty = log_wall(np.array([0.5]), boundaries=(0.0, 1.0))
    assert penalty.sum() == 0.0


def test_log_wall_outside_bounds():
    penalty = log_wall(np.array([2.0]), boundaries=(0.0, 1.0))
    assert penalty.sum() > 0.0


def test_calculate_concentration_series_length():
    conc = calculate_concentration_series(100.0, 2, 8)
    assert len(conc) == 8


def test_calculate_concentration_series_has_zero():
    conc = calculate_concentration_series(100.0, 2, 5, has_zero=True)
    assert conc[0] == 0.0


def test_calculate_concentration_series_no_zero():
    conc = calculate_concentration_series(100.0, 2, 5, has_zero=False)
    assert all(c > 0 for c in conc)


def test_calculate_concentration_series_descending():
    conc = calculate_concentration_series(100.0, 2, 5, ascending=False, has_zero=False)
    assert conc[0] > conc[-1]


def test_calculate_concentration_series_invalid_inputs():
    with pytest.raises(ValueError):
        calculate_concentration_series(-1.0, 2, 5)
    with pytest.raises(ValueError):
        calculate_concentration_series(100.0, 0.5, 5)
