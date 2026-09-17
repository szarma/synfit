"""Invalid curve directions raise ValueError at every public entry point."""
import pandas as pd
import pytest

from synfit import default_bounds, default_fit_config
from synfit.data import FitConfig
from synfit.joint_marginal import JointMarginalFit, default_joint_marginal_config
from synfit.matrix import MatrixFit
from synfit.single import SingleDrugFit, SingleDrugFitWithError
from tests.helpers import matrix_from_config


_CORE_KEYS = ("top", "bottom", "log_c50_a", "hill_a", "log_c50_b", "hill_b")


def _curve() -> pd.DataFrame:
    conc = [0.01, 0.1, 1.0, 10.0, 100.0]
    y = [0.9, 0.8, 0.5, 0.2, 0.1]
    return pd.DataFrame({"concentration": conc, "y": y, "replicate": ["A"] * len(conc)})


def _assert_invalid_direction(exc_info, *, bad: str = "sideways"):
    message = str(exc_info.value)
    assert bad in message
    assert "inhibition" in message
    assert "activation" in message


def test_default_fit_config_rejects_invalid_direction():
    with pytest.raises(ValueError) as exc_info:
        default_fit_config(_curve(), direction="sideways")
    _assert_invalid_direction(exc_info)


def test_default_bounds_rejects_invalid_direction():
    with pytest.raises(ValueError) as exc_info:
        default_bounds(_curve(), direction="sideways")
    _assert_invalid_direction(exc_info)


@pytest.mark.parametrize("direction", ["inhibition", "activation"])
def test_default_fit_config_accepts_valid_direction(direction):
    cfg = default_fit_config(_curve(), direction=direction)
    assert cfg.direction == direction


@pytest.mark.parametrize("direction", ["inhibition", "activation"])
def test_default_bounds_accepts_valid_direction(direction):
    bounds = default_bounds(_curve(), direction=direction)
    assert bounds.effect_0[0] < bounds.effect_0[1]
    assert bounds.effect_inf[0] < bounds.effect_inf[1]


def test_invalid_direction_is_rejected_before_data_derivation():
    """Direction is validated before any data-derived computation."""
    empty = pd.DataFrame({"concentration": [], "y": [], "replicate": []})
    with pytest.raises(ValueError) as exc_info:
        default_fit_config(empty, direction="sideways")
    _assert_invalid_direction(exc_info)


def test_fit_config_rejects_invalid_direction():
    with pytest.raises(ValueError) as exc_info:
        FitConfig(direction="sideways")
    _assert_invalid_direction(exc_info)


@pytest.mark.parametrize("direction", ["inhibition", "activation"])
def test_fit_config_accepts_valid_direction(direction):
    assert FitConfig(direction=direction).direction == direction


@pytest.mark.parametrize("direction", ["inhibition", "activation"])
def test_single_drug_fit_accepts_valid_direction(direction):
    fitter = SingleDrugFit(_curve(), FitConfig(direction=direction))
    assert fitter.config.direction == direction


def test_single_drug_fit_rejects_invalid_direction():
    with pytest.raises(ValueError) as exc_info:
        SingleDrugFit(_curve(), FitConfig(direction="sideways"))
    _assert_invalid_direction(exc_info)


def test_single_drug_fit_with_error_rejects_invalid_direction():
    df = _curve().assign(y_err=0.05)
    with pytest.raises(ValueError) as exc_info:
        SingleDrugFitWithError(df, FitConfig(direction="sideways"))
    _assert_invalid_direction(exc_info)


@pytest.mark.parametrize("direction", ["inhibition", "activation"])
def test_single_drug_fit_with_error_accepts_valid_direction(direction):
    df = _curve().assign(y_err=0.05)
    fitter = SingleDrugFitWithError(df, FitConfig(direction=direction))
    assert fitter.config.direction == direction


@pytest.mark.parametrize("which", ["direction_a", "direction_b"])
def test_default_joint_marginal_config_rejects_invalid_direction(which):
    data_a = data_b = _curve()
    kwargs = {which: "sideways"}
    with pytest.raises(ValueError) as exc_info:
        default_joint_marginal_config(data_a, data_b, **kwargs)
    _assert_invalid_direction(exc_info)


@pytest.mark.parametrize("direction_a,direction_b", [
    ("inhibition", "inhibition"),
    ("activation", "activation"),
    ("inhibition", "activation"),
])
def test_default_joint_marginal_config_accepts_valid_directions(direction_a, direction_b):
    result = default_joint_marginal_config(
        _curve(), _curve(), direction_a=direction_a, direction_b=direction_b,
    )
    for key in _CORE_KEYS:
        assert key in result


@pytest.mark.parametrize("which", ["direction_a", "direction_b"])
def test_joint_marginal_fit_rejects_invalid_direction(which):
    kwargs = {which: "sideways"}
    with pytest.raises(ValueError) as exc_info:
        JointMarginalFit(_curve(), _curve(), **kwargs)
    _assert_invalid_direction(exc_info)


@pytest.mark.parametrize("direction_a,direction_b", [
    ("inhibition", "inhibition"),
    ("activation", "activation"),
    ("inhibition", "activation"),
])
def test_joint_marginal_fit_accepts_valid_directions(direction_a, direction_b):
    fitter = JointMarginalFit(
        _curve(), _curve(), direction_a=direction_a, direction_b=direction_b,
    )
    assert fitter.direction_a == direction_a
    assert fitter.direction_b == direction_b


@pytest.mark.parametrize("which", ["direction_horizontal", "direction_vertical"])
def test_matrix_fit_rejects_invalid_direction(which):
    replicates, conc_hor, conc_ver = matrix_from_config()
    kwargs = {which: "sideways"}
    with pytest.raises(ValueError) as exc_info:
        MatrixFit(replicates, conc_hor, conc_ver, **kwargs)
    _assert_invalid_direction(exc_info)


@pytest.mark.parametrize("direction", ["inhibition", "activation"])
def test_matrix_fit_accepts_valid_matching_directions(direction):
    replicates, conc_hor, conc_ver = matrix_from_config(
        noise_model="gaussian" if direction == "activation" else "lognormal",
    )
    fitter = MatrixFit(
        replicates, conc_hor, conc_ver,
        direction_horizontal=direction, direction_vertical=direction,
    )
    assert fitter.direction_horizontal == fitter.direction_vertical == direction
