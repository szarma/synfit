"""Regression: compound_add_mult must not crash fitters with AttributeError."""

import numpy as np
import pytest

from synfit.data import FitConfig
from synfit.joint_marginal import JointMarginalFit
from synfit.matrix import MatrixFit
from synfit.noise import CompoundAddMult, from_dict, to_dict
from synfit.single import (
    SingleDrugFit,
    SingleDrugFitWithError,
    default_bounds,
    default_fit_config,
)
from synfit.synthetic import generate_single_drug

from tests.helpers import matrix_from_config

_COMPOUND_NOISE_FORMS = (
    "compound_add_mult",
    {"kind": "compound_add_mult"},
    CompoundAddMult(),
)


def _single_drug_df():
    config = {
        "seed": 42,
        "hill_params": {"c50": 1.0, "hill": 1.5, "effect_0": 0.9, "effect_inf": 0.05},
        "concentration_series": {
            "initial_conc": 100.0,
            "fold_dilutions": 10 ** (2 / 7),
            "length": 8,
            "has_zero": False,
        },
        "n_replicates": 3,
        "noise_model": "gaussian",
        "noise_sigma": 0.02,
        "noise_sigma_log": 0.02,
    }
    return generate_single_drug(config)


def _single_drug_with_error_df():
    df = _single_drug_df()
    return df.assign(y_err=0.05)


def _two_drug_data():
    df = _single_drug_df()
    return df, df.copy()


@pytest.mark.parametrize("noise", _COMPOUND_NOISE_FORMS)
def test_single_drug_fit_rejects_compound_noise(noise):
    with pytest.raises(ValueError, match=r"constant/linear/quadratic gaussian and lognormal"):
        SingleDrugFit(_single_drug_df(), FitConfig(noise=noise))


@pytest.mark.parametrize("noise", _COMPOUND_NOISE_FORMS)
def test_single_drug_fit_with_error_rejects_compound_noise(noise):
    with pytest.raises(ValueError, match=r"gaussian_constant and lognormal"):
        SingleDrugFitWithError(_single_drug_with_error_df(), FitConfig(noise=noise))


@pytest.mark.parametrize("noise", _COMPOUND_NOISE_FORMS)
def test_default_fit_config_rejects_compound_noise(noise):
    with pytest.raises(ValueError, match=r"constant/linear/quadratic gaussian and lognormal"):
        default_fit_config(_single_drug_df(), noise=noise)


@pytest.mark.parametrize("noise", _COMPOUND_NOISE_FORMS)
def test_default_bounds_rejects_compound_noise(noise):
    with pytest.raises(ValueError, match=r"constant/linear/quadratic gaussian and lognormal"):
        default_bounds(_single_drug_df(), noise=noise)


@pytest.mark.parametrize("noise", _COMPOUND_NOISE_FORMS)
def test_matrix_fit_rejects_compound_noise(noise):
    replicates, conc_h, conc_v = matrix_from_config()
    with pytest.raises(ValueError, match=r"constant gaussian/lognormal"):
        MatrixFit(replicates, conc_h, conc_v, noise=noise)


@pytest.mark.parametrize("noise", _COMPOUND_NOISE_FORMS)
def test_joint_marginal_fit_rejects_compound_noise(noise):
    data_a, data_b = _two_drug_data()
    with pytest.raises(ValueError, match=r"constant/linear/quadratic gaussian and lognormal"):
        JointMarginalFit(data_a, data_b, noise=noise)


def test_compound_add_mult_noise_spec_still_round_trips():
    spec = CompoundAddMult(sigma_log_init=0.25)
    assert from_dict(to_dict(spec)) == spec
