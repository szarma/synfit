"""Tests for the centralised parameter-role taxonomy.

These predicates are the single source of truth used by the optimiser
preconditioning, curve-kwarg unpacking, and variance handling across the
single-drug, matrix, and joint-marginal fits — so they must agree with the
names those fits actually emit.
"""
from synfit.param_roles import (
    ASYMPTOTE_PARAM_NAMES,
    VARIANCE_PARAM_NAMES,
    is_asymptote_param,
    is_log_param,
    is_variance_param,
)


def test_is_log_param_matches_log_prefixed_names():
    # single-drug / joint per-drug / matrix per-axis log locations
    for name in ("log_c50", "log_c50_a", "log_c50_b", "log_c50_hor", "log_c50_ver"):
        assert is_log_param(name)
    for name in ("hill", "effect_0", "top", "var_a", "asymmetry", "c50"):
        assert not is_log_param(name)


def test_is_variance_param_matches_polynomial_coefficients():
    for name in VARIANCE_PARAM_NAMES:
        assert is_variance_param(name)
    for name in ("log_c50", "hill", "effect_inf", "top", "var_d"):
        assert not is_variance_param(name)


def test_is_asymptote_param_covers_both_naming_conventions():
    # single-drug / matrix use effect_0/effect_inf; joint-marginal uses top/bottom
    for name in ("effect_0", "effect_inf", "top", "bottom"):
        assert is_asymptote_param(name)
    for name in ("log_c50", "hill", "asymmetry", "var_a", "var_b"):
        assert not is_asymptote_param(name)


def test_role_sets_are_disjoint():
    assert not (set(ASYMPTOTE_PARAM_NAMES) & set(VARIANCE_PARAM_NAMES))
    for name in (*ASYMPTOTE_PARAM_NAMES, *VARIANCE_PARAM_NAMES):
        assert not is_log_param(name)
