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


# --- Cross-check against the names the fitters actually emit ----------------
# Pin the predicates to the real parameter-name sources, not a hand-maintained
# list, so adding a parameter to a fit can't silently fall outside the taxonomy.

def _classify(name: str) -> set[str]:
    roles = set()
    if is_log_param(name):
        roles.add("log")
    if is_variance_param(name):
        roles.add("variance")
    if is_asymptote_param(name):
        roles.add("asymptote")
    return roles


def test_single_drug_param_names_each_have_at_most_one_role():
    from synfit.data import FitConfig

    names = FitConfig().fitting_parameters  # log_c50, hill, effect_0, effect_inf
    assert is_log_param("log_c50")
    assert is_asymptote_param("effect_0") and is_asymptote_param("effect_inf")
    # hill is a shape exponent: no log/variance/asymptote role.
    assert _classify("hill") == set()
    for name in names:
        assert len(_classify(name)) <= 1, f"{name} has overlapping roles"


def test_matrix_param_names_classified_by_taxonomy():
    from synfit.matrix import MatrixFit

    names = MatrixFit._PARAM_NAMES
    asymptotes = {n for n in names if is_asymptote_param(n)}
    logs = {n for n in names if is_log_param(n)}
    assert asymptotes == {"effect_0", "effect_inf"}
    assert logs == {"log_c50_hor", "log_c50_ver"}
    for name in names:
        assert len(_classify(name)) <= 1, f"{name} has overlapping roles"


def test_joint_marginal_param_names_classified_by_taxonomy():
    from synfit.joint_marginal import JointMarginalFit

    names = JointMarginalFit._full_param_names("4p", "5p")
    asymptotes = {n for n in names if is_asymptote_param(n)}
    logs = {n for n in names if is_log_param(n)}
    assert asymptotes == {"top", "bottom"}  # joint shares asymptotes as top/bottom
    assert logs == {"log_c50_a", "log_c50_b"}
    for name in names:
        assert len(_classify(name)) <= 1, f"{name} has overlapping roles"
