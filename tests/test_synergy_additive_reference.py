"""Compare Bliss, HSA, and Loewe against offline ``synergy`` package reference JSON."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from synfit.bliss import (
    bliss_deviation,
    bliss_independence,
    bliss_reference,
    hsa_deviation,
    hsa_reference,
)
from synfit.hill import hill_curve
from synfit.loewe import loewe_ci, loewe_reference
from tests.synergy_reference_conventions import (
    interior_mask,
    normalized_survival,
    package_synergy_to_synfit_deviation,
)

REF_JSON = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "data"
    / "bliss_hsa_loewe_reference_values.json"
)

EXPECTED_CASE_NAMES = frozenset(
    {
        "bliss_additive",
        "synergistic",
        "antagonistic",
        "mismatched_slopes_synergistic",
        "loewe_additive",
        "offset_inhibition_synergistic",
        "rectangular_perturbed_marginals",
        "activation_synergistic",
    }
)

_REQUIRED_VERSION_KEYS = (
    "synergy_package_version",
    "numpy_version",
    "scipy_version",
    "mpmath_version",
)

_REQUIRED_CASE_KEYS = frozenset(
    {
        "name",
        "conc_hor",
        "conc_ver",
        "c50_hor",
        "c50_ver",
        "hill_hor",
        "hill_ver",
        "effect_0",
        "effect_inf",
        "mean_matrix",
        "resp_hor_model",
        "resp_ver_model",
        "bliss_reference",
        "hsa_reference",
        "loewe_reference",
        "bliss_synergy",
        "hsa_synergy",
        "loewe_ci",
        "loewe_reference_exact",
    }
)

_TABULATED_CASE_KEYS = frozenset(
    {
        "bliss_reference_tabulated",
        "hsa_reference_tabulated",
        "hsa_synergy_tabulated",
        "resp_hor_observed",
        "resp_ver_observed",
    }
)

# Closed-form Hill marginals and analytic Bliss/HSA/Loewe CI — no slice optimiser.
# Observed max |synfit − package| on committed fixtures is ~4.44e-16; 1e-12 leaves
# margin for float evaluation order only (JSON round-trip is exact).
_CLOSED_FORM_ATOL = 1e-12

# loewe_reference uses Brent root-finding; mpmath ground truth (dps=50) differs
# by max ~3.47e-9 on loewe_additive (1e-8 covers all committed cases).
_LOEWE_REFERENCE_EXACT_ATOL = 1e-8

# synergy.Loewe.E_reference minimises squared residual via scipy minimize_scalar
# (default xatol=1e-5); measured max |synfit − package| ≈ 1.669e-6 on inhibition grids.
_LOEWE_REFERENCE_PKG_ATOL = 2e-6

# Upstream synergy 1.0.0 Loewe.E_reference cannot solve activation (weakest_E guard).
# See tests/synergy_reference_conventions.py — do not widen tolerances to match it.
_LOEWE_REFERENCE_PKG_EXCLUDED: dict[str, str] = {
    "activation_synergistic": (
        "E_reference uses weakest_E=max(Emax1,Emax2) and skips when either marginal "
        "E < weakest_E (inhibition-only). On this fixture, mode CI/delta_weakest "
        "returns 1.8 (=Emax) on interior cells where exact is 0.413–1.766 "
        "(max |package−exact| 1.387 at cell (1,1); exact 0.413 vs package 1.8); "
        "delta_nan is all NaN. synfit vs loewe_reference_exact max |Δ| 6.1e-9."
    ),
}


def _load_reference_cases() -> list[dict]:
    with REF_JSON.open() as f:
        payload = json.load(f)
    for key in _REQUIRED_VERSION_KEYS:
        assert key in payload and payload[key], f"fixture missing version field {key!r}"
    cases = payload["cases"]
    assert isinstance(cases, list) and cases, "fixture cases must be a non-empty list"
    names = [case["name"] for case in cases]
    assert len(names) == len(set(names)), "fixture case names must be unique"
    assert frozenset(names) == EXPECTED_CASE_NAMES, (
        f"unexpected case set: got {frozenset(names)!r}"
    )
    for case in cases:
        missing = _REQUIRED_CASE_KEYS - case.keys()
        assert not missing, f"{case['name']}: missing required keys {missing!r}"
        if case["name"] == "rectangular_perturbed_marginals":
            tab_missing = _TABULATED_CASE_KEYS - case.keys()
            assert not tab_missing, f"{case['name']}: missing tabulated keys {tab_missing!r}"
    return cases


def _normalized_bliss_from_package_marginals(
    resp_hor: np.ndarray,
    resp_ver: np.ndarray,
    e0: float,
    e_inf: float,
) -> np.ndarray:
    nh = np.clip(normalized_survival(resp_hor, e0, e_inf), 0.0, 1.0)
    nv = np.clip(normalized_survival(resp_ver, e0, e_inf), 0.0, 1.0)
    return np.outer(nv, nh)


def _assert_matching_finite_masks(
    a: np.ndarray,
    b: np.ndarray,
    region: np.ndarray,
    *,
    err_msg: str,
) -> np.ndarray:
    finite_a = np.isfinite(a) & region
    finite_b = np.isfinite(b) & region
    np.testing.assert_array_equal(finite_a, finite_b, err_msg=err_msg)
    assert finite_a.any(), err_msg + " (no finite cells in region)"
    return finite_a


class TestSynergyReferenceConventionCanary:
    def test_package_synergy_sign_mapping_scalar(self):
        # Hand-computed: scale = 2.5 - 0.4 = 2.1; synfit = -package / scale.
        mapped = package_synergy_to_synfit_deviation(np.array(0.21), 2.5, 0.4)
        assert float(mapped) == pytest.approx(-0.1)

    def test_normalized_survival_offset_scalar(self):
        # (1.0 - 0.4) / 2.1 = 2/7 — independent of synfit or synergy implementations.
        assert normalized_survival(np.array(1.0), 2.5, 0.4) == pytest.approx(2.0 / 7.0)


class TestBlissHsaLoeweSynergyReferenceJson:
    @pytest.fixture(scope="class")
    def cases(self):
        return _load_reference_cases()

    def test_bliss_independence_and_deviation(self, cases):
        for case in cases:
            ch = np.array(case["conc_hor"])
            cv = np.array(case["conc_ver"])
            m = np.array(case["mean_matrix"])
            e0, e_inf = case["effect_0"], case["effect_inf"]
            ref_syn = bliss_independence(
                ch,
                cv,
                case["c50_hor"],
                case["c50_ver"],
                case["hill_hor"],
                case["hill_ver"],
                e0,
                e_inf,
            )
            norm_syn = normalized_survival(ref_syn, e0, e_inf)
            resp_hor = np.array(case["resp_hor_model"])
            resp_ver = np.array(case["resp_ver_model"])
            norm_pkg = _normalized_bliss_from_package_marginals(resp_hor, resp_ver, e0, e_inf)
            interior = interior_mask(ch, cv)
            assert np.max(np.abs(norm_syn - norm_pkg)) < _CLOSED_FORM_ATOL

            ref_model = np.array(case["bliss_reference"])
            dev_syn = bliss_deviation(m, ref_model, e0, e_inf, ch, cv)
            dev_pkg = package_synergy_to_synfit_deviation(
                np.array(case["bliss_synergy"]), e0, e_inf
            )
            assert np.max(np.abs(dev_syn[interior] - dev_pkg[interior])) < _CLOSED_FORM_ATOL
            if case["name"] != "bliss_additive":
                assert np.max(np.abs(dev_pkg[interior])) > 0.01

    def test_bliss_reference_from_observed_marginals(self, cases):
        for case in cases:
            if "bliss_reference_tabulated" not in case:
                continue
            e0, e_inf = case["effect_0"], case["effect_inf"]
            resp_hor = np.array(case["resp_hor_observed"])
            resp_ver = np.array(case["resp_ver_observed"])
            ref_tab = np.array(case["bliss_reference_tabulated"])
            ref_syn = bliss_reference(resp_hor, resp_ver, effect_0=e0, effect_inf=e_inf)
            norm_syn = normalized_survival(ref_syn, e0, e_inf)
            norm_expected = _normalized_bliss_from_package_marginals(
                resp_hor, resp_ver, e0, e_inf
            )
            assert np.max(np.abs(norm_syn - norm_expected)) < _CLOSED_FORM_ATOL
            assert np.max(np.abs(ref_syn - ref_tab)) < _CLOSED_FORM_ATOL

    def test_hsa_reference_and_deviation(self, cases):
        for case in cases:
            ch = np.array(case["conc_hor"])
            cv = np.array(case["conc_ver"])
            m = np.array(case["mean_matrix"])
            e0, e_inf = case["effect_0"], case["effect_inf"]
            if "hsa_reference_tabulated" in case:
                resp_hor = np.array(case["resp_hor_observed"])
                resp_ver = np.array(case["resp_ver_observed"])
                ref_pkg = np.array(case["hsa_reference_tabulated"])
            else:
                resp_hor = hill_curve(
                    ch,
                    c50=case["c50_hor"],
                    hill=case["hill_hor"],
                    effect_0=e0,
                    effect_inf=e_inf,
                )
                resp_ver = hill_curve(
                    cv,
                    c50=case["c50_ver"],
                    hill=case["hill_ver"],
                    effect_0=e0,
                    effect_inf=e_inf,
                )
                ref_pkg = np.array(case["hsa_reference"])
            ref_syn = hsa_reference(resp_hor, resp_ver, effect_0=e0, effect_inf=e_inf)
            interior = interior_mask(ch, cv)
            assert np.max(np.abs(ref_syn - ref_pkg)) < _CLOSED_FORM_ATOL

            dev_syn = hsa_deviation(m, resp_hor, resp_ver, e0, e_inf, ch, cv)
            if "hsa_synergy_tabulated" in case:
                hsa_syn_pkg = np.array(case["hsa_synergy_tabulated"])
            else:
                hsa_syn_pkg = np.array(case["hsa_synergy"])
            dev_pkg = package_synergy_to_synfit_deviation(hsa_syn_pkg, e0, e_inf)
            assert np.max(np.abs(dev_syn[interior] - dev_pkg[interior])) < _CLOSED_FORM_ATOL

    def test_loewe_reference(self, cases):
        for case in cases:
            ch = np.array(case["conc_hor"])
            cv = np.array(case["conc_ver"])
            e0, e_inf = case["effect_0"], case["effect_inf"]
            ref_exact = np.array(case["loewe_reference_exact"])
            ref_syn = loewe_reference(
                ch,
                cv,
                case["c50_hor"],
                case["c50_ver"],
                case["hill_hor"],
                case["hill_ver"],
                e0,
                e_inf,
            )
            interior = interior_mask(ch, cv)
            assert np.all(np.isfinite(ref_exact[interior])), (
                f"{case['name']}: exact Loewe oracle must be finite on all interior cells"
            )
            compare = _assert_matching_finite_masks(
                ref_syn,
                ref_exact,
                interior,
                err_msg=f"{case['name']}: synfit vs exact Loewe finite-mask mismatch",
            )
            assert (
                np.max(np.abs(ref_syn[compare] - ref_exact[compare]))
                < _LOEWE_REFERENCE_EXACT_ATOL
            )
            ref_pkg = np.array(case["loewe_reference"])
            if case["name"] in _LOEWE_REFERENCE_PKG_EXCLUDED:
                continue
            compare_pkg = _assert_matching_finite_masks(
                ref_syn,
                ref_pkg,
                interior,
                err_msg=f"{case['name']}: synfit vs package Loewe finite-mask mismatch",
            )
            assert (
                np.max(np.abs(ref_syn[compare_pkg] - ref_pkg[compare_pkg]))
                < _LOEWE_REFERENCE_PKG_ATOL
            )

    def test_loewe_ci(self, cases):
        for case in cases:
            ch = np.array(case["conc_hor"])
            cv = np.array(case["conc_ver"])
            m = np.array(case["mean_matrix"])
            e0, e_inf = case["effect_0"], case["effect_inf"]
            ci_syn = loewe_ci(
                ch,
                cv,
                case["c50_hor"],
                case["c50_ver"],
                case["hill_hor"],
                case["hill_ver"],
                m,
                e0,
                e_inf,
            )
            ci_pkg = np.array(case["loewe_ci"])
            interior = interior_mask(ch, cv)
            compare = _assert_matching_finite_masks(
                ci_syn,
                ci_pkg,
                interior,
                err_msg=f"{case['name']}: synfit vs package Loewe CI finite-mask mismatch",
            )
            assert (
                np.max(np.abs(ci_syn[compare] - ci_pkg[compare])) < _CLOSED_FORM_ATOL
            )
            if case["name"] == "loewe_additive":
                assert np.max(np.abs(ci_pkg[compare] - 1.0)) < 1e-4
