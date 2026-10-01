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

# Closed-form Hill marginals and analytic Bliss/HSA/Loewe CI — no slice optimiser.
# Observed max |synfit − package| on committed fixtures is ~1.7e-16; 1e-12 leaves
# margin for float evaluation order only (JSON round-trip is exact).
_CLOSED_FORM_ATOL = 1e-12

# loewe_reference uses Brent root-finding; mpmath ground truth (dps=50) differs
# by max ~3.47e-9 on loewe_additive (1e-8 covers all committed cases).
_LOEWE_REFERENCE_EXACT_ATOL = 1e-8

# synergy.Loewe.E_reference minimises squared residual via scipy minimize_scalar
# (default xatol=1e-5); measured max |synfit − package| ≈ 1.66e-6 on inhibition grids.
_LOEWE_REFERENCE_PKG_ATOL = 2e-6

# Upstream synergy 1.0.0 Loewe.E_reference cannot solve activation (weakest_E guard).
# See tests/synergy_reference_conventions.py — do not widen tolerances to match it.
_LOEWE_REFERENCE_PKG_EXCLUDED: dict[str, str] = {
    "activation_synergistic": (
        "E_reference uses weakest_E=max(Emax1,Emax2) and skips when either marginal "
        "E < weakest_E (inhibition-only). On this fixture, mode CI/delta_weakest "
        "returns 1.8 (=Emax) on interior cells where exact is 0.413–2.05 "
        "(max |package−exact| 1.387 at cell (1,1); exact 0.413 vs package 1.8); "
        "delta_nan is all NaN. synfit vs loewe_reference_exact max |Δ| 6.1e-9."
    ),
}


def _hill_normalized_bliss_from_package(
    ch: np.ndarray,
    cv: np.ndarray,
    case: dict,
) -> np.ndarray:
    """Normalised Bliss independence from Hill marginals (package E1*E2 mapped to survival)."""
    e0, e_inf = case["effect_0"], case["effect_inf"]
    eh = hill_curve(
        ch,
        c50=case["c50_hor"],
        hill=case["hill_hor"],
        effect_0=e0,
        effect_inf=e_inf,
    )
    ev = hill_curve(
        cv,
        c50=case["c50_ver"],
        hill=case["hill_ver"],
        effect_0=e0,
        effect_inf=e_inf,
    )
    return np.outer(
        normalized_survival(ev, e0, e_inf),
        normalized_survival(eh, e0, e_inf),
    )


class TestBlissHsaLoeweSynergyReferenceJson:
    @pytest.fixture(scope="class")
    def cases(self):
        with REF_JSON.open() as f:
            return json.load(f)["cases"]

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
            norm_pkg = _hill_normalized_bliss_from_package(ch, cv, case)
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
            nh = np.clip(normalized_survival(resp_hor, e0, e_inf), 0.0, 1.0)
            nv = np.clip(normalized_survival(resp_ver, e0, e_inf), 0.0, 1.0)
            norm_expected = np.outer(nv, nh)
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
            assert np.nanmax(np.abs(ref_syn - ref_exact)) < _LOEWE_REFERENCE_EXACT_ATOL
            ref_pkg = np.array(case["loewe_reference"])
            interior = interior_mask(ch, cv)
            if case["name"] in _LOEWE_REFERENCE_PKG_EXCLUDED:
                continue
            assert (
                np.nanmax(np.abs(ref_syn[interior] - ref_pkg[interior]))
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
            assert (
                np.nanmax(np.abs(ci_syn[interior] - ci_pkg[interior]))
                < _CLOSED_FORM_ATOL
            )
            if case["name"] == "loewe_additive":
                assert np.max(np.abs(ci_pkg[interior] - 1.0)) < 1e-4
