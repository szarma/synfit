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
from synfit.loewe import loewe_ci
from tests.synergy_reference_conventions import (
    interior_mask,
    package_synergy_to_synfit_deviation,
)

REF_JSON = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "data"
    / "bliss_hsa_loewe_reference_values.json"
)

# Closed-form Hill marginals and analytic Bliss/HSA/Loewe CI — no slice optimiser.
# Observed max deviation vs synergy 1.0.0 reference is ~1e-16; 1e-9 leaves room
# for float JSON round-trip only.
_CLOSED_FORM_ATOL = 1e-9


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
            ref_pkg = np.array(case["bliss_reference"])
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
            interior = interior_mask(ch, cv)
            assert np.max(np.abs(ref_syn - ref_pkg)) < _CLOSED_FORM_ATOL

            dev_syn = bliss_deviation(m, ref_syn, e0, e_inf, ch, cv)
            dev_pkg = package_synergy_to_synfit_deviation(
                np.array(case["bliss_synergy"]), e0, e_inf
            )
            assert np.max(np.abs(dev_syn[interior] - dev_pkg[interior])) < _CLOSED_FORM_ATOL
            if case["name"] != "bliss_additive":
                assert np.max(np.abs(dev_pkg[interior])) > 0.01

    def test_bliss_observed_reference(self, cases):
        for case in cases:
            m = np.array(case["mean_matrix"])
            e0, e_inf = case["effect_0"], case["effect_inf"]
            resp_hor = m[0, :]
            resp_ver = m[:, 0]
            ref_indep = np.array(case["bliss_observed_reference"])
            ref_syn = bliss_reference(resp_hor, resp_ver, effect_0=e0, effect_inf=e_inf)
            assert np.max(np.abs(ref_syn - ref_indep)) < _CLOSED_FORM_ATOL

    def test_hsa_reference_and_deviation(self, cases):
        for case in cases:
            ch = np.array(case["conc_hor"])
            cv = np.array(case["conc_ver"])
            m = np.array(case["mean_matrix"])
            e0, e_inf = case["effect_0"], case["effect_inf"]
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
            dev_pkg = package_synergy_to_synfit_deviation(
                np.array(case["hsa_synergy"]), e0, e_inf
            )
            assert np.max(np.abs(dev_syn[interior] - dev_pkg[interior])) < _CLOSED_FORM_ATOL

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
            assert np.max(np.abs(ci_syn[interior] - ci_pkg[interior])) < _CLOSED_FORM_ATOL
            if case["name"] == "loewe_additive":
                assert np.max(np.abs(ci_pkg[interior] - 1.0)) < 1e-4
