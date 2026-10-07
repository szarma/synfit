import json

import numpy as np

from synfit.hill import hill_curve
from synfit.loewe import (
    _invert_hill,
    loewe_ci,
    loewe_reference,
    slope_mismatch_warning,
)
from synfit.synthetic import SCENARIOS_DIR, generate_matrix


def test_invert_hill_inside_range():
    c50, hill = 1.0, 1.0
    effect_0, effect_inf = 1.0, 0.0
    y = np.array([0.25, 0.5, 0.75])
    c = _invert_hill(y, c50, hill, effect_0, effect_inf)
    assert np.all(np.isfinite(c))
    assert np.all(c > 0)


def test_invert_hill_outside_range_nan():
    c50, hill = 1.0, 1.0
    effect_0, effect_inf = 1.0, 0.0
    y = np.array([-0.1, 1.5])
    c = _invert_hill(y, c50, hill, effect_0, effect_inf)
    assert np.all(np.isnan(c))


def test_invert_hill_zero_scale_all_nan():
    y = np.array([0.5])
    c = _invert_hill(y, 1.0, 1.0, 0.5, 0.5)
    assert np.isnan(c[0])


def test_loewe_ci_shape_and_finite_interior():
    conc_hor = np.array([0.0, 1.0, 10.0])
    conc_ver = np.array([0.0, 2.0])
    mean_m = np.array(
        [
            [1.0, 0.85, 0.5],
            [0.9, 0.7, 0.4],
        ],
        dtype=float,
    )
    ci = loewe_ci(
        conc_hor,
        conc_ver,
        c50_hor=5.0,
        c50_ver=8.0,
        hill_hor=1.5,
        hill_ver=1.2,
        mean_matrix=mean_m,
        effect_0=1.0,
        effect_inf=0.02,
    )
    assert ci.shape == mean_m.shape
    assert np.any(np.isfinite(ci))


def test_loewe_ci_matches_loop_reference():
    """Vectorized CI equals the previous element-wise definition."""
    conc_hor = np.array([0.0, 0.5, 2.0])
    conc_ver = np.array([0.0, 1.0])
    rng = np.random.default_rng(0)
    mean_m = 0.2 + 0.6 * rng.random((len(conc_ver), len(conc_hor)))

    c50_hor, c50_ver = 1.2, 0.8
    hill_hor, hill_ver = 1.1, 0.9
    effect_0, effect_inf = 1.0, 0.05

    ci_fast = loewe_ci(
        conc_hor,
        conc_ver,
        c50_hor,
        c50_ver,
        hill_hor,
        hill_ver,
        mean_m,
        effect_0,
        effect_inf,
    )

    n_ver, n_hor = mean_m.shape
    ci_slow = np.full((n_ver, n_hor), np.nan)
    for i in range(n_ver):
        for j in range(n_hor):
            # Mask zero-concentration edges — same rule as loewe_ci.
            if conc_hor[j] == 0 or conc_ver[i] == 0:
                continue
            e = mean_m[i, j]
            a_e = _invert_hill(np.array([e]), c50_hor, hill_hor, effect_0, effect_inf)[0]
            b_e = _invert_hill(np.array([e]), c50_ver, hill_ver, effect_0, effect_inf)[0]
            if np.isnan(a_e) or np.isnan(b_e) or a_e == 0 or b_e == 0:
                continue
            ci_slow[i, j] = conc_hor[j] / a_e + conc_ver[i] / b_e

    nan_fast = np.isnan(ci_fast)
    nan_slow = np.isnan(ci_slow)
    assert np.array_equal(nan_fast, nan_slow)
    assert np.allclose(ci_fast[~nan_fast], ci_slow[~nan_slow], equal_nan=True)


def test_loewe_ci_zero_conc_row_nan():
    conc_hor = np.array([0.0, 1.0])
    conc_ver = np.array([0.0, 1.0])
    mean_m = np.ones((2, 2))
    ci = loewe_ci(
        conc_hor,
        conc_ver,
        c50_hor=1.0,
        c50_ver=1.0,
        hill_hor=1.0,
        hill_ver=1.0,
        mean_matrix=mean_m,
        effect_0=1.0,
        effect_inf=0.0,
    )
    assert np.isnan(ci).all()


def test_loewe_ci_masks_zero_concentration_edges():
    # Use in-band responses at zero-conc cells — without the explicit mask those
    # would compute CI = 1 (a tautology). Verify they come back as NaN.
    conc_hor = np.array([0.0, 1.0, 4.0])
    conc_ver = np.array([0.0, 2.0, 6.0])
    c50_h, c50_v = 1.0, 2.0
    hill_h, hill_v = 1.0, 1.0
    effect_0, effect_inf = 1.0, 0.0
    from synfit.hill import hill_curve
    # Build a mean matrix with in-band values everywhere including the edges.
    mean_m = np.array([
        [1.0, 0.5, 0.2],
        [0.6, 0.35, 0.15],
        [0.3, 0.20, 0.10],
    ])
    ci = loewe_ci(
        conc_hor, conc_ver,
        c50_hor=c50_h, c50_ver=c50_v,
        hill_hor=hill_h, hill_ver=hill_v,
        mean_matrix=mean_m,
        effect_0=effect_0, effect_inf=effect_inf,
    )
    assert np.all(np.isnan(ci[0, :]))    # zero-conc_ver row
    assert np.all(np.isnan(ci[:, 0]))    # zero-conc_hor column
    assert np.any(np.isfinite(ci[1:, 1:]))  # interior has finite CI


def test_loewe_reference_shape_and_range():
    conc_hor = np.array([0.0, 0.5, 2.0, 10.0])
    conc_ver = np.array([0.0, 1.0, 5.0])
    effect_0, effect_inf = 1.0, 0.05
    ref = loewe_reference(
        conc_hor, conc_ver,
        c50_hor=1.0, c50_ver=2.0,
        hill_hor=1.0, hill_ver=1.2,
        effect_0=effect_0, effect_inf=effect_inf,
    )
    assert ref.shape == (3, 4)
    finite = ref[np.isfinite(ref)]
    assert np.all(finite >= effect_inf - 1e-6)
    assert np.all(finite <= effect_0 + 1e-6)


def test_loewe_reference_boundaries_match_single_drug():
    """Cells with one concentration = 0 must equal the other drug's Hill prediction."""
    conc_hor = np.array([0.0, 1.0, 10.0])
    conc_ver = np.array([0.0, 2.0, 8.0])
    ref = loewe_reference(
        conc_hor, conc_ver,
        c50_hor=1.5, c50_ver=3.0,
        hill_hor=1.1, hill_ver=0.9,
        effect_0=1.0, effect_inf=0.0,
    )
    # (0,0) = effect_0
    assert np.isclose(ref[0, 0], 1.0)
    # first row (conc_v = 0): equals horizontal-drug-alone Hill
    hor_alone = hill_curve(conc_hor, c50=1.5, hill=1.1, effect_0=1.0, effect_inf=0.0)
    assert np.allclose(ref[0, :], hor_alone)
    # first column (conc_h = 0): equals vertical-drug-alone Hill
    ver_alone = hill_curve(conc_ver, c50=3.0, hill=0.9, effect_0=1.0, effect_inf=0.0)
    assert np.allclose(ref[:, 0], ver_alone)


def test_ci_unity_on_loewe_additive_synthetic_dataset():
    """On a Loewe-additive synthetic matrix, `loewe_ci` must return ≈ 1
    everywhere on the interior (within assay noise).

    This is the Loewe counterpart to `matrix_independent` (which is
    Bliss-additive by construction): it proves CI deviates from 1 on
    Bliss-additive data not because of a bug but because the two null
    hypotheses genuinely disagree — on data that *is* Loewe-additive,
    CI collapses onto 1 as expected.

    Uses equal Hill slopes so Loewe additivity is theoretically unambiguous.
    """
    config = json.loads(
        (SCENARIOS_DIR / "matrix_loewe_additive" / "config.json").read_text()
    )
    replicates, conc_hor, conc_ver = generate_matrix(config)
    mean_matrix = np.mean(replicates, axis=0)

    h = config["horizontal_drug"]
    v = config["vertical_drug"]
    ci = loewe_ci(
        conc_hor, conc_ver,
        c50_hor=h["c50"], c50_ver=v["c50"],
        hill_hor=h["hill"], hill_ver=v["hill"],
        mean_matrix=mean_matrix,
        effect_0=config["effect_0"],
        effect_inf=config["effect_inf"],
    )

    # Interior cells (both concentrations > 0) where CI is defined.
    interior_mask = (
        (conc_hor[np.newaxis, :] > 0)
        & (conc_ver[:, np.newaxis] > 0)
        & np.isfinite(ci)
    )
    interior = ci[interior_mask]
    # Noise is lognormal σ=0.03 on three replicates → mean std on CI ≈ 0.05;
    # median should sit very close to 1, and no cell should wander far.
    assert interior.size >= 9, "need a meaningful interior sample"
    assert abs(np.median(interior) - 1.0) < 0.05
    assert np.mean(np.abs(interior - 1.0)) < 0.1
    assert np.all(np.abs(interior - 1.0) < 0.3)


def test_loewe_reference_satisfies_ci_equals_one():
    """For interior cells, solved E* must satisfy conc_h/A(E*) + conc_v/B(E*) = 1."""
    conc_hor = np.array([0.5, 2.0, 5.0])
    conc_ver = np.array([0.3, 1.5, 6.0])
    c50_h, c50_v = 1.2, 2.5
    hill_h, hill_v = 1.3, 0.8
    effect_0, effect_inf = 1.0, 0.05
    ref = loewe_reference(
        conc_hor, conc_ver,
        c50_hor=c50_h, c50_ver=c50_v,
        hill_hor=hill_h, hill_ver=hill_v,
        effect_0=effect_0, effect_inf=effect_inf,
    )
    for i, cv in enumerate(conc_ver):
        for j, ch in enumerate(conc_hor):
            e = ref[i, j]
            if not np.isfinite(e):
                continue
            a = _invert_hill(np.array([e]), c50_h, hill_h, effect_0, effect_inf)[0]
            b = _invert_hill(np.array([e]), c50_v, hill_v, effect_0, effect_inf)[0]
            ci = ch / a + cv / b
            assert np.isclose(ci, 1.0, atol=1e-5)


# 5-parameter (asymmetric) Hill support


def test_invert_hill_5p_roundtrip():
    # hill_curve evaluated then inverted must return the input concentration.
    c50, hill, S = 2.0, 1.5, 2.3
    effect_0, effect_inf = 1.0, 0.0
    conc = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 20.0])
    y = hill_curve(conc, c50=c50, hill=hill,
                   effect_0=effect_0, effect_inf=effect_inf, asymmetry=S)
    c_back = _invert_hill(y, c50, hill, effect_0, effect_inf, asymmetry=S)
    assert np.allclose(c_back, conc, rtol=1e-10, atol=1e-12)


def test_invert_hill_5p_activation_direction():
    # Same roundtrip with reversed asymptotes (activation assay).
    c50, hill, S = 1.0, 2.0, 0.5
    effect_0, effect_inf = 0.2, 1.2  # reversed: low at c=0, high at saturation
    conc = np.array([0.1, 1.0, 3.0, 10.0])
    y = hill_curve(conc, c50=c50, hill=hill,
                   effect_0=effect_0, effect_inf=effect_inf, asymmetry=S)
    c_back = _invert_hill(y, c50, hill, effect_0, effect_inf, asymmetry=S)
    assert np.allclose(c_back, conc, rtol=1e-10, atol=1e-12)


def test_invert_hill_none_asymmetry_equals_4p():
    # asymmetry=None should behave like asymmetry=1.0 (matches hill_curve).
    c50, hill = 1.0, 1.3
    effect_0, effect_inf = 1.0, 0.0
    y = np.array([0.2, 0.5, 0.8])
    c_4p = _invert_hill(y, c50, hill, effect_0, effect_inf)
    c_none = _invert_hill(y, c50, hill, effect_0, effect_inf, asymmetry=None)
    assert np.allclose(c_4p, c_none)


def test_loewe_ci_5p_equals_unity_on_sham_combination():
    # Sham combination: two drugs with identical 5p Hill parameters behave as a
    # single drug split into fractional doses. CI must equal 1 everywhere.
    c50, hill, S = 1.5, 1.8, 1.7
    effect_0, effect_inf = 1.0, 0.0
    conc_hor = np.array([0.0, 0.25, 0.5, 1.0, 2.0])
    conc_ver = np.array([0.0, 0.25, 0.5, 1.0, 2.0])
    # Sham surface: response at (c_h, c_v) is Hill(c_h + c_v)
    total = conc_hor[None, :] + conc_ver[:, None]
    mean = hill_curve(total, c50=c50, hill=hill,
                      effect_0=effect_0, effect_inf=effect_inf, asymmetry=S)
    ci = loewe_ci(
        conc_hor, conc_ver,
        c50_hor=c50, c50_ver=c50,
        hill_hor=hill, hill_ver=hill,
        mean_matrix=mean,
        effect_0=effect_0, effect_inf=effect_inf,
        asymmetry_hor=S, asymmetry_ver=S,
    )
    # Only interior cells are defined (edges have mean == effect_0 or pure single-drug)
    interior = ci[1:, 1:]
    assert np.all(np.isfinite(interior))
    assert np.allclose(interior, 1.0, atol=1e-8)


def test_loewe_reference_5p_sham_returns_single_hill():
    # For identical 5p params, the Loewe reference at (c_h, c_v) must equal
    # Hill(c_h + c_v) — same sham-combination identity as the 4p case.
    c50, hill, S = 2.0, 1.2, 2.5
    effect_0, effect_inf = 1.0, 0.0
    conc_hor = np.array([0.1, 0.5, 1.0, 3.0])
    conc_ver = np.array([0.1, 0.5, 1.0, 3.0])
    ref = loewe_reference(
        conc_hor, conc_ver,
        c50_hor=c50, c50_ver=c50,
        hill_hor=hill, hill_ver=hill,
        effect_0=effect_0, effect_inf=effect_inf,
        asymmetry_hor=S, asymmetry_ver=S,
    )
    total = conc_hor[None, :] + conc_ver[:, None]
    expected = hill_curve(total, c50=c50, hill=hill,
                          effect_0=effect_0, effect_inf=effect_inf, asymmetry=S)
    assert np.allclose(ref, expected, atol=1e-6)


def test_loewe_functions_default_to_4p():
    # Omitting asymmetry kwargs must reproduce the existing 4p results bit-for-bit.
    c50_h, hill_h = 1.0, 1.2
    c50_v, hill_v = 2.0, 0.9
    effect_0, effect_inf = 1.0, 0.0
    conc_hor = np.array([0.0, 0.3, 1.0, 3.0])
    conc_ver = np.array([0.0, 0.3, 1.0, 3.0])
    mean = np.array([
        [1.0, 0.85, 0.55, 0.30],
        [0.90, 0.70, 0.45, 0.25],
        [0.70, 0.50, 0.30, 0.15],
        [0.40, 0.25, 0.15, 0.05],
    ])
    ci_default = loewe_ci(conc_hor, conc_ver, c50_h, c50_v, hill_h, hill_v,
                          mean, effect_0, effect_inf)
    ci_explicit = loewe_ci(conc_hor, conc_ver, c50_h, c50_v, hill_h, hill_v,
                           mean, effect_0, effect_inf,
                           asymmetry_hor=1.0, asymmetry_ver=1.0)
    # NaNs live in the same cells, finite values are bit-equal
    assert np.array_equal(np.isnan(ci_default), np.isnan(ci_explicit))
    fin = ~np.isnan(ci_default)
    assert np.allclose(ci_default[fin], ci_explicit[fin], atol=0, rtol=0)


# Slope-mismatch warning thresholds


def test_slope_warning_none_for_matched_slopes():
    # Exact match, well above the neutral band -> no warning.
    assert slope_mismatch_warning(2.0, 2.0) is None


def test_slope_warning_suppressed_in_neutral_band():
    # Both slopes near 1 -> Loewe is well-behaved even with a 2x nominal ratio
    # within the band (0.9 and 1.05 differ by ~17%, still suppressed).
    assert slope_mismatch_warning(0.95, 1.05) is None
    assert slope_mismatch_warning(0.9, 1.1) is None


def test_slope_warning_yellow_at_moderate_ratio():
    w = slope_mismatch_warning(1.0, 1.5)
    assert w is not None
    assert w["severity"] == "yellow"
    assert w["ratio"] == 1.5
    assert "matched slopes" in w["message"]


def test_slope_warning_red_at_large_ratio():
    w = slope_mismatch_warning(0.5, 1.6)  # ratio = 3.2
    assert w is not None
    assert w["severity"] == "red"
    assert w["ratio"] > 3.0


def test_slope_warning_symmetric_in_argument_order():
    w1 = slope_mismatch_warning(0.5, 2.0)
    w2 = slope_mismatch_warning(2.0, 0.5)
    assert w1 is not None and w2 is not None
    assert w1["severity"] == w2["severity"]
    assert w1["ratio"] == w2["ratio"]


def test_slope_warning_rejects_invalid_slopes():
    assert slope_mismatch_warning(float("nan"), 1.0) is None
    assert slope_mismatch_warning(1.0, 0.0) is None
    assert slope_mismatch_warning(-1.0, 1.0) is None


def test_slope_warning_border_below_yellow():
    # Just under the ratio threshold -> no warning.
    assert slope_mismatch_warning(1.0, 1.49) is None


def test_slope_warning_tolerates_none_and_non_numeric():
    # API call sites pass values straight from fit-params dicts; a missing /
    # unparseable hill should silently suppress the warning, not raise.
    assert slope_mismatch_warning(None, 1.0) is None
    assert slope_mismatch_warning(1.0, None) is None
    assert slope_mismatch_warning(None, None) is None
    assert slope_mismatch_warning("not a number", 1.0) is None


def _load_sham_replicates():
    """Load matrix_loewe_sham/{rep1,rep2,rep3}.csv as a list of 2-D arrays."""
    bundle = SCENARIOS_DIR / "matrix_loewe_sham"
    config = json.loads((bundle / "config.json").read_text())
    csv_paths = sorted(bundle.glob("rep*.csv"))
    replicates = [np.loadtxt(p) for p in csv_paths]
    return config, replicates


def _edges_to_dataframes(replicates, conc_hor, conc_ver):
    """Replicate the production matrix path's edge-slice extraction.

    Mirrors ``MatrixFit._edge_slice`` and ``adapters.build_marginal_dataframes``:
    horizontal edge = first row of each replicate (vertical conc = 0);
    vertical edge   = first column of each replicate (horizontal conc = 0).
    """
    rows_h, rows_v = [], []
    for rep_ix, rep in enumerate(replicates):
        for j, c in enumerate(conc_hor):
            rows_h.append({"concentration": float(c), "y": float(rep[0, j]), "replicate": str(rep_ix)})
        for i, c in enumerate(conc_ver):
            rows_v.append({"concentration": float(c), "y": float(rep[i, 0]), "replicate": str(rep_ix)})
    import pandas as pd
    return pd.DataFrame(rows_h), pd.DataFrame(rows_v)


def test_loewe_sham_returns_unity():
    """Loewe sham combination — the same drug combined with itself.

    Strictest correctness check for the combination index: a drug
    combined with itself has by construction Loewe CI = 1 at every
    cell, modulo sampling noise. Any non-trivial deviation indicates a
    bug in either the synthetic generator (``loewe_reference``) or
    the inversion pipeline (``loewe_ci``).

    Mirrors the production matrix flow:
      1. Generate sham replicates from ``matrix_loewe_sham`` config.
      2. Estimate per-drug Hill marginals from the single-agent edges
         via ``JointMarginalFit`` (the same fitter the SPA uses for
         matrix datasets — *not* ``MatrixFit``).
      3. Compute Loewe CI from the fitted marginals + the mean matrix
         using ``loewe_ci``, the same call ``share_payload.py`` makes.
      4. Assert CI ≈ 1.0 in cells where inversion is well-defined.
    """
    from synfit.joint_marginal import JointMarginalFit
    from synfit.synthetic import calculate_concentration_series
    import pandas as pd  # noqa: F401  (re-export so np is available too)

    config, replicates = _load_sham_replicates()
    h = config["horizontal_drug"]
    v = config["vertical_drug"]
    conc_hor = calculate_concentration_series(
        initial_conc=h["concentration_series"]["initial_conc"],
        fold_dilutions=h["concentration_series"]["fold_dilutions"],
        length=h["concentration_series"]["length"],
        has_zero=h["concentration_series"].get("has_zero", False),
    )
    conc_ver = calculate_concentration_series(
        initial_conc=v["concentration_series"]["initial_conc"],
        fold_dilutions=v["concentration_series"]["fold_dilutions"],
        length=v["concentration_series"]["length"],
        has_zero=v["concentration_series"].get("has_zero", False),
    )

    data_h, data_v = _edges_to_dataframes(replicates, conc_hor, conc_ver)
    fit = JointMarginalFit(
        data_h, data_v,
        direction_a=h["direction"],
        direction_b=v["direction"],
        noise={"kind": "lognormal"},
    ).fit()

    mean_matrix = np.nanmean(np.stack(replicates), axis=0)
    ci = loewe_ci(
        conc_hor, conc_ver,
        c50_hor=fit.drug_a.c50, c50_ver=fit.drug_b.c50,
        hill_hor=fit.drug_a.hill, hill_ver=fit.drug_b.hill,
        mean_matrix=mean_matrix,
        effect_0=fit.top, effect_inf=fit.bottom,
    )

    valid = np.isfinite(ci)
    # For a sham, the marginals + edges share signal so most cells should
    # invert. Require at least half of the matrix (excluding the
    # zero-concentration row & column where CI is undefined) to be valid.
    n_inverttable_cells = (len(conc_hor) - 1) * (len(conc_ver) - 1)
    assert valid.sum() >= n_inverttable_cells // 2, (
        f"too few invertible cells: {valid.sum()} / "
        f"{n_inverttable_cells} interior cells; sham generator or inversion is broken"
    )

    valid_ci = ci[valid]
    median_ci = float(np.median(valid_ci))
    # Two-tier assertion. The median pins the systematic direction (any
    # implementation bug shifts the whole distribution; a working
    # round-trip gives a tight median near 1.0). The per-cell bound is
    # looser because Loewe inversion amplifies noise near the asymptotes
    # — cells where the observed response is close to ``top`` or
    # ``bottom`` have small dy/dC, so small noise on response becomes
    # large noise on the inverted dose.
    assert abs(median_ci - 1.0) < 0.03, (
        f"Loewe CI median for a sham should be within 3% of 1.0; "
        f"got median={median_ci:.4f}, min={valid_ci.min():.4f}, "
        f"max={valid_ci.max():.4f}. Either loewe_reference (generator) "
        f"or loewe_ci (metric) has a systematic bug."
    )
    np.testing.assert_allclose(
        valid_ci, 1.0, rtol=0.20,
        err_msg=(
            "Loewe CI on a sham combination has cells > 20% off unity — "
            f"min={valid_ci.min():.4f}, max={valid_ci.max():.4f}, "
            f"median={median_ci:.4f}. The median is well-pinned but at "
            "least one cell is far. This usually indicates an inversion "
            "instability near the asymptote, not a metric bug."
        ),
    )
