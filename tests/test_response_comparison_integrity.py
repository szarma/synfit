"""The comparison helpers must flag deliberately corrupted results."""

import copy

import numpy as np
import pytest

from tests.response_scaling_helpers import (
    compare_baseline_snapshots,
    compare_single_results,
    run_scale_case,
    case_by_id,
    extract_snapshot_record,
    assert_param_cov_equivariant,
    covariance_issues,
)
from tests.test_response_snapshots import build_baseline


def _native_and_scaled_single(s: float = 1e3):
    case = case_by_id("single_gaussian_constant_noisy")
    native = run_scale_case(case)
    from tests.response_scaling_helpers import fit_scaled_case

    scaled = fit_scaled_case(case, s)
    return native, scaled, s


def test_compare_single_results_flags_covariance_magnitude_corruption():
    native, scaled, s = _native_and_scaled_single()
    assert native.param_cov is not None and scaled.param_cov is not None
    corrupt = copy.deepcopy(scaled)
    corrupt.param_cov = corrupt.param_cov * 1e8
    with pytest.raises(AssertionError):
        compare_single_results(native, corrupt, s, check_covariance=True)


def test_compare_single_results_flags_param_cov_presence_mismatch():
    native, scaled, s = _native_and_scaled_single()
    corrupt = copy.deepcopy(scaled)
    corrupt.param_cov = None
    with pytest.raises(AssertionError):
        assert_param_cov_equivariant(native, corrupt, s)


def test_baseline_compare_flags_param_ci_cleared():
    reference = build_baseline()
    current = copy.deepcopy(reference)
    case_id = next(iter(current["cases"]))
    current["cases"][case_id]["param_ci"] = None
    issues = compare_baseline_snapshots(reference, current)
    assert any("param_ci presence" in m for m in issues)


def test_baseline_compare_flags_predict_ci_cleared():
    reference = build_baseline()
    current = copy.deepcopy(reference)
    case_id = next(iter(current["cases"]))
    current["cases"][case_id]["predict_ci"] = None
    issues = compare_baseline_snapshots(reference, current)
    assert any("predict_ci presence" in m for m in issues)


def test_baseline_compare_flags_nested_param_cov_row_change():
    reference = build_baseline()
    current = copy.deepcopy(reference)
    for case_id, rec in current["cases"].items():
        cov = rec["to_dict"].get("param_cov")
        if cov is not None and len(cov) > 1:
            # 0.1 in units of the reference SE product — well past the 2 % tolerance
            cov[1][0] += 0.1 * (cov[0][0] * cov[1][1]) ** 0.5
            break
    else:
        pytest.skip("no matrix-style param_cov in baseline")
    issues = compare_baseline_snapshots(reference, current)
    assert any("param_cov" in m for m in issues)


def test_baseline_compare_flags_covariance_scaled():
    reference = build_baseline()
    current = copy.deepcopy(reference)
    for case_id, rec in current["cases"].items():
        cov = rec["to_dict"].get("param_cov")
        if cov is not None:
            rec["to_dict"]["param_cov"] = (np.asarray(cov) * 1e8).tolist()
            break
    else:
        pytest.skip("no param_cov in baseline")
    issues = compare_baseline_snapshots(reference, current)
    assert any("param_cov" in m for m in issues)


def test_covariance_corruption_in_a_bound_active_row_is_flagged():
    """A zero reference SE must not hide corruption in that row."""
    ref = np.diag([1.0, 0.0])
    assert covariance_issues(ref, np.diag([1.0, 1e9]))
    assert not covariance_issues(ref, np.diag([1.0, 0.0]))


def test_snapshot_covariance_uses_one_tolerance():
    """A 1 % change in a variance is within the 2 % covariance tolerance; 5 % is not."""
    reference = build_baseline()
    for factor, flagged in ((1.01, False), (1.05, True)):
        current = copy.deepcopy(reference)
        cov = current["cases"]["single_gaussian_constant_noisy"]["to_dict"]["param_cov"]
        cov[1][1] *= factor
        issues = compare_baseline_snapshots(reference, current)
        assert bool(issues) is flagged, issues


@pytest.mark.parametrize(
    "field, value",
    [
        ("sigma", None),
        ("log_likelihood", None),
        ("aic", None),
        ("bic", None),
        ("rss", None),
        ("r2", None),
        ("success", False),
        ("response_scale", 1e50),
    ],
)
def test_compare_single_results_flags_dropped_metric_failure_and_scale(field, value):
    native, scaled, s = _native_and_scaled_single(1e-2)
    corrupt = copy.deepcopy(scaled)
    setattr(corrupt, field, value)
    with pytest.raises(AssertionError):
        compare_single_results(native, corrupt, s)
