#!/usr/bin/env python3
"""Compare or regenerate tests/data/response_baseline.json for native response fits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "tests" / "data" / "response_baseline.json"

# Allow ``from tests.response_scaling_helpers import ...`` when run as a script.
sys.path.insert(0, str(ROOT))

from tests.response_scaling_helpers import (  # noqa: E402
    SCALE_FIT_CASES,
    SNAPSHOT_CASE_IDS,
    case_by_id,
    extract_snapshot_record,
    run_scale_case,
    runtime_versions,
)


def build_baseline() -> dict:
    cases = {}
    for case_id in SNAPSHOT_CASE_IDS:
        case = case_by_id(case_id)
        result = run_scale_case(case)
        cases[case_id] = extract_snapshot_record(case, result)
    return {"versions": runtime_versions(), "cases": cases}


def _compare_scalar(name: str, ref, cur, rtol: float, atol: float) -> list[str]:
    if ref is None and cur is None:
        return []
    if ref is None or cur is None:
        return [f"{name}: {ref!r} -> {cur!r}"]
    if isinstance(ref, bool) or isinstance(ref, str):
        if ref != cur:
            return [f"{name}: {ref!r} -> {cur!r}"]
        return []
    if isinstance(ref, (int, float)) and isinstance(cur, (int, float)):
        if not np.isclose(ref, cur, rtol=rtol, atol=atol):
            return [f"{name}: {ref} -> {cur}"]
        return []
    return []


def _compare_dict(ref: dict, cur: dict, prefix: str = "") -> list[str]:
    issues: list[str] = []
    if set(ref.keys()) != set(cur.keys()):
        issues.append(f"{prefix}keys {sorted(ref.keys())} vs {sorted(cur.keys())}")
        return issues
    for key in ref:
        path = f"{prefix}{key}"
        rv, cv = ref[key], cur[key]
        if isinstance(rv, dict) and isinstance(cv, dict):
            issues.extend(_compare_dict(rv, cv, path + "."))
        elif isinstance(rv, list) and isinstance(cv, list):
            if len(rv) != len(cv):
                issues.append(f"{path}: len {len(rv)} vs {len(cv)}")
            else:
                for i, (a, b) in enumerate(zip(rv, cv)):
                    issues.extend(_compare_scalar(f"{path}[{i}]", a, b, 5e-4, 1e-7))
        else:
            # atol only guards values that are legitimately ~0; real magnitudes
            # are held to rtol (variance coefficients can be ~1e-4).
            rtol, atol = 5e-4, 1e-7
            if key in ("r2",):
                rtol, atol = 0.0, 1e-5
            if key in ("log_likelihood", "aic", "bic"):
                rtol, atol = 0.0, 1e-4
            issues.extend(_compare_scalar(path, rv, cv, rtol, atol))
    return issues


def _compare_cov(ref: list, cur: list) -> list[str]:
    r = np.asarray(ref, dtype=float)
    c = np.asarray(cur, dtype=float)
    se_r = np.sqrt(np.maximum(np.diag(r), 0.0))
    se_c = np.sqrt(np.maximum(np.diag(c), 0.0))
    std_r = r / np.outer(se_r, se_r)
    std_c = c / np.outer(se_c, se_c)
    if not np.allclose(std_c, std_r, rtol=0.02, atol=0.02, equal_nan=True):
        return ["param_cov: standardised entries differ >2%"]
    return []


def _compare_ci_widths(ref: dict, cur: dict) -> list[str]:
    issues = []
    conc = np.asarray(ref["conc"])
    w_ref = np.asarray(ref["hi"]) - np.asarray(ref["lo"])
    w_cur = np.asarray(cur["hi"]) - np.asarray(cur["lo"])
    rel = np.abs(w_cur - w_ref) / np.maximum(w_ref, 1e-12)
    if np.any(rel > 0.01):
        issues.append(f"predict_ci widths: max rel diff {float(np.max(rel)):.4g}")
    return issues


def compare_baselines(reference: dict, current: dict) -> list[str]:
    issues: list[str] = []
    ref_cases = reference["cases"]
    cur_cases = current["cases"]
    if set(ref_cases) != set(cur_cases):
        issues.append(f"case ids differ: {set(ref_cases)} vs {set(cur_cases)}")
        return issues
    for case_id in sorted(ref_cases):
        prefix = f"{case_id}: "
        rc, cc = ref_cases[case_id], cur_cases[case_id]
        if rc.get("fingerprint") != cc.get("fingerprint"):
            issues.append(prefix + "input fingerprint changed")
        issues.extend(_compare_dict(rc["to_dict"], cc["to_dict"], prefix + "to_dict."))
        pc_r, pc_c = rc.get("param_ci"), cc.get("param_ci")
        if pc_r is not None and pc_c is not None:
            for name in pc_r:
                lo_r, hi_r = pc_r[name]
                lo_c, hi_c = pc_c[name]
                w_r, w_c = hi_r - lo_r, hi_c - lo_c
                if abs(w_c - w_r) / max(w_r, 1e-12) > 0.01:
                    issues.append(prefix + f"param_ci width {name} differs >1%")
        pci_r, pci_c = rc.get("predict_ci"), cc.get("predict_ci")
        if pci_r and pci_c:
            issues.extend([prefix + m for m in _compare_ci_widths(pci_r, pci_c)])
        cov_r = rc["to_dict"].get("param_cov")
        cov_c = cc["to_dict"].get("param_cov")
        if cov_r is not None and cov_c is not None:
            issues.extend([prefix + m for m in _compare_cov(cov_r, cov_c)])
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Regenerate the committed baseline JSON.",
    )
    args = parser.parse_args()
    current = build_baseline()
    if args.write:
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_PATH.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {BASELINE_PATH}")
        return 0

    if not BASELINE_PATH.is_file():
        print(f"Missing baseline: {BASELINE_PATH} (run with --write)", file=sys.stderr)
        return 1
    reference = json.loads(BASELINE_PATH.read_text())
    issues = compare_baselines(reference, current)
    if reference.get("versions") != current.get("versions"):
        print(f"note: baseline recorded with {reference.get('versions')}, "
              f"running {current.get('versions')}")
    if issues:
        for line in issues:
            print(line)
        return 1
    print("OK — baseline matches current native fits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
