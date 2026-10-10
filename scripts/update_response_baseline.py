#!/usr/bin/env python3
"""Compare or regenerate tests/data/response_baseline.json for native response fits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "tests" / "data" / "response_baseline.json"

# Allow ``from tests.response_scaling_helpers import ...`` when run as a script.
sys.path.insert(0, str(ROOT))

from tests.response_scaling_helpers import (  # noqa: E402
    SCALE_FIT_CASES,
    SNAPSHOT_CASE_IDS,
    case_by_id,
    compare_baseline_snapshots,
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


def compare_baselines(reference: dict, current: dict) -> list[str]:
    return compare_baseline_snapshots(reference, current)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Overwrite tests/data/response_baseline.json with current native fits.",
    )
    args = parser.parse_args()
    current = build_baseline()
    if args.write:
        BASELINE_PATH.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {BASELINE_PATH}")
        return 0
    if not BASELINE_PATH.is_file():
        print(f"Missing baseline at {BASELINE_PATH}; run with --write", file=sys.stderr)
        return 1
    reference = json.loads(BASELINE_PATH.read_text())
    issues = compare_baselines(reference, current)
    if issues:
        print("Baseline drift:")
        for line in issues:
            print(" ", line)
        return 1
    print("Baseline matches current native fits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
