"""Committed baselines for native-scale fit outputs (stage 1 behaviour lock)."""

import json
from pathlib import Path

import pytest

from scripts.update_response_baseline import BASELINE_PATH, compare_baselines, build_baseline


def test_native_outputs():
    """Native (s=1) fits match tests/data/response_baseline.json (pytest never regenerates)."""
    assert BASELINE_PATH.is_file(), "missing baseline; run `just update-response-baseline`"
    reference = json.loads(BASELINE_PATH.read_text())
    current = build_baseline()
    issues = compare_baselines(reference, current)
    assert not issues, "baseline drift:\n" + "\n".join(issues)
