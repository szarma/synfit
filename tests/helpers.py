"""Shared test utilities for synfit tests."""

from __future__ import annotations

import numpy as np

from synfit.synthetic import generate_matrix

PNG_MAGIC = b"\x89PNG"


def matrix_from_config(
    synergy_factor: float = 0.0,
    seed: int = 42,
    noise_model: str = "lognormal",
    noise_sigma_log: float = 0.05,
    noise_sigma: float = 0.03,
    has_zero: bool = True,
):
    """Build synthetic matrix replicates and concentration vectors (shared by matrix/zip tests)."""
    config = {
        "seed": seed,
        "horizontal_drug": {
            "c50": 5.0,
            "hill": 1.5,
            "concentration_series": {
                "initial_conc": 50.0,
                "fold_dilutions": 3.0,
                "length": 6,
                "has_zero": has_zero,
            },
        },
        "vertical_drug": {
            "c50": 8.0,
            "hill": 1.2,
            "concentration_series": {
                "initial_conc": 80.0,
                "fold_dilutions": 3.0,
                "length": 6,
                "has_zero": has_zero,
            },
        },
        "effect_0": 1.0,
        "effect_inf": 0.02,
        "n_replicates": 3,
        "noise_model": noise_model,
        "noise_sigma_log": noise_sigma_log,
        "noise_sigma": noise_sigma,
        "synergy_factor": synergy_factor,
    }
    return generate_matrix(config)
