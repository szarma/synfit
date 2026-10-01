"""Generate ZIP reference matrices via the GPL-3 ``synergy`` package (run offline).

    uv run --no-project \\
        --with synergy==1.0.0 \\
        --with numpy==2.4.3 \\
        --with scipy==1.17.1 \\
        python scripts/zip_reference_values.py
"""
from __future__ import annotations

import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
from scipy import __version__ as scipy_version
from synergy.combination import ZIP
from synergy.single import Hill

OUT = Path(__file__).resolve().parents[1] / "tests" / "data" / "zip_reference_values.json"

REQUIRED_VERSIONS = {
    "synergy": "1.0.0",
    "numpy": "2.4.3",
    "scipy": "1.17.1",
}

CH = np.array([0.0, 0.1, 0.3, 1.0, 3.0, 10.0])
CV = np.array([0.0, 0.2, 0.6, 2.0, 6.0, 20.0])
C50_HOR, C50_VER = 1.0, 2.0
HILL_HOR, HILL_VER = 1.5, 1.0
E0, EMAX = 1.0, 0.0


def _check_required_versions() -> None:
    import numpy

    found = {
        "synergy": importlib.metadata.version("synergy"),
        "numpy": numpy.__version__,
        "scipy": scipy_version,
    }
    mismatches = [
        f"{name}: need {want}, got {found[name]}"
        for name, want in REQUIRED_VERSIONS.items()
        if found[name] != want
    ]
    if mismatches:
        print("Refusing to regenerate with mismatched dependency versions:", file=sys.stderr)
        for line in mismatches:
            print(f"  {line}", file=sys.stderr)
        raise SystemExit(1)


def _hill_survival(conc: np.ndarray, c50: float, hill: float) -> np.ndarray:
    c = np.asarray(conc, dtype=float)
    return E0 + (EMAX - E0) * c**hill / (c50**hill + c**hill)


def _bliss_surface() -> np.ndarray:
    fh = _hill_survival(CH, C50_HOR, HILL_HOR)
    fv = _hill_survival(CV, C50_VER, HILL_VER)
    return np.outer(fv, fh)


def _synergistic_surface(bliss: np.ndarray) -> np.ndarray:
    out = bliss.copy()
    for i in range(1, len(CV)):
        for j in range(1, len(CH)):
            out[i, j] = bliss[i, j] ** 1.25
    return out


def _potentiated_surface() -> np.ndarray:
    """Vertical drug's c50 drops up to 4x as the horizontal dose rises."""
    out = np.empty((len(CV), len(CH)))
    for j, ch in enumerate(CH):
        c50_ver = C50_VER / (1.0 + 3.0 * ch / (ch + 1.0))
        out[:, j] = _hill_survival(CV, c50_ver, HILL_VER) * _hill_survival(np.array([ch]), C50_HOR, HILL_HOR)[0]
    return out


def _synergy_zip_outputs(mean_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ch_grid = np.tile(CH, (len(CV), 1))
    cv_grid = np.repeat(CV, len(CH)).reshape(len(CV), len(CH))
    d1 = ch_grid.ravel()
    d2 = cv_grid.ravel()
    drug1 = Hill(E0=E0, Emax=EMAX, h=HILL_HOR, C=C50_HOR)
    drug2 = Hill(E0=E0, Emax=EMAX, h=HILL_VER, C=C50_VER)
    model = ZIP(drug1_model=drug1, drug2_model=drug2)
    synergy = model.fit(d1, d2, mean_matrix.ravel())
    reference = model.E_reference(d1, d2)
    fitted = reference - synergy
    n_ver, n_hor = mean_matrix.shape
    return (
        reference.reshape(n_ver, n_hor),
        fitted.reshape(n_ver, n_hor),
        synergy.reshape(n_ver, n_hor),
    )


def _case(name: str, mean_matrix: np.ndarray) -> dict:
    reference, fitted, synergy_delta = _synergy_zip_outputs(mean_matrix)
    return {
        "name": name,
        "conc_hor": CH.tolist(),
        "conc_ver": CV.tolist(),
        "c50_hor": C50_HOR,
        "c50_ver": C50_VER,
        "hill_hor": HILL_HOR,
        "hill_ver": HILL_VER,
        "effect_0": E0,
        "effect_inf": EMAX,
        "mean_matrix": mean_matrix.tolist(),
        "zip_reference": reference.tolist(),
        "zip_fitted": fitted.tolist(),
        "synergy_delta": synergy_delta.tolist(),
    }


def main() -> None:
    _check_required_versions()
    bliss = _bliss_surface()
    cases = [
        _case("synergistic", _synergistic_surface(bliss)),
        _case("potentiated", _potentiated_surface()),
    ]
    payload = {
        "synergy_package_version": REQUIRED_VERSIONS["synergy"],
        "numpy_version": REQUIRED_VERSIONS["numpy"],
        "scipy_version": REQUIRED_VERSIONS["scipy"],
        "cases": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
