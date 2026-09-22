"""Generate ZIP delta reference matrices via the GPL-3 ``synergy`` package (run offline).

    uv run --no-project --with synergy python scripts/zip_reference_values.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from synergy.combination import ZIP
from synergy.single import Hill

OUT = Path(__file__).resolve().parents[1] / "tests" / "data" / "zip_reference_values.json"

CH = np.array([0.0, 0.1, 0.3, 1.0, 3.0, 10.0])
CV = np.array([0.0, 0.2, 0.6, 2.0, 6.0, 20.0])
C50_HOR, C50_VER = 1.0, 2.0
HILL_HOR, HILL_VER = 1.5, 1.0
E0, EMAX = 1.0, 0.0


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
            out[i, j] = bliss[i, j] ** 1.12
    return out


def _antagonistic_surface(bliss: np.ndarray) -> np.ndarray:
    out = bliss.copy()
    for i in range(1, len(CV)):
        for j in range(1, len(CH)):
            out[i, j] = min(1.0, bliss[i, j] ** 0.97)
    return out


def _synergy_zip_delta(mean_matrix: np.ndarray) -> np.ndarray:
    ch_grid = np.tile(CH, (len(CV), 1))
    cv_grid = np.repeat(CV, len(CH)).reshape(len(CV), len(CH))
    d1 = ch_grid.ravel()
    d2 = cv_grid.ravel()
    drug1 = Hill(E0=E0, Emax=EMAX, h=HILL_HOR, C=C50_HOR)
    drug2 = Hill(E0=E0, Emax=EMAX, h=HILL_VER, C=C50_VER)
    model = ZIP(drug1_model=drug1, drug2_model=drug2)
    synergy = model.fit(d1, d2, mean_matrix.ravel())
    return synergy.reshape(len(CV), len(CH))


def _case(name: str, mean_matrix: np.ndarray) -> dict:
    synergy_delta = _synergy_zip_delta(mean_matrix)
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
        "synergy_delta": synergy_delta.tolist(),
    }


def main() -> None:
    bliss = _bliss_surface()
    cases = [
        _case("synergistic", _synergistic_surface(bliss)),
        _case("antagonistic", _antagonistic_surface(bliss)),
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"cases": cases}, indent=2) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
