"""Shared fixtures and comparisons for response-scale equivariance tests."""

from __future__ import annotations

import copy
import hashlib
import sys
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from synfit.data import FitConfig, FitResult
from synfit.joint_marginal import JointMarginalFit, JointMarginalResult
from synfit.matrix import MatrixFit, MatrixFitResult
from synfit.noise import (
    GaussianConstant,
    GaussianLinear,
    GaussianQuadratic,
    Lognormal,
    NoiseSpec,
)
from synfit.param_roles import is_variance_param
from synfit.single import SingleDrugFit, SingleDrugFitWithError
from synfit.synthetic import scenario_dir

from tests.helpers import matrix_from_config
from tests.test_covariance_scaling import _two_drug_frames
from tests.test_joint_marginal import _two_drug_hetero_data
from tests.test_variance_anchor import _hetero_curve

CONC_GRID = np.array([0.0, 0.01, 0.1, 1.0, 10.0, 100.0], dtype=float)

_FIT_CACHE: dict[str, Any] = {}


def single_drug_noisy_data() -> pd.DataFrame:
    name = "single_drug_noisy"
    return pd.read_csv(scenario_dir(name) / f"{name}.csv")


def single_drug_activation_data() -> pd.DataFrame:
    name = "single_drug_activation"
    return pd.read_csv(scenario_dir(name) / f"{name}.csv")


def scale_dataframe(df: pd.DataFrame, s: float) -> pd.DataFrame:
    out = df.copy()
    out["y"] = df["y"].values * s
    if "y_err" in df.columns:
        out["y_err"] = df["y_err"].values * s
    return out


def scale_matrix_replicates(replicates: list[np.ndarray], s: float) -> list[np.ndarray]:
    return [np.asarray(r, dtype=float) * s for r in replicates]


@dataclass(frozen=True)
class ScaleFitCase:
    """One row in the response-scaling regression matrix."""

    case_id: str
    kind: str  # single | joint | matrix | single_with_error
    noise: NoiseSpec | str
    data_builder: Callable[[], Any]
    fit_config: FitConfig | None = None
    direction: str | None = None  # activation override for single-drug scenarios

    def build_data(self) -> Any:
        return self.data_builder()

    def fit_config_for(self) -> FitConfig:
        if self.fit_config is not None:
            return copy.deepcopy(self.fit_config)
        if isinstance(self.noise, str):
            if self.noise in ("lognormal", "gaussian_constant"):
                return FitConfig(noise=self.noise) if self.noise == "lognormal" else FitConfig()
            return FitConfig(noise=self.noise)
        if self.direction == "activation":
            return FitConfig(noise=self.noise, direction="activation")
        return FitConfig(noise=self.noise)


def _matrix_data():
    reps, ch, cv = matrix_from_config(seed=42, noise_model="lognormal")
    return reps, ch, cv


def _single_with_error_data():
    return single_drug_noisy_data().assign(y_err=0.1)


SCALE_FIT_CASES: list[ScaleFitCase] = [
    ScaleFitCase("single_gaussian_constant_noisy", "single", GaussianConstant(), single_drug_noisy_data),
    ScaleFitCase("single_lognormal_noisy", "single", Lognormal(), single_drug_noisy_data),
    ScaleFitCase(
        "single_gaussian_linear_hetero",
        "single",
        GaussianLinear(),
        lambda: _hetero_curve(c=0.0),
    ),
    ScaleFitCase(
        "single_gaussian_quadratic_hetero",
        "single",
        GaussianQuadratic(),
        lambda: _hetero_curve(c=0.03),
    ),
    ScaleFitCase("joint_gaussian_constant", "joint", "gaussian_constant", lambda: _two_drug_frames()),
    ScaleFitCase("joint_lognormal", "joint", "lognormal", lambda: _two_drug_frames()),
    ScaleFitCase(
        "joint_gaussian_linear",
        "joint",
        GaussianLinear(),
        lambda: _two_drug_hetero_data(a=1e-4, b=0.02),
    ),
    ScaleFitCase(
        "joint_gaussian_quadratic",
        "joint",
        GaussianQuadratic(),
        lambda: _two_drug_hetero_data(a=1e-4, b=0.02, c=0.01),
    ),
    ScaleFitCase("matrix_gaussian", "matrix", "gaussian", _matrix_data),
    ScaleFitCase("matrix_lognormal", "matrix", "lognormal", _matrix_data),
    ScaleFitCase(
        "single_error_gaussian_noisy",
        "single_with_error",
        GaussianConstant(),
        _single_with_error_data,
    ),
    ScaleFitCase(
        "single_error_lognormal_noisy",
        "single_with_error",
        Lognormal(),
        _single_with_error_data,
    ),
    ScaleFitCase(
        "single_lognormal_activation",
        "single",
        Lognormal(),
        single_drug_activation_data,
        direction="activation",
    ),
]


SNAPSHOT_CASE_IDS = [c.case_id for c in SCALE_FIT_CASES if c.case_id != "single_lognormal_activation"] + [
    "single_lognormal_activation",
]


def run_scale_case(case: ScaleFitCase) -> Any:
    """Fit once at native scale; result is cached by ``case_id``."""
    if case.case_id in _FIT_CACHE:
        return _FIT_CACHE[case.case_id]

    data = case.build_data()
    if case.kind == "single":
        cfg = case.fit_config_for()
        result = SingleDrugFit(data, cfg).fit()
    elif case.kind == "single_with_error":
        cfg = case.fit_config_for()
        result = SingleDrugFitWithError(data, cfg).fit()
    elif case.kind == "joint":
        data_a, data_b = data
        result = JointMarginalFit(data_a, data_b, noise=case.noise).fit()
    elif case.kind == "matrix":
        reps, ch, cv = data
        em = "lognormal" if case.noise == "lognormal" else "gaussian"
        result = MatrixFit(reps, ch, cv, error_model=em).fit()
    else:
        raise ValueError(f"unknown kind {case.kind}")

    _FIT_CACHE[case.case_id] = result
    return result


def case_by_id(case_id: str) -> ScaleFitCase:
    for c in SCALE_FIT_CASES:
        if c.case_id == case_id:
            return c
    raise KeyError(case_id)


def _param_scale_factor(name: str, s: float) -> float:
    if name in ("effect_0", "effect_inf", "top", "bottom"):
        return s
    if name.startswith("var_"):
        key = name[4:]
        if key == "a":
            return s * s
        if key == "b":
            return s
        if key == "c":
            return 1.0
    return 1.0


def _sigma_scale_factor(error_model: str, s: float) -> float:
    return 1.0 if error_model == "lognormal" else s


def _variance_entry_scale(key: str, s: float) -> float:
    if key == "a":
        return s * s
    if key == "b":
        return s
    return 1.0


def n_likelihood_admitted(
    y: np.ndarray,
    y_pred: np.ndarray,
    noise: NoiseSpec | str,
    *,
    mask: np.ndarray | None = None,
    y_err: np.ndarray | None = None,
) -> int:
    """Count observations entering ``log_prob`` (mask + finite residuals)."""
    from synfit.noise import _coerce_noise_arg, _transform

    spec = _coerce_noise_arg(noise)
    y = np.asarray(y, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if isinstance(spec, Lognormal) or isinstance(
        spec, (GaussianConstant, GaussianLinear, GaussianQuadratic)
    ):
        y_t, y_pred_t = _transform(y, y_pred, spec)
        residuals = (y_t - y_pred_t)[mask]
        return int(np.sum(np.isfinite(residuals)))
    # compound: finite y and mu on original scale
    y_m = y[mask]
    mu_m = y_pred[mask]
    return int(np.sum(np.isfinite(y_m) & np.isfinite(mu_m)))


def covariance_issues(ref: np.ndarray, cur: np.ndarray, *, tol: float = 2e-2) -> list[str]:
    """Differences between two covariances in units of the reference SEs.

    Rows the reference fixes at zero (bound-active parameters) must be zero in
    ``cur`` too; standardising them by a zero SE would hide any corruption.
    """
    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)
    if ref.shape != cur.shape:
        return [f"param_cov shape {ref.shape} vs {cur.shape}"]
    ref_se = np.sqrt(np.maximum(np.diag(ref), 0.0))
    zero = ref_se == 0
    issues = []
    if np.any(cur[zero, :] != 0) or np.any(cur[:, zero] != 0):
        issues.append("param_cov: non-zero entries where the reference row is zero")
    keep = ~zero
    denom = np.outer(ref_se[keep], ref_se[keep])
    diff = np.abs(cur[np.ix_(keep, keep)] - ref[np.ix_(keep, keep)]) / denom
    if np.any(~np.isfinite(diff)) or np.any(diff > tol):
        issues.append(f"param_cov: max standardised difference {np.nanmax(diff):.3g}")
    return issues


def assert_param_cov_equivariant(
    native: FitResult,
    scaled: FitResult,
    s: float,
    *,
    tol: float = 2e-2,
) -> None:
    """Scaled covariance, back-transformed to native units, matches the native one."""
    if (native.param_cov is None) ^ (scaled.param_cov is None):
        raise AssertionError(
            f"param_cov presence mismatch: native={native.param_cov is not None}, "
            f"scaled={scaled.param_cov is not None}"
        )
    if native.param_cov is None:
        return
    assert native.param_names is not None and scaled.param_names is not None
    fac = np.array([_param_scale_factor(n, s) for n in native.param_names], dtype=float)
    issues = covariance_issues(native.param_cov, scaled.param_cov / np.outer(fac, fac), tol=tol)
    assert not issues, "; ".join(issues)


def _compare_drug_shape_and_asymptotes(
    native: FitResult,
    scaled: FitResult,
    s: float,
    *,
    rtol: float = 1e-3,
) -> None:
    """Per-drug fields embedded in joint/matrix results (no likelihood metrics)."""
    np.testing.assert_allclose(scaled.log_c50, native.log_c50, rtol=rtol)
    np.testing.assert_allclose(scaled.hill, native.hill, rtol=rtol)
    if native.asymmetry is not None:
        np.testing.assert_allclose(scaled.asymmetry, native.asymmetry, rtol=rtol)
    np.testing.assert_allclose(scaled.effect_0, native.effect_0 * s, rtol=rtol)
    np.testing.assert_allclose(scaled.effect_inf, native.effect_inf * s, rtol=rtol)
    np.testing.assert_allclose(scaled.c50, native.c50, rtol=rtol)


def _assert_same_presence(native, scaled, fields) -> None:
    """A metric missing on one side only is a difference, not something to skip."""
    for name in fields:
        assert (getattr(native, name) is None) == (getattr(scaled, name) is None), (
            f"{name} presence differs: native={getattr(native, name)!r}, "
            f"scaled={getattr(scaled, name)!r}"
        )


def _assert_success_and_scale(native, scaled, s: float) -> None:
    assert native.success, f"native fit failed: {native.message}"
    assert scaled.success, f"scaled fit failed: {scaled.message}"
    np.testing.assert_allclose(scaled.response_scale, native.response_scale * s, rtol=1e-9)


def compare_single_results(
    native: FitResult,
    scaled: FitResult,
    s: float,
    *,
    rtol: float = 1e-3,
    cov_rtol: float = 2e-2,
    check_covariance: bool = True,
) -> None:
    """Assert scaled fit matches native under response rescaling conventions."""
    _assert_success_and_scale(native, scaled, s)
    _assert_same_presence(
        native, scaled,
        ("sigma", "variance_params", "log_likelihood", "aic", "bic", "rss", "r2"),
    )
    np.testing.assert_allclose(scaled.log_c50, native.log_c50, rtol=rtol)
    np.testing.assert_allclose(scaled.hill, native.hill, rtol=rtol)
    if native.asymmetry is not None:
        np.testing.assert_allclose(scaled.asymmetry, native.asymmetry, rtol=rtol)
    np.testing.assert_allclose(scaled.effect_0, native.effect_0 * s, rtol=rtol)
    np.testing.assert_allclose(scaled.effect_inf, native.effect_inf * s, rtol=rtol)
    np.testing.assert_allclose(scaled.c50, native.c50, rtol=rtol)

    if native.sigma is not None:
        sf = _sigma_scale_factor(native.error_model, s)
        np.testing.assert_allclose(scaled.sigma, native.sigma * sf, rtol=rtol)

    if native.variance_params:
        assert scaled.variance_params is not None
        for key in native.variance_params:
            fac = _variance_entry_scale(key, s)
            np.testing.assert_allclose(
                scaled.variance_params[key],
                native.variance_params[key] * fac,
                rtol=rtol,
            )

    n = native.n_valid
    if native.log_likelihood is not None:
        np.testing.assert_allclose(
            scaled.log_likelihood,
            native.log_likelihood - n * np.log(s),
            rtol=1e-4,
            atol=1e-4,
        )
    if native.aic is not None:
        np.testing.assert_allclose(
            scaled.aic,
            native.aic + 2 * n * np.log(s),
            rtol=1e-4,
            atol=1e-4,
        )
    if native.bic is not None:
        np.testing.assert_allclose(
            scaled.bic,
            native.bic + 2 * n * np.log(s),
            rtol=1e-4,
            atol=1e-4,
        )

    if native.rss is not None:
        np.testing.assert_allclose(scaled.rss, native.rss * s * s, rtol=rtol)
    if native.r2 is not None:
        np.testing.assert_allclose(scaled.r2, native.r2, atol=1e-5)

    assert scaled.n_valid == native.n_valid
    assert scaled.n_total == native.n_total
    assert scaled.n_params == native.n_params

    if check_covariance:
        assert_param_cov_equivariant(native, scaled, s, tol=cov_rtol)

    d_native = native.to_dict()
    d_scaled = scaled.to_dict()
    assert set(d_scaled.keys()) == set(d_native.keys())


def compare_joint_results(
    native: JointMarginalResult,
    scaled: JointMarginalResult,
    s: float,
    *,
    rtol: float = 1e-3,
    cov_rtol: float = 2e-2,
    check_covariance: bool = True,
) -> None:
    _assert_success_and_scale(native, scaled, s)
    _assert_same_presence(
        native, scaled, ("sigma", "variance_params", "log_likelihood", "aic"),
    )
    np.testing.assert_allclose(scaled.top, native.top * s, rtol=rtol)
    np.testing.assert_allclose(scaled.bottom, native.bottom * s, rtol=rtol)
    _compare_drug_shape_and_asymptotes(native.drug_a, scaled.drug_a, s, rtol=rtol)
    _compare_drug_shape_and_asymptotes(native.drug_b, scaled.drug_b, s, rtol=rtol)

    if native.sigma is not None:
        sf = _sigma_scale_factor(native.error_model, s)
        np.testing.assert_allclose(scaled.sigma, native.sigma * sf, rtol=rtol)

    if native.variance_params:
        assert scaled.variance_params is not None
        for key in native.variance_params:
            np.testing.assert_allclose(
                scaled.variance_params[key],
                native.variance_params[key] * _variance_entry_scale(key, s),
                rtol=rtol,
            )

    n = native.n_data
    assert scaled.n_data == n
    if native.log_likelihood is not None:
        np.testing.assert_allclose(
            scaled.log_likelihood,
            native.log_likelihood - n * np.log(s),
            rtol=1e-4,
            atol=1e-4,
        )
    if native.aic is not None:
        np.testing.assert_allclose(
            scaled.aic,
            native.aic + 2 * n * np.log(s),
            rtol=1e-4,
            atol=1e-4,
        )

    if check_covariance:
        assert_param_cov_equivariant(native, scaled, s, tol=cov_rtol)

    d_native = native.to_dict()
    d_scaled = scaled.to_dict()
    assert set(d_scaled.keys()) == set(d_native.keys())


def compare_matrix_results(
    native: MatrixFitResult,
    scaled: MatrixFitResult,
    s: float,
    *,
    rtol: float = 1e-3,
    cov_rtol: float = 2e-2,
    check_covariance: bool = True,
) -> None:
    _assert_success_and_scale(native.horizontal, scaled.horizontal, s)
    _assert_success_and_scale(native.vertical, scaled.vertical, s)
    assert native.success and scaled.success
    _assert_same_presence(native, scaled, ("sigma",))
    np.testing.assert_allclose(scaled.effect_0, native.effect_0 * s, rtol=rtol)
    np.testing.assert_allclose(scaled.effect_inf, native.effect_inf * s, rtol=rtol)
    _compare_drug_shape_and_asymptotes(native.horizontal, scaled.horizontal, s, rtol=rtol)
    _compare_drug_shape_and_asymptotes(native.vertical, scaled.vertical, s, rtol=rtol)
    if native.sigma is not None:
        sf = _sigma_scale_factor(
            native.horizontal.error_model if native.horizontal.sigma else "gaussian",
            s,
        )
        np.testing.assert_allclose(scaled.sigma, native.sigma * sf, rtol=rtol)

    if check_covariance:
        assert_param_cov_equivariant(native, scaled, s, tol=cov_rtol)


def _compare_scalar_values(
    path: str,
    ref,
    cur,
    rtol: float,
    atol: float,
) -> list[str]:
    if ref is None and cur is None:
        return []
    if ref is None or cur is None:
        return [f"{path}: {ref!r} -> {cur!r}"]
    if isinstance(ref, bool) or isinstance(ref, str):
        if ref != cur:
            return [f"{path}: {ref!r} -> {cur!r}"]
        return []
    if isinstance(ref, (int, float)) and isinstance(cur, (int, float)):
        r_tol, a_tol = rtol, atol
        if path.endswith(("log_likelihood", "aic", "bic")):
            r_tol, a_tol = 0.0, 1e-4
        elif path.endswith("r2"):
            r_tol, a_tol = 0.0, 1e-5
        if not np.isclose(ref, cur, rtol=r_tol, atol=a_tol):
            return [f"{path}: {ref} -> {cur}"]
        return []
    return []


def _compare_nested(ref, cur, path: str = "", *, rtol: float, atol: float) -> list[str]:
    issues: list[str] = []
    if isinstance(ref, dict) and isinstance(cur, dict):
        if set(ref.keys()) != set(cur.keys()):
            issues.append(f"{path}keys {sorted(ref.keys())} vs {sorted(cur.keys())}")
            return issues
        for key in ref:
            if key == "param_cov":  # compared separately, on its own tolerance
                continue
            issues.extend(
                _compare_nested(ref[key], cur[key], f"{path}{key}.", rtol=rtol, atol=atol)
            )
        return issues
    if isinstance(ref, list) and isinstance(cur, list):
        if len(ref) != len(cur):
            issues.append(f"{path}: len {len(ref)} vs {len(cur)}")
            return issues
        for i, (a, b) in enumerate(zip(ref, cur)):
            issues.extend(_compare_nested(a, b, f"{path}[{i}]", rtol=rtol, atol=atol))
        return issues
    issues.extend(_compare_scalar_values(path.rstrip("."), ref, cur, rtol, atol))
    return issues


def compare_param_cov_baseline(ref: list, cur: list, *, tol: float = 2e-2) -> list[str]:
    return covariance_issues(np.asarray(ref, dtype=float), np.asarray(cur, dtype=float), tol=tol)


def compare_baseline_snapshots(reference: dict, current: dict) -> list[str]:
    """Return human-readable diffs between two baseline JSON payloads."""
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
        issues.extend(
            _compare_nested(rc["to_dict"], cc["to_dict"], prefix + "to_dict.", rtol=5e-4, atol=1e-7)
        )
        pc_r, pc_c = rc.get("param_ci"), cc.get("param_ci")
        if (pc_r is None) ^ (pc_c is None):
            issues.append(prefix + f"param_ci presence: {pc_r!r} vs {pc_c!r}")
        elif pc_r is not None and pc_c is not None:
            for name in pc_r:
                lo_r, hi_r = pc_r[name]
                lo_c, hi_c = pc_c[name]
                w_r, w_c = hi_r - lo_r, hi_c - lo_c
                if abs(w_c - w_r) / max(w_r, 1e-12) > 0.01:
                    issues.append(prefix + f"param_ci width {name} differs >1%")
        pci_r, pci_c = rc.get("predict_ci"), cc.get("predict_ci")
        if (pci_r is None) ^ (pci_c is None):
            issues.append(prefix + f"predict_ci presence: {pci_r!r} vs {pci_c!r}")
        elif pci_r and pci_c:
            conc = np.asarray(pci_r["conc"])
            w_ref = np.asarray(pci_r["hi"]) - np.asarray(pci_r["lo"])
            w_cur = np.asarray(pci_c["hi"]) - np.asarray(pci_c["lo"])
            rel = np.abs(w_cur - w_ref) / np.maximum(w_ref, 1e-12)
            if np.any(rel > 0.01):
                issues.append(prefix + f"predict_ci widths: max rel diff {float(np.max(rel)):.4g}")
        cov_r = rc["to_dict"].get("param_cov")
        cov_c = cc["to_dict"].get("param_cov")
        if (cov_r is None) ^ (cov_c is None):
            issues.append(prefix + f"param_cov presence: {cov_r!r} vs {cov_c!r}")
        elif cov_r is not None and cov_c is not None:
            issues.extend([prefix + m for m in compare_param_cov_baseline(cov_r, cov_c)])
    return issues


def fit_scaled_case(case: ScaleFitCase, s: float) -> Any:
    data = case.build_data()
    if case.kind == "single":
        return SingleDrugFit(scale_dataframe(data, s), case.fit_config_for()).fit()
    if case.kind == "single_with_error":
        return SingleDrugFitWithError(scale_dataframe(data, s), case.fit_config_for()).fit()
    if case.kind == "joint":
        data_a, data_b = data
        return JointMarginalFit(
            scale_dataframe(data_a, s), scale_dataframe(data_b, s), noise=case.noise,
        ).fit()
    if case.kind == "matrix":
        reps, ch, cv = data
        em = "lognormal" if case.noise == "lognormal" else "gaussian"
        return MatrixFit(
            scale_matrix_replicates(reps, s), ch, cv, error_model=em,
        ).fit()
    raise ValueError(case.kind)


def hill_predictions(result: FitResult, conc: np.ndarray) -> np.ndarray:
    from synfit.hill import hill_curve

    return hill_curve(
        conc,
        c50=result.c50,
        hill=result.hill,
        effect_0=result.effect_0,
        effect_inf=result.effect_inf,
        asymmetry=result.asymmetry,
    )


def config_fingerprint(case: ScaleFitCase) -> dict:
    """Hashable description of inputs for snapshot drift detection."""
    data = case.build_data()
    payload: dict[str, Any] = {"case_id": case.case_id, "kind": case.kind}
    if case.kind in ("single", "single_with_error"):
        cfg = case.fit_config_for()
        payload["config"] = {
            "noise": getattr(cfg.noise, "kind", str(cfg.noise)),
            "direction": cfg.direction,
        }
        payload["data_hash"] = _df_hash(data)
    elif case.kind == "joint":
        data_a, data_b = data
        payload["noise"] = getattr(case.noise, "kind", str(case.noise))
        payload["data_hash_a"] = _df_hash(data_a)
        payload["data_hash_b"] = _df_hash(data_b)
    else:
        reps, ch, cv = data
        payload["noise"] = case.noise
        payload["data_hash"] = hashlib.sha256(
            np.concatenate([r.ravel() for r in reps]).tobytes()
            + ch.tobytes()
            + cv.tobytes(),
        ).hexdigest()
    return payload


def _df_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()


def runtime_versions() -> dict[str, str]:
    import scipy

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def extract_snapshot_record(case: ScaleFitCase, result: Any) -> dict:
    """Serialisable baseline payload for one native fit."""
    rec: dict[str, Any] = {
        "case_id": case.case_id,
        "fingerprint": config_fingerprint(case),
        "versions": runtime_versions(),
        "to_dict": _to_dict_for_snapshot(result),
    }
    rec["param_ci"] = _param_ci_raw(result)
    rec["predict_ci"] = _predict_ci_raw(result, CONC_GRID)
    return rec


def _strip_message(d: dict) -> dict:
    out = dict(d)
    out.pop("message", None)
    return out


def _to_dict_for_snapshot(result: Any) -> dict:
    if isinstance(result, FitResult):
        return _strip_message(result.to_dict())
    if isinstance(result, (JointMarginalResult, MatrixFitResult)):
        return _strip_message(result.to_dict())
    raise TypeError(type(result))


def _param_ci_raw(result: Any) -> dict | None:
    if isinstance(result, FitResult):
        return result.param_ci()
    if isinstance(result, JointMarginalResult):
        return result._compute_joint_param_cis()
    if isinstance(result, MatrixFitResult):
        # Build from param_cov like FitResult.param_ci
        if result.param_cov is None or not result.param_names:
            return None
        from scipy.stats import norm

        z = norm.ppf(0.975)
        cis = {}
        for i, name in enumerate(result.param_names):
            val = None
            if name in ("effect_0", "effect_inf"):
                val = getattr(result, name)
            elif name.startswith("log_c50"):
                val = getattr(result.horizontal if "hor" in name else result.vertical, "log_c50", None)
            elif "hill" in name:
                drug = result.horizontal if "hor" in name else result.vertical
                val = drug.hill
            if val is None:
                continue
            se = float(np.sqrt(result.param_cov[i, i]))
            cis[name] = (val - z * se, val + z * se)
        return cis
    return None


def _predict_ci_raw(result: Any, conc: np.ndarray) -> dict | None:
    if isinstance(result, FitResult):
        band = result.predict_ci(conc)
        if band is None:
            return None
        lo, hi = band
        return {"conc": conc.tolist(), "lo": lo.tolist(), "hi": hi.tolist()}
    return None


def native_solution_x(fitter: SingleDrugFit, native: FitResult, s: float) -> np.ndarray:
    names = list(fitter.config.fitting_parameters)
    xs = []
    for name in names:
        val = getattr(native, name, None)
        if val is None and is_variance_param(name):
            val = (native.variance_params or {}).get(name[4:])
        if name in ("effect_0", "effect_inf"):
            val = val * s
        xs.append(val)
    return np.asarray(xs, dtype=float)


def result_to_x(fitter: SingleDrugFit, result: FitResult) -> np.ndarray:
    names = list(fitter.config.fitting_parameters)
    xs = []
    for name in names:
        val = getattr(result, name, None)
        if val is None and is_variance_param(name):
            val = (result.variance_params or {}).get(name[4:])
        xs.append(val)
    return np.asarray(xs, dtype=float)
