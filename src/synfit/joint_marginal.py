"""Joint fit of two single-drug dose-response curves with shared asymptotes.

Used when the two drugs come from the same experiment (same assay, same plate,
same controls), so the low- and high-concentration asymptotes ``effect_0`` /
``effect_inf`` must be identical across both drugs. Hill and C50 remain per-drug.
Each drug may independently use a 4- or 5-parameter Hill model; when 5p, the
asymmetry parameter is per-drug.

Per-parameter fit/fix + custom initials/bounds are supported via the
``param_config`` argument on the fitter.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import FitConfig, FitResult, validate_direction, variance_anchor_from_asymptotes
from .fitting import FitBase, PRIOR_PENALTY_WEIGHT, effective_observation_mask
from .hill import hill_curve, log_wall
from .param_roles import is_variance_param
from .noise import (
    GaussianConstant,
    GaussianLinear,
    GaussianQuadratic,
    Lognormal,
    NoiseSpec,
    apply_variance_param_domain,
    clamp_noise_spec_initials,
    error_model_name,
    fit_noise_scale,
    free_coefficients,
    from_dict as noise_from_dict,
    is_heteroscedastic_gaussian,
    require_supported_joint_marginal_fitting_noise,
    legacy_to_noise_spec,
    log_prob as noise_log_prob,
    noise_spec_with_variance_initials,
    scaled_variance_coefficient_defaults,
    variance_at,
    variance_initials_for_noise,
    variance_model_name,
)
from .single import _coerce_noise, _init_config_from_data, _robust_response_range


@dataclass
class JointMarginalResult:
    """Result of a joint two-drug fit with shared top/bottom.

    ``top`` is the high-signal asymptote of the shared plate; ``bottom`` is
    the low-signal asymptote. Each drug's ``effect_0`` / ``effect_inf`` on
    its own FitResult is derived from ``top`` / ``bottom`` via its direction:
    for an inhibitor ``effect_0 = top`` and ``effect_inf = bottom``; for an
    activator the mapping is reversed. Mixed pairs work correctly because
    only the physical plate asymptotes are shared.
    """

    drug_a: FitResult
    drug_b: FitResult
    top: float
    bottom: float
    success: bool
    message: str = ""
    param_cov: np.ndarray | None = field(default=None, repr=False)
    param_names: list | None = field(default=None, repr=False)
    # Goodness-of-fit on the joint data (both marginals, both drugs).
    log_likelihood: float | None = None
    aic: float | None = None
    n_params: int | None = None
    n_data: int | None = None
    error_model: str = "gaussian"
    # MLE noise scale σ̂ in the error model's natural likelihood space, computed
    # from the joint residuals across both drugs. Same value is stamped onto
    # drug_a.sigma and drug_b.sigma since they share the experiment/error model.
    # ``None`` when a non-constant variance model was fitted (use
    # ``variance_params`` instead).
    sigma: float | None = None
    # Variance model used for this fit (constant / linear / quadratic).
    variance_model: str = "constant"
    # Fitted variance polynomial coefficients σ² = a + b·d + c·d²
    # (d = μ − min(top, bottom)). Populated when ``variance_model`` is
    # non-constant; only the coefficients the model actually fits are present.
    variance_params: dict | None = None

    def plot_synergy(
        self,
        mean_matrix: np.ndarray,
        conc_horizontal: np.ndarray,
        conc_vertical: np.ndarray,
        *,
        x_label: str | None = None,
        y_label: str | None = None,
        title: str | None = None,
    ) -> bytes:
        """Plot four synergy score panels using this result's fitted curves.

        The matrix and dose grids follow ``synfit.plotting.synergy_heatmaps``.
        Fitted potencies, slopes, asymmetries and direction-appropriate
        asymptotes are supplied automatically. Both drugs must have the same
        direction. Returns PNG bytes.
        """
        if self.drug_a.direction != self.drug_b.direction:
            raise ValueError("Synergy plotting requires both drugs to have the same direction")
        from .plotting import synergy_heatmaps

        return synergy_heatmaps(
            mean_matrix, conc_horizontal, conc_vertical,
            c50_hor=self.drug_a.c50, c50_ver=self.drug_b.c50,
            hill_hor=self.drug_a.hill, hill_ver=self.drug_b.hill,
            effect_0=self.drug_a.effect_0, effect_inf=self.drug_a.effect_inf,
            asymmetry_hor=self.drug_a.asymmetry, asymmetry_ver=self.drug_b.asymmetry,
            x_label=x_label, y_label=y_label, title=title,
        )

    def predict_variance(self, mu: np.ndarray | float) -> np.ndarray | None:
        """σ²(μ) for this joint fit given a predicted response.

        Anchored at ``m = min(top, bottom)`` so reported σ² matches the fit.
        Falls back to the profiled ``sigma**2`` for constant variance; ``None``
        if neither is available.
        """
        mu = np.asarray(mu, dtype=float)
        if self.variance_params:
            a = float(self.variance_params.get("a", 0.0))
            b = float(self.variance_params.get("b", 0.0))
            c = float(self.variance_params.get("c", 0.0))
            return variance_at(
                mu, a, b, c,
                anchor=variance_anchor_from_asymptotes(self.top, self.bottom),
            )
        if self.sigma is not None:
            s = float(self.sigma)
            if self.error_model == "lognormal":
                s2 = s * s
                exp_s2 = np.exp(s2)
                return mu * mu * exp_s2 * np.expm1(s2)
            return np.full_like(mu, s * s)
        return None

    def to_dict(self) -> dict:
        """Serialise to the matrix-analysis fit_params shape.

        ``horizontal`` / ``vertical`` mirror the matrix axes (drug_a is fed as
        the horizontal drug, drug_b as the vertical — the sole context this
        class is used in). Top-level ``effect_0`` / ``effect_inf`` follow the
        inhibition convention (``effect_0 = top``) so that downstream synergy
        calculators (bliss_independence, HSA, Loewe, ZIP) see canonical values
        without extra remapping.

        Confidence intervals are derived from the *joint* parameter covariance
        and surfaced in two places: a top-level ``ci`` dict for the shared
        plate-level params (``top``, ``bottom``), and per-drug ``ci`` dicts
        injected onto ``horizontal`` / ``vertical`` for that drug's own
        parameters (``c50``, ``log_c50``, ``hill``, optionally ``asymmetry``).
        The drug-level ``FitResult.param_ci()`` would otherwise come back
        empty because those FitResults don't carry their own ``param_cov``.
        """
        joint_ci = self._compute_joint_param_cis()

        d = {
            "top": self.top,
            "bottom": self.bottom,
            "effect_0": self.top,
            "effect_inf": self.bottom,
            "horizontal": self.drug_a.to_dict(),
            "vertical": self.drug_b.to_dict(),
            "direction_horizontal": self.drug_a.direction,
            "direction_vertical": self.drug_b.direction,
            "success": self.success,
            "message": self.message,
            "fit_kind": "joint_marginal",
            "error_model": self.error_model,
            "aic": self.aic,
            "log_likelihood": self.log_likelihood,
            "n_params": self.n_params,
            "n_data": self.n_data,
            "sigma": self.sigma,
            "variance_model": self.variance_model,
            "variance_params": self.variance_params,
        }
        # Shared CIs at top level — top/bottom are joint params, identical
        # for both drugs, so keep them out of the per-drug dicts.
        shared_ci = {k: list(joint_ci[k]) for k in ("top", "bottom") if k in joint_ci}
        if shared_ci:
            d["ci"] = shared_ci
        # Per-drug CIs on the horizontal/vertical sub-dicts (these come from
        # the *joint* cov; the per-drug FitResults can't compute them alone).
        d["horizontal"]["ci"] = self._drug_ci_from_joint(joint_ci, suffix="a")
        d["vertical"]["ci"] = self._drug_ci_from_joint(joint_ci, suffix="b")

        if self.param_cov is not None:
            d["param_cov"] = self.param_cov.tolist()
        if self.param_names:
            d["param_names"] = list(self.param_names)
        return d

    def _compute_joint_param_cis(self, alpha: float = 0.05) -> dict[str, tuple[float, float]]:
        """Wald CI per joint parameter from the optimised covariance matrix.

        Names are joint-suffixed (``log_c50_a``, ``hill_b``, etc.). Shared
        params keep their bare names (``top``, ``bottom``).
        """
        if self.param_cov is None or not self.param_names:
            return {}
        from scipy.stats import norm as _norm
        z = _norm.ppf(1 - alpha / 2)

        # Resolve a value for each joint name from the FitResult fields.
        def _value(name: str) -> float | None:
            if name == "top":
                return self.top
            if name == "bottom":
                return self.bottom
            for suffix, drug in (("a", self.drug_a), ("b", self.drug_b)):
                tail = f"_{suffix}"
                if name.endswith(tail):
                    base = name[: -len(tail)]
                    return getattr(drug, base, None)
            return None

        cis: dict[str, tuple[float, float]] = {}
        for i, name in enumerate(self.param_names):
            se = float(np.sqrt(self.param_cov[i, i]))
            val = _value(name)
            if val is None:
                continue
            cis[name] = (val - z * se, val + z * se)
        return cis

    @staticmethod
    def _drug_ci_from_joint(
        joint_ci: dict[str, tuple[float, float]],
        *,
        suffix: str,
    ) -> dict[str, list[float]]:
        """Lift the joint CIs for one drug into the per-drug naming scheme.

        Maps ``log_c50_<suffix>`` → ``log_c50``, etc. Also derives ``c50``
        from ``log_c50`` via the monotonic transform, mirroring
        ``FitResult.param_ci``.
        """
        out: dict[str, list[float]] = {}
        for base in ("log_c50", "hill", "asymmetry"):
            joint_name = f"{base}_{suffix}"
            if joint_name in joint_ci:
                lo, hi = joint_ci[joint_name]
                out[base] = [round(float(lo), 6), round(float(hi), 6)]
        if "log_c50" in out:
            lo_log, hi_log = out["log_c50"]
            out["c50"] = [round(10 ** lo_log, 6), round(10 ** hi_log, 6)]
        return out


class JointMarginalFit(FitBase):
    """Fit two drugs' dose-response curves sharing plate-level ``top`` / ``bottom``.

    Parameters
    ----------
    data_a, data_b:
        DataFrames with columns ``concentration``, ``y``, ``replicate``.
    model_a, model_b:
        ``"4p"`` (default) or ``"5p"`` — selected per drug. 5p adds a per-drug
        asymmetry parameter.
    direction_a, direction_b:
        ``"inhibition"`` or ``"activation"`` per drug.
    error_model:
        Shared across both drugs (same experiment).
    param_config:
        Optional per-parameter overrides of the form
        ``{name: {"init": float, "lo": float, "hi": float, "fit": bool}}``.
        Valid names: ``top``, ``bottom``, ``log_c50_a``, ``hill_a``,
        ``asymmetry_a`` (5p only), ``log_c50_b``, ``hill_b``, ``asymmetry_b``
        (5p only). Any of the four entry keys may be omitted; missing values
        fall back to auto-derived defaults. ``fit=False`` holds the parameter
        constant at ``init`` (or the auto default if no init given).
    """

    # Canonical order for serialisation of the *full* param space.  Only the
    # subset whose ``fit`` flag is True actually enters the optimiser.
    _SHARED_NAMES = ("top", "bottom")
    _PER_DRUG_NAMES = ("log_c50", "hill", "asymmetry")

    @classmethod
    def from_matrix(
        cls,
        replicates: list[np.ndarray] | np.ndarray,
        conc_horizontal: np.ndarray,
        conc_vertical: np.ndarray,
        *,
        model_a: str = "4p",
        model_b: str = "4p",
        direction_a: str = "inhibition",
        direction_b: str = "inhibition",
        noise: NoiseSpec | dict | str | None = None,
        param_config: dict | None = None,
        valids: np.ndarray | None = None,
    ) -> JointMarginalFit:
        """Build a joint marginal fitter from replicate combination matrices.

        ``replicates`` has shape (n_replicates, n_vertical, n_horizontal).
        Each concentration vector must contain exactly one zero dose, at any
        position, and at least one positive dose. Drug A is horizontal and
        drug B vertical. Only the zero-dose edges enter the fit.

        Both complete edges are retained in ``data_a`` / ``data_b``. Their
        shared no-drug well is masked out of drug B's likelihood, so each
        physical observation contributes once. ``valids`` optionally supplies
        a Boolean inclusion mask with the same shape as ``replicates``;
        exclusions apply to both copies of the shared well. Other options
        follow the two-DataFrame constructor.
        """
        ch = np.asarray(conc_horizontal, dtype=float)
        cv = np.asarray(conc_vertical, dtype=float)
        for name, conc in (("conc_horizontal", ch), ("conc_vertical", cv)):
            if conc.ndim != 1 or not np.all(np.isfinite(conc)) or np.any(conc < 0):
                raise ValueError(f"{name} must be a one-dimensional finite non-negative array")
            if np.count_nonzero(conc == 0) != 1 or not np.any(conc > 0):
                raise ValueError(f"{name} must contain one zero dose and positive doses")
        matrices = np.asarray(replicates, dtype=float)
        if matrices.ndim != 3 or matrices.shape[0] == 0 or matrices.shape[1:] != (cv.size, ch.size):
            raise ValueError("replicates must have shape (n_replicates, n_vertical, n_horizontal), with n_replicates > 0")
        if valids is None:
            included = np.ones(matrices.shape, dtype=bool)
        else:
            included = np.asarray(valids, dtype=bool)
            if included.shape != matrices.shape:
                raise ValueError("valids must have the same shape as replicates")

        zero_h = int(np.flatnonzero(ch == 0)[0])
        zero_v = int(np.flatnonzero(cv == 0)[0])
        n_reps = matrices.shape[0]
        data_a = pd.DataFrame({
            "concentration": np.tile(ch, n_reps),
            "y": matrices[:, zero_v, :].ravel(),
            "replicate": np.repeat(np.arange(n_reps), ch.size),
        })
        data_b = pd.DataFrame({
            "concentration": np.tile(cv, n_reps),
            "y": matrices[:, :, zero_h].ravel(),
            "replicate": np.repeat(np.arange(n_reps), cv.size),
        })
        mask_a = included[:, zero_v, :].ravel()
        mask_b = included[:, :, zero_h].copy()
        mask_b[:, zero_v] = False
        return cls(
            data_a, data_b, model_a=model_a, model_b=model_b,
            direction_a=direction_a, direction_b=direction_b,
            noise=noise, param_config=param_config,
            valids_a=mask_a, valids_b=mask_b.ravel(),
        )

    def __init__(
        self,
        data_a: pd.DataFrame,
        data_b: pd.DataFrame,
        *,
        model_a: str = "4p",
        model_b: str = "4p",
        direction_a: str = "inhibition",
        direction_b: str = "inhibition",
        noise: NoiseSpec | dict | str | None = None,
        error_model: str | None = None,
        variance_model: str | None = None,
        param_config: dict | None = None,
        valids_a: np.ndarray | None = None,
        valids_b: np.ndarray | None = None,
    ):
        for name, df in (("data_a", data_a), ("data_b", data_b)):
            if not {"concentration", "y", "replicate"}.issubset(df.columns):
                raise ValueError(f"{name} must have columns: concentration, y, replicate")
        if model_a not in ("4p", "5p") or model_b not in ("4p", "5p"):
            raise ValueError("model_a and model_b must be '4p' or '5p'")
        validate_direction(direction_a, name="direction_a")
        validate_direction(direction_b, name="direction_b")

        self.data_a = data_a
        self.data_b = data_b
        self.model_a = model_a
        self.model_b = model_b
        self.direction_a = direction_a
        self.direction_b = direction_b
        self._mask_a = effective_observation_mask(
            valids_a,
            len(data_a),
            data_a["concentration"].values,
            data_a["y"].values,
        )
        self._mask_b = effective_observation_mask(
            valids_b,
            len(data_b),
            data_b["concentration"].values,
            data_b["y"].values,
        )

        if noise is not None and (error_model is not None or variance_model is not None):
            raise ValueError("Pass either noise or legacy error_model/variance_model, not both.")
        self.noise = (
            noise_from_dict(noise)
            if noise is not None
            else legacy_to_noise_spec(error_model or "gaussian", variance_model or "constant")
        )
        require_supported_joint_marginal_fitting_noise(self.noise)

        cfg_a = _init_config_from_data(
            data_a.loc[self._mask_a], direction=direction_a, noise=self.noise,
        )
        cfg_b = _init_config_from_data(
            data_b.loc[self._mask_b], direction=direction_b, noise=self.noise,
        )

        # Auto-defaults for each param (init + bounds). User overrides overlay
        # on top via ``param_config``.
        defaults, bounds = self._compute_defaults(cfg_a, cfg_b, direction_a, direction_b)

        self._response_scale = None
        full_names = self._full_param_names(model_a, model_b)
        if is_heteroscedastic_gaussian(self.noise):
            self._response_scale = _joint_response_scale(
                data_a, data_b, self._mask_a, self._mask_b,
            )
            self.noise = clamp_noise_spec_initials(
                self.noise, response_scale=self._response_scale,
            )
            var_defaults, var_bounds = _variance_coefficient_defaults(
                self.noise, self._response_scale,
            )
            for name in var_defaults:
                full_names.append(name)
                defaults[name] = var_defaults[name]
                bounds[name] = var_bounds[name]
            if var_defaults:
                self.noise = noise_spec_with_variance_initials(self.noise, var_defaults)

        cfg = param_config or {}

        optimized_names: list[str] = []
        optimized_x0: list[float] = []
        optimized_bounds: list[tuple[float, float]] = []
        fixed_values: dict[str, float] = {}

        for name in full_names:
            entry = cfg.get(name, {}) or {}
            init = float(entry.get("init", defaults[name]))
            lo = float(entry.get("lo", bounds[name][0]))
            hi = float(entry.get("hi", bounds[name][1]))
            fit_flag = bool(entry.get("fit", True))
            if is_variance_param(name):
                lo, hi, clamped_init = apply_variance_param_domain(
                    name, lo, hi, init,
                    response_scale=self._response_scale,
                )
                init = float(clamped_init)
            if fit_flag:
                optimized_names.append(name)
                optimized_x0.append(init)
                optimized_bounds.append((lo, hi))
            else:
                fixed_values[name] = init

        if not optimized_names:
            raise ValueError(
                "At least one parameter must have fit=True — cannot run an "
                "optimiser with no free parameters."
            )

        self._param_names = optimized_names
        self._x0 = optimized_x0
        self._bounds = optimized_bounds
        self._name_to_ix = {n: i for i, n in enumerate(optimized_names)}
        self._fixed_values = fixed_values

        super().__init__(FitConfig(noise=self.noise))

    @classmethod
    def _full_param_names(cls, model_a: str, model_b: str) -> list[str]:
        """All parameter names that physically belong to this fit's Hill model.

        4p drugs contribute ``log_c50_X`` + ``hill_X``; 5p adds ``asymmetry_X``.
        Shared ``top`` / ``bottom`` are always present.
        """
        names = list(cls._SHARED_NAMES)
        for suffix, model in (("a", model_a), ("b", model_b)):
            names.append(f"log_c50_{suffix}")
            names.append(f"hill_{suffix}")
            if model == "5p":
                names.append(f"asymmetry_{suffix}")
        return names

    @staticmethod
    def _compute_defaults(
        cfg_a, cfg_b, direction_a: str, direction_b: str,
    ) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
        """Derive per-parameter initial values and bounds from single-drug configs."""
        def _top_of(cfg, direction):
            return cfg.effect_inf if direction == "activation" else cfg.effect_0
        def _bot_of(cfg, direction):
            return cfg.effect_0 if direction == "activation" else cfg.effect_inf
        def _top_bounds(cfg, direction):
            return cfg.bounds.effect_inf if direction == "activation" else cfg.bounds.effect_0
        def _bot_bounds(cfg, direction):
            return cfg.bounds.effect_0 if direction == "activation" else cfg.bounds.effect_inf

        top_init = 0.5 * (_top_of(cfg_a, direction_a) + _top_of(cfg_b, direction_b))
        bot_init = 0.5 * (_bot_of(cfg_a, direction_a) + _bot_of(cfg_b, direction_b))
        top_lo_a, top_hi_a = _top_bounds(cfg_a, direction_a)
        top_lo_b, top_hi_b = _top_bounds(cfg_b, direction_b)
        bot_lo_a, bot_hi_a = _bot_bounds(cfg_a, direction_a)
        bot_lo_b, bot_hi_b = _bot_bounds(cfg_b, direction_b)

        defaults = {
            "top": top_init,
            "bottom": bot_init,
            "log_c50_a": cfg_a.log_c50,
            "hill_a": cfg_a.hill,
            "asymmetry_a": cfg_a.asymmetry,
            "log_c50_b": cfg_b.log_c50,
            "hill_b": cfg_b.hill,
            "asymmetry_b": cfg_b.asymmetry,
        }
        bounds = {
            "top": (min(top_lo_a, top_lo_b), max(top_hi_a, top_hi_b)),
            "bottom": (min(bot_lo_a, bot_lo_b), max(bot_hi_a, bot_hi_b)),
            "log_c50_a": cfg_a.bounds.log_c50,
            "hill_a": cfg_a.bounds.hill,
            "asymmetry_a": cfg_a.bounds.asymmetry,
            "log_c50_b": cfg_b.bounds.log_c50,
            "hill_b": cfg_b.bounds.hill,
            "asymmetry_b": cfg_b.bounds.asymmetry,
        }
        return defaults, bounds

    def _get(self, x: np.ndarray, name: str) -> float:
        """Return the current value of a named param — from x if optimised, else the fixed constant."""
        if name in self._fixed_values:
            return self._fixed_values[name]
        return float(x[self._name_to_ix[name]])

    def _x_to_variance_params(self, x: np.ndarray) -> tuple[float, float, float] | None:
        """Variance polynomial coefficients for this fit's ``x``.

        JointMarginalFit indexes parameters via ``_name_to_ix`` (its own
        ordering), not ``config.fitting_parameters``, so we override the
        ``FitBase`` default which assumes single-drug ordering. Coefficients
        not in the optimised list (i.e. fixed) come from ``_fixed_values`` or
        fall back to ``FitConfig`` defaults.
        """
        if isinstance(self.noise, (GaussianConstant, Lognormal)):
            return None
        defaults_cfg = FitConfig(noise=self.noise)
        a = self._get(x, "var_a") if "var_a" in self._name_to_ix or "var_a" in self._fixed_values else float(defaults_cfg.var_a)
        b = self._get(x, "var_b") if "var_b" in self._name_to_ix or "var_b" in self._fixed_values else float(defaults_cfg.var_b)
        c = self._get(x, "var_c") if "var_c" in self._name_to_ix or "var_c" in self._fixed_values else float(defaults_cfg.var_c)
        return a, b, c

    def _variance_anchor(self, x: np.ndarray) -> float:
        return variance_anchor_from_asymptotes(self._get(x, "top"), self._get(x, "bottom"))

    @staticmethod
    def _drug_asymptotes(top: float, bottom: float, direction: str) -> tuple[float, float]:
        """Map shared plate ``top`` / ``bottom`` to this drug's ``(effect_0, effect_inf)``.

        Inhibitor: response falls with dose, so effect_0 = top, effect_inf = bottom.
        Activator: response rises with dose, so effect_0 = bottom, effect_inf = top.
        """
        if direction == "activation":
            return bottom, top
        return top, bottom

    def _unpack(self, x: np.ndarray) -> dict:
        log_c50_a = self._get(x, "log_c50_a")
        log_c50_b = self._get(x, "log_c50_b")
        return {
            "top": self._get(x, "top"),
            "bottom": self._get(x, "bottom"),
            "log_c50_a": log_c50_a,
            "hill_a": self._get(x, "hill_a"),
            "c50_a": 10 ** log_c50_a,
            "log_c50_b": log_c50_b,
            "hill_b": self._get(x, "hill_b"),
            "c50_b": 10 ** log_c50_b,
            "asymmetry_a": self._get(x, "asymmetry_a") if self.model_a == "5p" else None,
            "asymmetry_b": self._get(x, "asymmetry_b") if self.model_b == "5p" else None,
        }

    def _log_prob_data(self, x: np.ndarray, **kwargs) -> float:
        mask_a = kwargs.get("mask_a", self._mask_a)
        mask_b = kwargs.get("mask_b", self._mask_b)
        p = self._unpack(x)
        e0_a, einf_a = self._drug_asymptotes(p["top"], p["bottom"], self.direction_a)
        e0_b, einf_b = self._drug_asymptotes(p["top"], p["bottom"], self.direction_b)
        ya_pred = hill_curve(
            self.data_a["concentration"].values,
            c50=p["c50_a"], hill=p["hill_a"],
            effect_0=e0_a, effect_inf=einf_a,
            asymmetry=p["asymmetry_a"] if p["asymmetry_a"] is not None else 1.0,
        )
        yb_pred = hill_curve(
            self.data_b["concentration"].values,
            c50=p["c50_b"], hill=p["hill_b"],
            effect_0=e0_b, effect_inf=einf_b,
            asymmetry=p["asymmetry_b"] if p["asymmetry_b"] is not None else 1.0,
        )
        var_params = self._x_to_variance_params(x)
        anchor = self._variance_anchor(x)
        ll_a = noise_log_prob(
            self.data_a["y"].values, ya_pred,
            self.config.noise,
            mask=mask_a,
            variance_params=var_params,
            variance_anchor=anchor,
        )
        ll_b = noise_log_prob(
            self.data_b["y"].values, yb_pred,
            self.config.noise,
            mask=mask_b,
            variance_params=var_params,
            variance_anchor=anchor,
        )
        return ll_a + ll_b

    def _log_prior_prob(self, x: np.ndarray) -> float:
        penalty = 0.0
        for v, bounds in zip(x, self._bounds):
            penalty += log_wall(np.array([v]), bounds).sum()
        return -penalty * PRIOR_PENALTY_WEIGHT

    def fit(self) -> JointMarginalResult:
        n_params = len(self._param_names)
        n_data = int(self._mask_a.sum() + self._mask_b.sum())
        if n_data < n_params + 1:
            raise ValueError(
                f"Need at least {n_params + 1} valid data points for {n_params} parameters; "
                f"got {n_data}."
            )

        x_opt, success, message, pcov = self._run_minimize(
            self._x0, self._bounds, mask_a=self._mask_a, mask_b=self._mask_b,
        )
        p = self._unpack(x_opt)

        em_str = error_model_name(self.noise)
        vm_str = variance_model_name(self.noise)

        e0_a, einf_a = self._drug_asymptotes(p["top"], p["bottom"], self.direction_a)
        e0_b, einf_b = self._drug_asymptotes(p["top"], p["bottom"], self.direction_b)

        ya_pred = hill_curve(
            self.data_a["concentration"].values,
            c50=p["c50_a"], hill=p["hill_a"],
            effect_0=e0_a, effect_inf=einf_a,
            asymmetry=p["asymmetry_a"] if p["asymmetry_a"] is not None else 1.0,
        )
        yb_pred = hill_curve(
            self.data_b["concentration"].values,
            c50=p["c50_b"], hill=p["hill_b"],
            effect_0=e0_b, effect_inf=einf_b,
            asymmetry=p["asymmetry_b"] if p["asymmetry_b"] is not None else 1.0,
        )
        # σ̂ is shared across the two drugs (same plate, same error model) and
        # only meaningful under the constant-variance branch — for non-constant
        # the fitter has solved for the polynomial directly, so σ varies with μ
        # and ``variance_params`` is the answer instead.
        variance_params: dict | None = None
        if isinstance(self.noise, (GaussianConstant, Lognormal)):
            y_joint = np.concatenate([self.data_a["y"].values, self.data_b["y"].values])
            y_pred_joint = np.concatenate([ya_pred, yb_pred])
            mask_joint = np.concatenate([self._mask_a, self._mask_b])
            sigma_hat = fit_noise_scale(
                y_joint, y_pred_joint, self.config.noise, mask=mask_joint,
            )
        else:
            sigma_hat = None
            var_tuple = self._x_to_variance_params(x_opt)
            if var_tuple is not None:
                free = set(free_coefficients(self.noise))
                entries: dict[str, float] = {}
                if "var_a" in free:
                    entries["a"] = float(var_tuple[0])
                if "var_b" in free:
                    entries["b"] = float(var_tuple[1])
                if "var_c" in free:
                    entries["c"] = float(var_tuple[2])
                variance_params = entries

        drug_a = FitResult(
            c50=p["c50_a"], log_c50=p["log_c50_a"],
            hill=p["hill_a"], effect_0=e0_a, effect_inf=einf_a,
            success=success,
            n_valid=int(self._mask_a.sum()), n_total=len(self.data_a),
            message=message, asymmetry=p["asymmetry_a"],
            direction=self.direction_a, error_model=em_str,
            sigma=sigma_hat,
            variance_model=vm_str,
            variance_params=variance_params,
        )
        drug_b = FitResult(
            c50=p["c50_b"], log_c50=p["log_c50_b"],
            hill=p["hill_b"], effect_0=e0_b, effect_inf=einf_b,
            success=success,
            n_valid=int(self._mask_b.sum()), n_total=len(self.data_b),
            message=message, asymmetry=p["asymmetry_b"],
            direction=self.direction_b, error_model=em_str,
            sigma=sigma_hat,
            variance_model=vm_str,
            variance_params=variance_params,
        )

        log_like = float(self._log_prob_data(x_opt))
        aic = float(-2.0 * log_like + 2.0 * n_params)

        return JointMarginalResult(
            drug_a=drug_a,
            drug_b=drug_b,
            top=p["top"],
            bottom=p["bottom"],
            success=success,
            message=message,
            param_cov=pcov,
            param_names=list(self._param_names),
            log_likelihood=log_like,
            aic=aic,
            n_params=n_params,
            n_data=n_data,
            error_model=em_str,
            sigma=sigma_hat,
            variance_model=vm_str,
            variance_params=variance_params,
        )


def fit_joint_marginal_auto(
    data_a: pd.DataFrame,
    data_b: pd.DataFrame,
    *,
    model_a: str = "4p",
    model_b: str = "4p",
    direction_a: str = "inhibition",
    direction_b: str = "inhibition",
    variance_model: str = "constant",
    param_config: dict | None = None,
) -> JointMarginalResult:
    """Run the joint marginal fit with both error models, return the lower-AIC one.

    Mirrors ``SingleDrugFit``'s auto-selection path: fit once gaussian, once
    lognormal, compare AIC (valid across the two thanks to the Jacobian
    correction baked into ``noise.log_prob``), return the winner. If only one
    of the two fits succeeds, that one wins by default.

    When ``variance_model`` is non-constant the lognormal branch is skipped —
    the polynomial σ²(μ) is gaussian-only, same rule the single-drug auto
    path follows.
    """
    validate_direction(direction_a, name="direction_a")
    validate_direction(direction_b, name="direction_b")
    base_noise = legacy_to_noise_spec("gaussian", variance_model)
    error_models = ("gaussian",) if not isinstance(base_noise, GaussianConstant) else ("gaussian", "lognormal")
    results: dict[str, JointMarginalResult] = {}
    for em in error_models:
        try:
            fitter = JointMarginalFit(
                data_a, data_b,
                model_a=model_a, model_b=model_b,
                direction_a=direction_a, direction_b=direction_b,
                noise=legacy_to_noise_spec(em, variance_model),
                param_config=param_config,
            )
            results[em] = fitter.fit()
        except Exception:
            # Non-positive responses under lognormal, etc. — skip.
            continue

    if not results:
        raise ValueError("Both gaussian and lognormal joint-marginal fits failed.")
    if len(results) == 1:
        return next(iter(results.values()))

    g_aic = results["gaussian"].aic
    l_aic = results["lognormal"].aic
    if g_aic is None or l_aic is None:
        return results["gaussian"]
    return results["gaussian"] if g_aic <= l_aic else results["lognormal"]


def _joint_response_scale(
    data_a: pd.DataFrame,
    data_b: pd.DataFrame,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> float:
    """Robust response range of the pooled plate (both drugs)."""
    y = np.concatenate([
        data_a.loc[mask_a, "y"].to_numpy(),
        data_b.loc[mask_b, "y"].to_numpy(),
    ])
    return _robust_response_range(y)


def _variance_coefficient_defaults(
    noise: NoiseSpec,
    s: float,
) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    """Init and bounds for free variance coefficients.

    Same source :class:`JointMarginalFit` uses when no ``param_config`` is
    supplied: response-scaled bounds/initials (``a = 1e-3·s²``, ``b = 0``,
    ``c = 0``) with user-given ``a_init`` / ``b_init`` / ``c_init`` kept
    after domain clamping. Shared with :func:`default_joint_marginal_config`
    so the public helper cannot drift from the fitter.
    """
    _, scaled_bounds = scaled_variance_coefficient_defaults(s)
    user_inits = variance_initials_for_noise(noise, s)
    defaults: dict[str, float] = {}
    bounds: dict[str, tuple[float, float]] = {}
    for name in free_coefficients(noise):
        defaults[name] = float(user_inits[name])
        bounds[name] = tuple(scaled_bounds[name])
    return defaults, bounds


def default_joint_marginal_config(
    data_a: pd.DataFrame,
    data_b: pd.DataFrame,
    *,
    model_a: str = "4p",
    model_b: str = "4p",
    direction_a: str = "inhibition",
    direction_b: str = "inhibition",
    noise: "NoiseSpec | dict | str | None" = None,
) -> dict[str, dict[str, float]]:
    """Data-driven default initials and bounds for a joint marginal fit.

    Public entry point to the same derivation :class:`JointMarginalFit` applies
    when no ``param_config`` is supplied: initials seeded from each drug's
    response and concentration ranges, asymptote bounds that extend half a
    dynamic range past each drug's observed extremes, then merged across the two
    drugs (union of the per-drug bound intervals). Noise-aware: lognormal floors
    all asymptote lower bounds strictly positive (> 0).

    Parameters
    ----------
    data_a, data_b:
        Per-drug replicate DataFrames with columns ``concentration``, ``y``,
        ``replicate``.  ``data_a`` maps to the horizontal drug (drug_a /
        ``_a`` suffixed params); ``data_b`` to the vertical (``_b``).
    model_a, model_b:
        ``"4p"`` (default) or ``"5p"``.  5p adds an asymmetry parameter per
        drug.  Asymmetry entries are present in the returned dict *only* when
        the corresponding model is ``"5p"``.
    direction_a, direction_b:
        ``"inhibition"`` or ``"activation"`` per drug.
    noise:
        Accepts a :class:`NoiseSpec`, a tagged dict (``{"kind": "lognormal"}``),
        or a kind string (``"lognormal"`` / ``"gaussian"``).  Lognormal forces
        the asymptote lower bounds positive; the same coercion
        :class:`JointMarginalFit` applies.

    Returns
    -------
    dict
        Flat mapping from joint parameter name to ``{"init": float, "lo":
        float, "hi": float}``.  Always includes ``top``, ``bottom``,
        ``log_c50_a``, ``hill_a``, ``log_c50_b``, ``hill_b``.  Includes
        ``asymmetry_a`` only when ``model_a == "5p"``; ``asymmetry_b`` only
        when ``model_b == "5p"``.  For heteroscedastic Gaussian noise
        (``gaussian_linear`` / ``gaussian_quadratic``) also includes the free
        variance coefficients (``var_a``, ``var_b``, and ``var_c`` for
        quadratic) with response-scaled initials and bounds matching
        :class:`JointMarginalFit` when no ``param_config`` is supplied
        (``var_a ∈ [1e-12·s², 10·s²]``, ``var_b ∈ [0, 10·s]``, ``var_c ∈
        [0, 10]``, default ``a = 1e-3·s²``). User-given ``a_init`` /
        ``b_init`` / ``c_init`` are kept after clamping to ``a > 0``,
        ``b ≥ 0``, ``c ≥ 0``. Constant Gaussian and lognormal return no
        ``var_*`` entries.

    Consumers that surface editable bounds in a UI should source their defaults
    here rather than re-deriving them, so the derivation has a single source of
    truth — identical to what the fitter sees when no overrides are supplied.
    """
    if model_a not in ("4p", "5p") or model_b not in ("4p", "5p"):
        raise ValueError("model_a and model_b must be '4p' or '5p'")
    validate_direction(direction_a, name="direction_a")
    validate_direction(direction_b, name="direction_b")

    coerced = _coerce_noise(noise)
    require_supported_joint_marginal_fitting_noise(coerced)
    mask_a = effective_observation_mask(
        None, len(data_a), data_a["concentration"].values, data_a["y"].values,
    )
    mask_b = effective_observation_mask(
        None, len(data_b), data_b["concentration"].values, data_b["y"].values,
    )
    cfg_a = _init_config_from_data(
        data_a.loc[mask_a], direction=direction_a, noise=coerced,
    )
    cfg_b = _init_config_from_data(
        data_b.loc[mask_b], direction=direction_b, noise=coerced,
    )
    defaults, bounds = JointMarginalFit._compute_defaults(cfg_a, cfg_b, direction_a, direction_b)

    include = ["top", "bottom", "log_c50_a", "hill_a", "log_c50_b", "hill_b"]
    if model_a == "5p":
        include.append("asymmetry_a")
    if model_b == "5p":
        include.append("asymmetry_b")

    if coerced is not None and is_heteroscedastic_gaussian(coerced):
        var_defaults, var_bounds = _variance_coefficient_defaults(
            coerced, _joint_response_scale(data_a, data_b, mask_a, mask_b),
        )
        defaults.update(var_defaults)
        bounds.update(var_bounds)
        include.extend(var_defaults)

    return {
        name: {
            "init": float(defaults[name]),
            "lo": float(bounds[name][0]),
            "hi": float(bounds[name][1]),
        }
        for name in include
    }
