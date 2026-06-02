import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass

from .data import FitConfig, FitResult
from .bliss import bliss_independence
from .fitting import PRIOR_PENALTY_WEIGHT, FitBase
from .hill import log_wall
from .param_roles import is_log_param
from .noise import (
    GaussianConstant,
    Lognormal,
    NoiseSpec,
    error_model_name,
    fit_noise_scale,
    from_dict as noise_from_dict,
    log_prob as noise_log_prob,
)
from .single import SingleDrugFit, _init_config_from_data

logger = logging.getLogger(__name__)


@dataclass
class MatrixFitResult:
    """Result of a matrix drug-drug interaction fit."""

    horizontal: FitResult
    vertical: FitResult
    effect_0: float
    effect_inf: float
    success: bool
    message: str = ""
    warnings: list[str] | None = None
    direction_horizontal: str = "inhibition"
    direction_vertical: str = "inhibition"
    # MLE noise scale σ̂ pooled across all replicates and cells, in the error
    # model's natural likelihood space. Stamped onto horizontal.sigma and
    # vertical.sigma as well.
    sigma: float | None = None
    # Parameter covariance (6x6, row/col order = ``MatrixFit._PARAM_NAMES``),
    # or None when the Hessian was not invertible at the optimum.
    param_cov: np.ndarray | None = None
    param_names: list[str] | None = None

    def to_dict(self) -> dict:
        d = {
            "horizontal": self.horizontal.to_dict(),
            "vertical": self.vertical.to_dict(),
            "effect_0": self.effect_0,
            "effect_inf": self.effect_inf,
            "success": self.success,
            "message": self.message,
            "direction_horizontal": self.direction_horizontal,
            "direction_vertical": self.direction_vertical,
            "sigma": self.sigma,
        }
        if self.param_names is not None:
            d["param_names"] = list(self.param_names)
        if self.param_cov is not None:
            d["param_cov"] = np.asarray(self.param_cov).tolist()
        if self.warnings:
            d["warnings"] = self.warnings
        return d


class MatrixFit(FitBase):
    """
    Fit drug-drug interaction matrix data using Bliss independence.

    replicates: list of 2D arrays, each (n_vertical, n_horizontal)
    conc_horizontal: 1D array of horizontal drug concentrations
    conc_vertical: 1D array of vertical drug concentrations
    valids: optional list of boolean 2D arrays, same shapes as replicates
    """

    def __init__(
        self,
        replicates: list[np.ndarray],
        conc_horizontal: np.ndarray,
        conc_vertical: np.ndarray,
        valids: list[np.ndarray] | None = None,
        noise: NoiseSpec | dict | str | None = None,
        error_model: str | None = None,
        direction_horizontal: str = "inhibition",
        direction_vertical: str = "inhibition",
    ):
        if not replicates:
            raise ValueError("At least one replicate matrix is required")
        expected_shape = (len(conc_vertical), len(conc_horizontal))
        for i, rep in enumerate(replicates):
            if rep.shape != expected_shape:
                raise ValueError(
                    f"Replicate {i} has shape {rep.shape}, expected {expected_shape} "
                    f"(len(conc_vertical) x len(conc_horizontal))"
                )

        self.replicates = replicates
        self.conc_horizontal = np.asarray(conc_horizontal, dtype=float)
        self.conc_vertical = np.asarray(conc_vertical, dtype=float)
        self.valids = valids
        if noise is not None and error_model is not None:
            raise ValueError("Pass either noise or legacy error_model, not both.")
        self.noise = noise_from_dict(noise if noise is not None else (error_model or "gaussian"))
        if not isinstance(self.noise, (GaussianConstant, Lognormal)):
            raise ValueError("MatrixFit currently supports constant gaussian/lognormal noise only.")
        # FitBase carries the noise spec and supplies the scale-relative
        # optimiser (_run_minimize) and covariance estimator. The matrix
        # parameter vector doesn't match the single-drug FitConfig schema, so —
        # like JointMarginalFit — we pass explicit _x0/_bounds to _run_minimize
        # and override _log_prob_data / _log_prior_prob below.
        super().__init__(FitConfig(noise=self.noise))
        self.direction_horizontal = direction_horizontal
        self.direction_vertical = direction_vertical

        if self.conc_horizontal.size == 0 or self.conc_vertical.size == 0:
            raise ValueError("Concentration arrays must be non-empty")

        self.zero_edge_missing = False
        if not np.isclose(self.conc_horizontal.flat[0], 0.0) or not np.isclose(
            self.conc_vertical.flat[0], 0.0
        ):
            self.zero_edge_missing = True
            logger.warning(
                "Matrix concentrations do not start at 0. "
                "Treating the lowest-concentration row/column as the single-agent edge. "
                "For best results, include a zero-concentration control."
            )

        self.mean_matrix = np.nanmean(np.stack(replicates), axis=0)

        # extract edge slices for single-drug pre-fits
        self._hor_df = self._edge_slice(axis="horizontal")
        self._ver_df = self._edge_slice(axis="vertical")

        self.synfit_hor = SingleDrugFit(
            self._hor_df,
            config=_init_config_from_data(
                self._hor_df,
                direction=direction_horizontal,
                noise=self.noise,
            ),
        )
        self.synfit_ver = SingleDrugFit(
            self._ver_df,
            config=_init_config_from_data(
                self._ver_df,
                direction=direction_vertical,
                noise=self.noise,
            ),
        )

        # pre-fit edge drugs independently
        self._hor_result = self.synfit_hor.fit()
        self._ver_result = self.synfit_ver.fit()

        # shared effect_0/effect_inf initialised from edge pre-fits
        effect_0_init = max(self._hor_result.effect_0, self._ver_result.effect_0)
        effect_inf_init = min(self._hor_result.effect_inf, self._ver_result.effect_inf)

        # 6 parameters: log_c50_hor, log_c50_ver, hill_hor, hill_ver, effect_0, effect_inf
        self._x0 = [
            self._hor_result.log_c50,
            self._ver_result.log_c50,
            self._hor_result.hill,
            self._ver_result.hill,
            effect_0_init,
            effect_inf_init,
        ]
        self._bounds = [
            self.synfit_hor.config.bounds.log_c50,
            self.synfit_ver.config.bounds.log_c50,
            (0.1, 4.0),
            (0.1, 4.0),
            (effect_0_init * 0.5, effect_0_init * 2),
            (0.0, effect_inf_init * 2 + 1e-6),
        ]

    def _edge_slice(self, axis: str) -> pd.DataFrame:
        """Extract the single-drug edge for the named axis.

        axis="horizontal": first row (vertical concentration 0) → horizontal drug alone.
        axis="vertical": first column (horizontal concentration 0) → vertical drug alone.
        """
        rows = []
        for rep in self.replicates:
            if axis == "horizontal":
                values = rep[0, :]  # first row: no vertical drug
                concs = self.conc_horizontal
            else:
                values = rep[:, 0]  # first column: no horizontal drug
                concs = self.conc_vertical
            for conc, y in zip(concs, values):
                rows.append({"concentration": conc, "y": y, "replicate": "edge"})
        return pd.DataFrame(rows)

    _PARAM_NAMES = ("log_c50_hor", "log_c50_ver", "hill_hor", "hill_ver", "effect_0", "effect_inf")

    def _unpack_x(self, x: np.ndarray) -> dict:
        out = {}
        for name, v in zip(self._PARAM_NAMES, x):
            if is_log_param(name):
                out[name[4:]] = 10**v
            else:
                out[name] = v
        return out

    def _log_prob_data(self, x: np.ndarray, **kwargs) -> float:
        p = self._unpack_x(x)
        bliss = bliss_independence(
            self.conc_horizontal, self.conc_vertical,
            c50_hor=p["c50_hor"], c50_ver=p["c50_ver"],
            hill_hor=p["hill_hor"], hill_ver=p["hill_ver"],
            effect_0=p["effect_0"], effect_inf=p["effect_inf"],
        )
        total_Z = 0.0
        for i, rep in enumerate(self.replicates):
            mask = self.valids[i].ravel() if self.valids else None
            total_Z += noise_log_prob(
                rep.ravel(), bliss.ravel(),
                self.noise,
                mask=mask,
            )
        return total_Z

    def _log_prior_prob(self, x: np.ndarray) -> float:
        penalty = 0.0
        for v, bounds in zip(x, self._bounds):
            penalty += log_wall(np.array([v]), bounds).sum()
        return -penalty * PRIOR_PENALTY_WEIGHT

    def fit(self) -> MatrixFitResult:
        n_params = 6
        if self.valids:
            n_data = int(sum(int(v.sum()) for v in self.valids))
        else:
            n_data = int(sum(r.size for r in self.replicates))
        if n_data < n_params + 1:
            raise ValueError(
                f"Need at least {n_params + 1} valid matrix cells for {n_params} parameters; "
                f"got {n_data}."
            )

        x_opt, success, message, pcov = self._run_minimize(self._x0, self._bounds)
        p = self._unpack_x(x_opt)

        # Pool residuals across all replicates and matrix cells to estimate σ̂
        # in the error model's natural space. Same scale used for both edges
        # since the matrix fit shares one error model across the whole plate.
        bliss = bliss_independence(
            self.conc_horizontal, self.conc_vertical,
            c50_hor=p["c50_hor"], c50_ver=p["c50_ver"],
            hill_hor=p["hill_hor"], hill_ver=p["hill_ver"],
            effect_0=p["effect_0"], effect_inf=p["effect_inf"],
        )
        y_all = np.concatenate([rep.ravel() for rep in self.replicates])
        y_pred_all = np.tile(bliss.ravel(), len(self.replicates))
        if self.valids:
            mask_all = np.concatenate([v.ravel() for v in self.valids]).astype(bool)
        else:
            mask_all = None
        sigma_hat = fit_noise_scale(
            y_all, y_pred_all, self.noise, mask=mask_all,
        )
        em_str = error_model_name(self.noise)

        hor_result = FitResult(
            c50=p["c50_hor"], log_c50=np.log10(p["c50_hor"]),
            hill=p["hill_hor"], effect_0=p["effect_0"], effect_inf=p["effect_inf"],
            success=success, n_valid=len(self._hor_df), n_total=len(self._hor_df),
            direction=self.direction_horizontal,
            error_model=em_str, sigma=sigma_hat,
        )
        ver_result = FitResult(
            c50=p["c50_ver"], log_c50=np.log10(p["c50_ver"]),
            hill=p["hill_ver"], effect_0=p["effect_0"], effect_inf=p["effect_inf"],
            success=success, n_valid=len(self._ver_df), n_total=len(self._ver_df),
            direction=self.direction_vertical,
            error_model=em_str, sigma=sigma_hat,
        )
        warnings = []
        if self.zero_edge_missing:
            warnings.append(
                "No zero-concentration control found. "
                "The lowest-concentration row/column was used as the single-agent edge."
            )

        return MatrixFitResult(
            horizontal=hor_result,
            vertical=ver_result,
            effect_0=p["effect_0"],
            effect_inf=p["effect_inf"],
            success=success,
            message=str(message),
            warnings=warnings or None,
            direction_horizontal=self.direction_horizontal,
            direction_vertical=self.direction_vertical,
            sigma=sigma_hat,
            param_cov=pcov,
            param_names=list(self._PARAM_NAMES),
        )

