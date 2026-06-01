import numpy as np
from scipy.optimize import minimize, approx_fprime

from .data import FitConfig
from .hill import log_wall

# Large multiplier so soft-boundary violations dominate the objective
PRIOR_PENALTY_WEIGHT = 1000

# Step size for numerical Hessian (central differences)
_HESS_EPS = 1e-5


class FitBase:
    """
    Base class for scipy-minimize fitting.

    Child classes implement _log_prob_data(x, **kwargs).
    """

    def __init__(self, config: FitConfig | None = None):
        self.config = config or FitConfig()

    # Names of variance polynomial coefficients that may appear in the
    # optimiser parameter list. Stripped out of the curve kwargs so they
    # don't get passed into hill_curve.
    _VARIANCE_PARAM_NAMES = ("var_a", "var_b", "var_c")

    def _x_to_kwargs(self, x: np.ndarray) -> dict:
        """Convert parameter array to hill_curve kwargs (un-logs log_ params).

        Variance polynomial coefficients (``var_a`` etc.) are filtered out —
        retrieve them via ``_x_to_variance_params`` instead.
        """
        out = {}
        for par, v in zip(self.config.fitting_parameters, x):
            if par in self._VARIANCE_PARAM_NAMES:
                continue
            if par.startswith("log_"):
                out[par[4:]] = 10**v
            else:
                out[par] = v
        return out

    def _x_to_variance_params(self, x: np.ndarray) -> tuple[float, float, float] | None:
        """Return ``(a, b, c)`` for the heteroscedastic σ² polynomial.

        Falls back to the FitConfig's static initial values for any coefficient
        not in ``fitting_parameters`` (i.e. a fixed coefficient such as ``c=0``
        for the linear model). Returns ``None`` for the constant model — that
        path is handled by the profile-likelihood branch of ``log_prob``.
        """
        from .noise import GaussianLinear, GaussianQuadratic  # avoid cycle at import time
        if not isinstance(self.config.noise, (GaussianLinear, GaussianQuadratic)):
            return None
        defaults = {
            "var_a": float(self.config.var_a),
            "var_b": float(self.config.var_b),
            "var_c": float(self.config.var_c),
        }
        for par, v in zip(self.config.fitting_parameters, x):
            if par in self._VARIANCE_PARAM_NAMES:
                defaults[par] = float(v)
        return defaults["var_a"], defaults["var_b"], defaults["var_c"]

    def _log_prior_prob(self, x: np.ndarray) -> float:
        bounds_obj = self.config.bounds
        penalty = 0.0
        for par, v in zip(self.config.fitting_parameters, x):
            bounds = getattr(bounds_obj, par)
            penalty += log_wall(np.array([v]), bounds).sum()
        return -penalty * PRIOR_PENALTY_WEIGHT

    def _log_prob_data(self, x: np.ndarray, **kwargs) -> float:
        raise NotImplementedError

    def _objective(self, x: np.ndarray, **kwargs) -> float:
        return -(self._log_prob_data(x, **kwargs) + self._log_prior_prob(x))

    def _get_x0_and_bounds(self):
        """Build initial values and bounds arrays from config."""
        x0 = [getattr(self.config, par) for par in self.config.fitting_parameters]
        bounds = [getattr(self.config.bounds, par) for par in self.config.fitting_parameters]
        return x0, bounds

    def _run_minimize(self, x0, bounds, **kwargs) -> tuple[np.ndarray, bool, str, np.ndarray | None]:
        # Optimise in scale-relative coordinates z = x / scale so L-BFGS-B sees
        # O(1) variables regardless of the response magnitude. With raw
        # parameters, a large-magnitude asymptote (e.g. effect_inf ~ 1e5) carries
        # a correspondingly tiny gradient, so the projected-gradient stopping
        # criterion (pgtol) is met prematurely and the fit under-converges —
        # which makes the result depend on the absolute scale of the data.
        # Flooring the scale at 1.0 leaves already-O(1) parameters unchanged.
        x0 = np.asarray(x0, dtype=float)
        scale = np.maximum(np.abs(x0), 1.0)
        z0 = x0 / scale
        z_bounds = (
            [(lo / sc, hi / sc) for (lo, hi), sc in zip(bounds, scale)]
            if bounds is not None
            else None
        )

        def objective_z(z):
            return self._objective(z * scale, **kwargs)

        res = minimize(objective_z, z0, bounds=z_bounds, method="L-BFGS-B")
        x_opt = res.x * scale
        pcov = self._estimate_covariance(x_opt, bounds, **kwargs)
        return x_opt, res.success, res.message, pcov

    def _estimate_covariance(
        self,
        x_opt: np.ndarray,
        bounds: list[tuple[float, float]] | None = None,
        **kwargs,
    ) -> np.ndarray | None:
        """Estimate parameter covariance from the numerical Hessian at the optimum.

        When a parameter is pinned at one of its bounds, the unconstrained
        Hessian is not positive-definite at the optimum (the gradient in that
        direction doesn't vanish — the bound holds it back). Naively inverting
        the full Hessian then produces negative diagonals, masking the genuine
        information present in the *interior* parameters. Strategy:

        1. Identify active-bound parameters (within tolerance scaled to bound width).
        2. Take the principal sub-block of the Hessian over interior parameters.
        3. Invert that sub-block; place it back into a full-size matrix.
        4. Rows/cols for bound-active parameters stay zero — they're effectively
           fixed at the bound, so their CI is degenerate but interior CIs survive.

        If the interior sub-block is singular or has negative diagonals, return None.
        """
        n = len(x_opt)

        # Probe the Hessian in a *scaled* parameter space so the finite-difference
        # step is relative to each parameter's magnitude rather than a fixed
        # absolute ``_HESS_EPS``. Large-magnitude parameters — e.g. ELISA/RFU
        # asymptotes of order 1e4 — otherwise receive a ~1e-9 *relative* step,
        # deep in floating-point cancellation noise; the resulting second
        # derivative comes out with the wrong sign, the inverse Hessian has a
        # negative diagonal, and the whole covariance is rejected. Flooring the
        # scale at 1.0 leaves O(1) parameters (log_c50, hill, variance
        # coefficients) on their original, already-adequate step, so this only
        # changes behaviour for the ill-scaled case.
        x_opt = np.asarray(x_opt, dtype=float)
        scale = np.maximum(np.abs(x_opt), 1.0)
        z_opt = x_opt / scale

        def neg_ll_scaled(z):
            return -self._log_prob_data(z * scale, **kwargs)

        # Hessian in scaled coordinates: H_z[i, j] = scale_i * scale_j * H_x[i, j].
        H = np.zeros((n, n))
        eps = _HESS_EPS
        for i in range(n):
            def grad_i(z):
                return approx_fprime(z, neg_ll_scaled, eps)[i]
            H[i, :] = approx_fprime(z_opt, grad_i, eps)

        # Symmetrise
        H = 0.5 * (H + H.T)

        # Back-transform the scaled covariance to original units:
        # Σ_x = D Σ_z D with D = diag(scale), i.e. Σ_x[i, j] = scale_i scale_j Σ_z[i, j].
        # Positive scales preserve diagonal signs, so the negative-variance
        # rejection below is equivalent whether checked before or after.
        outer_scale = np.outer(scale, scale)

        # Identify bound-active parameters. Tolerance scales with bound width so
        # very narrow bounds don't get false positives from finite-difference noise.
        active = np.zeros(n, dtype=bool)
        if bounds is not None:
            for i, (lo, hi) in enumerate(bounds):
                width = float(hi) - float(lo)
                tol = max(1e-8, 1e-6 * width)
                if abs(x_opt[i] - lo) <= tol or abs(x_opt[i] - hi) <= tol:
                    active[i] = True

        try:
            if not np.any(active):
                pcov = np.linalg.inv(H) * outer_scale
                if np.any(np.diag(pcov) < 0):
                    return None
                return pcov

            # Invert only the interior sub-block; zero-fill bound-active rows/cols.
            interior = ~active
            if not np.any(interior):
                return None
            H_int = H[np.ix_(interior, interior)]
            pcov_int = np.linalg.inv(H_int)
            if np.any(np.diag(pcov_int) < 0):
                return None
            pcov = np.zeros((n, n))
            pcov[np.ix_(interior, interior)] = pcov_int
            pcov = pcov * outer_scale
            return pcov
        except np.linalg.LinAlgError:
            return None
