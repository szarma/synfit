from importlib.metadata import version as _pkg_version, PackageNotFoundError as _PackageNotFoundError

try:
    __version__ = _pkg_version("synfit")
except _PackageNotFoundError:  # pragma: no cover - source tree without installed metadata
    __version__ = "0.0.0+unknown"

from .hill import hill_curve, log_wall, calculate_concentration_series
from .bliss import bliss_independence, bliss_reference, hsa_reference
from .loewe import loewe_ci, loewe_reference
from .zip import zip_delta, zip_reference
from .data import FitConfig, FitBounds, FitResult
from .noise import NoiseSpec, log_prob as noise_log_prob
from .single import (
    SingleDrugFit,
    SingleDrugFitWithError,
    default_bounds,
    default_fit_config,
)
from .matrix import MatrixFit
