"""Static matplotlib figures for synfit results."""
import io
from collections.abc import Mapping

# Force the non-interactive Agg backend before any pyplot import. Django's
# threaded runserver crashes the default TkAgg backend with "main thread is
# not in main loop" when figures are created off the main thread (e.g. during
# demo pre-seed). Agg is fine for server-side PNG generation, which is all
# this module does.
import matplotlib
matplotlib.use("Agg", force=True)

import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.patheffects as mpe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np
import pandas as pd

from .hill import hill_curve
from .data import FitResult
from .bliss import bliss_independence, bliss_deviation, bliss_reference, hsa_deviation
from .loewe import loewe_ci
from .zip import zip_delta

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.facecolor": "#fafafa",
        "figure.facecolor": "white",
        "axes.grid": False,
        "savefig.dpi": 180,
        "savefig.transparent": False,
    }
)

_FIT_BLUE = "#2563eb"
_EXCLUDED = "#ef4444"


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("#fafafa")


def _fmt_conc(c: float) -> str:
    """Axis tick label for a concentration, rounded to three significant digits."""
    return f"{c:.3g}"


def _symlog_linthresh(min_positive_concentration: float) -> float:
    """Start the linear region at the lower decade of the smallest dose."""
    if not np.isfinite(min_positive_concentration) or min_positive_concentration <= 0:
        raise ValueError("min_positive_concentration must be finite and positive")
    return float(10.0 ** np.floor(np.log10(min_positive_concentration)))


def _save_fig_png(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def raw_scatter_plot(data: pd.DataFrame) -> bytes:
    """
    Scatter plot of raw readout vs concentration (log x-axis).

    No fitted curve — just the data. Intended as a pre-fit preview.
    Expected columns: concentration, y, replicate.
    Returns PNG bytes.
    """
    fig, ax = plt.subplots(figsize=(5, 3.5))
    _style_axes(ax)

    conc_nonzero = data.query("concentration > 0")["concentration"]
    has_zero = (data["concentration"] == 0).any()

    if has_zero:
        linthresh = _symlog_linthresh(float(conc_nonzero.min()))
        ax.set_xscale("symlog", linthresh=linthresh)
    else:
        ax.set_xscale("log")

    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (rep, grp) in enumerate(data.groupby("replicate")):
        color = palette[i % len(palette)]
        ax.scatter(
            grp["concentration"].values,
            grp["y"].values,
            s=36,
            color=color,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.6,
            label=str(rep),
        )

    ax.set_xlabel("Concentration")
    ax.set_ylabel("Response")
    if data["replicate"].nunique() > 1:
        ax.legend(fontsize=7, framealpha=0.7)
    fig.tight_layout()
    return _save_fig_png(fig)


def raw_replicate_heatmap(data: pd.DataFrame) -> bytes:
    """
    Heatmap of readout values: concentration (columns) × replicate (rows).

    Expected columns: concentration, y, replicate.
    Returns PNG bytes.
    """
    pivot = data.pivot_table(index="replicate", columns="concentration", values="y")
    pivot = pivot.sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(max(5, pivot.shape[1] * 0.5 + 1.5), max(2, pivot.shape[0] * 0.5 + 1)))
    _style_axes(ax)

    m = np.ma.masked_invalid(pivot.values.astype(float))
    im = ax.imshow(m, aspect="auto", cmap="viridis")

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([_fmt_conc(float(c)) for c in pivot.columns], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([str(rep) for rep in pivot.index], fontsize=7)
    ax.set_xlabel("Concentration")
    ax.set_ylabel("Replicate")

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, fraction=0.035, pad=0.02)
    cbar.set_label("Response", fontsize=9)

    if pivot.shape[0] <= 10 and pivot.shape[1] <= 15:
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if not np.isfinite(val):
                    continue
                txt = ax.text(
                    j, i, f"{val:.2g}",
                    ha="center", va="center", fontsize=6, color="#0f172a",
                )
                txt.set_path_effects(
                    [mpe.withStroke(linewidth=2.5, foreground="white", alpha=0.85)]
                )

    fig.tight_layout()
    return _save_fig_png(fig)


def dose_response_plot(
    data: pd.DataFrame,
    result: FitResult,
    valids: np.ndarray | None = None,
    *,
    reference: Mapping[str, float] | None = None,
    reference_label: str = "Ground truth",
    x_label: str | None = None,
    title: str | None = None,
) -> bytes:
    """
    Dose-response scatter, fitted curve, and optional reference curve.

    Returns PNG bytes.
    Expected columns: concentration, y, replicate.

    ``reference`` may contain ``c50``, ``hill``, ``effect_0``, ``effect_inf``
    and optionally ``asymmetry`` — for example, parameters used to generate a
    synthetic dataset. It is plotted as a dashed curve. ``x_label``
    and ``title`` label the finished plot. The x-axis is symmetric-log when
    data contains zero concentration and log otherwise. A lognormal fit uses a
    log y-axis only when every plotted response and confidence limit is positive.
    """
    required_columns = {"concentration", "y", "replicate"}
    if not required_columns.issubset(data.columns):
        raise ValueError("data must have columns: concentration, y, replicate")

    concentration = np.asarray(data["concentration"], dtype=float)
    positive = concentration[np.isfinite(concentration) & (concentration > 0)]
    if positive.size == 0:
        raise ValueError("dose_response_plot requires at least one positive concentration")

    if valids is None:
        valids = np.ones(len(data), dtype=bool)
    else:
        valids = np.asarray(valids, dtype=bool)
        if valids.shape != (len(data),):
            raise ValueError("valids must be a one-dimensional mask aligned with data")

    reference_kwargs: dict[str, float] | None = None
    if reference is not None:
        required_reference = {"c50", "hill", "effect_0", "effect_inf"}
        missing = required_reference.difference(reference)
        if missing:
            raise ValueError(
                "reference must contain c50, hill, effect_0 and effect_inf; "
                f"missing {', '.join(sorted(missing))}"
            )
        reference_kwargs = {name: float(reference[name]) for name in required_reference}
        if "asymmetry" in reference:
            reference_kwargs["asymmetry"] = float(reference["asymmetry"])

    fig, ax = plt.subplots(figsize=(5, 3.5))
    _style_axes(ax)

    conc_nonzero = positive
    has_zero = np.any(concentration == 0)

    xr = np.geomspace(conc_nonzero.min() * 0.2, conc_nonzero.max() * 4)
    if has_zero:
        xr = np.r_[0.0, xr]
    yr = hill_curve(
        xr,
        c50=result.c50,
        hill=result.hill,
        effect_0=result.effect_0,
        effect_inf=result.effect_inf,
        asymmetry=result.asymmetry,
    )

    if has_zero:
        linthresh = _symlog_linthresh(float(conc_nonzero.min()))
        ax.set_xscale("symlog", linthresh=linthresh)
        ax.plot(xr, yr, color=_FIT_BLUE, lw=1.5, label="Fit")
    else:
        ax.semilogx(xr, yr, color=_FIT_BLUE, lw=1.5, label="Fit")

    reference_y: np.ndarray | None = None
    if reference_kwargs is not None:
        reference_y = hill_curve(xr, **reference_kwargs)
        ax.plot(
            xr,
            reference_y,
            color="crimson",
            lw=1.5,
            linestyle="--",
            label=reference_label,
        )

    # Confidence band (delta method)
    ci_band = result.predict_ci(xr)
    if ci_band is not None:
        lo, hi = ci_band
        ax.fill_between(xr, lo, hi, color=_FIT_BLUE, alpha=0.15, linewidth=0, label="95% CI")

    # Lognormal fits use a multiplicative response model. Show that geometry on
    # a logarithmic y-axis only when every rendered curve/data value is valid
    # there; otherwise retain a readable linear scale.
    y_for_scale = [np.asarray(data["y"], dtype=float), yr]
    if ci_band is not None:
        y_for_scale.extend(ci_band)
    if reference_y is not None:
        y_for_scale.append(reference_y)
    finite_y = np.concatenate([values[np.isfinite(values)] for values in y_for_scale])
    if result.error_model == "lognormal" and finite_y.size and np.all(finite_y > 0):
        ax.set_yscale("log")

    plot_data = data.copy()
    plot_data["_valid"] = valids
    excluded_labelled = False
    for i, (_, grp) in enumerate(plot_data.groupby("replicate")):
        mask = grp["_valid"].to_numpy()
        ax.plot(
            grp["concentration"].values[mask],
            grp["y"].values[mask], "o", mfc='w', mec='gray',
            label="Observations" if i == 0 else None,
        )
        if (~mask).any():
            ax.scatter(
                grp["concentration"].values[~mask],
                grp["y"].values[~mask],
                s=40,
                color=_EXCLUDED,
                marker="x",
                linewidths=1.5,
                label=None if excluded_labelled else "excluded",
            )
            excluded_labelled = True

    ax.set_xlabel("Concentration" if x_label is None else x_label)
    ax.set_ylabel("Response")
    if title is not None:
        ax.set_title(title)

    cis = result.param_ci()
    if result.direction == "activation":
        label = "EC₅₀"
    else:
        label = "IC₅₀"

    if cis and "c50" in cis:
        c50_lo, c50_hi = cis["c50"]
        annotation = f"{label} = {result.c50:.3g} [{c50_lo:.3g}, {c50_hi:.3g}]"
    else:
        annotation = f"{label} = {result.c50:.3g}"

    if cis and "hill" in cis:
        h_lo, h_hi = cis["hill"]
        annotation += f"\nHill = {result.hill:.2f} [{h_lo:.2f}, {h_hi:.2f}]"
    else:
        annotation += f"\nHill = {result.hill:.2f}"
    ax.text(
        0.05,
        0.05,
        annotation,
        transform=ax.transAxes,
        va="bottom",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=9, loc="best")

    fig.tight_layout()
    return _save_fig_png(fig)


def matrix_heatmap(
    matrix: np.ndarray,
    conc_horizontal: np.ndarray,
    conc_vertical: np.ndarray,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
    norm: mcolors.Normalize | None = None,
    cbar_label: str | None = None,
    title: str | None = None,
) -> bytes:
    """
    Heatmap of a drug-drug interaction matrix.

    Returns PNG bytes.

    Pass a pre-built ``norm`` (e.g. ``LogNorm``, ``Normalize``) to control the
    colour scale. If omitted, falls back to ``Normalize(vmin, vmax)``.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    _style_axes(ax)

    n_ver, n_hor = matrix.shape
    m = np.ma.masked_invalid(matrix.astype(float))
    finite = m.compressed()

    if norm is None:
        if vmin is not None and vmax is not None:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        else:
            norm = mcolors.Normalize(
                vmin=float(np.nanmin(matrix)) if finite.size else 0.0,
                vmax=float(np.nanmax(matrix)) if finite.size else 1.0,
            )

    im = ax.imshow(m, aspect="auto", origin="lower", cmap=cmap, norm=norm)

    xlabels = [_fmt_conc(float(c)) for c in conc_horizontal]
    ylabels = [_fmt_conc(float(c)) for c in conc_vertical]
    ax.set_xticks(range(n_hor))
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n_ver))
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xlabel("Horizontal drug concentration")
    ax.set_ylabel("Vertical drug concentration")

    if title:
        ax.set_title(title, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, shrink=0.42, fraction=0.035, pad=0.02)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=9)

    if n_ver <= 10 and n_hor <= 10:
        for i in range(n_ver):
            for j in range(n_hor):
                val = matrix[i, j]
                if not np.isfinite(val):
                    continue
                txt = ax.text(
                    j,
                    i,
                    f"{val:.2g}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="#0f172a",
                )
                txt.set_path_effects(
                    [mpe.withStroke(linewidth=2.5, foreground="white", alpha=0.85)]
                )

    fig.tight_layout()
    return _save_fig_png(fig)


def raw_matrix_heatmap(
    matrix: np.ndarray,
    conc_horizontal: np.ndarray,
    conc_vertical: np.ndarray,
    label: str = "",
) -> bytes:
    """
    Heatmap of raw replicate response values using a logarithmic colour scale.

    Raises ``ValueError`` if any value is ≤ 0 (``LogNorm`` requires strictly positive data).
    Returns PNG bytes.
    """
    if np.any(matrix <= 0):
        raise ValueError(
            "Raw matrix contains non-positive values; LogNorm requires strictly positive data."
        )
    norm = mcolors.LogNorm(vmin=float(np.nanmin(matrix)), vmax=float(np.nanmax(matrix)))
    return matrix_heatmap(
        matrix,
        conc_horizontal,
        conc_vertical,
        cmap="plasma",
        norm=norm,
        cbar_label="Response",
        title=label or None,
    )


def hsa_heatmap(
    mean_matrix: np.ndarray,
    conc_horizontal: np.ndarray,
    conc_vertical: np.ndarray,
    resp_horizontal: np.ndarray,
    resp_vertical: np.ndarray,
    effect_0: float,
    effect_inf: float,
) -> bytes:
    """
    Heatmap of deviation from HSA (Highest Single Agent) null model.

    Negative (red) = synergy, positive (blue) = antagonism. Fixed ``[-1, 1]`` colour range.
    Returns PNG bytes.
    """
    dev = hsa_deviation(
        mean_matrix,
        resp_horizontal,
        resp_vertical,
        effect_0=effect_0,
        effect_inf=effect_inf,
        conc_hor=conc_horizontal,
        conc_ver=conc_vertical,
    )
    return matrix_heatmap(
        dev,
        conc_horizontal,
        conc_vertical,
        norm=mcolors.Normalize(vmin=-1.0, vmax=1.0),
        cmap="bwr",
        cbar_label="HSA deviation",
    )


def deviation_heatmap(
    mean_matrix: np.ndarray,
    conc_horizontal: np.ndarray,
    conc_vertical: np.ndarray,
    c50_hor: float,
    c50_ver: float,
    hill_hor: float,
    hill_ver: float,
    effect_0: float,
    effect_inf: float,
) -> bytes:
    """
    Heatmap of deviation from Bliss independence.

    Negative (red) = synergy, positive (blue) = antagonism. Fixed ``[-1, 1]`` colour range.
    Returns PNG bytes.
    """
    bliss = bliss_independence(
        conc_horizontal,
        conc_vertical,
        c50_hor=c50_hor,
        c50_ver=c50_ver,
        hill_hor=hill_hor,
        hill_ver=hill_ver,
        effect_0=effect_0,
        effect_inf=effect_inf,
    )
    dev = bliss_deviation(mean_matrix, bliss, effect_0=effect_0, effect_inf=effect_inf,
                          conc_hor=conc_horizontal, conc_ver=conc_vertical)
    return matrix_heatmap(
        dev,
        conc_horizontal,
        conc_vertical,
        norm=mcolors.Normalize(vmin=-1.0, vmax=1.0),
        cmap="bwr",
        cbar_label="Bliss deviation",
    )


def zip_heatmap(
    mean_matrix: np.ndarray,
    conc_horizontal: np.ndarray,
    conc_vertical: np.ndarray,
    c50_hor: float,
    c50_ver: float,
    hill_hor: float,
    hill_ver: float,
    effect_0: float,
    effect_inf: float,
) -> bytes:
    """
    Heatmap of ZIP delta-scores (Zero Interaction Potency).

    Negative (red) = synergy, positive (blue) = antagonism. Fixed ``[-1, 1]`` colour range.
    Returns PNG bytes.
    """
    delta = zip_delta(
        mean_matrix,
        conc_horizontal,
        conc_vertical,
        c50_hor=c50_hor,
        c50_ver=c50_ver,
        hill_hor=hill_hor,
        hill_ver=hill_ver,
        effect_0=effect_0,
        effect_inf=effect_inf,
    )
    return matrix_heatmap(
        delta,
        conc_horizontal,
        conc_vertical,
        norm=mcolors.Normalize(vmin=-1.0, vmax=1.0),
        cmap="bwr",
        cbar_label="ZIP δ-score",
    )


def loewe_heatmap(
    mean_matrix: np.ndarray,
    conc_horizontal: np.ndarray,
    conc_vertical: np.ndarray,
    c50_hor: float,
    c50_ver: float,
    hill_hor: float,
    hill_ver: float,
    effect_0: float,
    effect_inf: float,
) -> bytes:
    """
    Heatmap of the Loewe Combination Index.

    CI < 1 (blue) = synergy, CI = 1 (white) = additivity, CI > 1 (red) = antagonism.
    Colour scale: ``LogNorm([0.25, 4])`` centred at 1 in log space.
    Cells where the observed response is outside the single-agent range are NaN (grey).
    Returns PNG bytes.
    """
    ci = loewe_ci(
        conc_horizontal,
        conc_vertical,
        c50_hor=c50_hor,
        c50_ver=c50_ver,
        hill_hor=hill_hor,
        hill_ver=hill_ver,
        mean_matrix=mean_matrix,
        effect_0=effect_0,
        effect_inf=effect_inf,
    )
    return matrix_heatmap(
        ci,
        conc_horizontal,
        conc_vertical,
        norm=mcolors.LogNorm(vmin=0.25, vmax=4.0),
        cmap="bwr",
        cbar_label="Combination Index",
    )


def synergy_heatmaps(
    mean_matrix: np.ndarray,
    conc_horizontal: np.ndarray,
    conc_vertical: np.ndarray,
    *,
    c50_hor: float,
    c50_ver: float,
    hill_hor: float,
    hill_ver: float,
    effect_0: float,
    effect_inf: float,
    asymmetry_hor: float | None = None,
    asymmetry_ver: float | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    title: str | None = None,
) -> bytes:
    """Plot Bliss, HSA, Loewe and ZIP scores on the combination-dose grid.

    The observed zero-dose matrix edges provide the Bliss and HSA single-agent
    responses.  Zero-dose rows and columns are excluded from every panel,
    because they are single-agent observations rather than combination scores.
    Bliss, HSA and ZIP use a fixed linear range [-1, 1], with negative values
    denoting synergy. Loewe CI uses a logarithmic range [0.25, 4], with CI < 1
    denoting synergy. Blue denotes synergy, white the null and red antagonism.
    """
    matrix = np.asarray(mean_matrix, dtype=float)
    ch = np.asarray(conc_horizontal, dtype=float)
    cv = np.asarray(conc_vertical, dtype=float)
    if matrix.shape != (len(cv), len(ch)):
        raise ValueError("mean_matrix shape must be (len(conc_vertical), len(conc_horizontal))")
    if ch.ndim != 1 or cv.ndim != 1 or not np.all(np.isfinite(ch)) or not np.all(np.isfinite(cv)):
        raise ValueError("concentration arrays must be one-dimensional and finite")
    if np.any(ch < 0) or np.any(cv < 0):
        raise ValueError("concentrations must be non-negative")

    zero_hor = np.flatnonzero(ch == 0)
    zero_ver = np.flatnonzero(cv == 0)
    if len(zero_hor) != 1 or len(zero_ver) != 1:
        raise ValueError("synergy_heatmaps requires one zero-dose row and column")
    interior_hor = ch > 0
    interior_ver = cv > 0
    if not np.any(interior_hor) or not np.any(interior_ver):
        raise ValueError("synergy_heatmaps requires positive doses for both drugs")

    resp_horizontal = matrix[zero_ver[0], :]
    resp_vertical = matrix[:, zero_hor[0]]
    bliss = bliss_reference(resp_horizontal, resp_vertical, effect_0, effect_inf)
    bliss_delta = bliss_deviation(matrix, bliss, effect_0, effect_inf, ch, cv)
    hsa_delta = hsa_deviation(matrix, resp_horizontal, resp_vertical, effect_0, effect_inf, ch, cv)
    loewe = loewe_ci(
        ch, cv, c50_hor, c50_ver, hill_hor, hill_ver, matrix, effect_0, effect_inf,
        asymmetry_hor=1.0 if asymmetry_hor is None else asymmetry_hor,
        asymmetry_ver=1.0 if asymmetry_ver is None else asymmetry_ver,
    )
    zip_score = zip_delta(
        matrix, ch, cv, c50_hor, c50_ver, hill_hor, hill_ver, effect_0, effect_inf,
        asymmetry_hor=asymmetry_hor, asymmetry_ver=asymmetry_ver,
    )

    panels = [
        (bliss_delta, "Bliss", "Fractional deviation", mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)),
        (hsa_delta, "HSA", "Fractional deviation", mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)),
        (loewe, "Loewe", "Combination Index", mcolors.LogNorm(vmin=0.25, vmax=4)),
        (zip_score, "ZIP", "Fractional delta", mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)),
    ]
    colors = mcolors.LinearSegmentedColormap.from_list(
        "synergy", ["#1b3a8c", "#ffffff", "#a31515"], N=256,
    )
    colors.set_bad("#d4d4d4")
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    x_values, y_values = ch[interior_hor], cv[interior_ver]
    for ax, (values, panel_title, colorbar_label, norm) in zip(axes.flat, panels):
        _style_axes(ax)
        interior = values[np.ix_(interior_ver, interior_hor)]
        image = ax.imshow(np.ma.masked_invalid(interior), origin="lower", aspect="auto", cmap=colors, norm=norm)
        ax.set_box_aspect(interior.shape[0] / interior.shape[1])
        ax.set_title(panel_title)
        ax.set_xticks(range(len(x_values)), [_fmt_conc(float(x)) for x in x_values], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(y_values)), [_fmt_conc(float(y)) for y in y_values], fontsize=7)
        ax.set_xlabel(x_label or "Horizontal drug concentration")
        ax.set_ylabel(y_label or "Vertical drug concentration")
        colorbar = fig.colorbar(image, ax=ax, shrink=0.82, label=colorbar_label, extend="both")
        if isinstance(norm, mcolors.LogNorm):
            colorbar.set_ticks([0.25, 0.5, 1, 2, 4], labels=["0.25", "0.5", "1", "2", "4"])
    if title:
        fig.suptitle(title)
    fig.subplots_adjust(wspace=.4, hspace=.4)
    return _save_fig_png(fig)
