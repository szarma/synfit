"""Static matplotlib figures for synfit results."""
import io

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
from .bliss import bliss_independence, bliss_deviation, hsa_deviation
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
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("bottom", "left"):
        sp = ax.spines[side]
        sp.set_linewidth(0.8)
        sp.set_color("#94a3b8")


def _fmt_conc(c: float) -> str:
    """Axis tick label for a concentration (scientific if |c| < 0.01 and nonzero)."""
    if c == 0:
        return "0"
    ac = abs(c)
    if ac < 0.01:
        s = np.format_float_scientific(c, precision=0, exp_digits=1)
        return s.replace("e+0", "e").replace("e-0", "e-").replace("E", "e")
    return f"{c:g}"


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
        linthresh = float(conc_nonzero.min()) / 10.0
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

    xlabels = [_fmt_conc(float(c)) for c in pivot.columns]
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([str(r) for r in pivot.index], fontsize=7)
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
) -> bytes:
    """
    Dose-response scatter + fitted curve.

    Returns PNG bytes.
    Expected columns: concentration, y, replicate.
    """
    fig, ax = plt.subplots(figsize=(5, 3.5))
    _style_axes(ax)

    conc_nonzero = data.query("concentration > 0")["concentration"]
    has_zero = (data["concentration"] == 0).any()

    xr = np.geomspace(conc_nonzero.min() * 0.2, conc_nonzero.max() * 4)
    yr = hill_curve(
        xr,
        c50=result.c50,
        hill=result.hill,
        effect_0=result.effect_0,
        effect_inf=result.effect_inf,
        asymmetry=result.asymmetry,
    )

    if has_zero:
        linthresh = float(conc_nonzero.min()) / 10.0
        ax.set_xscale("symlog", linthresh=linthresh)
        ax.plot(xr, yr, color=_FIT_BLUE, lw=1.5, label="fit")
    else:
        ax.semilogx(xr, yr, color=_FIT_BLUE, lw=1.5, label="fit")

    # Confidence band (delta method)
    ci_band = result.predict_ci(xr)
    if ci_band is not None:
        lo, hi = ci_band
        ax.fill_between(xr, lo, hi, color=_FIT_BLUE, alpha=0.15, linewidth=0, label="95% CI")

    if valids is None:
        valids = np.ones(len(data), dtype=bool)

    for rep, grp in data.groupby("replicate"):
        mask = valids[grp.index.values]
        ax.scatter(
            grp["concentration"].values[mask],
            grp["y"].values[mask],
            s=36,
            color=_FIT_BLUE,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.6,
        )
        if (~mask).any():
            ax.scatter(
                grp["concentration"].values[~mask],
                grp["y"].values[~mask],
                s=40,
                color=_EXCLUDED,
                marker="x",
                linewidths=1.5,
                label="excluded",
            )

    ax.set_xlabel("Concentration")
    ax.set_ylabel("Response")

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
        fontsize=9,
        bbox={
            "facecolor": "#f1f5f9",
            "edgecolor": "#cbd5e1",
            "boxstyle": "round,pad=0.35",
        },
    )

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
