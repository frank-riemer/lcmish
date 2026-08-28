"""Publication-friendly one-page PDF summaries for LCMish fits.

The layout is deliberately familiar to users of traditional MRS fitting software:
observed spectrum and fit at the top, residuals immediately below, and numerical
fit results on the same page. It is not a reproduction of LCModel output and is
labelled accordingly.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .models import FitResult


def _format_number(value: float, *, precision: int = 4) -> str:
    """Compact formatting that remains readable in a dense one-page table."""
    value = float(value)
    if not np.isfinite(value):
        return "inf" if value > 0 else "-inf" if value < 0 else "nan"
    if value == 0:
        return "0"
    if abs(value) >= 1e4 or abs(value) < 1e-3:
        return f"{value:.{max(1, precision - 1)}e}"
    return f"{value:.{precision}g}"


def _fit_metrics(result: "FitResult") -> dict[str, float]:
    """Return simple descriptive fit metrics without imposing QC thresholds."""
    data = np.asarray(result.data, dtype=float)
    fit = np.asarray(result.fit, dtype=float)
    residual = np.asarray(result.residual, dtype=float)
    if result.data_imag is not None and result.residual_imag is not None:
        data_norm = np.hypot(
            np.linalg.norm(data), np.linalg.norm(np.asarray(result.data_imag, dtype=float))
        )
        residual_norm = np.hypot(
            np.linalg.norm(residual),
            np.linalg.norm(np.asarray(result.residual_imag, dtype=float)),
        )
    else:
        data_norm = np.linalg.norm(data)
        residual_norm = np.linalg.norm(residual)
    data_norm = max(float(data_norm), np.finfo(float).tiny)
    relative_residual = float(residual_norm / data_norm)
    if np.std(data) <= np.finfo(float).tiny or np.std(fit) <= np.finfo(float).tiny:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(data, fit)[0, 1])
    ss_tot = float(np.sum((data - np.mean(data)) ** 2))
    r_squared = float(1.0 - np.sum(residual**2) / ss_tot) if ss_tot > 0 else float("nan")
    return {
        "fit_correlation": correlation,
        "relative_residual": relative_residual,
        "r_squared": r_squared,
    }


def save_pdf_report(
    result: "FitResult",
    path: str | Path,
    *,
    title: str = "LCMish fit summary",
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write an LCModel-style, single-page PDF fit summary.

    Parameters
    ----------
    result
        Completed :class:`~lcmish.models.FitResult`.
    path
        Destination PDF path.
    title
        Page title. Kept separate from provenance labelling.
    metadata
        Optional short metadata fields to show in the numerical summary.

    Notes
    -----
    The report is intentionally *LCModel-style* only in the broad sense of putting
    the spectrum, fit, residual and concentration table on one page. It is not an
    LCModel ``.ps``/``.pdf`` output clone and should not be described as one.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ppm = np.asarray(result.ppm, dtype=float)
    data = np.asarray(result.data, dtype=float)
    fit = np.asarray(result.fit, dtype=float)
    baseline = np.asarray(result.baseline, dtype=float)
    residual = np.asarray(result.residual, dtype=float)
    metrics = _fit_metrics(result)

    merged_meta = dict(result.metadata)
    if metadata:
        merged_meta.update(metadata)

    # A4 portrait, because one-page spectroscopy summaries tend to end up in
    # lab books, supplementary PDFs and printers whose world view remains A4.
    fig = plt.figure(figsize=(8.27, 11.69))
    grid = fig.add_gridspec(
        5,
        3,
        height_ratios=(0.34, 3.6, 1.15, 0.18, 3.2),
        width_ratios=(1.2, 1.2, 1.0),
        hspace=0.16,
        wspace=0.16,
    )

    ax_title = fig.add_subplot(grid[0, :])
    ax_title.axis("off")
    ax_title.text(0.0, 0.72, title, fontsize=15, fontweight="bold", va="center")
    status = "fit converged" if result.success else "fit did not converge"
    ax_title.text(
        1.0,
        0.72,
        f"LCMish  |  {status}",
        fontsize=9.5,
        ha="right",
        va="center",
    )
    ax_title.text(
        0.0,
        0.12,
        "One-page linear-combination fit summary (LCModel-style layout; not LCModel output)",
        fontsize=8.3,
        va="center",
    )

    ax_spec = fig.add_subplot(grid[1, :])
    ax_spec.plot(ppm, data, linewidth=0.85, label="data")
    ax_spec.plot(ppm, fit, linewidth=1.05, label="fit")
    ax_spec.plot(ppm, baseline, linewidth=0.8, linestyle="--", label="baseline")
    ax_spec.set_xlim(float(np.max(ppm)), float(np.min(ppm)))
    ax_spec.set_ylabel("signal (a.u.)")
    ax_spec.tick_params(labelbottom=False)
    ax_spec.legend(frameon=False, ncol=3, fontsize=8, loc="upper right")
    ax_spec.set_title("Observed spectrum and fitted model", fontsize=9.5, loc="left")

    ax_res = fig.add_subplot(grid[2, :], sharex=ax_spec)
    ax_res.plot(ppm, residual, linewidth=0.75, label="real residual")
    if result.residual_imag is not None:
        ax_res.plot(
            ppm,
            np.asarray(result.residual_imag, dtype=float),
            linewidth=0.65,
            linestyle=":",
            label="imaginary residual",
        )
        ax_res.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper right")
    ax_res.axhline(0.0, linewidth=0.55)
    ax_res.set_xlim(float(np.max(ppm)), float(np.min(ppm)))
    ax_res.set_ylabel("residual")
    ax_res.set_xlabel("chemical shift (ppm)")
    ax_res.set_title("Residual", fontsize=9.5, loc="left")

    ax_sep = fig.add_subplot(grid[3, :])
    ax_sep.axis("off")
    ax_sep.plot([0, 1], [0.5, 0.5], transform=ax_sep.transAxes, linewidth=0.6)

    table_ax = fig.add_subplot(grid[4, 0:2])
    info_ax = fig.add_subplot(grid[4, 2])
    table_ax.axis("off")
    info_ax.axis("off")

    rows = result.summary_rows()
    headers = ["Component", "Amplitude", "SE", "CRLB-ish %"]
    cell_text = [
        [
            str(row["component"]),
            _format_number(row["amplitude"]),
            _format_number(row["conditional_se"]),
            _format_number(row["conditional_crlb_like_percent"], precision=3),
        ]
        for row in rows
    ]
    table = table_ax.table(
        cellText=cell_text,
        colLabels=headers,
        colWidths=[0.31, 0.24, 0.21, 0.24],
        cellLoc="right",
        colLoc="right",
        loc="upper left",
        bbox=[0.0, 0.0, 1.0, 0.95],
    )
    table.auto_set_font_size(False)
    # Keep all supplied components on the one-page report. Large 1H basis sets
    # will be small in print but remain crisp and zoomable in the vector PDF.
    fontsize = 7.3 if len(rows) <= 16 else 6.4 if len(rows) <= 24 else 5.4
    table.set_fontsize(fontsize)
    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(0.35)
        if c == 0:
            cell.get_text().set_ha("left")
        if r == 0:
            cell.get_text().set_fontweight("bold")
    table_ax.set_title("Components", fontsize=9.5, loc="left", pad=3)

    info_lines = [
        "Fit summary",
        f"success: {result.success}",
        f"cost: {_format_number(result.cost)}",
        f"correlation: {_format_number(metrics['fit_correlation'])}",
        f"relative residual: {_format_number(metrics['relative_residual'])}",
        f"R^2: {_format_number(metrics['r_squared'])}",
        "",
        "Nonlinear parameters",
    ]
    for key, value in result.nonlinear.items():
        info_lines.append(f"{key}: {_format_number(value)}")

    # Metadata is deliberately selective. Dumping an arbitrary nested dictionary
    # into an A4 page is a surprisingly efficient way of making all of it useless.
    simple_meta = []
    for key, value in merged_meta.items():
        if isinstance(value, (str, int, float, bool)) and len(str(value)) <= 42:
            simple_meta.append((str(key), str(value)))
    if simple_meta:
        info_lines += ["", "Metadata"]
        for key, value in simple_meta[:7]:
            info_lines.append(f"{key}: {value}")

    info_ax.text(
        0.0,
        0.98,
        "\n".join(info_lines),
        va="top",
        ha="left",
        fontsize=7.0,
        linespacing=1.28,
        family="monospace",
    )

    fig.text(
        0.5,
        0.012,
        "LCMish reports conditional linearised CRLB-like uncertainty, not LCModel %SD. "
        "Always inspect the fit; the optimiser has never met your metabolite.",
        ha="center",
        va="bottom",
        fontsize=6.6,
    )

    with PdfPages(path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return path
