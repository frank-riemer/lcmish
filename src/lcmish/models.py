"""Core data containers for LCMish."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import csv
import json

import numpy as np


@dataclass
class SpectralData:
    """A single complex time-domain MR spectroscopy signal."""

    fid: np.ndarray
    dwell_time_s: float
    transmitter_mhz: float
    reference_ppm: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.fid = np.asarray(self.fid, dtype=np.complex128).reshape(-1)
        if self.fid.size < 8:
            raise ValueError("SpectralData requires at least 8 complex points")
        if self.dwell_time_s <= 0:
            raise ValueError("dwell_time_s must be positive")
        if self.transmitter_mhz <= 0:
            raise ValueError("transmitter_mhz must be positive")

    @property
    def npoints(self) -> int:
        return int(self.fid.size)

    def time_axis(self) -> np.ndarray:
        return np.arange(self.npoints, dtype=float) * self.dwell_time_s

    def ppm_axis(self, nfft: int | None = None) -> np.ndarray:
        nfft = int(nfft or self.npoints)
        freq_hz = np.fft.fftshift(np.fft.fftfreq(nfft, d=self.dwell_time_s))
        # Positive frequency offsets appear at lower ppm in the conventional MRS axis.
        return self.reference_ppm - freq_hz / self.transmitter_mhz


@dataclass
class BasisSet:
    """Time-domain basis signals and their labels."""

    names: list[str]
    fids: np.ndarray
    dwell_time_s: float | None = None
    transmitter_mhz: float | None = None
    reference_ppm: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.names = [str(x).strip() for x in self.names]
        self.fids = np.asarray(self.fids, dtype=np.complex128)
        if self.fids.ndim != 2:
            raise ValueError("BasisSet.fids must have shape (components, points)")
        if self.fids.shape[0] != len(self.names):
            raise ValueError("BasisSet names and component count do not match")
        if len(set(self.names)) != len(self.names):
            raise ValueError("Basis component names must be unique")

    @property
    def ncomponents(self) -> int:
        return len(self.names)

    @property
    def npoints(self) -> int:
        return int(self.fids.shape[1])


@dataclass(frozen=True)
class GroupConfig:
    """Shared nonlinear terms for a collection of basis components."""

    name: str
    members: tuple[str, ...]
    shift_bounds_ppm: tuple[float, float] | None = None
    lorentzian_bounds_hz: tuple[float, float] | None = None
    initial_shift_ppm: float = 0.0
    initial_lorentzian_hz: float = 0.0


@dataclass(frozen=True)
class FitConfig:
    """Numerical configuration for one spectral fit."""

    ppm_range: tuple[float, float]
    zero_fill_factor: int = 2
    nonnegative_amplitudes: bool = True
    baseline_knots: int = 14
    baseline_lambda: float = 1e-2
    global_shift_bounds_ppm: tuple[float, float] = (-0.25, 0.25)
    phase0_bounds_deg: tuple[float, float] = (-90.0, 90.0)
    phase1_bounds_deg_per_ppm: tuple[float, float] = (-30.0, 30.0)
    lorentzian_bounds_hz: tuple[float, float] = (0.0, 40.0)
    gaussian_bounds_hz: tuple[float, float] = (0.0, 40.0)
    initial_global_shift_ppm: float = 0.0
    initial_phase0_deg: float = 0.0
    initial_phase1_deg_per_ppm: float = 0.0
    initial_lorentzian_hz: float = 3.0
    initial_gaussian_hz: float = 3.0
    groups: tuple[GroupConfig, ...] = ()
    max_nfev: int = 300


@dataclass
class FitResult:
    """Result of a single LCMish fit."""

    names: list[str]
    ppm: np.ndarray
    data: np.ndarray
    fit: np.ndarray
    baseline: np.ndarray
    residual: np.ndarray
    components: dict[str, np.ndarray]
    amplitudes: np.ndarray
    amplitude_se: np.ndarray
    crlb_percent: np.ndarray
    nonlinear: dict[str, float]
    success: bool
    message: str
    cost: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for i, name in enumerate(self.names):
            rows.append(
                {
                    "component": name,
                    "amplitude": float(self.amplitudes[i]),
                    "conditional_se": float(self.amplitude_se[i]),
                    "conditional_crlb_like_percent": float(self.crlb_percent[i]),
                }
            )
        return rows

    def save_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.summary_rows()
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def save_components_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = ["ppm", "data", "fit", "baseline", "residual", *self.names]
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for i, ppm in enumerate(self.ppm):
                row = {
                    "ppm": float(ppm),
                    "data": float(self.data[i]),
                    "fit": float(self.fit[i]),
                    "baseline": float(self.baseline[i]),
                    "residual": float(self.residual[i]),
                }
                for name in self.names:
                    row[name] = float(self.components[name][i])
                writer.writerow(row)

    def save_table(self, path: str | Path, *, title: str = "LCMish fit", metadata: dict | None = None) -> None:
        path = Path(path)
        merged = dict(self.metadata)
        if metadata:
            merged.update(metadata)
        lines = [title, "=" * len(title), "", f"success: {self.success}", f"message: {self.message}", f"cost: {self.cost:.8g}", ""]
        for key, value in self.nonlinear.items():
            lines.append(f"{key}: {value:.8g}")
        lines += ["", "component\tamplitude\tconditional_se\tconditional_crlb_like_percent"]
        for row in self.summary_rows():
            lines.append(
                f"{row['component']}\t{row['amplitude']:.8g}\t{row['conditional_se']:.8g}\t{row['conditional_crlb_like_percent']:.5g}"
            )
        if merged:
            lines += ["", "metadata:", json.dumps(merged, indent=2, default=str)]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n")

    def save_coord(self, path: str | Path, *, title: str = "LCMish fit") -> None:
        """Write a simple, human-readable coordinate-style file.

        This is intentionally not claimed to be byte-compatible with LCModel .coord.
        """
        self.save_table(path, title=title)

    def save_checkpoint_npz(self, path: str | Path, metadata: dict | None = None) -> None:
        payload_meta = dict(self.metadata)
        if metadata:
            payload_meta.update(metadata)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            names=np.asarray(self.names, dtype=object),
            ppm=self.ppm,
            data=self.data,
            fit=self.fit,
            baseline=self.baseline,
            residual=self.residual,
            amplitudes=self.amplitudes,
            amplitude_se=self.amplitude_se,
            crlb_percent=self.crlb_percent,
            nonlinear_json=json.dumps(self.nonlinear),
            metadata_json=json.dumps(payload_meta, default=str),
        )

    def save_pdf(
        self,
        path: str | Path,
        *,
        title: str = "LCMish fit summary",
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Write a one-page PDF with spectrum, fit, residuals and result table."""
        from .report import save_pdf_report

        return save_pdf_report(self, path, title=title, metadata=metadata)

    def save_pdf_report(
        self,
        path: str | Path,
        *,
        title: str = "LCMish fit summary",
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Alias for :meth:`save_pdf` for callers who prefer the explicit name."""
        return self.save_pdf(path, title=title, metadata=metadata)

    def plot(self, path: str | Path | None = None, *, title: str = "LCMish fit"):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.plot(self.ppm, self.data, lw=1.0, label="data")
        ax.plot(self.ppm, self.fit, lw=1.1, label="fit")
        ax.plot(self.ppm, self.baseline, lw=0.9, ls="--", label="baseline")
        offset = np.nanmin(self.data) - 0.15 * max(np.ptp(self.data), 1.0)
        ax.plot(self.ppm, self.residual + offset, lw=0.8, label="residual (offset)")
        ax.axhline(offset, lw=0.5)
        ax.set_xlim(max(self.ppm), min(self.ppm))
        ax.set_xlabel("ppm")
        ax.set_ylabel("a.u.")
        ax.set_title(title)
        ax.legend(frameon=False, ncol=4)
        fig.tight_layout()
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=180, bbox_inches="tight")
            plt.close(fig)
        return fig


@dataclass
class FitAudit:
    """Multistart fitting record."""

    best: FitResult
    trials: list[FitResult]
    best_index: int
