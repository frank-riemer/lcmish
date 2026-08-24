"""Experimental local NAD-region fitting for phosphorus MRS.

The single-spectrum model is acquisition-agnostic. The convenience CSI route
is deliberately narrower: it expects reconstructed complex 2-D CSI data and
an explicit study-specific voxel mask, then performs voxel QC, PCr alignment,
coherent combination and the local fit. It is not a universal preprocessing
pipeline for other acquisition types.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from scipy.optimize import least_squares, lsq_linear

from .compat import trapezoid
from .models import CSIData, SpectralData


@dataclass(frozen=True)
class P31RedoxConfig:
    """Literature-constrained local NAD+/NADH model for unedited 31P spectra.

    Chemical shifts use PCr = 0 ppm. The NAD+ two-spin constants are from
    Lu et al. (Magn Reson Med 2014;71:1959-1972): D=0.3211 ppm and J=20.03 Hz.
    Components are normalized by phosphorus count, so equal NAD+ and NADH
    coefficients represent equal molecule amounts.
    """

    ppm_range: tuple[float, float] = (-9.0, -6.5)
    nad_plus_center_ppm: float = -8.312
    nadh_ppm: float = -8.130
    alpha_atp_center_ppm: float = -7.560
    nad_spin_separation_ppm: float = 0.3211
    nad_j_hz: float = 20.03
    alpha_atp_j_hz: float = 15.5
    shift_bounds_ppm: tuple[float, float] = (-0.08, 0.08)
    alpha_relative_shift_bounds_ppm: tuple[float, float] = (-0.15, 0.15)
    phase0_bounds_deg: tuple[float, float] = (-30.0, 30.0)
    phase1_bounds_deg_per_ppm: tuple[float, float] = (-20.0, 20.0)
    nad_linewidth_bounds_hz: tuple[float, float] = (4.0, 35.0)
    alpha_extra_linewidth_bounds_hz: tuple[float, float] = (0.0, 12.0)
    initial_nad_linewidth_hz: float = 10.0
    initial_alpha_extra_linewidth_hz: float = 1.5
    baseline_order: int = 2
    include_nucleotide_sugar_nuisance: bool = False
    nucleotide_sugar_center_ppm: float = -8.20
    nucleotide_sugar_j_hz: float = 20.5
    bootstrap_repeats: int = 0
    random_seed: int = 20260821


@dataclass
class P31RedoxResult:
    """Result of the experimental single-spectrum local NAD-region fit."""

    names: tuple[str, ...]
    amplitudes: np.ndarray
    amplitude_se: np.ndarray
    crlb_percent: np.ndarray
    nonlinear: dict[str, float]
    ppm: np.ndarray
    data: np.ndarray
    fit: np.ndarray
    baseline: np.ndarray
    components: dict[str, np.ndarray]
    residual: np.ndarray
    success: bool
    message: str
    cost: float
    fit_correlation: float
    relative_residual: float
    bootstrap_amplitudes: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def amplitude(self, name: str) -> float:
        return float(self.amplitudes[self.names.index(name)])

    def component_is_estimable(
        self, name: str, *, max_crlb_percent: float | None = None
    ) -> bool:
        index = self.names.index(name)
        amplitude = float(self.amplitudes[index])
        if not np.isfinite(amplitude) or amplitude <= np.finfo(float).eps:
            return False
        crlb = float(self.crlb_percent[index])
        return max_crlb_percent is None or (
            np.isfinite(crlb) and crlb <= float(max_crlb_percent)
        )

    @property
    def apparent_redox_ratio(self) -> float:
        """Return NAD+/NADH, or NaN when either component is on the boundary."""
        if not (
            self.component_is_estimable("NAD_plus")
            and self.component_is_estimable("NADH")
        ):
            return float("nan")
        return self.amplitude("NAD_plus") / self.amplitude("NADH")

    @property
    def nad_plus_over_nadh(self) -> float:
        """Alias for :attr:`apparent_redox_ratio`."""
        return self.apparent_redox_ratio


@dataclass(frozen=True)
class P31CSIRedoxQCConfig:
    """Explicit QC thresholds for the masked 2-D CSI redox workflow.

    Thresholds are required because they are study- and acquisition-specific.
    They should be prespecified and reported rather than treated as universal
    LCMish defaults.
    """

    pcr_snr_min: float
    min_retained_voxels: int
    local_fit_correlation_min: float
    local_relative_residual_max: float
    max_component_crlb_percent: float | None = None
    pcr_window_ppm: tuple[float, float] = (-0.6, 0.6)
    noise_windows_ppm: tuple[tuple[float, float], ...] = (
        (8.5, 9.8),
        (-19.8, -18.0),
    )
    phase_window_ppm: tuple[float, float] = (-0.6, 0.6)
    target_pcr_ppm: float = 0.0
    begin_time_s: float = 0.0
    max_nuisance_log_ratio_change: float | None = None

    def __post_init__(self) -> None:
        if self.pcr_snr_min < 0:
            raise ValueError("pcr_snr_min must be non-negative")
        if self.min_retained_voxels < 1:
            raise ValueError("min_retained_voxels must be at least one")
        if not -1.0 <= self.local_fit_correlation_min <= 1.0:
            raise ValueError("local_fit_correlation_min must be between -1 and 1")
        if self.local_relative_residual_max <= 0:
            raise ValueError("local_relative_residual_max must be positive")


@dataclass
class P31CSIPreparationResult:
    """Auditable output of masked voxel QC, alignment and combination."""

    combined: SpectralData
    supplied_mask: np.ndarray
    retained_mask: np.ndarray
    pcr_snr: np.ndarray
    retained_voxel_indices: tuple[tuple[int, int], ...]
    observed_pcr_ppm: np.ndarray
    zero_order_phase_deg: np.ndarray
    excluded_reasons: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_masked(self) -> int:
        return int(np.count_nonzero(self.supplied_mask))

    @property
    def n_retained(self) -> int:
        return int(np.count_nonzero(self.retained_mask))


@dataclass
class P31CSIRedoxResult:
    """Result and QC audit for the masked 2-D CSI convenience workflow."""

    preparation: P31CSIPreparationResult
    primary: P31RedoxResult
    nuisance: P31RedoxResult | None
    qc_pass: bool
    qc_reasons: tuple[str, ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def apparent_redox_ratio(self) -> float:
        """Return the ratio only when workflow and component QC both pass."""
        return self.primary.apparent_redox_ratio if self.qc_pass else float("nan")


def nad_plus_ab_pattern(
    transmitter_mhz: float, config: P31RedoxConfig | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return the field-dependent NAD+ quartet positions and relative areas."""
    config = config or P31RedoxConfig()
    delta_hz = config.nad_spin_separation_ppm * float(transmitter_mhz)
    splitting_hz = np.sqrt(delta_hz**2 + config.nad_j_hz**2)
    outer_hz = (splitting_hz + config.nad_j_hz) / 2.0
    inner_hz = (splitting_hz - config.nad_j_hz) / 2.0
    offsets_hz = np.array([-outer_hz, -inner_hz, inner_hz, outer_hz])
    weak = (splitting_hz - config.nad_j_hz) / splitting_hz
    strong = (splitting_hz + config.nad_j_hz) / splitting_hz
    weights = np.array([weak, strong, strong, weak]) / 2.0
    positions = config.nad_plus_center_ppm + offsets_hz / float(transmitter_mhz)
    return positions, weights


def _multiplet_fid(
    time: np.ndarray,
    transmitter_mhz: float,
    positions_ppm: np.ndarray | list[float],
    weights: np.ndarray | list[float],
    linewidth_hz: float,
    shift_ppm: float,
) -> np.ndarray:
    positions = np.asarray(positions_ppm, dtype=float) + float(shift_ppm)
    weights_array = np.asarray(weights, dtype=float)
    oscillators = np.exp(
        -1j
        * 2.0
        * np.pi
        * positions[:, None]
        * float(transmitter_mhz)
        * time[None, :]
    )
    decay = np.exp(-np.pi * float(linewidth_hz) * time)
    return (weights_array @ oscillators) * decay


def _component_spectra(
    data: SpectralData,
    config: P31RedoxConfig,
    nfft: int,
    shift_ppm: float,
    nad_linewidth_hz: float,
    alpha_extra_linewidth_hz: float,
    alpha_relative_shift_ppm: float,
    phase0_deg: float,
    phase1_deg_per_ppm: float,
) -> tuple[tuple[str, ...], np.ndarray]:
    time = data.time_axis()
    nad_positions, nad_weights = nad_plus_ab_pattern(data.transmitter_mhz, config)
    fids = [
        _multiplet_fid(
            time,
            data.transmitter_mhz,
            nad_positions,
            nad_weights,
            nad_linewidth_hz,
            shift_ppm,
        ),
        _multiplet_fid(
            time,
            data.transmitter_mhz,
            [config.nadh_ppm],
            [2.0],
            nad_linewidth_hz,
            shift_ppm,
        ),
        _multiplet_fid(
            time,
            data.transmitter_mhz,
            [
                config.alpha_atp_center_ppm
                + alpha_relative_shift_ppm
                - config.alpha_atp_j_hz / (2.0 * data.transmitter_mhz),
                config.alpha_atp_center_ppm
                + alpha_relative_shift_ppm
                + config.alpha_atp_j_hz / (2.0 * data.transmitter_mhz),
            ],
            [0.5, 0.5],
            nad_linewidth_hz + alpha_extra_linewidth_hz,
            shift_ppm,
        ),
    ]
    names = ["NAD_plus", "NADH", "alpha_ATP"]
    if config.include_nucleotide_sugar_nuisance:
        half_split = config.nucleotide_sugar_j_hz / (2.0 * data.transmitter_mhz)
        fids.append(
            _multiplet_fid(
                time,
                data.transmitter_mhz,
                [
                    config.nucleotide_sugar_center_ppm - half_split,
                    config.nucleotide_sugar_center_ppm + half_split,
                ],
                [0.5, 0.5],
                nad_linewidth_hz,
                shift_ppm,
            )
        )
        names.append("nucleotide_sugar_nuisance")
    spectra = np.fft.fftshift(np.fft.fft(np.stack(fids), n=nfft, axis=1), axes=1)
    ppm = data.ppm_axis(nfft)
    phase = np.deg2rad(
        phase0_deg + phase1_deg_per_ppm * (ppm - config.alpha_atp_center_ppm)
    )
    spectra *= np.exp(1j * phase)[None, :]
    return tuple(names), spectra


def fit_p31_redox(
    data: SpectralData,
    config: P31RedoxConfig | None = None,
    *,
    nfft: int | None = None,
) -> P31RedoxResult:
    """Fit alpha-ATP, NAD+ and NADH in the local upfield alpha-ATP region."""
    config = config or P31RedoxConfig()
    nfft = int(nfft or max(4096, 4 * data.npoints))
    ppm_full = data.ppm_axis(nfft)
    spectrum_full = data.spectrum(nfft)
    lo, hi = sorted(config.ppm_range)
    mask = (ppm_full >= lo) & (ppm_full <= hi)
    ppm = ppm_full[mask]
    y = spectrum_full.real[mask]
    if ppm.size < 32:
        raise ValueError("The redox fitting window contains fewer than 32 points")
    x = (ppm - ppm.mean()) / max(float(np.ptp(ppm)) / 2.0, np.finfo(float).eps)
    baseline_columns = np.stack(
        [x**degree for degree in range(config.baseline_order + 1)], axis=1
    )
    cache: dict[str, Any] = {}

    def solve_linear(theta: np.ndarray, target: np.ndarray = y):
        names, spectra = _component_spectra(
            data,
            config,
            nfft,
            theta[0],
            theta[1],
            theta[2],
            theta[3],
            theta[4],
            theta[5],
        )
        component_matrix = spectra[:, mask].real.T
        design = np.column_stack([component_matrix, baseline_columns])
        nmet = len(names)
        lower = np.r_[
            np.zeros(nmet), np.full(baseline_columns.shape[1], -np.inf)
        ]
        linear = lsq_linear(
            design,
            target,
            bounds=(lower, np.full(design.shape[1], np.inf)),
            method="trf",
        )
        cache.update(names=names, design=design, linear=linear)
        return design @ linear.x, linear

    lower = np.array(
        [
            config.shift_bounds_ppm[0],
            config.nad_linewidth_bounds_hz[0],
            config.alpha_extra_linewidth_bounds_hz[0],
            config.alpha_relative_shift_bounds_ppm[0],
            config.phase0_bounds_deg[0],
            config.phase1_bounds_deg_per_ppm[0],
        ]
    )
    upper = np.array(
        [
            config.shift_bounds_ppm[1],
            config.nad_linewidth_bounds_hz[1],
            config.alpha_extra_linewidth_bounds_hz[1],
            config.alpha_relative_shift_bounds_ppm[1],
            config.phase0_bounds_deg[1],
            config.phase1_bounds_deg_per_ppm[1],
        ]
    )
    initial = np.array(
        [
            0.0,
            config.initial_nad_linewidth_hz,
            config.initial_alpha_extra_linewidth_hz,
            0.0,
            0.0,
            0.0,
        ]
    )
    nonlinear = least_squares(
        lambda theta: solve_linear(theta)[0] - y,
        initial,
        bounds=(lower, upper),
        max_nfev=180,
    )
    fitted, linear = solve_linear(nonlinear.x)
    names = cache["names"]
    design = cache["design"]
    nmet = len(names)
    amplitudes = linear.x[:nmet]
    baseline = baseline_columns @ linear.x[nmet:]
    component_curves = design[:, :nmet].T * amplitudes[:, None]
    residual_values = y - fitted
    dof = max(1, y.size - linear.x.size - nonlinear.x.size)
    noise_variance = float(residual_values @ residual_values / dof)
    covariance = noise_variance * np.linalg.pinv(design.T @ design)
    amplitude_se = np.sqrt(np.maximum(0.0, np.diag(covariance)[:nmet]))

    boot = None
    if config.bootstrap_repeats > 0:
        rng = np.random.default_rng(config.random_seed)
        boot = np.full((config.bootstrap_repeats, nmet), np.nan)
        for index in range(config.bootstrap_repeats):
            y_boot = fitted + rng.choice(
                residual_values, size=residual_values.size, replace=True
            )

            def boot_solve(theta: np.ndarray):
                return solve_linear(theta, y_boot)

            boot_nonlinear = least_squares(
                lambda theta: boot_solve(theta)[0] - y_boot,
                nonlinear.x,
                bounds=(lower, upper),
                max_nfev=100,
            )
            boot[index] = boot_solve(boot_nonlinear.x)[1].x[:nmet]
        amplitude_se = np.nanstd(boot, axis=0, ddof=1)

    crlb = np.full(nmet, np.inf)
    positive = amplitudes > np.finfo(float).eps
    crlb[positive] = 100.0 * amplitude_se[positive] / amplitudes[positive]
    correlation = (
        float(np.corrcoef(y, fitted)[0, 1])
        if np.std(y) > 0 and np.std(fitted) > 0
        else float("nan")
    )
    relative_residual = float(
        np.linalg.norm(residual_values)
        / max(np.linalg.norm(y), np.finfo(float).eps)
    )
    return P31RedoxResult(
        names=names,
        amplitudes=amplitudes,
        amplitude_se=amplitude_se,
        crlb_percent=crlb,
        nonlinear={
            "common_shift_ppm": float(nonlinear.x[0]),
            "nad_linewidth_hz": float(nonlinear.x[1]),
            "alpha_extra_linewidth_hz": float(nonlinear.x[2]),
            "alpha_relative_shift_ppm": float(nonlinear.x[3]),
            "phase0_deg": float(nonlinear.x[4]),
            "phase1_deg_per_ppm": float(nonlinear.x[5]),
        },
        ppm=ppm,
        data=y,
        fit=fitted,
        baseline=baseline,
        components={
            name: component_curves[index]
            for index, name in enumerate(names)
        },
        residual=residual_values,
        success=bool(nonlinear.success and linear.success),
        message=str(nonlinear.message),
        cost=float(np.sum(residual_values**2)),
        fit_correlation=correlation,
        relative_residual=relative_residual,
        bootstrap_amplitudes=boot,
        metadata={
            "model": "Lu_2014_field_specific_NAD_plus_AB_quartet_and_NADH_singlet",
            "experimental": True,
            "unedited_spectrum_warning": True,
            "phosphorus_count_normalized": True,
            "nfft": nfft,
            "nucleotide_sugar_nuisance": config.include_nucleotide_sugar_nuisance,
        },
    )


def redox_nuisance_sensitivity(
    data: SpectralData,
    config: P31RedoxConfig | None = None,
    *,
    nfft: int | None = None,
) -> tuple[P31RedoxResult, P31RedoxResult]:
    """Fit the same spectrum without and with a nucleotide-sugar term."""
    config = config or P31RedoxConfig()
    primary = fit_p31_redox(
        data,
        replace(config, include_nucleotide_sugar_nuisance=False),
        nfft=nfft,
    )
    nuisance = fit_p31_redox(
        data,
        replace(config, include_nucleotide_sugar_nuisance=True),
        nfft=nfft,
    )
    return primary, nuisance


def _window_mask(ppm: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    return (ppm >= min(window)) & (ppm <= max(window))


def _integrate(
    ppm: np.ndarray, values: np.ndarray, window: tuple[float, float]
) -> np.ndarray:
    mask = _window_mask(ppm, window)
    if np.count_nonzero(mask) < 2:
        raise ValueError(f"Integration window {window} contains fewer than two points")
    order = np.argsort(ppm[mask])
    return trapezoid(values[..., mask][..., order], ppm[mask][order], axis=-1)


def p31_csi_pcr_snr(
    data: CSIData,
    *,
    pcr_window_ppm: tuple[float, float] = (-0.6, 0.6),
    noise_windows_ppm: tuple[tuple[float, float], ...] = (
        (8.5, 9.8),
        (-19.8, -18.0),
    ),
) -> np.ndarray:
    """Return a robust, phase-independent PCr peak-SNR map."""
    ppm = data.ppm_axis()
    spectra = data.spectra()
    pcr_mask = _window_mask(ppm, pcr_window_ppm)
    noise_mask = np.zeros(ppm.shape, dtype=bool)
    for window in noise_windows_ppm:
        noise_mask |= _window_mask(ppm, window)
    if np.count_nonzero(pcr_mask) < 3:
        raise ValueError("PCr window contains fewer than three spectral points")
    if np.count_nonzero(noise_mask) < 8:
        raise ValueError("Noise windows contain fewer than eight spectral points")
    noise = spectra[..., noise_mask]
    real_sd = np.median(
        np.abs(noise.real - np.median(noise.real, axis=-1, keepdims=True)),
        axis=-1,
    ) / 0.67448975
    imag_sd = np.median(
        np.abs(noise.imag - np.median(noise.imag, axis=-1, keepdims=True)),
        axis=-1,
    ) / 0.67448975
    complex_sd = np.sqrt((real_sd**2 + imag_sd**2) / 2.0)
    peak = np.max(np.abs(spectra[..., pcr_mask]), axis=-1)
    return peak / np.maximum(complex_sd, np.finfo(float).tiny)


def _quadratic_peak_ppm(
    ppm: np.ndarray, magnitude: np.ndarray, window: tuple[float, float]
) -> float:
    indices = np.flatnonzero(_window_mask(ppm, window))
    if indices.size < 3:
        raise ValueError("PCr search window contains fewer than three points")
    peak = int(indices[np.argmax(magnitude[indices])])
    result = float(ppm[peak])
    if 0 < peak < ppm.size - 1:
        local_ppm = ppm[peak - 1 : peak + 2]
        local_magnitude = magnitude[peak - 1 : peak + 2]
        coefficients = np.polyfit(local_ppm, local_magnitude, 2)
        if coefficients[0] < 0:
            vertex = float(-coefficients[1] / (2.0 * coefficients[0]))
            if min(local_ppm) <= vertex <= max(local_ppm):
                result = vertex
    return result


def prepare_p31_csi_redox(
    data: CSIData,
    voxel_mask: np.ndarray,
    qc: P31CSIRedoxQCConfig,
) -> P31CSIPreparationResult:
    """QC, PCr-align and coherently combine masked 2-D CSI voxels.

    The supplied mask is interpreted only as a study-specific inclusion mask;
    LCMish does not infer anatomy or scanner orientation.
    """
    supplied_mask = np.asarray(voxel_mask, dtype=bool)
    if supplied_mask.shape != data.spatial_shape:
        raise ValueError("voxel_mask must match the 2-D CSI spatial shape")
    if not np.any(supplied_mask):
        raise ValueError("voxel_mask selects no voxels")
    snr = p31_csi_pcr_snr(
        data,
        pcr_window_ppm=qc.pcr_window_ppm,
        noise_windows_ppm=qc.noise_windows_ppm,
    )
    retained_mask = supplied_mask & np.isfinite(snr) & (snr >= qc.pcr_snr_min)
    indices = tuple(
        tuple(int(value) for value in index) for index in np.argwhere(retained_mask)
    )
    if not indices:
        raise ValueError("No masked CSI voxel passed the configured PCr-SNR threshold")

    excluded_reasons: dict[str, str] = {}
    for row, column in np.argwhere(supplied_mask & ~retained_mask):
        key = f"{int(row)},{int(column)}"
        excluded_reasons[key] = "non-finite or below-threshold PCr SNR"

    ppm = data.ppm_axis()
    time = np.arange(data.npoints, dtype=float) * data.dwell_time_s
    corrected_spectra: list[np.ndarray] = []
    observed: list[float] = []
    phases: list[float] = []
    for row, column in indices:
        raw_spectrum = data.voxel(row, column).spectrum()
        observed_ppm = _quadratic_peak_ppm(
            ppm, np.abs(raw_spectrum), qc.pcr_window_ppm
        )
        shift_hz = (observed_ppm - qc.target_pcr_ppm) * data.transmitter_mhz
        aligned_fid = data.fids[row, column] * np.exp(
            1j * 2.0 * np.pi * shift_hz * time
        )
        aligned_spectrum = np.fft.fftshift(np.fft.fft(aligned_fid))
        phase = -float(np.angle(_integrate(ppm, aligned_spectrum, qc.phase_window_ppm)))
        phased = aligned_spectrum * np.exp(1j * phase)
        if np.real(_integrate(ppm, phased, qc.phase_window_ppm)) < 0:
            phased = -phased
            phase += np.pi
        frequency_hz = (data.reference_ppm - ppm) * data.transmitter_mhz
        phased *= np.exp(-1j * 2.0 * np.pi * frequency_hz * qc.begin_time_s)
        corrected_spectra.append(phased)
        observed.append(observed_ppm)
        phases.append(float((np.degrees(phase) + 180.0) % 360.0 - 180.0))

    corrected_fids = np.fft.ifft(
        np.fft.ifftshift(np.asarray(corrected_spectra), axes=-1), axis=-1
    )
    combined_fid = np.mean(corrected_fids, axis=0)
    metadata = dict(data.metadata)
    metadata.update(
        {
            "workflow": "experimental_masked_2d_csi_redox",
            "mask_is_study_specific": True,
            "n_masked_voxels": int(np.count_nonzero(supplied_mask)),
            "n_retained_voxels": len(indices),
            "retained_voxel_indices": [list(index) for index in indices],
            "voxel_combination": "equal_weight_complex_mean_after_PCr_alignment",
            "pcr_snr_min": float(qc.pcr_snr_min),
            "begin_time_s": float(qc.begin_time_s),
        }
    )
    combined = SpectralData(
        combined_fid,
        data.dwell_time_s,
        data.transmitter_mhz,
        data.reference_ppm,
        metadata,
    )
    return P31CSIPreparationResult(
        combined=combined,
        supplied_mask=supplied_mask,
        retained_mask=retained_mask,
        pcr_snr=snr,
        retained_voxel_indices=indices,
        observed_pcr_ppm=np.asarray(observed),
        zero_order_phase_deg=np.asarray(phases),
        excluded_reasons=excluded_reasons,
        metadata=metadata,
    )


def fit_p31_csi_redox(
    data: CSIData,
    voxel_mask: np.ndarray,
    qc: P31CSIRedoxQCConfig,
    redox_config: P31RedoxConfig | None = None,
    *,
    run_nucleotide_sugar_sensitivity: bool = True,
    nfft: int | None = None,
) -> P31CSIRedoxResult:
    """Run the explicitly masked 2-D CSI preparation, QC and local redox fit.

    Passing these checks does not by itself validate participant-level redox
    quantification. For a study using group-composite spectra, call
    :func:`prepare_p31_csi_redox` per scan and construct the prespecified
    composites in the study analysis before calling :func:`fit_p31_redox`.
    """
    preparation = prepare_p31_csi_redox(data, voxel_mask, qc)
    redox_config = redox_config or P31RedoxConfig()
    if run_nucleotide_sugar_sensitivity:
        primary, nuisance = redox_nuisance_sensitivity(
            preparation.combined, redox_config, nfft=nfft
        )
    else:
        primary = fit_p31_redox(preparation.combined, redox_config, nfft=nfft)
        nuisance = None

    reasons: list[str] = []
    if preparation.n_retained < qc.min_retained_voxels:
        reasons.append(
            f"retained {preparation.n_retained} voxels; requires {qc.min_retained_voxels}"
        )
    if not primary.success:
        reasons.append("local redox optimizer did not converge")
    if not np.isfinite(primary.fit_correlation) or (
        primary.fit_correlation < qc.local_fit_correlation_min
    ):
        reasons.append(
            f"local fit correlation {primary.fit_correlation:.4g} below threshold"
        )
    if not np.isfinite(primary.relative_residual) or (
        primary.relative_residual > qc.local_relative_residual_max
    ):
        reasons.append(
            f"local relative residual {primary.relative_residual:.4g} above threshold"
        )
    for component in ("NAD_plus", "NADH"):
        if not primary.component_is_estimable(
            component, max_crlb_percent=qc.max_component_crlb_percent
        ):
            reasons.append(f"{component} is boundary-limited or insufficiently precise")

    diagnostics: dict[str, Any] = {
        "primary_apparent_redox_ratio": primary.apparent_redox_ratio,
        "nuisance_apparent_redox_ratio": None,
        "nuisance_log_ratio_change": None,
    }
    if nuisance is not None:
        primary_ratio = primary.apparent_redox_ratio
        nuisance_ratio = nuisance.apparent_redox_ratio
        diagnostics["nuisance_apparent_redox_ratio"] = nuisance_ratio
        if (
            np.isfinite(primary_ratio)
            and primary_ratio > 0
            and np.isfinite(nuisance_ratio)
            and nuisance_ratio > 0
        ):
            log_change = float(abs(np.log(nuisance_ratio / primary_ratio)))
            diagnostics["nuisance_log_ratio_change"] = log_change
            if (
                qc.max_nuisance_log_ratio_change is not None
                and log_change > qc.max_nuisance_log_ratio_change
            ):
                reasons.append("apparent ratio is not stable to nucleotide-sugar modelling")
        elif qc.max_nuisance_log_ratio_change is not None:
            reasons.append("nucleotide-sugar sensitivity ratio is not estimable")

    return P31CSIRedoxResult(
        preparation=preparation,
        primary=primary,
        nuisance=nuisance,
        qc_pass=not reasons,
        qc_reasons=tuple(reasons),
        diagnostics=diagnostics,
    )
