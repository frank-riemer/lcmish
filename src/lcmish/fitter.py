"""Linear-combination spectral fitting with transparent numerical assumptions."""
from __future__ import annotations

from dataclasses import replace
import math
from typing import Iterable

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import least_squares, lsq_linear

from .models import BasisSet, FitAudit, FitConfig, FitResult, GroupConfig, SpectralData


def _next_pow_two(n: int) -> int:
    return 1 << (int(n - 1).bit_length())


def _resample_or_pad_fid(fid: np.ndarray, n: int) -> np.ndarray:
    fid = np.asarray(fid, dtype=np.complex128).reshape(-1)
    if fid.size >= n:
        return fid[:n].copy()
    out = np.zeros(n, dtype=np.complex128)
    out[: fid.size] = fid
    return out


def _fortran_nint(value: float) -> int:
    return int(np.copysign(np.floor(abs(value) + 0.5), value))


def _resize_unrearranged_spectrum(spec: np.ndarray, n: int) -> np.ndarray:
    """Truncate/zero-fill the two halves of an unrearranged FFT like MYBASI."""
    source = np.asarray(spec, dtype=np.complex128).reshape(-1)
    out = np.zeros(int(n), dtype=np.complex128)
    nhalf = min(source.size, out.size) // 2
    out[:nhalf] = source[:nhalf]
    out[-nhalf:] = source[-nhalf:]
    return out


def _match_basis_time_grid(
    basis: BasisSet, data: SpectralData, model_npoints: int
) -> np.ndarray:
    """Return model FIDs on the data grid, following LCModel where possible.

    LCModel constructs its basis functions over ``NDATA = 2*NUNFIL``, not
    merely over the acquired ``NUNFIL`` samples.  This matters when a long,
    narrow-band basis retains signal after the acquired data have ended.
    Generic basis inputs keep their historical acquired-duration semantics
    and are zero-filled after ``data.npoints``.
    """
    model_npoints = int(model_npoints)
    if model_npoints < data.npoints:
        raise ValueError("model_npoints cannot be shorter than the acquired data")
    if basis.dwell_time_s is None:
        return np.asarray(
            [
                _resample_or_pad_fid(
                    _resample_or_pad_fid(fid, data.npoints), model_npoints
                )
                for fid in basis.fids
            ]
        )

    basis_dt = float(basis.dwell_time_s)
    if bool(basis.metadata.get("lcmodel_basis")):
        # Recreate the zero-filled frequency array read by MYBASI, then perform
        # its bandwidth conversion by preserving the two unrearranged FFT halves.
        ndatab = int(basis.metadata.get("stored_ndatab", 2 * basis.npoints))
        basis_f0 = basis.metadata.get("basis_hzpppm") or basis.transmitter_mhz
        basis_f0 = float(basis_f0) if basis_f0 is not None else data.transmitter_mhz
        rndata_freq = (
            ndatab * basis_dt * basis_f0
            / (float(data.dwell_time_s) * float(data.transmitter_mhz))
        )
        ndata_freq = 2 * _fortran_nint(0.5 * rndata_freq)
        # LCModel 6.3 uses BWTOLR=.001 by default to avoid an immaterial
        # bandwidth conversion when the requested and internal grids agree.
        bwtolr = float(basis.metadata.get("bwtolr", 0.001))
        if abs(1.0 - rndata_freq / model_npoints) <= bwtolr:
            ndata_freq = model_npoints
        if ndata_freq <= 0:
            raise ValueError("LCModel basis/data grid conversion produced no samples")
        rows = []
        for fid in basis.fids:
            fid_zf = np.zeros(ndatab, dtype=np.complex128)
            fid_zf[: min(fid.size, ndatab)] = fid[:ndatab]
            stored_grid = np.fft.fft(fid_zf) / np.sqrt(ndatab)
            data_bw_grid = _resize_unrearranged_spectrum(stored_grid, ndata_freq)
            converted = np.fft.ifft(data_bw_grid) * np.sqrt(ndata_freq)
            rows.append(_resample_or_pad_fid(converted, model_npoints))
        out = np.asarray(rows)
    elif np.isclose(basis_dt, data.dwell_time_s, rtol=1e-7, atol=1e-12):
        out = np.asarray(
            [
                _resample_or_pad_fid(
                    _resample_or_pad_fid(fid, data.npoints), model_npoints
                )
                for fid in basis.fids
            ]
        )
    else:
        # Generic non-LCModel basis: interpolate its complex FID in physical time.
        t_basis = np.arange(basis.npoints, dtype=float) * basis_dt
        t_data = np.arange(data.npoints, dtype=float) * data.dwell_time_s
        rows = []
        for fid in basis.fids:
            rows.append(
                np.interp(t_data, t_basis, fid.real, left=0.0, right=0.0)
                + 1j * np.interp(t_data, t_basis, fid.imag, left=0.0, right=0.0)
            )
        out = np.asarray(
            [_resample_or_pad_fid(fid, model_npoints) for fid in rows]
        )

    reference_delta_hz = (
        float(data.reference_ppm) - float(basis.reference_ppm)
    ) * data.transmitter_mhz
    if not np.isclose(reference_delta_hz, 0.0, atol=1e-12):
        t = np.arange(model_npoints, dtype=float) * data.dwell_time_s
        out *= np.exp(1j * 2.0 * np.pi * reference_delta_hz * t)[None, :]
    return out


def _bspline_matrix(x: np.ndarray, n_basis: int, degree: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if n_basis <= degree:
        raise ValueError("baseline_knots must exceed spline degree")
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if math.isclose(xmin, xmax):
        return np.ones((x.size, 1))
    internal_count = n_basis - degree - 1
    internal = np.linspace(xmin, xmax, internal_count + 2)[1:-1] if internal_count > 0 else np.array([])
    knots = np.r_[[xmin] * (degree + 1), internal, [xmax] * (degree + 1)]
    cols = []
    for i in range(n_basis):
        coeff = np.zeros(n_basis)
        coeff[i] = 1.0
        cols.append(BSpline(knots, coeff, degree, extrapolate=True)(x))
    return np.column_stack(cols)


def _second_difference(n: int) -> np.ndarray:
    if n < 3:
        return np.zeros((0, n))
    d = np.zeros((n - 2, n))
    for i in range(n - 2):
        d[i, i : i + 3] = (1.0, -2.0, 1.0)
    return d


def _group_map(groups: tuple[GroupConfig, ...], names: list[str]) -> dict[str, GroupConfig]:
    present = set(names)
    mapping: dict[str, GroupConfig] = {}
    for group in groups:
        for member in group.members:
            if member in present and member not in mapping:
                mapping[member] = group
    return mapping


def _parameter_spec(config: FitConfig, basis: BasisSet):
    keys = ["global_shift_ppm", "phase0_deg", "phase1_deg_per_ppm", "lorentzian_hz", "gaussian_hz"]
    x0 = [
        config.initial_global_shift_ppm,
        config.initial_phase0_deg,
        config.initial_phase1_deg_per_ppm,
        config.initial_lorentzian_hz,
        config.initial_gaussian_hz,
    ]
    lower = [
        config.global_shift_bounds_ppm[0],
        config.phase0_bounds_deg[0],
        config.phase1_bounds_deg_per_ppm[0],
        config.lorentzian_bounds_hz[0],
        config.gaussian_bounds_hz[0],
    ]
    upper = [
        config.global_shift_bounds_ppm[1],
        config.phase0_bounds_deg[1],
        config.phase1_bounds_deg_per_ppm[1],
        config.lorentzian_bounds_hz[1],
        config.gaussian_bounds_hz[1],
    ]
    present = set(basis.names)
    active_groups: list[GroupConfig] = []
    for group in config.groups:
        if not any(member in present for member in group.members):
            continue
        active_groups.append(group)
        if group.shift_bounds_ppm is not None:
            keys.append(f"group_shift_ppm:{group.name}")
            x0.append(group.initial_shift_ppm)
            lower.append(group.shift_bounds_ppm[0])
            upper.append(group.shift_bounds_ppm[1])
        if group.lorentzian_bounds_hz is not None:
            keys.append(f"group_lorentzian_hz:{group.name}")
            x0.append(group.initial_lorentzian_hz)
            lower.append(group.lorentzian_bounds_hz[0])
            upper.append(group.lorentzian_bounds_hz[1])
    return keys, np.asarray(x0, float), np.asarray(lower, float), np.asarray(upper, float), active_groups


def _components_for_params(
    params: dict[str, float],
    *,
    data: SpectralData,
    basis: BasisSet,
    nfft: int,
    ppm: np.ndarray,
    group_by_name: dict[str, GroupConfig],
) -> np.ndarray:
    """Return complex metabolite spectra in the acquired-data phase frame."""
    t = np.arange(nfft, dtype=float) * data.dwell_time_s
    phase0 = np.deg2rad(params["phase0_deg"])
    phase1 = np.deg2rad(params["phase1_deg_per_ppm"])
    pivot = data.reference_ppm
    phase = np.exp(1j * (phase0 + phase1 * (ppm - pivot)))
    matched_fids = _match_basis_time_grid(basis, data, nfft)
    out = []
    for name, fid in zip(basis.names, matched_fids):
        shift_ppm = params["global_shift_ppm"]
        lor_hz = params["lorentzian_hz"]
        group = group_by_name.get(name)
        if group is not None:
            shift_ppm += params.get(f"group_shift_ppm:{group.name}", 0.0)
            lor_hz += params.get(f"group_lorentzian_hz:{group.name}", 0.0)
        shift_hz = shift_ppm * data.transmitter_mhz
        gauss_hz = max(params["gaussian_hz"], 0.0)
        lor = np.exp(-np.pi * max(lor_hz, 0.0) * t)
        # Approximate Gaussian FWHM broadening in the time domain.
        gauss = np.exp(-((np.pi * gauss_hz * t) ** 2) / (4.0 * np.log(2.0)))
        freq = np.exp(1j * 2.0 * np.pi * shift_hz * t)
        transformed = fid * lor * gauss * freq
        spec = np.fft.fftshift(np.fft.fft(transformed, n=nfft)) * phase
        out.append(spec)
    return np.column_stack(out)


def _regularised_linear_solution(
    y: np.ndarray,
    design: np.ndarray,
    *,
    n_metab: int,
    baseline_blocks: tuple[tuple[int, int], ...],
    config: FitConfig,
) -> np.ndarray:
    """Solve a real-valued bounded linear problem with smooth baselines."""
    regularisers = []
    if config.baseline_lambda > 0:
        for start, size in baseline_blocks:
            if size < 3:
                continue
            d2 = _second_difference(size)
            reg = np.zeros((d2.shape[0], design.shape[1]), dtype=float)
            reg[:, start : start + size] = np.sqrt(config.baseline_lambda) * d2
            regularisers.append(reg)
    if regularisers:
        reg = np.vstack(regularisers)
        a_aug = np.vstack([design, reg])
        y_aug = np.r_[y, np.zeros(reg.shape[0])]
    else:
        a_aug, y_aug = design, y
    lower = np.r_[
        np.zeros(n_metab)
        if config.nonnegative_amplitudes
        else np.full(n_metab, -np.inf),
        np.full(design.shape[1] - n_metab, -np.inf),
    ]
    upper = np.full(design.shape[1], np.inf)
    solved = lsq_linear(
        a_aug,
        y_aug,
        bounds=(lower, upper),
        method="trf",
        lsmr_tol="auto",
        max_iter=300,
    )
    return solved.x


def _solve_linear_real(y, metab, baseline, config: FitConfig):
    n_metab = metab.shape[1]
    n_base = baseline.shape[1]
    design = np.column_stack([metab, baseline])
    coeff = _regularised_linear_solution(
        y,
        design,
        n_metab=n_metab,
        baseline_blocks=((n_metab, n_base),),
        config=config,
    )
    return coeff, design


def _solve_linear_complex(y, metab, baseline, config: FitConfig):
    """Fit complex data with real metabolite amplitudes and complex splines."""
    n_metab = metab.shape[1]
    n_base = baseline.shape[1]
    zeros = np.zeros_like(baseline)
    design = np.block(
        [
            [metab.real, baseline, zeros],
            [metab.imag, zeros, baseline],
        ]
    )
    target = np.r_[y.real, y.imag]
    coeff = _regularised_linear_solution(
        target,
        design,
        n_metab=n_metab,
        baseline_blocks=((n_metab, n_base), (n_metab + n_base, n_base)),
        config=config,
    )
    baseline_complex = (
        baseline @ coeff[n_metab : n_metab + n_base]
        + 1j * baseline @ coeff[n_metab + n_base :]
    )
    prediction = metab @ coeff[:n_metab] + baseline_complex
    return coeff, design, target, prediction, baseline_complex


def fit_spectrum(data: SpectralData, basis: BasisSet, config: FitConfig) -> FitResult:
    """Fit one spectrum using nonlinear nuisance parameters and linear amplitudes."""
    fit_domain = str(config.fit_domain).strip().lower()
    if fit_domain not in {"complex", "real"}:
        raise ValueError("fit_domain must be 'complex' or 'real'")
    nfft = _next_pow_two(data.npoints * max(1, int(config.zero_fill_factor)))
    ppm_all = data.ppm_axis(nfft)
    spectrum_all = np.fft.fftshift(np.fft.fft(data.fid, n=nfft))
    lo, hi = sorted(config.ppm_range)
    mask = (ppm_all >= lo) & (ppm_all <= hi)
    if np.count_nonzero(mask) < 32:
        raise ValueError("Fit ppm range contains too few spectral points")
    ppm = ppm_all[mask]
    y_complex = spectrum_all[mask]
    y = y_complex if fit_domain == "complex" else y_complex.real
    baseline = _bspline_matrix(ppm, max(5, int(config.baseline_knots)))

    keys, x0, lower, upper, _active_groups = _parameter_spec(config, basis)
    group_by_name = _group_map(config.groups, basis.names)

    cache: dict[str, object] = {}

    def evaluate(x):
        params = dict(zip(keys, map(float, x)))
        metab_all = _components_for_params(params, data=data, basis=basis, nfft=nfft, ppm=ppm_all, group_by_name=group_by_name)
        metab = metab_all[mask]
        if fit_domain == "complex":
            coeff, design, target, pred_complex, baseline_complex = _solve_linear_complex(
                y_complex, metab, baseline, config
            )
            pred = np.r_[pred_complex.real, pred_complex.imag]
            residual = target - pred
            cache["target"] = target
            cache["pred_complex"] = pred_complex
            cache["baseline_complex"] = baseline_complex
        else:
            coeff, design = _solve_linear_real(y, metab.real, baseline, config)
            pred = design @ coeff
            residual = y - pred
        cache["params"] = params
        cache["metab"] = metab
        cache["coeff"] = coeff
        cache["design"] = design
        cache["pred"] = pred
        return residual

    optimum = least_squares(evaluate, x0, bounds=(lower, upper), max_nfev=config.max_nfev, method="trf")
    residual = evaluate(optimum.x)
    params = cache["params"]
    metab = np.asarray(cache["metab"])
    coeff = np.asarray(cache["coeff"])
    design = np.asarray(cache["design"])
    pred = np.asarray(cache["pred"])
    n_metab = basis.ncomponents
    amps = coeff[:n_metab]
    phase0 = np.deg2rad(params["phase0_deg"])
    phase1 = np.deg2rad(params["phase1_deg_per_ppm"])
    phase = np.exp(
        1j * (phase0 + phase1 * (ppm - float(data.reference_ppm)))
    )
    phase_correction = np.conj(phase)

    if fit_domain == "complex":
        baseline_raw = np.asarray(cache["baseline_complex"])
        fit_raw = np.asarray(cache["pred_complex"])
        data_display = y_complex * phase_correction
        fit_display = fit_raw * phase_correction
        baseline_display = baseline_raw * phase_correction
        residual_display = (y_complex - fit_raw) * phase_correction
        component_complex = {
            name: metab[:, i] * amps[i] * phase_correction
            for i, name in enumerate(basis.names)
        }
        components = {name: value.real for name, value in component_complex.items()}
        components_imag = {name: value.imag for name, value in component_complex.items()}
    else:
        base_coeff = coeff[n_metab:]
        baseline_fit = baseline @ base_coeff
        data_display = y.astype(np.complex128)
        fit_display = pred.astype(np.complex128)
        baseline_display = baseline_fit.astype(np.complex128)
        residual_display = residual.astype(np.complex128)
        components = {
            name: metab[:, i].real * amps[i]
            for i, name in enumerate(basis.names)
        }
        components_imag = {}

    target_size = 2 * y_complex.size if fit_domain == "complex" else y.size
    dof = max(1, target_size - design.shape[1] - len(keys))
    sigma2 = float(np.dot(residual, residual) / dof)
    try:
        cov = sigma2 * np.linalg.pinv(design.T @ design)
        se = np.sqrt(np.maximum(np.diag(cov)[:n_metab], 0.0))
    except np.linalg.LinAlgError:
        se = np.full(n_metab, np.nan)
    crlb = np.where(np.abs(amps) > np.finfo(float).eps, 100.0 * se / np.abs(amps), np.inf)

    return FitResult(
        names=list(basis.names),
        ppm=ppm,
        data=data_display.real,
        fit=fit_display.real,
        baseline=baseline_display.real,
        residual=residual_display.real,
        components=components,
        amplitudes=amps,
        amplitude_se=se,
        crlb_percent=crlb,
        nonlinear=params,
        success=bool(optimum.success),
        message=str(optimum.message),
        cost=float(optimum.cost),
        metadata={
            "nfft": int(nfft),
            "fit_ppm_range": [float(lo), float(hi)],
            "fit_domain": fit_domain,
            "display_domain": (
                "phase-corrected complex spectrum" if fit_domain == "complex" else "legacy unphased real spectrum"
            ),
            "phase_convention": (
                "phase parameters rotate the basis into the acquired-data frame; "
                "the inverse rotation is applied to data and fit for display"
            ),
            "uncertainty": "conditional linearised CRLB-like estimate; not LCModel %SD",
        },
        data_imag=data_display.imag if fit_domain == "complex" else None,
        fit_imag=fit_display.imag if fit_domain == "complex" else None,
        baseline_imag=baseline_display.imag if fit_domain == "complex" else None,
        residual_imag=residual_display.imag if fit_domain == "complex" else None,
        components_imag=components_imag,
    )


def fit_spectrum_multistart(
    data: SpectralData,
    basis: BasisSet,
    config: FitConfig,
    *,
    starts: Iterable[dict[str, float]] = ({},),
) -> FitAudit:
    """Run several initial conditions and retain the smallest-cost fit."""
    trials: list[FitResult] = []
    for override in starts:
        allowed = {
            "initial_global_shift_ppm",
            "initial_phase0_deg",
            "initial_phase1_deg_per_ppm",
            "initial_lorentzian_hz",
            "initial_gaussian_hz",
        }
        updates = {k: v for k, v in override.items() if k in allowed}
        trial_config = replace(config, **updates)
        trials.append(fit_spectrum(data, basis, trial_config))
    if not trials:
        raise ValueError("starts must contain at least one initial condition")
    costs = np.asarray([x.cost if np.isfinite(x.cost) else np.inf for x in trials])
    best_index = int(np.argmin(costs))
    return FitAudit(best=trials[best_index], trials=trials, best_index=best_index)
