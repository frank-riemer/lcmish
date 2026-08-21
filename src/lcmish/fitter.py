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
    t = np.arange(data.npoints, dtype=float) * data.dwell_time_s
    phase0 = np.deg2rad(params["phase0_deg"])
    phase1 = np.deg2rad(params["phase1_deg_per_ppm"])
    pivot = data.reference_ppm
    phase = np.exp(1j * (phase0 + phase1 * (ppm - pivot)))
    out = []
    for name, source in zip(basis.names, basis.fids):
        fid = _resample_or_pad_fid(source, data.npoints)
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
        out.append(spec.real)
    return np.column_stack(out)


def _solve_linear(y, metab, baseline, config: FitConfig):
    n_metab = metab.shape[1]
    n_base = baseline.shape[1]
    design = np.column_stack([metab, baseline])
    if n_base >= 3 and config.baseline_lambda > 0:
        d2 = _second_difference(n_base)
        reg = np.column_stack([np.zeros((d2.shape[0], n_metab)), np.sqrt(config.baseline_lambda) * d2])
        a_aug = np.vstack([design, reg])
        y_aug = np.r_[y, np.zeros(reg.shape[0])]
    else:
        a_aug, y_aug = design, y
    lower = np.r_[np.zeros(n_metab) if config.nonnegative_amplitudes else np.full(n_metab, -np.inf), np.full(n_base, -np.inf)]
    upper = np.full(n_metab + n_base, np.inf)
    solved = lsq_linear(a_aug, y_aug, bounds=(lower, upper), method="trf", lsmr_tol="auto", max_iter=300)
    return solved.x, design


def fit_spectrum(data: SpectralData, basis: BasisSet, config: FitConfig) -> FitResult:
    """Fit one spectrum using nonlinear nuisance parameters and linear amplitudes."""
    nfft = _next_pow_two(data.npoints * max(1, int(config.zero_fill_factor)))
    ppm_all = data.ppm_axis(nfft)
    spectrum_all = np.fft.fftshift(np.fft.fft(data.fid, n=nfft)).real
    lo, hi = sorted(config.ppm_range)
    mask = (ppm_all >= lo) & (ppm_all <= hi)
    if np.count_nonzero(mask) < 32:
        raise ValueError("Fit ppm range contains too few spectral points")
    ppm = ppm_all[mask]
    y = spectrum_all[mask]
    baseline = _bspline_matrix(ppm, max(5, int(config.baseline_knots)))

    keys, x0, lower, upper, _active_groups = _parameter_spec(config, basis)
    group_by_name = _group_map(config.groups, basis.names)

    cache: dict[str, object] = {}

    def evaluate(x):
        params = dict(zip(keys, map(float, x)))
        metab_all = _components_for_params(params, data=data, basis=basis, nfft=nfft, ppm=ppm_all, group_by_name=group_by_name)
        metab = metab_all[mask]
        coeff, design = _solve_linear(y, metab, baseline, config)
        pred = design @ coeff
        cache["params"] = params
        cache["metab"] = metab
        cache["coeff"] = coeff
        cache["design"] = design
        cache["pred"] = pred
        return y - pred

    optimum = least_squares(evaluate, x0, bounds=(lower, upper), max_nfev=config.max_nfev, method="trf")
    residual = evaluate(optimum.x)
    params = cache["params"]
    metab = np.asarray(cache["metab"])
    coeff = np.asarray(cache["coeff"])
    design = np.asarray(cache["design"])
    pred = np.asarray(cache["pred"])
    n_metab = basis.ncomponents
    amps = coeff[:n_metab]
    base_coeff = coeff[n_metab:]
    baseline_fit = baseline @ base_coeff
    components = {name: metab[:, i] * amps[i] for i, name in enumerate(basis.names)}

    dof = max(1, y.size - design.shape[1] - len(keys))
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
        data=y,
        fit=pred,
        baseline=baseline_fit,
        residual=residual,
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
            "uncertainty": "conditional linearised CRLB-like estimate; not LCModel %SD",
        },
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
