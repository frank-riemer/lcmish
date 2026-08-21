from __future__ import annotations

import numpy as np

from lcmish import BasisSet, FitConfig, SpectralData, fit_spectrum, fit_spectrum_multistart
from lcmish.fitter import _match_basis_time_grid


def _peak(ppm, t, f0, decay_hz=6.0):
    return np.exp(-np.pi * decay_hz * t) * np.exp(1j * 2 * np.pi * ppm * f0 * t)


def test_synthetic_fit_recovers_relative_amplitudes():
    n = 256
    dwell = 1 / 2500
    f0 = 51.7
    t = np.arange(n) * dwell
    basis = BasisSet(
        ["PCr", "Pi"],
        np.asarray([_peak(0.0, t, f0), _peak(4.5, t, f0)]),
        dwell_time_s=dwell,
        transmitter_mhz=f0,
    )
    data = SpectralData(1.4 * basis.fids[0] + 0.35 * basis.fids[1], dwell, f0)
    config = FitConfig(
        ppm_range=(-2, 7),
        zero_fill_factor=2,
        baseline_knots=7,
        baseline_lambda=1.0,
        global_shift_bounds_ppm=(-1e-4, 1e-4),
        phase0_bounds_deg=(-1e-3, 1e-3),
        phase1_bounds_deg_per_ppm=(-1e-3, 1e-3),
        lorentzian_bounds_hz=(0, 1e-3),
        gaussian_bounds_hz=(0, 1e-3),
        initial_lorentzian_hz=0,
        initial_gaussian_hz=0,
        max_nfev=50,
    )
    result = fit_spectrum(data, basis, config)
    assert result.success
    ratio = result.amplitudes[1] / result.amplitudes[0]
    assert np.isclose(ratio, 0.25, rtol=0.05)
    assert np.linalg.norm(result.residual) / np.linalg.norm(result.data) < 0.03


def test_multistart_returns_audit():
    n = 128
    dwell = 1 / 2000
    f0 = 51.7
    t = np.arange(n) * dwell
    basis = BasisSet(["PCr"], np.asarray([_peak(0.0, t, f0)]))
    data = SpectralData(basis.fids[0].copy(), dwell, f0)
    config = FitConfig(ppm_range=(-3, 3), baseline_knots=6, max_nfev=20)
    audit = fit_spectrum_multistart(data, basis, config, starts=({}, {"initial_phase0_deg": 5.0}))
    assert len(audit.trials) == 2
    assert 0 <= audit.best_index < 2


def test_fit_respects_basis_dwell_time():
    data_n = 256
    basis_n = 512
    data_dwell = 0.0005
    basis_dwell = 0.00025
    f0 = 50.0
    frequency_hz = 125.0
    t_basis = np.arange(basis_n) * basis_dwell
    t_data = np.arange(data_n) * data_dwell
    basis = BasisSet(
        ["X"],
        np.asarray([np.exp(-8 * t_basis) * np.exp(1j * 2 * np.pi * frequency_hz * t_basis)]),
        dwell_time_s=basis_dwell,
        transmitter_mhz=f0,
    )
    data = SpectralData(
        np.exp(-8 * t_data) * np.exp(1j * 2 * np.pi * frequency_hz * t_data),
        data_dwell,
        f0,
    )
    config = FitConfig(
        ppm_range=(-4, 0),
        zero_fill_factor=2,
        baseline_knots=6,
        baseline_lambda=1e8,
        global_shift_bounds_ppm=(-1e-4, 1e-4),
        phase0_bounds_deg=(-1e-3, 1e-3),
        phase1_bounds_deg_per_ppm=(-1e-3, 1e-3),
        lorentzian_bounds_hz=(0, 1e-3),
        gaussian_bounds_hz=(0, 1e-3),
        initial_lorentzian_hz=0,
        initial_gaussian_hz=0,
        max_nfev=50,
    )
    result = fit_spectrum(data, basis, config)
    assert result.success
    assert np.linalg.norm(result.residual) / np.linalg.norm(result.data) < 0.01


def test_lcmodel_basis_retains_tail_to_internal_model_length():
    ndatab = 512
    acquired_n = 128
    model_n = 256
    basis_dt = 0.0005
    data_dt = 0.001
    f0 = 50.0
    t = np.arange(ndatab) * basis_dt
    full_fid = np.exp(-4.0 * t) * np.exp(1j * 2 * np.pi * 80.0 * t)
    basis = BasisSet(
        ["X"],
        np.asarray([full_fid[: ndatab // 2]]),
        dwell_time_s=basis_dt,
        transmitter_mhz=f0,
        metadata={"lcmodel_basis": True, "stored_ndatab": ndatab},
    )
    data = SpectralData(np.zeros(acquired_n), data_dt, f0)

    matched = _match_basis_time_grid(basis, data, model_n)

    assert matched.shape == (1, model_n)
    assert np.linalg.norm(matched[0, acquired_n:]) > 0.01 * np.linalg.norm(
        matched[0, :acquired_n]
    )


def test_one_page_pdf_report(tmp_path):
    n = 128
    dwell = 1 / 2000
    f0 = 51.7
    t = np.arange(n) * dwell
    basis = BasisSet(
        ["PCr", "Pi"],
        np.asarray([_peak(0.0, t, f0), _peak(4.5, t, f0)]),
    )
    data = SpectralData(1.0 * basis.fids[0] + 0.2 * basis.fids[1], dwell, f0)
    config = FitConfig(
        ppm_range=(-2, 7),
        baseline_knots=6,
        global_shift_bounds_ppm=(-1e-4, 1e-4),
        phase0_bounds_deg=(-1e-3, 1e-3),
        phase1_bounds_deg_per_ppm=(-1e-3, 1e-3),
        lorentzian_bounds_hz=(0, 1e-3),
        gaussian_bounds_hz=(0, 1e-3),
        initial_lorentzian_hz=0,
        initial_gaussian_hz=0,
        max_nfev=30,
    )
    result = fit_spectrum(data, basis, config)
    out = tmp_path / "fit_summary.pdf"
    returned = result.save_pdf(out, title="LCMish test fit")
    assert returned == out
    payload = out.read_bytes()
    assert payload.startswith(b"%PDF")
    assert len(payload) > 10_000
