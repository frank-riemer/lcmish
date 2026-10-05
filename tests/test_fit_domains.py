from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from lcmish import BasisSet, FitConfig, P31RedoxConfig, SpectralData, fit_p31_redox, fit_spectrum
from lcmish.domains import magnitude_model, normalise_fit_domain
from lcmish.redox import _component_spectra
from lcmish.cli import build_parser


@pytest.mark.parametrize("alias,canonical", [("both", "complex"), ("imaginary", "imag"), ("mag", "magnitude")])
def test_domain_aliases(alias, canonical):
    assert normalise_fit_domain(alias) == canonical


def test_magnitude_preserves_interference_and_component_allocation():
    metab = np.array([[1+1j, -0.4-0.7j], [2j, 1-1j]], dtype=complex)
    amplitudes = np.array([1.0, 0.5])
    baseline = np.ones((2, 1))
    prediction, base, jac, curves = magnitude_model(metab, amplitudes, baseline, np.array([0.1]))
    expected = np.abs(metab @ amplitudes)
    np.testing.assert_allclose(prediction, expected + 0.1)
    np.testing.assert_allclose(curves.sum(axis=1), expected)
    assert not np.allclose(expected, np.abs(metab) @ amplitudes)
    for index in range(2):
        step = np.zeros(2)
        step[index] = 1e-6
        numerical = (np.abs(metab @ (amplitudes+step)) - np.abs(metab @ (amplitudes-step))) / 2e-6
        np.testing.assert_allclose(jac[:, index], numerical, rtol=1e-6)


@pytest.mark.parametrize("domain", ["real", "imag", "complex", "magnitude"])
def test_general_fitter_recovers_amplitudes_in_each_domain(domain):
    n, dwell, f0 = 256, 1/2500, 51.7
    time = np.arange(n)*dwell
    fids = np.array([np.exp(-np.pi*7*time) * np.exp(-2j*np.pi*centre*f0*time) for centre in [0, 4.8]])
    basis = BasisSet(["PCr", "Pi"], fids, dwell, f0)
    data = SpectralData((fids[0]+0.3*fids[1])*np.exp(0.3j), dwell, f0)
    phase = np.rad2deg(0.3)
    config = FitConfig(
        ppm_range=(-2, 7), fit_domain=domain, baseline_knots=6, baseline_lambda=1,
        global_shift_bounds_ppm=(-1e-4, 1e-4), phase0_bounds_deg=(phase-0.01, phase+0.01),
        initial_phase0_deg=phase, phase1_bounds_deg_per_ppm=(-0.01, 0.01),
        lorentzian_bounds_hz=(0, 1e-3), gaussian_bounds_hz=(0, 1e-3),
        initial_lorentzian_hz=0, initial_gaussian_hz=0, max_nfev=100,
    )
    result = fit_spectrum(data, basis, config)
    assert result.success
    np.testing.assert_allclose(result.amplitudes, [1, 0.3], rtol=0.005)
    np.testing.assert_allclose(sum(result.components.values())+result.baseline, result.fit, atol=1e-8)
    assert result.metadata["fit_domain"] == domain
    assert (result.residual_imag is not None) == (domain == "complex")
    if domain == "magnitude":
        assert "phase0_deg" not in result.nonlinear
        assert not result.metadata["phase_estimated"]
        assert not result.metadata["magnitude_noise_correction"]


def _redox_data(domain):
    n, dwell, f0 = 1024, 1/2500, 49.892
    empty = SpectralData(np.zeros(n, complex), dwell, f0)
    config = P31RedoxConfig(
        fit_domain=domain, baseline_order=1,
        initial_nad_linewidth_hz=8, initial_alpha_extra_linewidth_hz=2,
        phase0_bounds_deg=(16.9, 17.1), phase1_bounds_deg_per_ppm=(-0.01, 0.01),
    )
    # The redox optimiser starts at phase 0; keep zero in its allowed range.
    config = replace(config, phase0_bounds_deg=(-20, 20))
    _, spectra = _component_spectra(empty, config, n, 0, 8, 2, 0, 17, 0)
    signal = np.array([0.3, 0.075, 2.8]) @ spectra
    return SpectralData(np.fft.ifft(np.fft.ifftshift(signal)), dwell, f0), config


@pytest.mark.parametrize("domain", ["real", "imag", "complex", "magnitude"])
def test_redox_fitter_recovers_all_domains(domain):
    data, config = _redox_data(domain)
    result = fit_p31_redox(data, config, nfft=data.npoints)
    assert result.success
    np.testing.assert_allclose(result.amplitudes, [0.3, 0.075, 2.8], rtol=0.03, atol=0.002)
    assert np.isclose(result.apparent_redox_ratio, 4, rtol=0.04)
    assert result.relative_residual < 0.005
    assert result.metadata["fit_domain"] == domain
    np.testing.assert_allclose(sum(result.components.values())+result.baseline, result.fit, atol=1e-8)
    if domain == "complex":
        assert result.residual_imag is not None
        np.testing.assert_allclose(sum(result.components_imag.values())+result.baseline_imag, result.fit_imag, atol=1e-8)
    if domain == "magnitude":
        assert "phase0_deg" not in result.nonlinear
        assert "phase1_deg_per_ppm" not in result.nonlinear


@pytest.mark.parametrize("domain", ["complex", "magnitude"])
def test_redox_bootstrap_selected_domain(domain):
    data, config = _redox_data(domain)
    result = fit_p31_redox(data, replace(config, bootstrap_repeats=3), nfft=data.npoints)
    assert result.bootstrap_amplitudes.shape == (3, 3)
    assert np.all(np.isfinite(result.bootstrap_amplitudes))
    assert np.all(np.isfinite(result.amplitude_se))


def test_invalid_redox_domain_is_rejected():
    with pytest.raises(ValueError, match="fit_domain"):
        P31RedoxConfig(fit_domain="power")


def test_linked_sugar_magnitude_recovers_exact_coherent_model():
    n, dwell, f0 = 1024, 1/2500, 49.892
    empty = SpectralData(np.zeros(n, complex), dwell, f0)
    config = P31RedoxConfig(
        ppm_range=(-10.4, -6.5), fit_domain="magnitude",
        include_linked_nucleotide_sugars=True, baseline_order=1,
        initial_nad_linewidth_hz=10, initial_alpha_extra_linewidth_hz=2,
    )
    _, spectra = _component_spectra(
        empty, config, n, 0, 10, 2, 0, 27, -5,
        linked_nucleotide_sugar_extra_linewidth_hz=4,
    )
    truth = [0.3, 0.075, 2.8, 0.18]
    fid = np.fft.ifft(np.fft.ifftshift(np.asarray(truth) @ spectra))
    result = fit_p31_redox(SpectralData(fid, dwell, f0), config, nfft=n)
    assert result.success
    np.testing.assert_allclose(result.amplitudes, truth, rtol=0.01)


def test_cli_exposes_domain_selection():
    args = build_parser().parse_args([
        "data.RAW", "basis.BASIS", "--ppm-min", "-20", "--ppm-max", "10",
        "--fit-domain", "magnitude",
    ])
    assert args.fit_domain == "magnitude"


@pytest.mark.parametrize("domain", ["imag", "magnitude", "complex"])
def test_general_domain_exports_and_reports(tmp_path, domain):
    n, dwell, f0 = 256, 1/2500, 51.7
    t = np.arange(n)*dwell
    fid = np.exp(-np.pi*8*t)*np.exp(0.3j)
    data = SpectralData(fid, dwell, f0)
    basis = BasisSet(["PCr"], fid[None], dwell, f0)
    result = fit_spectrum(data, basis, FitConfig(
        ppm_range=(-3, 3), fit_domain=domain, baseline_knots=6,
    ))
    result.save_components_csv(tmp_path / "components.csv")
    result.save_pdf(tmp_path / "fit.pdf")
    result.save_checkpoint_npz(tmp_path / "fit.npz")
    assert (tmp_path / "fit.pdf").stat().st_size > 1000
    assert np.load(tmp_path / "fit.npz", allow_pickle=True)["data"].shape == result.data.shape
