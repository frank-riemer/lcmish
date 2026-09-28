from __future__ import annotations

import numpy as np
import pytest

from lcmish.redox import _component_spectra

from lcmish import (
    CSIData,
    P31AnchorPhaseConfig,
    P31CSIRedoxQCConfig,
    P31RedoxConfig,
    SpectralData,
    apply_p31_phase_correction,
    estimate_p31_anchor_phase,
    fit_p31_csi_redox,
    fit_p31_redox,
    fit_p31_redox_anchor_informed,
    nad_plus_ab_pattern,
)


def _multiplet(t, f0, positions, weights, linewidth):
    return (
        np.asarray(weights)
        @ np.exp(-1j * 2 * np.pi * np.asarray(positions)[:, None] * f0 * t[None, :])
    ) * np.exp(-np.pi * linewidth * t)


def _synthetic_p31_fid(n=1024, dwell=1 / 2500.0, f0=49.892088):
    t = np.arange(n) * dwell
    config = P31RedoxConfig(baseline_order=1)
    positions, weights = nad_plus_ab_pattern(f0, config)
    nad_plus = _multiplet(t, f0, positions, weights, 10.0)
    nadh = _multiplet(t, f0, [config.nadh_ppm], [2.0], 10.0)
    alpha = _multiplet(
        t,
        f0,
        [
            config.alpha_atp_center_ppm - config.alpha_atp_j_hz / (2 * f0),
            config.alpha_atp_center_ppm + config.alpha_atp_j_hz / (2 * f0),
        ],
        [0.5, 0.5],
        12.0,
    )
    pcr = _multiplet(t, f0, [0.0], [1.0], 8.0)
    fid = 3.0 * pcr + 0.30 * nad_plus + 0.075 * nadh + 2.8 * alpha
    return fid, dwell, f0, config


def _linked_nucleotide_sugar(t, f0, config, linewidth=14.0):
    half_split = config.linked_nucleotide_sugar_j_hz / (2 * f0)
    upfield = config.linked_nucleotide_sugar_upfield_center_ppm
    overlap = config.linked_nucleotide_sugar_overlap_center_ppm
    return _multiplet(
        t,
        f0,
        [
            upfield - half_split,
            upfield + half_split,
            overlap - half_split,
            overlap + half_split,
        ],
        [0.5, 0.5, 0.5, 0.5],
        linewidth,
    )


def _synthetic_anchor_fid(
    n=1024,
    dwell=1 / 2500.0,
    f0=49.892088,
    alpha_phase_offset_deg=0.0,
):
    fid, dwell, f0, config = _synthetic_p31_fid(n, dwell, f0)
    t = np.arange(n) * dwell
    alpha = _multiplet(
        t,
        f0,
        [
            config.alpha_atp_center_ppm - config.alpha_atp_j_hz / (2 * f0),
            config.alpha_atp_center_ppm + config.alpha_atp_j_hz / (2 * f0),
        ],
        [0.5, 0.5],
        12.0,
    )
    fid += 2.8 * (
        np.exp(1j * np.deg2rad(alpha_phase_offset_deg)) - 1.0
    ) * alpha
    fid += 1.0 * _multiplet(t, f0, [4.85], [1.0], 9.0)
    fid += 1.7 * _multiplet(t, f0, [-2.65, -2.35], [0.5, 0.5], 12.0)
    fid += 1.4 * _multiplet(t, f0, [-16.10, -15.78], [0.5, 0.5], 14.0)
    return fid, dwell, f0, config


def _phase_fid(fid, dwell, f0, phase0_deg, phase1_deg_per_ppm):
    data = SpectralData(fid, dwell, f0)
    ppm = data.ppm_axis()
    phase = phase0_deg + phase1_deg_per_ppm * ppm
    spectrum = data.spectrum() * np.exp(1j * np.deg2rad(phase))
    return np.fft.ifft(np.fft.ifftshift(spectrum))


def test_3t_nad_plus_pattern_has_expected_ab_geometry():
    positions, weights = nad_plus_ab_pattern(49.892088)
    assert np.allclose(positions, [-8.7699, -8.3683, -8.2557, -7.8541], atol=2e-4)
    assert np.isclose(weights.sum(), 2.0)
    assert np.isclose(weights[0] / weights[1], 0.123, atol=0.002)


def test_local_redox_fit_recovers_synthetic_amplitudes():
    rng = np.random.default_rng(18)
    fid, dwell, f0, config = _synthetic_p31_fid()
    fid = fid + 0.0005 * (rng.normal(size=fid.size) + 1j * rng.normal(size=fid.size))
    result = fit_p31_redox(SpectralData(fid, dwell, f0), config)
    assert result.success
    assert np.allclose(result.amplitudes[:3], [0.30, 0.075, 2.8], rtol=0.08, atol=0.005)
    assert np.isclose(result.apparent_redox_ratio, 4.0, rtol=0.12)
    assumptions = result.metadata["acquisition_model_assumptions"]
    assert assumptions["basis_approximation"] == "idealized_proton_decoupled"
    assert assumptions["nad_ratio_interpretation"] == "apparent_spectral_ratio"
    assert assumptions["phosphorus_phosphorus_coupling_retained"] is True
    for key in (
        "proton_coupling_simulated",
        "decoupling_pulse_train_simulated",
        "partial_acquisition_decoupling_simulated",
        "noe_correction_applied_by_fitter",
        "saturation_correction_applied_by_fitter",
        "excitation_profile_correction_applied_by_fitter",
        "receive_response_correction_applied_by_fitter",
        "acquisition_conditions_inferred_from_headers",
        "linked_sugar_relative_response_fitted",
    ):
        assert assumptions[key] is False
    assert assumptions["linked_sugar_partner_area_ratio"] is None


def test_linked_nucleotide_sugar_fit_uses_both_phosphate_regions():
    rng = np.random.default_rng(181)
    fid, dwell, f0, _ = _synthetic_p31_fid()
    config = P31RedoxConfig(
        ppm_range=(-10.4, -6.5),
        baseline_order=1,
        include_linked_nucleotide_sugars=True,
    )
    time = np.arange(fid.size) * dwell
    fid += 0.18 * _linked_nucleotide_sugar(time, f0, config)
    fid += 0.0005 * (
        rng.normal(size=fid.size) + 1j * rng.normal(size=fid.size)
    )
    result = fit_p31_redox(SpectralData(fid, dwell, f0), config)
    assert result.success
    assert result.names[:4] == (
        "NAD_plus",
        "NADH",
        "alpha_ATP",
        "UDP_sugars_linked",
    )
    assert np.allclose(
        result.amplitudes[:4], [0.30, 0.075, 2.8, 0.18], rtol=0.10
    )
    assert np.isclose(result.apparent_redox_ratio, 4.0, rtol=0.12)
    assert result.metadata["acquisition_model_assumptions"][
        "linked_sugar_partner_area_ratio"
    ] == 1.0


def test_linked_sugar_basis_is_one_equal_area_two_phosphorus_component():
    fid, dwell, f0, _ = _synthetic_p31_fid()
    data = SpectralData(fid, dwell, f0)
    config = P31RedoxConfig(
        ppm_range=(-10.4, -6.5), include_linked_nucleotide_sugars=True,
    )
    names, spectra = _component_spectra(
        data, config, fid.size, 0.0, 10.0, 2.0, 0.0, 0.0, 0.0,
        linked_nucleotide_sugar_extra_linewidth_hz=4.0,
    )
    assert names.count("UDP_sugars_linked") == 1
    sugar_fid = np.fft.ifft(np.fft.ifftshift(spectra[-1]))
    expected = _linked_nucleotide_sugar(data.time_axis(), f0, config, 14.0)
    np.testing.assert_allclose(sugar_fid, expected, atol=1e-12)
    # Integrated complex spectral area equals the t=0 weight sum. The NAD
    # and pooled sugar components each carry two phosphorus weights; alpha
    # ATP carries one. This is bookkeeping, not an NOE calibration.
    component_fids = np.fft.ifft(np.fft.ifftshift(spectra, axes=1), axis=1)
    np.testing.assert_allclose(component_fids[:, 0], [2.0, 2.0, 1.0, 2.0], atol=1e-12)


def test_linked_nucleotide_sugar_model_requires_partner_window():
    with pytest.raises(ValueError, match="covering both"):
        P31RedoxConfig(include_linked_nucleotide_sugars=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        P31RedoxConfig(
            ppm_range=(-10.4, -6.5),
            include_nucleotide_sugar_nuisance=True,
            include_linked_nucleotide_sugars=True,
        )


def test_non_nad_anchor_phase_recovers_known_linear_phase():
    rng = np.random.default_rng(67)
    fid, dwell, f0, _ = _synthetic_anchor_fid()
    phased = _phase_fid(fid, dwell, f0, 21.0, -8.5)
    phased += 0.0005 * (
        rng.normal(size=phased.size) + 1j * rng.normal(size=phased.size)
    )
    data = SpectralData(phased, dwell, f0)
    estimate = estimate_p31_anchor_phase(data)
    assert estimate.anchor_names == ("Pi", "PCr", "gamma_ATP", "beta_ATP")
    assert np.isclose(estimate.phase0_deg, 21.0, atol=3.0)
    assert np.isclose(estimate.phase1_deg_per_ppm, -8.5, atol=0.5)
    assert estimate.phase_residual_rms_deg < 4.0

    corrected = apply_p31_phase_correction(
        data,
        estimate.phase0_deg,
        estimate.phase1_deg_per_ppm,
        pivot_ppm=estimate.pivot_ppm,
    )
    corrected_estimate = estimate_p31_anchor_phase(corrected)
    assert abs(corrected_estimate.phase0_deg) < 1.0
    assert abs(corrected_estimate.phase1_deg_per_ppm) < 0.15


def test_anchor_phase_does_not_use_nad_window():
    fid, dwell, f0, _ = _synthetic_anchor_fid()
    data = SpectralData(_phase_fid(fid, dwell, f0, -17.0, 6.0), dwell, f0)
    before = estimate_p31_anchor_phase(data, nfft=data.npoints)
    ppm = data.ppm_axis()
    spectrum = data.spectrum()
    nad_window = (ppm >= -9.0) & (ppm <= -6.5)
    spectrum[nad_window] += 20.0 * np.exp(1j * 1.3)
    altered = SpectralData(
        np.fft.ifft(np.fft.ifftshift(spectrum)), dwell, f0
    )
    after = estimate_p31_anchor_phase(altered, nfft=altered.npoints)
    assert np.isclose(after.phase0_deg, before.phase0_deg, atol=1e-10)
    assert np.isclose(
        after.phase1_deg_per_ppm, before.phase1_deg_per_ppm, atol=1e-10
    )


def test_anchor_informed_redox_fit_recovers_synthetic_ratio():
    rng = np.random.default_rng(91)
    fid, dwell, f0, config = _synthetic_anchor_fid()
    phased = _phase_fid(fid, dwell, f0, 28.0, -9.0)
    phased += 0.0005 * (
        rng.normal(size=phased.size) + 1j * rng.normal(size=phased.size)
    )
    result = fit_p31_redox_anchor_informed(
        SpectralData(phased, dwell, f0),
        config,
        P31AnchorPhaseConfig(
            residual_phase0_bounds_deg=(-2.0, 2.0),
            residual_phase1_bounds_deg_per_ppm=(-1.0, 1.0),
        ),
    )
    assert result.fit.success
    assert np.isclose(result.apparent_redox_ratio, 4.0, rtol=0.15)
    assert result.fit.metadata["phase_strategy"] == "non_NAD_anchor_informed"


def test_anchor_informed_fit_separates_alpha_offset_from_nad_phase():
    fid, dwell, f0, config = _synthetic_anchor_fid(alpha_phase_offset_deg=-18.0)
    phased = _phase_fid(fid, dwell, f0, 24.0, -7.5)
    result = fit_p31_redox_anchor_informed(
        SpectralData(phased, dwell, f0),
        config,
    )
    assert result.fit.success
    assert np.isclose(result.apparent_redox_ratio, 4.0, rtol=0.15)
    assert np.isclose(
        result.fit.nonlinear["alpha_phase_offset_deg"], -18.0, atol=2.5
    )
    assert abs(result.fit.nonlinear["phase1_deg_per_ppm"]) <= 2.0


def test_masked_2d_csi_workflow_retains_auditable_voxel_qc():
    rng = np.random.default_rng(42)
    fid, dwell, f0, config = _synthetic_p31_fid()
    fids = np.zeros((2, 2, fid.size), dtype=np.complex128)
    for row, column, scale in ((0, 0, 1.0), (0, 1, 0.9), (1, 0, 1.1)):
        noise = 0.001 * (
            rng.normal(size=fid.size) + 1j * rng.normal(size=fid.size)
        )
        fids[row, column] = scale * fid + noise
    fids[1, 1] = 0.25 * (
        rng.normal(size=fid.size) + 1j * rng.normal(size=fid.size)
    )
    csi = CSIData(fids, dwell, f0)
    mask = np.ones((2, 2), dtype=bool)
    qc = P31CSIRedoxQCConfig(
        pcr_snr_min=10.0,
        min_retained_voxels=3,
        local_fit_correlation_min=0.98,
        local_relative_residual_max=0.15,
    )
    result = fit_p31_csi_redox(
        csi,
        mask,
        qc,
        config,
        run_nucleotide_sugar_sensitivity=False,
    )
    assert result.preparation.n_masked == 4
    assert result.preparation.n_retained == 3
    assert result.preparation.retained_mask[1, 1] == 0
    assert result.preparation.excluded_reasons["1,1"]
    assert result.qc_pass
    assert np.isclose(result.apparent_redox_ratio, 4.0, rtol=0.20)
