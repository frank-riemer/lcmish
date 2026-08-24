from __future__ import annotations

import numpy as np

from lcmish import (
    CSIData,
    P31CSIRedoxQCConfig,
    P31RedoxConfig,
    SpectralData,
    fit_p31_csi_redox,
    fit_p31_redox,
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
