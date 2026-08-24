"""Synthetic example of the experimental masked 2-D 31P-CSI redox route.

Replace the synthetic array and mask with reconstructed complex CSI data and a
mask whose anatomical orientation has been verified for the study. The QC
values below are examples, not universal defaults.
"""
from __future__ import annotations

import numpy as np

from lcmish import (
    CSIData,
    P31CSIRedoxQCConfig,
    P31RedoxConfig,
    fit_p31_csi_redox,
    nad_plus_ab_pattern,
)


def multiplet(time, f0, positions_ppm, weights, linewidth_hz):
    oscillators = np.exp(
        -1j
        * 2
        * np.pi
        * np.asarray(positions_ppm)[:, None]
        * f0
        * time[None, :]
    )
    return (np.asarray(weights) @ oscillators) * np.exp(-np.pi * linewidth_hz * time)


rng = np.random.default_rng(2026)
npoints = 1024
dwell_time_s = 1 / 2500
transmitter_mhz = 49.892088
time = np.arange(npoints) * dwell_time_s
redox_config = P31RedoxConfig()
nad_positions, nad_weights = nad_plus_ab_pattern(transmitter_mhz, redox_config)

base_fid = (
    3.0 * multiplet(time, transmitter_mhz, [0.0], [1.0], 8.0)
    + 0.30 * multiplet(time, transmitter_mhz, nad_positions, nad_weights, 10.0)
    + 0.075 * multiplet(
        time, transmitter_mhz, [redox_config.nadh_ppm], [2.0], 10.0
    )
    + 2.8
    * multiplet(
        time,
        transmitter_mhz,
        [
            redox_config.alpha_atp_center_ppm
            - redox_config.alpha_atp_j_hz / (2 * transmitter_mhz),
            redox_config.alpha_atp_center_ppm
            + redox_config.alpha_atp_j_hz / (2 * transmitter_mhz),
        ],
        [0.5, 0.5],
        12.0,
    )
)

# Explicit shape: row, column, time. No anatomical orientation is inferred.
fids = np.zeros((4, 4, npoints), dtype=np.complex128)
for row in range(4):
    for column in range(4):
        noise = 0.002 * (
            rng.normal(size=npoints) + 1j * rng.normal(size=npoints)
        )
        fids[row, column] = base_fid + noise

csi = CSIData(fids, dwell_time_s, transmitter_mhz)

# This stands in for a verified, study-specific anatomical voxel mask.
voxel_mask = np.zeros(csi.spatial_shape, dtype=bool)
voxel_mask[1:3, 1:3] = True

qc = P31CSIRedoxQCConfig(
    pcr_snr_min=10.0,
    min_retained_voxels=3,
    local_fit_correlation_min=0.85,
    local_relative_residual_max=0.55,
    # Set a prespecified CRLB-like threshold if validated for the acquisition.
    max_component_crlb_percent=None,
)

result = fit_p31_csi_redox(csi, voxel_mask, qc, redox_config)
print("retained voxels:", result.preparation.retained_voxel_indices)
print("QC pass:", result.qc_pass, result.qc_reasons)
print("primary apparent NAD+/NADH:", result.primary.apparent_redox_ratio)
print(
    "with nucleotide-sugar nuisance:",
    None if result.nuisance is None else result.nuisance.apparent_redox_ratio,
)
print("reportable apparent ratio:", result.apparent_redox_ratio)
