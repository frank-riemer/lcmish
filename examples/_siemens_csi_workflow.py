"""Shared output and QC code for the Siemens CSI examples."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lcmish import CSIData, P31CSIRedoxQCConfig, fit_p31_csi_redox


def run_workflow(
    csi: CSIData,
    mask_path: Path,
    output: Path,
    *,
    pcr_snr_min: float,
    min_retained_voxels: int,
) -> None:
    """Run the current masked 31P-CSI route and save auditable outputs."""
    output.mkdir(parents=True, exist_ok=True)
    mask = np.asarray(np.load(mask_path), dtype=bool)
    if mask.shape != csi.spatial_shape:
        raise ValueError(
            f"mask shape {mask.shape} does not match CSI shape {csi.spatial_shape}"
        )
    if not np.any(mask):
        raise ValueError("mask selects no voxels")

    qc = P31CSIRedoxQCConfig(
        pcr_snr_min=pcr_snr_min,
        min_retained_voxels=min_retained_voxels,
        local_fit_correlation_min=0.85,
        local_relative_residual_max=0.55,
        max_component_crlb_percent=None,
    )
    result = fit_p31_csi_redox(csi, mask, qc)
    preparation = result.preparation
    primary = result.primary

    np.save(output / "pcr_snr.npy", preparation.pcr_snr)
    np.save(output / "retained_mask.npy", preparation.retained_mask)
    np.savez_compressed(
        output / "combined_and_fit.npz",
        combined_fid=preparation.combined.fid,
        ppm=primary.ppm,
        data=primary.data,
        fit=primary.fit,
        baseline=primary.baseline,
        residual=primary.residual,
        amplitudes=primary.amplitudes,
        crlb_percent=primary.crlb_percent,
        component_names=np.asarray(primary.names),
    )

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    snr_image = axes[0].imshow(preparation.pcr_snr, origin="lower", cmap="viridis")
    axes[0].contour(
        preparation.retained_mask.astype(float),
        levels=[0.5],
        colors="white",
        linewidths=1.0,
    )
    figure.colorbar(snr_image, ax=axes[0], label="PCr peak SNR")
    axes[0].set_title("PCr SNR and retained mask")
    axes[0].set_xlabel("column")
    axes[0].set_ylabel("row")
    axes[1].plot(primary.ppm, primary.data, color="black", label="data")
    axes[1].plot(primary.ppm, primary.fit, color="tab:red", label="fit")
    axes[1].plot(primary.ppm, primary.residual, color="0.55", label="residual")
    axes[1].invert_xaxis()
    axes[1].set_title("Combined NAD-region fit")
    axes[1].set_xlabel("ppm")
    axes[1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "workflow_summary.png", dpi=180)
    plt.close(figure)

    summary = {
        "source": csi.metadata,
        "csi_shape": list(csi.spatial_shape),
        "spectral_points": csi.npoints,
        "dwell_time_s": csi.dwell_time_s,
        "transmitter_mhz": csi.transmitter_mhz,
        "supplied_mask": str(mask_path),
        "masked_voxels": preparation.n_masked,
        "retained_voxels": preparation.n_retained,
        "retained_voxel_indices": [list(index) for index in preparation.retained_voxel_indices],
        "qc_pass": result.qc_pass,
        "qc_reasons": list(result.qc_reasons),
        "fit_success": primary.success,
        "fit_correlation": primary.fit_correlation,
        "relative_residual": primary.relative_residual,
        "apparent_nad_plus_over_nadh": result.apparent_redox_ratio,
        "absolute_concentration_calibrated": False,
        "orientation_was_inferred": False,
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n"
    )
    print(json.dumps(summary, indent=2, allow_nan=True))
