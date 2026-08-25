"""Explicit Siemens 2-D CSI reconstruction helpers.

These readers intentionally expose acquisition-specific choices.  They do not
infer anatomical orientation, voxel masks, or an arbitrary Twix dimension
layout.  Validate those choices against scanner reconstruction or a phantom
before using the result quantitatively.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .models import CSIData


def reconstruct_siemens_csi_array(
    kspace: np.ndarray,
    *,
    dwell_time_s: float,
    transmitter_mhz: float,
    reference_ppm: float = 0.0,
    reverse_second_spatial_axis: bool = True,
    conjugate_for_nmr: bool = True,
    metadata: dict[str, Any] | None = None,
) -> CSIData:
    """Reconstruct Siemens 2-D CSI shaped ``(time, line, average, segment)``.

    Averages are combined coherently before a centered spatial FFT.  The axis
    reversal and complex conjugation reproduce one validated Siemens 31P CSI
    convention, but remain explicit because other sequences can differ.
    """
    array = np.asarray(kspace, dtype=np.complex128)
    if array.ndim != 4:
        raise ValueError("kspace must have shape (time, line, average, segment)")
    if min(array.shape) < 1:
        raise ValueError("spectral, CSI, and average dimensions must be non-zero")

    combined = np.mean(array, axis=2)
    spatial = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(combined, axes=(1, 2)), axes=(1, 2)),
        axes=(1, 2),
    )
    if reverse_second_spatial_axis:
        spatial = spatial[:, :, ::-1]
    if conjugate_for_nmr:
        spatial = np.conj(spatial)

    output_metadata = dict(metadata or {})
    output_metadata.update(
        {
            "source_axes": ["time", "line", "average", "segment"],
            "average_combination": "complex_mean",
            "spatial_transform": "centered_fft2",
            "second_spatial_axis_reversed": bool(reverse_second_spatial_axis),
            "complex_conjugated_for_nmr_ppm": bool(conjugate_for_nmr),
            "orientation_requires_independent_validation": True,
        }
    )
    return CSIData(
        np.moveaxis(spatial, 0, -1),
        dwell_time_s,
        transmitter_mhz,
        reference_ppm,
        output_metadata,
    )


def read_siemens_twix_csi(
    path: str | Path,
    *,
    reference_ppm: float = 0.0,
    remove_oversampling: bool = True,
    reverse_second_spatial_axis: bool = True,
    conjugate_for_nmr: bool = True,
) -> CSIData:
    """Read a Siemens Twix CSI image stream with the validated four-axis layout.

    ``pymapvbvd`` is imported only when this function is called.  Unsupported
    layouts fail visibly instead of being guessed.
    """
    try:
        import mapvbvd
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Twix CSI support requires pymapvbvd; install `lcmish[twix]`"
        ) from exc

    twix = mapvbvd.mapVBVD(str(path), quiet=True)
    if isinstance(twix, list):
        twix = twix[-1]
    if not hasattr(twix, "image"):
        raise ValueError("Twix file has no image MDH stream recognised by pymapvbvd")
    image = twix.image
    image.squeeze = True
    image.flagRemoveOS = bool(remove_oversampling)
    array = np.asarray(image[""])
    dimensions = tuple(image.sqzDims)
    if array.ndim != 4 or dimensions != ("Col", "Lin", "Ave", "Seg"):
        raise ValueError(
            f"Unsupported Twix image layout {array.shape}, {dimensions}; "
            "expected (Col, Lin, Ave, Seg)"
        )

    header = twix.hdr
    try:
        frequency_hz = float(
            header["MeasYaps"][
                ("sTXSPEC", "asNucleusInfo", "0", "lFrequency")
            ]
        )
        raw_dwell_time_s = (
            float(header["MeasYaps"][("sRXSPEC", "alDwellTime", "0")]) * 1e-9
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Could not read transmitter frequency or dwell time from Twix") from exc
    dwell_time_s = raw_dwell_time_s * (2.0 if remove_oversampling else 1.0)
    nucleus = str(
        header.get("MeasYaps", {}).get(
            ("sTXSPEC", "asNucleusInfo", "0", "tNucleus"), ""
        )
    ).strip('"')

    return reconstruct_siemens_csi_array(
        array,
        dwell_time_s=dwell_time_s,
        transmitter_mhz=frequency_hz / 1e6,
        reference_ppm=reference_ppm,
        reverse_second_spatial_axis=reverse_second_spatial_axis,
        conjugate_for_nmr=conjugate_for_nmr,
        metadata={
            "source_format": "Siemens Twix",
            "source_path_name": Path(path).name,
            "nucleus": nucleus,
            "raw_dwell_time_s": raw_dwell_time_s,
            "readout_oversampling_removed": bool(remove_oversampling),
            "averages": int(array.shape[2]),
        },
    )


def read_siemens_mrs_dicom_csi(
    path: str | Path,
    *,
    spatial_shape: tuple[int, int],
    dwell_time_s: float,
    transmitter_mhz: float,
    reference_ppm: float = 0.0,
    conjugate_for_nmr: bool | None = None,
) -> CSIData:
    """Read a Siemens MR Spectroscopy DICOM payload into a 2-D CSI grid.

    Acquisition values are explicit because exported or anonymized DICOMs may
    omit the relevant functional groups.  Both standard SpectroscopyData
    ``(5600,0020)`` and Siemens private ``(7fe1,1010)`` payloads are supported.
    Their historically observed complex-sign conventions differ; override the
    default only after independent validation.
    """
    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "DICOM CSI support requires pydicom; install `pydicom`"
        ) from exc

    rows, columns = (int(value) for value in spatial_shape)
    if rows < 1 or columns < 1:
        raise ValueError("spatial_shape values must be positive")
    dataset = pydicom.dcmread(str(path), force=True)
    standard_tag = (0x5600, 0x0020)
    private_tag = (0x7FE1, 0x1010)
    if standard_tag in dataset:
        payload_tag = standard_tag
        default_conjugation = True
        sign_rule = "standard_spectroscopy_data_observed_convention"
    elif private_tag in dataset:
        payload_tag = private_tag
        default_conjugation = False
        sign_rule = "siemens_private_payload_observed_convention"
    else:
        raise ValueError(
            "DICOM contains neither standard (5600,0020) nor supported "
            "Siemens private (7fe1,1010) spectroscopy data"
        )
    selected_conjugation = (
        default_conjugation if conjugate_for_nmr is None else bool(conjugate_for_nmr)
    )

    payload = dataset[payload_tag].value
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        payload = bytes(payload)
    byte_order = ">" if getattr(dataset, "is_little_endian", True) is False else "<"
    floats = np.frombuffer(payload, dtype=f"{byte_order}f4")
    if floats.size % 2:
        raise ValueError("DICOM SpectroscopyData contains an odd number of floats")
    values = floats[0::2] + 1j * floats[1::2]
    voxel_count = rows * columns
    if values.size % voxel_count:
        raise ValueError("DICOM SpectroscopyData is incompatible with spatial_shape")
    points = values.size // voxel_count
    if points < 8:
        raise ValueError("DICOM SpectroscopyData contains fewer than eight points per voxel")
    fids = values.reshape(rows, columns, points)
    if selected_conjugation:
        fids = np.conj(fids)

    return CSIData(
        fids,
        dwell_time_s,
        transmitter_mhz,
        reference_ppm,
        metadata={
            "source_format": "Siemens DICOM MR Spectroscopy",
            "source_path_name": Path(path).name,
            "payload_tag": f"({payload_tag[0]:04x},{payload_tag[1]:04x})",
            "payload_dtype": f"{byte_order}f4 interleaved complex",
            "complex_conjugated_for_nmr_ppm": selected_conjugation,
            "complex_sign_rule": sign_rule,
            "complex_sign_was_explicitly_overridden": (
                selected_conjugation != default_conjugation
            ),
            "orientation_requires_independent_validation": True,
        },
    )


# Compatibility with the name used by the original private tutorial.
read_mrs_dicom_csi = read_siemens_mrs_dicom_csi
