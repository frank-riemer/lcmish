"""NIfTI-MRS input support.

LCMish treats NIfTI-MRS as its vendor-neutral spectroscopy interchange format.
This reader deliberately extracts one FID at a time: coil combination, dynamic
averaging and other preprocessing steps must be performed explicitly upstream.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .models import SpectralData

_NIFTI_MRS_ECODE = 44
_TIME_UNIT_SCALE = {
    "sec": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "msec": 1e-3,
    "millisecond": 1e-3,
    "milliseconds": 1e-3,
    "usec": 1e-6,
    "microsecond": 1e-6,
    "microseconds": 1e-6,
}


def _decode_intent_name(header: Any) -> str:
    try:
        intent = header.get_intent()[2]
    except Exception as exc:  # pragma: no cover - defensive for unusual nibabel headers
        raise ValueError("Could not read NIfTI intent name") from exc
    if isinstance(intent, bytes):
        intent = intent.decode("ascii", errors="replace")
    return str(intent).strip().strip("\x00")


def _read_mrs_extension(header: Any) -> dict[str, Any]:
    extensions = getattr(header, "extensions", None)
    if extensions is None or not hasattr(extensions, "get_codes"):
        raise ValueError("NIfTI file has no readable header extensions")
    codes = list(extensions.get_codes())
    if _NIFTI_MRS_ECODE not in codes:
        raise ValueError(
            "NIfTI file does not contain the required NIfTI-MRS JSON extension "
            f"(ecode {_NIFTI_MRS_ECODE})"
        )
    content = extensions[codes.index(_NIFTI_MRS_ECODE)].get_content()
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="strict")
    content = str(content).rstrip("\x00 \t\r\n")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("NIfTI-MRS header extension is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("NIfTI-MRS JSON extension must contain an object")
    return payload


def _dwell_time_seconds(header: Any) -> tuple[float, str | None]:
    try:
        raw = float(header["pixdim"][4])
    except Exception as exc:
        raise ValueError("Could not read NIfTI-MRS dwell time from pixdim[4]") from exc
    if not np.isfinite(raw) or raw <= 0:
        raise ValueError("NIfTI-MRS dwell time in pixdim[4] must be positive")

    unit: str | None = None
    try:
        unit = header.get_xyzt_units()[1]
    except Exception:
        pass
    if isinstance(unit, bytes):
        unit = unit.decode("ascii", errors="replace")
    unit = str(unit).lower() if unit not in (None, "") else None

    # Early NIfTI-MRS files sometimes omitted the NIfTI time-unit flag while
    # storing pixdim[4] in seconds. Preserve that historical convention but
    # record the assumption in metadata so it is never invisible.
    if unit in (None, "unknown"):
        return raw, None
    if unit not in _TIME_UNIT_SCALE:
        raise ValueError(f"Unsupported NIfTI time unit {unit!r} for pixdim[4]")
    return raw * _TIME_UNIT_SCALE[unit], unit


def _dimension_labels(ndim: int, metadata: dict[str, Any]) -> list[str]:
    labels = ["x", "y", "z"]
    defaults = {5: "DIM_COIL", 6: "DIM_DYN", 7: "DIM_INDIRECT_0"}
    for nifti_dim in range(5, ndim + 1):
        labels.append(str(metadata.get(f"dim_{nifti_dim}") or defaults.get(nifti_dim, f"dim_{nifti_dim}")))
    return labels


def _select_fid(
    data: np.ndarray,
    *,
    index: tuple[int, ...] | None,
    dim_labels: list[str],
) -> tuple[np.ndarray, tuple[int, ...]]:
    if data.ndim < 4:
        raise ValueError(
            f"NIfTI-MRS data must have at least four dimensions (x, y, z, time); got shape {data.shape}"
        )

    nonspectral_axes = [0, 1, 2, *range(4, data.ndim)]
    nonspectral_shape = tuple(int(data.shape[axis]) for axis in nonspectral_axes)

    if index is None:
        if any(size != 1 for size in nonspectral_shape):
            shape_desc = ", ".join(
                f"{label}={size}" for label, size in zip(dim_labels, nonspectral_shape)
            )
            raise ValueError(
                "NIfTI-MRS file contains more than one FID "
                f"({shape_desc}). LCMish will not silently choose/average/coil-combine data. "
                "Pass index=(x, y, z, ...) explicitly, or preprocess to a single FID first."
            )
        selected = tuple(0 for _ in nonspectral_axes)
    else:
        selected = tuple(int(i) for i in index)
        if len(selected) != len(nonspectral_axes):
            raise ValueError(
                f"index must contain {len(nonspectral_axes)} integers for dimensions "
                f"{tuple(dim_labels)}; got {len(selected)}"
            )

    selector: list[int | slice] = [slice(None)] * data.ndim
    normalised: list[int] = []
    for axis, requested, size, label in zip(nonspectral_axes, selected, nonspectral_shape, dim_labels):
        idx = requested + size if requested < 0 else requested
        if idx < 0 or idx >= size:
            raise IndexError(f"Index {requested} is out of range for {label} with size {size}")
        selector[axis] = idx
        normalised.append(idx)

    fid = np.asarray(data[tuple(selector)], dtype=np.complex128)
    if fid.ndim != 1:
        raise ValueError(f"Internal NIfTI-MRS selection did not produce one FID; got shape {fid.shape}")
    return fid, tuple(normalised)


def read_nifti_mrs(
    path: str | Path,
    *,
    index: tuple[int, ...] | None = None,
    reference_ppm: float | None = None,
) -> SpectralData:
    """Read one time-domain FID from a NIfTI-MRS ``.nii``/``.nii.gz`` file.

    Parameters
    ----------
    path
        NIfTI-MRS file. The required spectroscopy JSON extension (ecode 44)
        and ``mrs_vM_m`` intent name are checked.
    index
        Explicit indices for every non-spectral dimension, in storage order:
        ``(x, y, z, dim_5, dim_6, dim_7)`` as applicable. It may be omitted
        only when every non-spectral dimension is singleton.
    reference_ppm
        Chemical-shift value at the centre frequency. If omitted, LCMish uses
        ``SpecFreqChemShift`` when present and otherwise falls back to 0 ppm.

    Notes
    -----
    NIfTI-MRS dimension 4 is the complex time-domain signal. LCMish does not
    automatically combine coils, average dynamics, add/subtract edit states,
    or choose an MRSI voxel. Those are preprocessing decisions, not file I/O.
    """
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError(
            "NIfTI-MRS support is optional. Install LCMish with "
            "`pip install lcmish[nifti]` (or install nibabel directly)."
        ) from exc

    path = Path(path)
    img = nib.load(str(path))
    header = img.header

    intent_name = _decode_intent_name(header)
    if not intent_name.lower().startswith("mrs_v"):
        raise ValueError(
            f"{path} does not declare NIfTI-MRS conformance in intent_name "
            f"(found {intent_name!r})"
        )

    mrs_meta = _read_mrs_extension(header)
    frequencies = mrs_meta.get("SpectrometerFrequency")
    nuclei = mrs_meta.get("ResonantNucleus")
    if not isinstance(frequencies, (list, tuple)) or not frequencies:
        raise ValueError("NIfTI-MRS metadata must contain SpectrometerFrequency as a non-empty array")
    if not isinstance(nuclei, (list, tuple)) or not nuclei:
        raise ValueError("NIfTI-MRS metadata must contain ResonantNucleus as a non-empty array")

    transmitter_mhz = float(frequencies[0])
    if not np.isfinite(transmitter_mhz) or transmitter_mhz <= 0:
        raise ValueError("NIfTI-MRS SpectrometerFrequency must be positive")
    nucleus = str(nuclei[0])

    dwell_time_s, time_unit = _dwell_time_seconds(header)
    data = np.asanyarray(img.dataobj)
    if not np.iscomplexobj(data):
        raise ValueError("NIfTI-MRS data must be stored as complex time-domain samples")

    dim_labels = _dimension_labels(data.ndim, mrs_meta)
    fid, selected = _select_fid(data, index=index, dim_labels=dim_labels)

    warnings: list[str] = []
    if time_unit is None:
        warnings.append("NIfTI time unit was unspecified; pixdim[4] was interpreted as seconds")

    if reference_ppm is None:
        if mrs_meta.get("SpecFreqChemShift") is not None:
            ref_ppm = float(mrs_meta["SpecFreqChemShift"])
            reference_source = "NIfTI-MRS SpecFreqChemShift"
        else:
            ref_ppm = 0.0
            reference_source = "default 0 ppm (SpecFreqChemShift absent)"
            warnings.append(
                "SpecFreqChemShift is absent; reference_ppm defaults to 0.0. "
                "Pass reference_ppm explicitly if the spectral centre is not 0 ppm."
            )
    else:
        ref_ppm = float(reference_ppm)
        reference_source = "user override"

    metadata: dict[str, Any] = {
        "source": str(path),
        "format": "NIfTI-MRS",
        "intent_name": intent_name,
        "shape": [int(v) for v in data.shape],
        "selected_index": list(selected),
        "dimension_labels": dim_labels,
        "nucleus": nucleus,
        "reference_ppm_source": reference_source,
        "nifti_mrs_header": mrs_meta,
    }
    affine = getattr(img, "affine", None)
    if affine is not None:
        metadata["affine"] = np.asarray(affine, dtype=float).tolist()
    if warnings:
        metadata["warnings"] = warnings

    return SpectralData(
        fid=fid,
        dwell_time_s=dwell_time_s,
        transmitter_mhz=transmitter_mhz,
        reference_ppm=ref_ppm,
        metadata=metadata,
    )
