"""Input readers for spectra and LCModel-style basis files."""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from .models import BasisSet, SpectralData

_FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?")


def _numbers(text: str) -> list[float]:
    return [float(x.replace("D", "E").replace("d", "e")) for x in _FLOAT_RE.findall(text)]


def _parse_assignments(block: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, raw in re.findall(r"(?im)^\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*([^,\n/]+)", block):
        out[key.upper()] = raw.strip().strip("'\"")
    return out


def read_raw(
    path: str | Path,
    *,
    dwell_time_s: float,
    transmitter_mhz: float,
    reference_ppm: float = 0.0,
) -> SpectralData:
    """Read a common LCModel-style `.RAW` file.

    Numeric values after the final namelist terminator are interpreted as
    real/imaginary pairs. Header fields are preserved as metadata where possible.
    """
    path = Path(path)
    text = path.read_text(errors="replace")
    end_positions = [m.end() for m in re.finditer(r"(?im)^\s*\$END\s*$", text)]
    body_start = end_positions[-1] if end_positions else 0
    header = text[:body_start]
    values = _numbers(text[body_start:])
    if len(values) < 16 or len(values) % 2:
        raise ValueError(f"Could not parse complex real/imaginary pairs from {path}")
    fid = np.asarray(values[0::2]) + 1j * np.asarray(values[1::2])
    return SpectralData(
        fid=fid,
        dwell_time_s=dwell_time_s,
        transmitter_mhz=transmitter_mhz,
        reference_ppm=reference_ppm,
        metadata={"source": str(path), **_parse_assignments(header)},
    )


def read_basis(
    path: str | Path,
    *,
    transmitter_mhz: float | None = None,
    reference_ppm: float = 0.0,
) -> BasisSet:
    """Read common LCModel `.BASIS` files into time-domain component signals.

    LCModel basis files encountered in practice contain namelist blocks followed
    by real/imaginary spectral pairs. LCMish parses each `$BASIS` component,
    applies integer `ISHIFT` by rolling the stored spectral vector, and converts
    the stored spectrum to a complex FID with an inverse FFT.

    The LCModel format has historical variants; unusual files should be checked
    explicitly by plotting reconstructed component spectra before fitting.
    """
    path = Path(path)
    text = path.read_text(errors="replace")

    global_blocks = list(re.finditer(r"(?is)\$(?:BASIS1|SEQPAR)\b(.*?)\$END", text))
    global_meta: dict[str, str] = {}
    for match in global_blocks:
        global_meta.update(_parse_assignments(match.group(1)))

    dwell = None
    for key in ("BADELT", "DELTAT"):
        if key in global_meta:
            try:
                dwell = float(global_meta[key].replace("D", "E"))
                break
            except ValueError:
                pass

    component_matches = list(re.finditer(r"(?is)\$BASIS\b(.*?)\$END", text))
    if not component_matches:
        raise ValueError(f"No $BASIS component blocks found in {path}")

    names: list[str] = []
    fids: list[np.ndarray] = []
    component_meta: list[dict[str, str]] = []

    for i, match in enumerate(component_matches):
        attrs = _parse_assignments(match.group(1))
        name = attrs.get("METABO") or attrs.get("METAB") or attrs.get("ID") or f"component_{i+1}"
        data_start = match.end()
        data_end = component_matches[i + 1].start() if i + 1 < len(component_matches) else len(text)
        values = _numbers(text[data_start:data_end])
        if len(values) < 8:
            continue
        if len(values) % 2:
            values = values[:-1]
        spec = np.asarray(values[0::2]) + 1j * np.asarray(values[1::2])
        if "ISHIFT" in attrs:
            try:
                spec = np.roll(spec, int(float(attrs["ISHIFT"])))
            except ValueError:
                pass
        fid = np.fft.ifft(np.fft.ifftshift(spec))
        names.append(name.strip())
        fids.append(fid)
        component_meta.append(attrs)

    if not fids:
        raise ValueError(f"No basis component numeric data found in {path}")

    n = min(len(x) for x in fids)
    arr = np.asarray([x[:n] for x in fids], dtype=np.complex128)
    return BasisSet(
        names=names,
        fids=arr,
        dwell_time_s=dwell,
        transmitter_mhz=transmitter_mhz,
        reference_ppm=reference_ppm,
        metadata={"source": str(path), "header": global_meta, "components": component_meta},
    )


def read_spectrum(
    path: str | Path,
    *,
    dwell_time_s: float | None = None,
    transmitter_mhz: float | None = None,
    reference_ppm: float | None = None,
    index: tuple[int, ...] | None = None,
) -> SpectralData:
    """Read a spectrum using the file extension to select a safe reader.

    NIfTI-MRS (``.nii``/``.nii.gz``) is the preferred vendor-neutral input.
    LCModel-style ``.RAW`` is also supported, but requires explicit dwell time
    and transmitter frequency because those values are not reliably encoded in
    every RAW file. Scanner raw formats are deliberately not auto-interpreted.
    """
    path = Path(path)
    lower = path.name.lower()
    if lower.endswith((".nii", ".nii.gz")):
        from .nifti import read_nifti_mrs

        return read_nifti_mrs(path, index=index, reference_ppm=reference_ppm)
    if lower.endswith(".raw"):
        if dwell_time_s is None or transmitter_mhz is None:
            raise ValueError(
                "LCModel RAW input requires dwell_time_s and transmitter_mhz. "
                "NIfTI-MRS input carries these values in its metadata."
            )
        return read_raw(
            path,
            dwell_time_s=float(dwell_time_s),
            transmitter_mhz=float(transmitter_mhz),
            reference_ppm=0.0 if reference_ppm is None else float(reference_ppm),
        )
    raise ValueError(
        f"Unsupported spectrum input {path.name!r}. Use NIfTI-MRS (.nii/.nii.gz) "
        "or LCModel-style .RAW; Siemens Twix is available separately via read_twix()."
    )


# Friendly short alias for interactive use.
read = read_spectrum
