"""Input readers for spectra and LCModel-style basis files."""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from .models import BasisSet, SpectralData

_FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?")
_LCMODEL_CENTER_PPM = 4.65


def _numbers(text: str) -> list[float]:
    return [float(x.replace("D", "E").replace("d", "e")) for x in _FLOAT_RE.findall(text)]


def _parse_assignments(block: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, raw in re.findall(r"(?im)^\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*([^,\n/]+)", block):
        out[key.upper()] = raw.strip().strip("'\"")
    return out


def _fortran_nint(value: float) -> int:
    """Return Fortran NINT semantics (nearest integer, ties away from zero)."""
    return int(np.copysign(np.floor(abs(value) + 0.5), value))


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
    """Read an LCModel `.BASIS` file using LCModel ``MYBASI`` conventions.

    The stored arrays are zero-filled, non-rearranged frequency-domain spectra.
    Each component is scaled by ``TRAMP/(VOLUME*CONC)``, shifted by ``ISHIFT``
    plus LCModel's carrier-grid correction from 4.65 ppm, inverse transformed
    with unitary normalization, and reduced to its unfilled time-domain half.
    ``reference_ppm`` is the carrier chemical shift of the data to be fitted.
    """
    path = Path(path)
    text = path.read_text(errors="replace")

    global_blocks = list(re.finditer(r"(?is)\$(?:BASIS1|SEQPAR)\b(.*?)\$END", text))
    global_meta: dict[str, str] = {}
    for match in global_blocks:
        global_meta.update(_parse_assignments(match.group(1)))

    try:
        dwell = float(global_meta["BADELT"].replace("D", "E"))
        ndatab = int(float(global_meta["NDATAB"].replace("D", "E")))
    except (KeyError, ValueError) as exc:
        raise ValueError(f"BASIS1/BADELT and BASIS1/NDATAB are required in {path}") from exc
    if dwell <= 0 or ndatab <= 0 or ndatab % 2:
        raise ValueError("BASIS1/BADELT must be positive and NDATAB must be positive and even")

    basis_f0 = None
    if "HZPPPM" in global_meta:
        try:
            basis_f0 = float(global_meta["HZPPPM"].replace("D", "E"))
        except ValueError:
            pass
    if basis_f0 is None:
        basis_f0 = transmitter_mhz
    if basis_f0 is None and not np.isclose(reference_ppm, _LCMODEL_CENTER_PPM):
        raise ValueError("HZPPPM or transmitter_mhz is required for LCModel carrier correction")

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
        needed = 2 * ndatab
        if len(values) < needed:
            raise ValueError(
                f"Basis component {name!r} has {len(values) // 2} complex points; "
                f"BASIS1/NDATAB declares {ndatab}"
            )
        values = values[:needed]
        spec = np.asarray(values[0::2]) + 1j * np.asarray(values[1::2])

        try:
            conc = float(attrs.get("CONC", "1").replace("D", "E"))
            tramp = float(attrs.get("TRAMP", "1").replace("D", "E"))
            volume = float(attrs.get("VOLUME", "1").replace("D", "E"))
        except ValueError as exc:
            raise ValueError(f"Invalid CONC, TRAMP, or VOLUME for {name!r}") from exc
        if min(conc, tramp, volume) <= 0:
            raise ValueError(f"CONC, TRAMP, and VOLUME must be positive for {name!r}")
        spec *= tramp / (volume * conc)

        try:
            ishift = int(float(attrs.get("ISHIFT", "0").replace("D", "E")))
        except ValueError as exc:
            raise ValueError(f"Invalid ISHIFT for {name!r}") from exc
        reference_shift = 0
        if basis_f0 is not None:
            ppm_increment = 1.0 / (dwell * ndatab * basis_f0)
            reference_shift = _fortran_nint(
                (_LCMODEL_CENTER_PPM - float(reference_ppm)) / ppm_increment
            )
        total_shift = ishift + reference_shift
        if total_shift:
            # LCModel: BASISF(J)=stored(ICYCLE(J+total_shift,NDATAB)).
            spec = np.roll(spec, -total_shift)

        fid_zf = np.fft.ifft(spec) * np.sqrt(ndatab)
        fid = fid_zf[: ndatab // 2].copy()
        names.append(name.strip())
        fids.append(fid)
        component_meta.append(
            {
                **attrs,
                "REFERENCE_SHIFT": str(reference_shift),
                "TOTAL_SHIFT": str(total_shift),
            }
        )

    if not fids:
        raise ValueError(f"No basis component numeric data found in {path}")

    arr = np.asarray(fids, dtype=np.complex128)
    return BasisSet(
        names=names,
        fids=arr,
        dwell_time_s=dwell,
        transmitter_mhz=transmitter_mhz if transmitter_mhz is not None else basis_f0,
        reference_ppm=reference_ppm,
        metadata={
            "source": str(path),
            "header": global_meta,
            "components": component_meta,
            "lcmodel_basis": True,
            "stored_ndatab": ndatab,
            "basis_hzpppm": basis_f0,
        },
    )


def write_basis(
    path: str | Path,
    basis: BasisSet,
    *,
    stored_ndatab: int | None = None,
    idbasi: str = "LCMish generated basis",
) -> Path:
    """Write a :class:`BasisSet` in LCModel-style ``.BASIS`` form.

    The writer is the inverse of :func:`read_basis`: arrays are stored on the
    unrearranged frequency grid with unitary normalization.  ``ISHIFT``
    cancels LCModel's 4.65-ppm carrier-grid correction so that the supplied
    ``basis.reference_ppm`` is preserved.
    """
    path = Path(path)
    if basis.dwell_time_s is None or basis.transmitter_mhz is None:
        raise ValueError("Writing .BASIS requires dwell_time_s and transmitter_mhz")
    ndatab = int(stored_ndatab or 2 * basis.npoints)
    if ndatab <= 0 or ndatab % 2 or ndatab < basis.npoints:
        raise ValueError("stored_ndatab must be even and at least basis.npoints")

    dwell = float(basis.dwell_time_s)
    f0 = float(basis.transmitter_mhz)
    ppm_increment = 1.0 / (dwell * ndatab * f0)
    reference_shift = _fortran_nint(
        (_LCMODEL_CENTER_PPM - float(basis.reference_ppm)) / ppm_increment
    )
    ishift = -reference_shift

    lines = [
        " $SEQPAR",
        f" HZPPPM= {f0:.10E},",
        " SEQ='LCMish generated',",
        " $END",
        " $BASIS1",
        f" IDBASI='{idbasi[:72]}',",
        " FMTBAS='(6E16.8)',",
        f" BADELT= {dwell:.10E},",
        f" NDATAB= {ndatab},",
        " $END",
    ]
    for name, fid in zip(basis.names, basis.fids):
        fid_zf = np.zeros(ndatab, dtype=np.complex128)
        fid_zf[: basis.npoints] = np.asarray(fid, dtype=np.complex128)
        stored = np.fft.fft(fid_zf) / np.sqrt(ndatab)
        lines.extend(
            [
                " $BASIS",
                f" ID='{name[:20]}',",
                f" METABO='{name[:6]}',",
                " CONC= 1.0,",
                " TRAMP= 1.0,",
                " VOLUME= 1.0,",
                f" ISHIFT= {ishift},",
                " $END",
            ]
        )
        values = np.empty(2 * ndatab, dtype=float)
        values[0::2] = stored.real
        values[1::2] = stored.imag
        for start in range(0, values.size, 6):
            lines.append(" ".join(f"{value:16.8E}" for value in values[start : start + 6]))
    path.write_text("\n".join(lines) + "\n")
    return path


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
