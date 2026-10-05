"""Fit one prepared 1H FID from MATLAB v7 or NIfTI-MRS.

This is an API example, not a preprocessing or absolute-concentration pipeline.
Use an acquisition-matched 1H basis and validate settings/QC independently.
MATLAB contract: one complex row/column vector named fid, plus scalar
dwell_time_s, transmitter_mhz and reference_ppm. No implicit grid flattening.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.io import loadmat

import lcmish
from lcmish import FitConfig, SpectralData, fit_spectrum, read, read_basis


def _scalar(mat: dict, key: str) -> float:
    if key not in mat:
        raise ValueError(f"MATLAB export needs scalar {key!r}")
    value = np.asarray(mat[key])
    if value.size != 1 or np.iscomplexobj(value):
        raise ValueError(f"{key} must be one real scalar")
    number = float(value.item())
    if not np.isfinite(number):
        raise ValueError(f"{key} must be finite")
    return number


def load_matlab_fid(path: str | Path) -> SpectralData:
    """Load one prepared vector; metadata are required rather than guessed."""
    mat = loadmat(path)
    if "fid" not in mat:
        raise ValueError("MATLAB export needs one complex vector named 'fid'")
    fid = np.asarray(mat["fid"]).squeeze()
    if fid.ndim != 1 or not np.iscomplexobj(fid):
        raise ValueError("Expected one complex vector, not a real signal or MRSI grid")
    if not np.all(np.isfinite(fid)):
        raise ValueError("FID must contain finite samples")
    return SpectralData(
        fid=fid,
        dwell_time_s=_scalar(mat, "dwell_time_s"),
        transmitter_mhz=_scalar(mat, "transmitter_mhz"),
        reference_ppm=_scalar(mat, "reference_ppm"),
        metadata={"source": str(path), "format": "MATLAB prepared FID"},
    )


def load_prepared_fid(
    path: str | Path,
    *,
    index: tuple[int, ...] | None = None,
    reference_ppm: float | None = None,
) -> SpectralData:
    """Load MATLAB or NIfTI-MRS without inferring the chemical-shift reference."""
    if reference_ppm is not None and not np.isfinite(reference_ppm):
        raise ValueError("reference_ppm must be finite")
    path = Path(path)
    if path.suffix.lower() == ".mat":
        if index is not None:
            raise ValueError("Select a voxel in MATLAB before exporting this vector")
        data = load_matlab_fid(path)
        if reference_ppm is not None:
            raise ValueError("MATLAB reference_ppm must be supplied in the export")
    elif str(path).lower().endswith((".nii", ".nii.gz")):
        data = read(path, index=index, reference_ppm=reference_ppm)
        if data.metadata["reference_ppm_source"].startswith("default"):
            raise ValueError("Missing chemical-shift reference; supply verified --ref-ppm")
        if str(data.metadata["nucleus"]).strip() != "1H":
            raise ValueError("This example expects ResonantNucleus='1H'")
    else:
        raise ValueError("This example accepts prepared .mat or NIfTI-MRS files")
    if not np.all(np.isfinite(data.fid)) or not np.isfinite(data.reference_ppm):
        raise ValueError("FID and chemical-shift reference must be finite")
    return data


def main() -> None:
    # This batch example writes reports and does not require an interactive GUI.
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("basis", type=Path, help="Acquisition-matched 1H LCModel-style BASIS")
    parser.add_argument("--out", type=Path, default=Path("proton_fit"))
    parser.add_argument("--ref-ppm", type=float, help="Verified NIfTI spectral-centre ppm")
    parser.add_argument("--index", type=int, nargs="+", help="All NIfTI non-time axes, zero-based")
    parser.add_argument("--ppm-min", type=float, default=0.5)
    parser.add_argument("--ppm-max", type=float, default=4.2)
    parser.add_argument("--fit-domain", choices=("complex", "real", "imag", "magnitude"), default="complex")
    args = parser.parse_args()
    if not np.isfinite([args.ppm_min, args.ppm_max]).all() or args.ppm_min >= args.ppm_max:
        parser.error("ppm-min and ppm-max must be finite and increasing")
    data = load_prepared_fid(
        args.input,
        index=None if args.index is None else tuple(args.index),
        reference_ppm=args.ref_ppm,
    )
    basis = read_basis(
        args.basis, transmitter_mhz=data.transmitter_mhz, reference_ppm=data.reference_ppm,
    )
    config = FitConfig(
        ppm_range=(args.ppm_min, args.ppm_max), fit_domain=args.fit_domain, max_nfev=500,
    )
    result = fit_spectrum(data, basis, config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    prefix = str(args.out)
    result.save_csv(prefix + ".csv")
    result.save_components_csv(prefix + "_components.csv")
    result.save_pdf(prefix + ".pdf", title="LCMish prepared 1H example")
    audit = {
        "lcmish_version": lcmish.__version__,
        "input": str(args.input), "basis": str(args.basis),
        "dwell_time_s": data.dwell_time_s,
        "transmitter_mhz": data.transmitter_mhz,
        "reference_ppm": data.reference_ppm,
        "selected_index": data.metadata.get("selected_index"),
        "input_warnings": data.metadata.get("warnings", []),
        "config": asdict(config), "success": result.success,
        "message": result.message, "nonlinear": result.nonlinear,
        "units": "fitted signal amplitudes; no water concentration calibration",
    }
    with Path(prefix + "_audit.json").open("w") as handle:
        json.dump(audit, handle, indent=2)
    print(result.success, result.message)
    print(result.summary_rows())
    print("Inspect both residual channels, bounds and basis adequacy before interpretation.")
    if not result.success:
        raise SystemExit("Fit did not converge; diagnostics were saved, but reject it pending review.")


if __name__ == "__main__":
    main()
