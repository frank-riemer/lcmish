#!/usr/bin/env python3
"""Example: read Siemens MR Spectroscopy DICOM and run the masked 31P workflow."""
from __future__ import annotations

import argparse
from pathlib import Path

from _siemens_csi_workflow import run_workflow
from lcmish.csi import read_siemens_mrs_dicom_csi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dicom", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True, help="Verified 2-D boolean .npy mask")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--basis",
        type=Path,
        help="LCModel .BASIS file; defaults to the bundled Siemens 3 T starter basis",
    )
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--columns", type=int, required=True)
    parser.add_argument("--dwell", type=float, required=True, help="Dwell time in seconds")
    parser.add_argument("--f0", type=float, required=True, help="31P transmitter frequency in MHz")
    parser.add_argument("--pcr-snr-min", type=float, default=10.0)
    parser.add_argument("--min-retained-voxels", type=int, default=3)
    parser.add_argument(
        "--conjugate-for-nmr",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override payload-specific sign only after independent validation",
    )
    args = parser.parse_args()

    csi = read_siemens_mrs_dicom_csi(
        args.dicom,
        spatial_shape=(args.rows, args.columns),
        dwell_time_s=args.dwell,
        transmitter_mhz=args.f0,
        conjugate_for_nmr=args.conjugate_for_nmr,
    )
    run_workflow(
        csi,
        args.mask,
        args.output,
        basis_path=args.basis,
        pcr_snr_min=args.pcr_snr_min,
        min_retained_voxels=args.min_retained_voxels,
    )


if __name__ == "__main__":
    main()
