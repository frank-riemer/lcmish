#!/usr/bin/env python3
"""Example: reconstruct Siemens Twix 31P CSI and run the masked workflow."""
from __future__ import annotations

import argparse
from pathlib import Path

from _siemens_csi_workflow import run_workflow
from lcmish.csi import read_siemens_twix_csi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--twix", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True, help="Verified 2-D boolean .npy mask")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pcr-snr-min", type=float, default=10.0)
    parser.add_argument("--min-retained-voxels", type=int, default=3)
    parser.add_argument(
        "--remove-oversampling",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--reverse-second-spatial-axis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sequence-specific; validate against scanner reconstruction",
    )
    parser.add_argument(
        "--conjugate-for-nmr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sequence-specific; validate ATP and Pi sides of PCr",
    )
    args = parser.parse_args()

    csi = read_siemens_twix_csi(
        args.twix,
        remove_oversampling=args.remove_oversampling,
        reverse_second_spatial_axis=args.reverse_second_spatial_axis,
        conjugate_for_nmr=args.conjugate_for_nmr,
    )
    run_workflow(
        csi,
        args.mask,
        args.output,
        pcr_snr_min=args.pcr_snr_min,
        min_retained_voxels=args.min_retained_voxels,
    )


if __name__ == "__main__":
    main()
