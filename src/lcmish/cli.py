"""Command-line interface for LCMish."""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import p31_brain_config, p31_brain_grouped_config
from .fitter import fit_spectrum
from .io import read_basis, read_spectrum


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fit an MRS spectrum with LCMish.")
    p.add_argument("spectrum", type=Path, help="NIfTI-MRS (.nii/.nii.gz) or LCModel-style .RAW")
    p.add_argument("basis", type=Path, help="LCModel-style .BASIS file")
    p.add_argument("--dwell", type=float, help="Dwell time in seconds (required for .RAW only)")
    p.add_argument("--f0", type=float, help="Transmitter frequency in MHz (required for .RAW only)")
    p.add_argument(
        "--ref-ppm",
        type=float,
        default=None,
        help="Chemical-shift value at spectral centre; NIfTI-MRS uses SpecFreqChemShift when available",
    )
    p.add_argument(
        "--index",
        type=int,
        nargs="+",
        default=None,
        metavar="I",
        help="Explicit non-spectral NIfTI-MRS indices: x y z [dim5 [dim6 [dim7]]]",
    )
    p.add_argument("--ppm-min", type=float, required=True)
    p.add_argument("--ppm-max", type=float, required=True)
    p.add_argument("--grouped-31p", action="store_true", help="Use the grouped 31P starting configuration")
    p.add_argument("--out", type=Path, default=Path("lcmish_fit"))
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    data = read_spectrum(
        args.spectrum,
        dwell_time_s=args.dwell,
        transmitter_mhz=args.f0,
        reference_ppm=args.ref_ppm,
        index=None if args.index is None else tuple(args.index),
    )
    basis = read_basis(
        args.basis,
        transmitter_mhz=data.transmitter_mhz,
        reference_ppm=data.reference_ppm,
    )
    factory = p31_brain_grouped_config if args.grouped_31p else p31_brain_config
    config = factory((args.ppm_min, args.ppm_max))
    result = fit_spectrum(data, basis, config)
    prefix = args.out
    result.save_csv(prefix.with_suffix(".csv"))
    result.save_components_csv(prefix.with_name(prefix.name + "_spectrum.csv"))
    result.save_table(prefix.with_suffix(".table"))
    result.save_checkpoint_npz(prefix.with_suffix(".npz"))
    result.save_pdf(prefix.with_suffix(".pdf"), title=f"LCMish {prefix.name}")
    result.plot(prefix.with_suffix(".png"), title=f"LCMish {prefix.name}")
    print(f"success={result.success} cost={result.cost:.6g}")
    for row in result.summary_rows():
        print(
            f"{row['component']}: {row['amplitude']:.6g} "
            f"(conditional CRLB-like {row['conditional_crlb_like_percent']:.2f}%)"
        )
    return 0 if result.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
