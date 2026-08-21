"""LCMish — transparent linear-combination modelling for MR spectroscopy."""

from .compat import trapezoid
from .config import p31_brain_config, p31_brain_grouped_config
from .fitter import fit_spectrum, fit_spectrum_multistart
from .io import read, read_basis, read_raw, read_spectrum
from .nifti import read_nifti_mrs
from .models import BasisSet, FitAudit, FitConfig, FitResult, GroupConfig, SpectralData
from .report import save_pdf_report
from .twix import TwixData, read_twix

__version__ = "0.2.1"

__all__ = [
    "__version__",
    "BasisSet",
    "SpectralData",
    "GroupConfig",
    "FitConfig",
    "FitResult",
    "FitAudit",
    "read",
    "read_spectrum",
    "read_nifti_mrs",
    "read_basis",
    "read_raw",
    "fit_spectrum",
    "fit_spectrum_multistart",
    "p31_brain_config",
    "p31_brain_grouped_config",
    "TwixData",
    "read_twix",
    "trapezoid",
    "save_pdf_report",
]
