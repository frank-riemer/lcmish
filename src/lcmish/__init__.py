"""LCMish — transparent linear-combination modelling for MR spectroscopy."""

from .compat import trapezoid
from .config import p31_brain_config, p31_brain_grouped_config
from .fitter import fit_spectrum, fit_spectrum_multistart
from .io import read, read_basis, read_raw, read_spectrum
from .nifti import read_nifti_mrs
from .models import BasisSet, CSIData, FitAudit, FitConfig, FitResult, GroupConfig, SpectralData
from .redox import (
    P31CSIPreparationResult,
    P31CSIRedoxQCConfig,
    P31CSIRedoxResult,
    P31RedoxConfig,
    P31RedoxResult,
    fit_p31_csi_redox,
    fit_p31_redox,
    nad_plus_ab_pattern,
    p31_csi_pcr_snr,
    prepare_p31_csi_redox,
    redox_nuisance_sensitivity,
)
from .report import save_pdf_report
from .twix import TwixData, read_twix

__version__ = "0.3.0.dev0"

__all__ = [
    "__version__",
    "BasisSet",
    "CSIData",
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
    "P31RedoxConfig",
    "P31RedoxResult",
    "P31CSIRedoxQCConfig",
    "P31CSIPreparationResult",
    "P31CSIRedoxResult",
    "nad_plus_ab_pattern",
    "fit_p31_redox",
    "redox_nuisance_sensitivity",
    "p31_csi_pcr_snr",
    "prepare_p31_csi_redox",
    "fit_p31_csi_redox",
]
