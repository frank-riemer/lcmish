"""LCMish — transparent linear-combination modelling for MR spectroscopy."""

from .compat import trapezoid
from .config import p31_brain_config, p31_brain_grouped_config
from .assets import P31_BRAIN_BASIS_FILENAME, load_p31_brain_basis
from .csi import (
    read_mrs_dicom_csi,
    read_siemens_mrs_dicom_csi,
    read_siemens_twix_csi,
    reconstruct_siemens_csi_array,
)
from .fitter import fit_spectrum, fit_spectrum_multistart
from .io import read, read_basis, read_raw, read_spectrum, write_basis
from .nifti import read_nifti_mrs
from .models import BasisSet, CSIData, FitAudit, FitConfig, FitResult, GroupConfig, SpectralData
from .redox import (
    P31AnchorInformedRedoxResult,
    P31AnchorPhaseConfig,
    P31AnchorPhaseEstimate,
    P31CSIPreparationResult,
    P31CSIRedoxQCConfig,
    P31CSIRedoxResult,
    P31RedoxConfig,
    P31RedoxResult,
    apply_p31_phase_correction,
    estimate_p31_anchor_phase,
    fit_p31_csi_redox,
    fit_p31_redox,
    fit_p31_redox_anchor_informed,
    nad_plus_ab_pattern,
    p31_csi_pcr_snr,
    prepare_p31_csi_redox,
    redox_nuisance_sensitivity,
)
from .report import save_pdf_report
from .twix import TwixData, read_twix

__version__ = "0.4.0"

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
    "write_basis",
    "read_raw",
    "fit_spectrum",
    "fit_spectrum_multistart",
    "p31_brain_config",
    "p31_brain_grouped_config",
    "TwixData",
    "read_twix",
    "reconstruct_siemens_csi_array",
    "read_siemens_twix_csi",
    "read_siemens_mrs_dicom_csi",
    "read_mrs_dicom_csi",
    "trapezoid",
    "save_pdf_report",
    "P31RedoxConfig",
    "P31RedoxResult",
    "P31AnchorPhaseConfig",
    "P31AnchorPhaseEstimate",
    "P31AnchorInformedRedoxResult",
    "P31CSIRedoxQCConfig",
    "P31CSIPreparationResult",
    "P31CSIRedoxResult",
    "nad_plus_ab_pattern",
    "fit_p31_redox",
    "estimate_p31_anchor_phase",
    "apply_p31_phase_correction",
    "fit_p31_redox_anchor_informed",
    "redox_nuisance_sensitivity",
    "p31_csi_pcr_snr",
    "prepare_p31_csi_redox",
    "fit_p31_csi_redox",
    "P31_BRAIN_BASIS_FILENAME",
    "load_p31_brain_basis",
]
