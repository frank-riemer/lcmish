import lcmish


def test_version_lineage():
    assert lcmish.__version__ == "0.3.0"


def test_expected_public_api():
    for name in (
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
        "save_pdf_report",
        "fit_p31_redox",
        "redox_nuisance_sensitivity",
        "prepare_p31_csi_redox",
        "fit_p31_csi_redox",
        "load_p31_brain_basis",
        "reconstruct_siemens_csi_array",
        "read_siemens_twix_csi",
        "read_siemens_mrs_dicom_csi",
    ):
        assert hasattr(lcmish, name)
