import lcmish


def test_version_lineage():
    assert lcmish.__version__ == "0.2.1"


def test_expected_public_api():
    for name in (
        "read",
        "read_spectrum",
        "read_nifti_mrs",
        "read_basis",
        "read_raw",
        "fit_spectrum",
        "fit_spectrum_multistart",
        "p31_brain_config",
        "p31_brain_grouped_config",
        "save_pdf_report",
    ):
        assert hasattr(lcmish, name)
