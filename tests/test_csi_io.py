from __future__ import annotations

import numpy as np
import pytest

from lcmish.csi import reconstruct_siemens_csi_array, read_siemens_mrs_dicom_csi
from lcmish.models import CSIData


def test_reconstruct_siemens_csi_array_round_trip():
    rng = np.random.default_rng(12)
    expected = rng.normal(size=(12, 4, 3)) + 1j * rng.normal(size=(12, 4, 3))
    kspace = np.fft.fftshift(
        np.fft.ifft2(
            np.fft.ifftshift(expected[:, :, ::-1], axes=(1, 2)),
            axes=(1, 2),
        ),
        axes=(1, 2),
    )
    kspace = np.repeat(kspace[:, :, None, :], 2, axis=2)

    result = reconstruct_siemens_csi_array(
        kspace,
        dwell_time_s=0.001,
        transmitter_mhz=49.9,
        conjugate_for_nmr=False,
    )

    assert isinstance(result, CSIData)
    assert result.fids.shape == (4, 3, 12)
    assert np.allclose(np.moveaxis(result.fids, -1, 0), expected)
    assert result.metadata["orientation_requires_independent_validation"] is True


def test_reconstruct_rejects_unknown_layout():
    with pytest.raises(ValueError, match="kspace must have shape"):
        reconstruct_siemens_csi_array(
            np.ones((8, 2, 2), complex),
            dwell_time_s=0.001,
            transmitter_mhz=49.9,
        )


@pytest.mark.parametrize(
    ("payload_tag", "expected_conjugation"),
    [((0x5600, 0x0020), True), ((0x7FE1, 0x1010), False)],
)
def test_dicom_payload_sign_convention(tmp_path, payload_tag, expected_conjugation):
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()
    path = tmp_path / "spectroscopy.dcm"
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    source = np.arange(8, dtype="<f4") + 1j * np.arange(10, 18, dtype="<f4")
    interleaved = np.empty(source.size * 2, dtype="<f4")
    interleaved[0::2] = source.real
    interleaved[1::2] = source.imag
    dataset.add_new(payload_tag, "OB", interleaved.tobytes())
    dataset.save_as(path, enforce_file_format=True)

    result = read_siemens_mrs_dicom_csi(
        path,
        spatial_shape=(1, 1),
        dwell_time_s=0.001,
        transmitter_mhz=50.0,
    )

    expected = np.conj(source) if expected_conjugation else source
    assert np.allclose(result.fids[0, 0], expected)
    assert result.metadata["complex_conjugated_for_nmr_ppm"] is expected_conjugation
