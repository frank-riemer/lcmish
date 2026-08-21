from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest

from lcmish import read, read_nifti_mrs


class _Extension:
    def __init__(self, payload):
        self._payload = payload

    def get_content(self):
        return json.dumps(self._payload).encode("utf-8")


class _Extensions(list):
    def get_codes(self):
        return [44 for _ in self]


class _Header(dict):
    def __init__(self, payload, *, dwell=0.0005, unit="sec", intent="mrs_v0_11"):
        super().__init__(pixdim=np.array([1, 1, 1, 1, dwell, 1, 1, 1], dtype=float))
        self.extensions = _Extensions([_Extension(payload)])
        self._unit = unit
        self._intent = intent

    def get_intent(self):
        return ("none", (), self._intent)

    def get_xyzt_units(self):
        return ("mm", self._unit)


class _Image:
    def __init__(self, data, payload, **header_kwargs):
        self.dataobj = data
        self.header = _Header(payload, **header_kwargs)
        self.affine = np.eye(4)


def _install_fake_nibabel(monkeypatch, image):
    fake = types.ModuleType("nibabel")
    fake.load = lambda path: image
    monkeypatch.setitem(sys.modules, "nibabel", fake)


def _payload(**extra):
    out = {
        "SpectrometerFrequency": [51.7],
        "ResonantNucleus": ["31P"],
        "SpecFreqChemShift": 0.25,
    }
    out.update(extra)
    return out


def test_read_single_fid_nifti_mrs(monkeypatch, tmp_path):
    fid = np.arange(16, dtype=float) + 1j * np.arange(16, dtype=float)[::-1]
    image = _Image(fid.reshape(1, 1, 1, 16), _payload(), dwell=0.5, unit="msec")
    _install_fake_nibabel(monkeypatch, image)

    path = tmp_path / "svs.nii.gz"
    data = read_nifti_mrs(path)

    assert data.npoints == 16
    assert np.allclose(data.fid, fid)
    assert np.isclose(data.dwell_time_s, 0.0005)
    assert np.isclose(data.transmitter_mhz, 51.7)
    assert np.isclose(data.reference_ppm, 0.25)
    assert data.metadata["nucleus"] == "31P"
    assert data.metadata["format"] == "NIfTI-MRS"


def test_generic_read_detects_nifti_mrs(monkeypatch, tmp_path):
    fid = np.ones(16, dtype=np.complex64)
    image = _Image(fid.reshape(1, 1, 1, 16), _payload())
    _install_fake_nibabel(monkeypatch, image)

    data = read(tmp_path / "anything.nii.gz")
    assert data.npoints == 16
    assert data.metadata["format"] == "NIfTI-MRS"


def test_higher_dimensions_require_explicit_index(monkeypatch, tmp_path):
    arr = np.zeros((1, 1, 1, 16, 2), dtype=np.complex64)
    arr[0, 0, 0, :, 0] = 1 + 1j
    arr[0, 0, 0, :, 1] = 2 + 3j
    image = _Image(arr, _payload(dim_5="DIM_DYN"))
    _install_fake_nibabel(monkeypatch, image)
    path = tmp_path / "dyn.nii.gz"

    with pytest.raises(ValueError, match="will not silently choose/average/coil-combine"):
        read_nifti_mrs(path)

    data = read_nifti_mrs(path, index=(0, 0, 0, 1))
    assert np.all(data.fid == 2 + 3j)
    assert data.metadata["selected_index"] == [0, 0, 0, 1]
    assert data.metadata["dimension_labels"][-1] == "DIM_DYN"


def test_reference_ppm_can_be_overridden(monkeypatch, tmp_path):
    fid = np.ones(16, dtype=np.complex64)
    image = _Image(fid.reshape(1, 1, 1, 16), _payload())
    _install_fake_nibabel(monkeypatch, image)

    data = read_nifti_mrs(tmp_path / "svs.nii", reference_ppm=4.7)
    assert np.isclose(data.reference_ppm, 4.7)
    assert data.metadata["reference_ppm_source"] == "user override"


def test_non_mrs_nifti_is_rejected(monkeypatch, tmp_path):
    fid = np.ones(16, dtype=np.complex64)
    image = _Image(fid.reshape(1, 1, 1, 16), _payload(), intent="none")
    _install_fake_nibabel(monkeypatch, image)

    with pytest.raises(ValueError, match="does not declare NIfTI-MRS"):
        read_nifti_mrs(tmp_path / "image.nii.gz")


def test_real_nibabel_roundtrip_if_available(tmp_path):
    """Integration test run in CI when the optional NIfTI dependency is installed."""
    nib = pytest.importorskip("nibabel")

    fid = (np.arange(32, dtype=np.float32) + 1j * np.arange(32, dtype=np.float32)[::-1]).astype(
        np.complex64
    )
    img = nib.Nifti2Image(fid.reshape(1, 1, 1, 32), np.eye(4))
    img.header.set_intent("none", name="mrs_v0_11")
    img.header["pixdim"][4] = 0.00025
    img.header.set_xyzt_units("mm", "sec")
    payload = _payload(Manufacturer="Synthetic Scanner")
    img.header.extensions.append(
        nib.nifti1.Nifti1Extension(44, json.dumps(payload).encode("utf-8"))
    )
    path = tmp_path / "roundtrip.nii.gz"
    nib.save(img, path)

    data = read_nifti_mrs(path)
    assert np.allclose(data.fid, fid)
    assert np.isclose(data.dwell_time_s, 0.00025)
    assert np.isclose(data.transmitter_mhz, 51.7)
    assert np.isclose(data.reference_ppm, 0.25)
