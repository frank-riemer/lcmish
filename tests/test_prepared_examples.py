"""Input-contract and smoke tests for the prepared MATLAB/1H documentation.

These test software behaviour, not the physical validity of a clinical basis.
"""
from __future__ import annotations

import importlib.util
import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.io import savemat

from lcmish import BasisSet, FitConfig, SpectralData, fit_spectrum, write_basis
import lcmish


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "fit_prepared_1h.py"
spec = importlib.util.spec_from_file_location("prepared_1h_example", EXAMPLE)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


@pytest.mark.parametrize("relative", ["README.md", "examples/README.md"])
def test_readme_python_examples_parse(relative):
    readme = EXAMPLE.parents[1] / relative
    # Check syntax without pretending placeholder acquisition files are available.
    for block in readme.read_text().split("```python\n")[1:]:
        ast.parse(block.split("```", 1)[0])


def _export(fid=None, **overrides):
    values = {
        "fid": np.exp(-np.arange(64) / 12) * np.exp(1j * np.arange(64) / 5),
        "dwell_time_s": 0.0005,
        "transmitter_mhz": 127.7,
        "reference_ppm": 4.7,
    }
    if fid is not None:
        values["fid"] = fid
    values.update(overrides)
    return values


@pytest.mark.parametrize("column", [False, True])
def test_matlab_complex_row_and_column_vectors(tmp_path, column):
    export = _export()
    fid = export["fid"]
    export["fid"] = fid[:, None] if column else fid[None, :]
    path = tmp_path / "fid.mat"
    savemat(path, export)
    data = example.load_prepared_fid(path)
    np.testing.assert_array_equal(data.fid, fid)
    assert data.dwell_time_s == 0.0005
    assert data.transmitter_mhz == 127.7
    assert data.reference_ppm == 4.7


@pytest.mark.parametrize("fid", [np.ones(64), np.ones((64, 3, 2), dtype=complex)])
def test_matlab_rejects_real_signal_and_multivoxel_grid(tmp_path, fid):
    path = tmp_path / "bad.mat"
    savemat(path, _export(fid))
    with pytest.raises(ValueError, match="one complex vector"):
        example.load_matlab_fid(path)


@pytest.mark.parametrize("key,value", [
    ("reference_ppm", np.nan), ("transmitter_mhz", [127.7, 127.8]),
    ("dwell_time_s", -0.0005), ("reference_ppm", 4.7 + 1j),
])
def test_matlab_rejects_invalid_metadata(tmp_path, key, value):
    path = tmp_path / "bad.mat"
    savemat(path, _export(**{key: value}))
    with pytest.raises(ValueError):
        example.load_matlab_fid(path)


def test_matlab_requires_metadata_and_finite_fid(tmp_path):
    path = tmp_path / "missing.mat"
    export = _export()
    del export["reference_ppm"]
    savemat(path, export)
    with pytest.raises(ValueError, match="reference_ppm"):
        example.load_matlab_fid(path)
    export = _export()
    export["fid"][10] = np.nan + 1j
    savemat(path, export)
    with pytest.raises(ValueError, match="finite"):
        example.load_matlab_fid(path)


def _nifti(path, fid, ref=None, nucleus="1H"):
    nib = pytest.importorskip("nibabel")
    meta = {"SpectrometerFrequency": [127.7], "ResonantNucleus": [nucleus]}
    if ref is not None:
        meta["SpecFreqChemShift"] = ref
    image = nib.Nifti2Image(fid.reshape(1, 1, 1, -1), np.eye(4))
    image.header.set_intent("none", name="mrs_v0_9")
    image.header.set_xyzt_units("mm", "sec")
    image.header["pixdim"][4] = 0.0005
    image.header.extensions.append(nib.nifti1.Nifti1Extension(44, json.dumps(meta).encode()))
    nib.save(image, path)


def test_nifti_reference_is_explicit_or_from_header(tmp_path):
    fid = _export()["fid"]
    path = tmp_path / "fid.nii.gz"
    _nifti(path, fid)
    with pytest.raises(ValueError, match="Missing chemical-shift reference"):
        example.load_prepared_fid(path)
    data = example.load_prepared_fid(path, reference_ppm=4.7)
    np.testing.assert_array_equal(data.fid, fid)
    assert data.reference_ppm == 4.7
    _nifti(path, fid, ref=4.7)
    assert example.load_prepared_fid(path).reference_ppm == 4.7
    _nifti(path, fid, ref=0.0, nucleus="31P")
    with pytest.raises(ValueError, match="1H"):
        example.load_prepared_fid(path)


def _basis(nucleus="1H"):
    f0, ref, peaks, window = {
        "1H": (127.7, 4.7, [2.01, 3.03], (0.5, 4.2)),
        "31P": (51.7, 0.0, [0.0, 4.8], (-2.0, 7.0)),
        "13C": (32.1, 100.0, [90.0, 110.0], (85.0, 115.0)),
    }[nucleus]
    t = np.arange(256) * 0.0005
    fids = np.asarray([
        np.exp(-np.pi * 6 * t) * np.exp(2j * np.pi * (ref - ppm) * f0 * t)
        for ppm in peaks
    ])
    basis = BasisSet(["comp_A", "comp_B"], fids, 0.0005, f0, ref)
    data = SpectralData(fids[0] + 0.3 * fids[1], 0.0005, f0, ref)
    return basis, data, peaks, window


@pytest.mark.parametrize("nucleus", ["1H", "31P", "13C"])
def test_general_api_reference_and_amplitudes_for_multiple_nuclei(nucleus):
    basis, data, peaks, window = _basis(nucleus)
    ppm = data.ppm_axis()
    for signal, centre in zip(basis.fids, peaks):
        measured = ppm[np.argmax(np.abs(np.fft.fftshift(np.fft.fft(signal))))]
        assert abs(measured - centre) < 1 / (data.npoints * data.dwell_time_s * data.transmitter_mhz)
    config = FitConfig(
        ppm_range=window, baseline_knots=6, baseline_lambda=1,
        global_shift_bounds_ppm=(-1e-4, 1e-4),
        phase0_bounds_deg=(-1e-3, 1e-3),
        phase1_bounds_deg_per_ppm=(-1e-3, 1e-3),
        lorentzian_bounds_hz=(0, 1e-3), gaussian_bounds_hz=(0, 1e-3),
        initial_lorentzian_hz=0, initial_gaussian_hz=0, max_nfev=50,
    )
    result = fit_spectrum(data, basis, config)
    assert result.success
    assert result.metadata["fit_domain"] == "complex"
    assert np.isclose(result.amplitudes[1] / result.amplitudes[0], 0.3, rtol=0.01)


def test_prepared_example_end_to_end_matlab_and_nifti(tmp_path):
    basis, data, _, _ = _basis()
    basis_path = tmp_path / "synthetic_1h.BASIS"
    write_basis(basis_path, basis)
    mat_path = tmp_path / "prepared.mat"
    savemat(mat_path, _export(data.fid))
    nifti_path = tmp_path / "prepared.nii.gz"
    _nifti(nifti_path, data.fid, ref=4.7)
    for source in (mat_path, nifti_path):
        out = tmp_path / source.stem
        # Match the package under test, including pytest's source-directory
        # selection or an explicitly selected wheel, in the child interpreter.
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(lcmish.__file__).resolve().parents[1])
        completed = subprocess.run([
            sys.executable, str(EXAMPLE), str(source), str(basis_path), "--out", str(out),
        ], capture_output=True, text=True, timeout=60, env=env)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert Path(str(out) + ".pdf").is_file()
        assert Path(str(out) + ".csv").is_file()
        audit = json.loads(Path(str(out) + "_audit.json").read_text())
        assert audit["success"]
        assert audit["config"]["fit_domain"] == "complex"
        assert audit["reference_ppm"] == 4.7
        assert audit["units"].startswith("fitted signal")
