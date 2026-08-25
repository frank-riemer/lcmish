from __future__ import annotations

from importlib.resources import files
import hashlib
import json

import numpy as np

from lcmish import P31_BRAIN_BASIS_FILENAME, SpectralData, load_p31_brain_basis


EXPECTED_SHA256 = "4b6eafe3cbf244204a35e726a752cfa000eeec47163d1dd4b33f85949c9840b8"


def _peak_ppm(fid, basis):
    data = SpectralData(
        fid[:1024],
        basis.dwell_time_s,
        basis.transmitter_mhz,
        basis.reference_ppm,
    )
    ppm = data.ppm_axis(16384)
    return float(ppm[np.argmax(np.abs(data.spectrum(16384)))])


def test_bundled_basis_is_present_and_unchanged():
    package = files("lcmish")
    resource = package.joinpath("data", P31_BRAIN_BASIS_FILENAME)
    content = resource.read_bytes()
    assert hashlib.sha256(content).hexdigest() == EXPECTED_SHA256
    metadata = json.loads(
        package.joinpath("data", P31_BRAIN_BASIS_FILENAME + ".json").read_text()
    )
    assert metadata["acquired_points"] == 1024
    assert metadata["model_points"] == 2048
    assert metadata["lcmodel_stored_ndatab"] == 4096
    assert "NOPARK" not in json.dumps(metadata).upper()


def test_bundled_basis_loads_with_expected_components_and_grid():
    basis = load_p31_brain_basis()
    assert basis.names == [
        "PE",
        "PC",
        "Pi_ex",
        "Pi",
        "GPE",
        "GPC",
        "PCr",
        "ATPg",
        "ATPa",
        "ATPb",
        "NAD+",
        "NADH",
    ]
    assert basis.npoints == 2048
    assert basis.dwell_time_s == 0.0005
    assert basis.transmitter_mhz == 49.891996
    assert basis.metadata["stored_ndatab"] == 4096
    assert basis.metadata["bundled_with_lcmish"] is True


def test_bundled_basis_peak_positions():
    basis = load_p31_brain_basis()
    for name, expected in {
        "PE": 6.76,
        "PC": 6.24,
        "Pi": 4.82,
        "GPE": 3.50,
        "GPC": 2.95,
        "PCr": 0.0,
        "NADH": -8.13,
    }.items():
        assert np.isclose(
            _peak_ppm(basis.fids[basis.names.index(name)], basis),
            expected,
            atol=0.003,
        )
