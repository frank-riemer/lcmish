from __future__ import annotations

import numpy as np

from lcmish import read_basis, read_raw


def test_read_raw_pairs(tmp_path):
    path = tmp_path / "x.RAW"
    path.write_text("$NMID\n ID='x',\n$END\n 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16\n")
    data = read_raw(path, dwell_time_s=0.001, transmitter_mhz=50.0)
    assert data.npoints == 8
    assert data.fid[0] == 1 + 2j
    assert data.fid[-1] == 15 + 16j


def test_read_simple_basis(tmp_path):
    path = tmp_path / "x.BASIS"
    pairs1 = " ".join(f"{float(i+1)} 0" for i in range(16))
    pairs2 = " ".join(f"{float(2*(i+1))} 0" for i in range(16))
    path.write_text(
        "$BASIS1\n BADELT=0.001,\n NDATAB=16,\n$END\n"
        "$BASIS\n METABO='A',\n ISHIFT=0,\n$END\n" + pairs1 + "\n"
        "$BASIS\n METABO='B',\n ISHIFT=0,\n$END\n" + pairs2 + "\n"
    )
    basis = read_basis(path, transmitter_mhz=50.0)
    assert basis.names == ["A", "B"]
    assert basis.fids.shape == (2, 16)
    assert np.isclose(basis.dwell_time_s, 0.001)
