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
    ndatab = 16
    dwell = 0.001
    f0 = 50.0
    reference_ppm = 4.65
    fid_a = np.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.complex128)
    fid_b = 2 * fid_a

    def stored_pairs(fid):
        zf = np.r_[fid, np.zeros(ndatab // 2, dtype=np.complex128)]
        spec = np.fft.fft(zf) / np.sqrt(ndatab)
        return " ".join(f"{x.real:.17g} {x.imag:.17g}" for x in spec)

    path.write_text(
        f"$SEQPAR\n HZPPPM={f0},\n$END\n"
        f"$BASIS1\n BADELT={dwell},\n NDATAB={ndatab},\n$END\n"
        "$BASIS\n METABO='A',\n CONC=2,\n TRAMP=4,\n VOLUME=2,\n ISHIFT=0,\n$END\n"
        + stored_pairs(fid_a)
        + "\n$NMUSED\n PPMSCA=8.44,\n$END\n"
        "$BASIS\n METABO='B',\n ISHIFT=0,\n$END\n"
        + stored_pairs(fid_b)
        + "\n"
    )
    basis = read_basis(path, reference_ppm=reference_ppm)
    assert basis.names == ["A", "B"]
    assert basis.fids.shape == (2, 8)
    assert np.isclose(basis.dwell_time_s, dwell)
    assert np.allclose(basis.fids[0], fid_a)
    assert np.allclose(basis.fids[1], fid_b)


def test_read_basis_applies_lcmodel_shift_and_scaling(tmp_path):
    ndatab = 32
    dwell = 0.001
    f0 = 50.0
    reference_ppm = 0.0
    fid = np.exp(1j * 2 * np.pi * 25.0 * np.arange(ndatab // 2) * dwell)
    zf = np.r_[fid, np.zeros(ndatab // 2, dtype=np.complex128)]
    shifted_spec = np.fft.fft(zf) / np.sqrt(ndatab)
    ppm_increment = 1.0 / (dwell * ndatab * f0)
    reference_shift = int(np.floor((4.65 - reference_ppm) / ppm_increment + 0.5))
    ishift = -3
    stored = np.roll(shifted_spec, ishift + reference_shift)
    pairs = " ".join(f"{x.real:.17g} {x.imag:.17g}" for x in stored)
    path = tmp_path / "shifted.BASIS"
    path.write_text(
        f"$SEQPAR\n HZPPPM={f0},\n$END\n"
        f"$BASIS1\n BADELT={dwell},\n NDATAB={ndatab},\n$END\n"
        f"$BASIS\n METABO='X',\n CONC=2,\n TRAMP=6,\n VOLUME=3,\n ISHIFT={ishift},\n$END\n"
        + pairs
        + "\n"
    )
    basis = read_basis(path, reference_ppm=reference_ppm)
    assert np.allclose(basis.fids[0], fid)
    assert basis.metadata["components"][0]["REFERENCE_SHIFT"] == str(reference_shift)
