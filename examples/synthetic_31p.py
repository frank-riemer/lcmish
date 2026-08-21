"""Tiny synthetic LCMish example; no external basis set required."""
from __future__ import annotations

import numpy as np

from lcmish import BasisSet, FitConfig, SpectralData, fit_spectrum

rng = np.random.default_rng(7)
n = 512
dwell = 1 / 3000
f0 = 51.7
t = np.arange(n) * dwell


def peak(ppm, decay_hz=8.0):
    hz = ppm * f0
    return np.exp(-np.pi * decay_hz * t) * np.exp(1j * 2 * np.pi * hz * t)

basis = BasisSet(
    names=["PCr", "Pi", "NAD-ish"],
    fids=np.asarray([peak(0.0), peak(4.8), peak(-8.3, 12.0)]),
    dwell_time_s=dwell,
    transmitter_mhz=f0,
)

fid = 1.0 * basis.fids[0] + 0.22 * basis.fids[1] + 0.07 * basis.fids[2]
fid += 0.004 * (rng.normal(size=n) + 1j * rng.normal(size=n))
data = SpectralData(fid, dwell, f0)

config = FitConfig(
    ppm_range=(-12, 8),
    global_shift_bounds_ppm=(-0.05, 0.05),
    phase0_bounds_deg=(-15, 15),
    phase1_bounds_deg_per_ppm=(-2, 2),
    lorentzian_bounds_hz=(0, 10),
    gaussian_bounds_hz=(0, 10),
    baseline_knots=10,
)
result = fit_spectrum(data, basis, config)
print(result.summary_rows())
result.plot("synthetic_31p_fit.png", title="LCMish synthetic 31P example")
result.save_pdf("synthetic_31p_fit.pdf", title="LCMish synthetic 31P example")
