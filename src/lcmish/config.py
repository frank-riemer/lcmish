"""Convenience fitting configurations."""
from __future__ import annotations

from .models import FitConfig, GroupConfig


def p31_brain_config(ppm_range: tuple[float, float] = (-20.0, 10.0)) -> FitConfig:
    """A conservative generic starting point for brain 31P spectra."""
    return FitConfig(
        ppm_range=ppm_range,
        zero_fill_factor=2,
        baseline_knots=16,
        baseline_lambda=2e-2,
        global_shift_bounds_ppm=(-0.20, 0.20),
        phase0_bounds_deg=(-60.0, 60.0),
        phase1_bounds_deg_per_ppm=(-15.0, 15.0),
        lorentzian_bounds_hz=(0.0, 35.0),
        gaussian_bounds_hz=(0.0, 30.0),
        initial_lorentzian_hz=4.0,
        initial_gaussian_hz=3.0,
    )


def p31_brain_grouped_config(ppm_range: tuple[float, float] = (-20.0, 10.0)) -> FitConfig:
    """31P starting configuration with shared metabolite-group nuisance terms.

    Groups are only activated for members that are present in the supplied basis.
    The numerical bounds are deliberately modest and should be reported/validated
    for a particular acquisition rather than treated as universal constants.
    """
    base = p31_brain_config(ppm_range)
    groups = (
        GroupConfig("ATP", ("ATPa", "ATPb", "ATPg"), (-0.12, 0.12), (0.0, 15.0)),
        GroupConfig("NAD", ("NAD", "NADP", "NADH", "NAD+", "NAD_plus"), (-0.12, 0.12), (0.0, 15.0)),
        GroupConfig("PME", ("PE", "PC"), (-0.12, 0.12), (0.0, 12.0)),
        GroupConfig("PDE", ("GPE", "GPC"), (-0.12, 0.12), (0.0, 12.0)),
    )
    return FitConfig(**{**base.__dict__, "groups": groups})
