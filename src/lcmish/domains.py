"""Shared spectral-domain projections and coherent magnitude bookkeeping."""
from __future__ import annotations

import numpy as np


def normalise_fit_domain(value: str) -> str:
    domain = str(value).strip().lower()
    domain = {"both": "complex", "imaginary": "imag", "mag": "magnitude"}.get(domain, domain)
    if domain not in {"complex", "real", "imag", "magnitude"}:
        raise ValueError("fit_domain must be 'complex', 'real', 'imag' or 'magnitude'")
    return domain


def project_spectrum(signal: np.ndarray, domain: str) -> np.ndarray:
    signal = np.asarray(signal)
    if domain == "complex":
        return np.r_[signal.real, signal.imag]
    if domain == "imag":
        return signal.imag
    if domain == "magnitude":
        return np.abs(signal)
    return signal.real


def domain_design(metab: np.ndarray, baseline: np.ndarray, domain: str) -> np.ndarray:
    if domain == "complex":
        zeros = np.zeros_like(baseline)
        return np.block([[metab.real, baseline, zeros], [metab.imag, zeros, baseline]])
    if domain == "magnitude":
        # Used only to initialise a nonlinear coherent-magnitude fit.
        return np.column_stack([np.abs(metab), baseline])
    return np.column_stack([metab.imag if domain == "imag" else metab.real, baseline])


def magnitude_model(metab: np.ndarray, amplitudes: np.ndarray, baseline: np.ndarray, coefficients: np.ndarray):
    """Return |sum of complex components| + additive scalar baseline and Jacobian.

    The amplitude Jacobian also gives additive interference-aware component
    allocations: their sum equals the coherent signal magnitude, but individual
    allocations may be negative and are not isolated component magnitudes.
    """
    coherent = metab @ amplitudes
    magnitude = np.abs(coherent)
    unit_conjugate = np.divide(
        np.conj(coherent), magnitude,
        out=np.zeros_like(coherent), where=magnitude > np.finfo(float).tiny,
    )
    derivative = (metab * unit_conjugate[:, None]).real
    baseline_curve = baseline @ coefficients
    prediction = magnitude + baseline_curve
    design = np.column_stack([derivative, baseline])
    components = derivative * amplitudes[None, :]
    return prediction, baseline_curve, design, components
