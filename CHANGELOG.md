# Changelog

## 0.2.1 — 2026-08-20

First public-facing LCMish release, retaining the internal PyLCModel version lineage.

- Renamed the project and Python namespace from PyLCModel to **LCMish**.
- Kept the internal version number **0.2.1** rather than restarting at 0.1.
- Added grouped metabolite shift/linewidth support and multistart fitting API.
- Added LCModel-style `.RAW` and `.BASIS` readers.
- Added NIfTI-MRS `.nii` / `.nii.gz` input as the preferred vendor-neutral path, with metadata-aware dwell time/frequency/reference handling and explicit indexing for MRSI or higher-dimensional data.
- Added the `lcmish.read(...)` / `read_spectrum(...)` auto-reader for NIfTI-MRS and LCModel-style RAW files.
- Added conditional amplitude uncertainty estimates, explicitly labelled as CRLB-like rather than LCModel `%SD`.
- Added optional Siemens Twix access through `pymapvbvd`.
- Added table, CSV, checkpoint and figure outputs.
- Added an LCModel-style (but clearly LCMish-labelled) one-page PDF summary with spectrum, fit, residuals, parameters and metabolite table; PDF replaces any need for PostScript as the default human-readable report.
- Fixed NumPy compatibility by using `numpy.trapezoid` when available, with a fallback for older NumPy versions. No monkey-patching of NumPy is required in user scripts.
- Removed study-specific paths, randomisation data, voxel choices and NOPARK-specific batch logic from the public core.
- Added explicit licensing, third-party provenance and AI-assisted development disclosures.
