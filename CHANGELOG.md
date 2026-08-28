# Changelog

## 0.3.1 — 2026-08-28

- Fixed a phase-domain defect in the general linear-combination fitter. LCMish
  now fits both real and imaginary spectral channels by default instead of
  discarding the imaginary channel and rotating only a real-projected basis.
- Fit reports and exported component curves now use the consistently
  phase-corrected real channel. The corresponding imaginary data, fit,
  baseline, residual and component channels remain available for QC.
- Added an explicit legacy `fit_domain="real"` option for reproducibility. It
  retains the historical unphased real-only behaviour and should not be used
  as the default for new quantitative work.
- Added regression coverage for a first-order phase ramp exceeding 180 degrees
  across the fit range, including a resonance that was inverted in the acquired
  real channel before phase correction.
- Clarified that NIfTI-MRS preserves complex data and phase-cycle/dynamic
  dimensions but does not itself perform phase correction, coil combination or
  vendor/sequence-specific reconstruction.

## 0.3.0 — 2026-08-25

- Added a bundled, provenance-documented experimental human-brain 31P starter
  basis for a Haukeland Siemens 3 T CSI preset: 1024 acquired complex points,
  0.5 ms dwell time and 49.891996 MHz transmitter frequency.
- The ready-to-use LCModel-style `.BASIS` file and JSON provenance sidecar are
  included in the wheel and source distribution.
- Added PE, PC, extracellular/intracellular Pi, GPE, GPC, PCr, alpha/beta/gamma
  ATP, NAD+ and NADH components. Complex nucleotide-sugar, blood-nuisance and
  membrane-background models remain deliberately excluded pending validation.
- Added an explicitly experimental, literature-constrained local 31P NAD-region
  fitter for NAD+, NADH and neighbouring alpha-ATP.
- Added field-dependent NAD+ AB-quartet generation, phosphorus-count
  normalization, conditional uncertainty, residual bootstrap and an optional
  overlapping nucleotide-sugar sensitivity term.
- Added an auditable convenience workflow for reconstructed complex 2-D CSI:
  callers supply a study-specific voxel mask and explicit QC thresholds;
  LCMish performs PCr-SNR filtering, voxel-wise PCr alignment, phase correction,
  coherent combination and the local redox fit.
- The reported quantity is deliberately named the *apparent* NAD+/NADH ratio,
  and is withheld when the workflow or component-identifiability checks fail.
- This workflow requires adaptation and independent validation for other
  acquisitions, localization schemes, scanners or voxel-selection strategies.
- Added explicit Siemens Twix 2-D CSI reconstruction and Siemens MR
  Spectroscopy DICOM payload readers, with runnable masked-workflow examples.
- Added `pydicom` and combined Siemens optional-dependency groups.

## 0.2.2 — 2026-08-24

- Corrected LCModel `.BASIS` parsing to read exactly `NDATAB` complex values per component.
- Added LCModel-compatible `TRAMP/(VOLUME*CONC)` scaling, `ISHIFT` direction, 4.65-ppm carrier correction and unitary inverse-FFT normalization.
- Added LCModel-compatible basis/data bandwidth conversion, including field-strength compensation, `BWTOLR` handling and the internal `NDATA=2*NUNFIL` model duration.
- Added regression tests for basis scaling, shifting, dwell-time conversion and preservation of narrow-band basis tails.
- Validated the corrected behavior privately against a locally compiled LCModel reference and FSL-MRS on a matched 12-component synthetic 31P case.

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
