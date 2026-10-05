# Changelog

## 0.4.0 — 2026-10-05

- Unified the general and redox domain selectors: real, imaginary, complex
  (both channels) and coherent magnitude, with aliases. Existing general
  complex and redox real defaults are retained. Complex redox results retain
  both channels and bootstrap paired residuals.
- Added nonlinear coherent-magnitude fitting, with an additive scalar baseline,
  amplitude-Jacobian uncertainty and interference-aware component allocations.
  Common phase is not estimated from magnitude and magnitude-noise bias is not
  corrected. Omitted coherent tails can bias local magnitude redox fits.
- Exposed fitting-domain choice in the CLI and prepared proton example.
- Consolidated the README introduction, put redox first, removed segmented bold
  emphasis and moved detailed workflows to the examples guide. Redox phase
  estimation and linked sugars are documented as part of the same codebase.

- Refocused the README on lightweight native multinuclear frequency-domain
  fitting without an external fitting binary or proprietary runtime. Corrected
  installation and general complex-fitting claims, and documented preprocessing,
  spectral references, water handling, configuration and MRSI array safeguards.
- Added direct MATLAB complex-vector and prepared NIfTI-MRS proton routes, a
  runnable prepared 1H example and regression tests for its input contract.
  These are API examples, not validation of proton acquisitions or absolute
  concentration calibration.
- Corrected the synthetic 31P example's oscillator sign to match the documented
  `ppm = reference_ppm - frequency_hz / transmitter_mhz` convention.
- Documented the idealized proton-decoupled redox basis, the absence of
  WALTZ-4 pulse-train/partial-acquisition simulation and metabolite-specific
  NOE or saturation correction, and the equal-response assumption linking
  the sugar partners. Added machine-readable acquisition-model assumptions
  to fit metadata without changing numerical fitting or defaults.
- Added an explicit bounded residual-phase example and distinguished hard
  parameter bounds from probabilistic priors, conditional fit precision from
  model validity, and single-spectrum ratio checks from CSI workflow QC.
- Added experimental non-NAD phase estimation from Pi, PCr, gamma-ATP and
  beta-ATP, with explicit SNR and phase-residual QC.
- Added native-grid phase correction and an anchor-informed NAD-region helper
  that records the complete phase audit alongside the fit.
- Added an optional alpha-ATP-only phase nuisance so that local alpha-ATP model
  mismatch need not be absorbed by a large, unconstrained NAD-window phase
  ramp. The existing `fit_p31_redox()` behavior remains unchanged by default.
- Added synthetic regression tests for global phase recovery, NAD-window
  independence, apparent-redox recovery and alpha-ATP phase separation.
- Added an opt-in pooled nucleotide-sugar sensitivity model for
  proton-decoupled data. Equal-area phosphorus-phosphorus doublets near -9.8
  and -8.2 ppm share one non-negative amplitude, so the separated partner can
  constrain the contribution overlapping NAD.
- Added a separate linked-sugar linewidth nuisance, explicit validation that
  the fit window includes both partners, and synthetic recovery tests. Legacy
  redox defaults and the original unlinked nuisance model remain unchanged.

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
- Removed study-specific paths, randomisation data, voxel choices and batch logic from the public core.
- Added explicit licensing, third-party provenance and AI-assisted development disclosures.
