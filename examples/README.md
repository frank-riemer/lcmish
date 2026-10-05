# LCMish examples and workflows

These examples make data preparation and model choices explicit. They are starting points for validation, not scanner-independent recipes. The general fitter is multinuclear; the specialised NAD/redox model is for ³¹P.

## NIfTI-MRS and RAW

Read a prepared single-FID NIfTI-MRS file:

```python
import lcmish

data = lcmish.read("prepared.nii.gz")
print(data.dwell_time_s, data.transmitter_mhz, data.reference_ppm)
print(data.metadata.get("warnings", []))
if data.metadata["reference_ppm_source"].startswith("default"):
    raise ValueError("Supply a verified spectral-centre reference before fitting")
```

The reader uses `SpecFreqChemShift` when available; otherwise it defaults to 0 ppm and stores a metadata warning. Pass `reference_ppm=<verified centre>` if needed. Water-centred proton data commonly need approximately 4.7 ppm, not 0 ppm; verify the actual convention rather than assuming it.

Select one MRSI voxel and every remaining non-time dimension explicitly:

```python
data = lcmish.read("mrsi.nii.gz", index=(0, 0, 0), reference_ppm=4.7)
```

Add dimension 5–7 indices when those dimensions exist. Python indices are zero-based. Loading does not combine coils, average dynamics, resolve phase cycling or correct vendor reconstruction. Reduce those dimensions before fitting where appropriate, but preserve the spatial voxels and geometry when building MRSI maps.

For LCModel-style RAW, provide acquisition metadata explicitly. This is a PCr-centred ³¹P illustration, not proton metadata:

```python
data = lcmish.read(
    "prepared.RAW", dwell_time_s=1/3000,
    transmitter_mhz=51.7, reference_ppm=0.0,
)
```

NIfTI-MRS is useful for interchange and metadata, but is not mandatory. GE P-files/ScanArchive need a validated external decoder/reconstruction before any of these routes.

## Complex vectors from MATLAB

To save just one already reconstructed and preprocessed complex time-domain vector:

```matlab
save('prepared_fid.mat', 'fid', '-v7');
```

Load it and provide its metadata separately:

```python
import numpy as np
from scipy.io import loadmat
from lcmish import SpectralData

fid = np.asarray(loadmat("prepared_fid.mat")["fid"]).squeeze()
if fid.ndim != 1 or not np.iscomplexobj(fid) or not np.all(np.isfinite(fid)):
    raise ValueError("Expected one finite complex FID vector, not a voxel grid")

# Illustrative proton values: replace with the actual acquisition metadata.
dwell_time_s = 0.0005
transmitter_mhz = 127.7
reference_ppm = 4.7  # Only for a verified water-centred FID in this convention.
data = SpectralData(fid, dwell_time_s, transmitter_mhz, reference_ppm)
```

No RAW conversion or NIfTI step is needed. Save the FID, not `real(fid)`, `abs(fid)` or an FFT display spectrum. `lcmish.read()` does not auto-read MAT files. The shape check accepts MATLAB row/column vectors but prevents the core constructor from silently flattening a multidimensional grid.

Prefer saving metadata with the vector:

```matlab
% Set these scalars from the acquisition and verified processing.
save('prepared_fid.mat', 'fid', 'dwell_time_s', ...
     'transmitter_mhz', 'reference_ppm', '-v7');
```

```python
mat = loadmat("prepared_fid.mat")
fid = np.asarray(mat["fid"]).squeeze()
if fid.ndim != 1 or not np.iscomplexobj(fid) or not np.all(np.isfinite(fid)):
    raise ValueError("Expected one finite complex FID vector")
data = SpectralData(
    fid,
    dwell_time_s=float(mat["dwell_time_s"].item()),
    transmitter_mhz=float(mat["transmitter_mhz"].item()),
    reference_ppm=float(mat["reference_ppm"].item()),
)
```

SciPy reads these `-v7` files; `-v7.3` is HDF5 and needs another reader or resaving as `-v7`. The runnable [prepared proton example](fit_prepared_1h.py) validates the vector and metadata.

A vector alone cannot supply dwell time (seconds), nucleus-specific transmitter frequency (MHz), or the chemical shift at zero Hz:

```text
ppm = reference_ppm - frequency_hz / transmitter_mhz
```

`reference_ppm` labels the centre; it does not shift or phase the FID. A genuinely water-centred ¹H signal might use 4.7 ppm, one shifted so creatine is at zero Hz might use 3.03 ppm, and a PCr-centred ³¹P signal uses 0 ppm. Check several known peaks, data/basis agreement, frequency sign and acquisition delay; do not blindly conjugate data or correct a delay twice.

## Fitting modes

Both fitters accept the same `fit_domain` selector:

| Value | Fitted data |
|---|---|
| `complex` (alias `both`) | Real and imaginary channels jointly |
| `real` | Real channel only |
| `imag` (alias `imaginary`) | Imaginary channel only |
| `magnitude` (alias `mag`) | Absolute value of the complex spectrum |

```python
from lcmish import FitConfig, P31RedoxConfig

general_config = FitConfig(ppm_range=(-20, 10), fit_domain="complex")
redox_config = P31RedoxConfig(fit_domain="complex")
# Use "real", "imag" or "magnitude" in either configuration as appropriate.
```

General fitting defaults to `complex`; redox defaults to `real` to preserve its existing behaviour. In complex general fits, phase parameters rotate the basis into the acquired-data frame; the inverse rotation is used consistently for display. Imaginary curves remain available for QC. Single-channel results display the fitted channel without inventing an unused channel; complex redox results retain both acquired-frame channels.

Magnitude fitting is nonlinear in amplitudes:

```text
magnitude prediction = abs(sum of complex component signals) + scalar baseline
```

Taking the magnitude of each basis component and adding those magnitudes would lose interference between overlapping signals. That approximation is used only to initialise the nonlinear fit. Common zero/first-order phase cancels in magnitude and is not estimated; a relative alpha-ATP phase term can still affect interference if enabled. The general magnitude fit requires nonnegative amplitudes.

Magnitude component curves are interference-aware allocations whose sum equals the coherent signal magnitude. They are not isolated absolute component spectra and can be negative where interference is destructive. Magnitude noise has a positive floor, especially at low SNR; no Rician/noncentral-noise bias correction is applied. It is an optional model/sensitivity route, not a remedy for poor reconstruction or unresolved NADH.

Magnitude local fits are also sensitive to coherent tails from peaks outside the fitted window: their interference with NAD cannot necessarily be represented by an additive scalar baseline. Synthetic recovery with an exact local model does not guarantee recovery in a whole-spectrum mixture with omitted signals. Validate basis completeness and compare modes before interpreting a magnitude redox ratio.

## Whole-spectrum fitting

Use `data` from either input route and a matched basis:

```python
from lcmish import FitConfig, read_basis, fit_spectrum

basis = read_basis(
    "matched.BASIS", transmitter_mhz=data.transmitter_mhz,
    reference_ppm=data.reference_ppm,
)
config = FitConfig(ppm_range=(-20, 10), fit_domain="complex")
result = fit_spectrum(data, basis, config)
print(result.success, result.message)
result.save_csv("fit.csv")
result.save_components_csv("components.csv")
result.save_pdf("fit.pdf")
result.plot("fit.png")
```

The polynomial redox baseline is separate from this regularised B-spline model. Baseline flexibility should be checked for absorption of metabolite signal and structured residuals.

| Generic setting | Default | Meaning |
|---|---|---|
| `ppm_range` | Required | Region included in fitting |
| `fit_domain` | `complex` | Which spectral channel(s) to fit |
| `zero_fill_factor` | 2 | FFT interpolation, not new acquired information |
| `nonnegative_amplitudes` | True | Constrain component weights |
| `baseline_knots` | 14 | Spline basis size, not ppm knot spacing |
| `baseline_lambda` | 0.01 | Larger values discourage baseline curvature |
| `global_shift_bounds_ppm` | ±0.25 | Common residual shift |
| `phase0_bounds_deg` | ±90 | Zero-order phase bounds |
| `phase1_bounds_deg_per_ppm` | ±30 | Phase-ramp bounds |
| Lorentzian/Gaussian bounds | 0–40 Hz | Additional broadening |
| `max_nfev` | 300 | Optimiser limit, not a quality criterion |

³¹P convenience configurations have different starting settings:

```python
from lcmish import p31_brain_grouped_config, fit_spectrum_multistart

config = p31_brain_grouped_config((-20, 10))
for group in config.groups:
    print(group.name, [name for name in group.members if name in basis.names])
audit = fit_spectrum_multistart(
    data, basis, config,
    starts=({}, {"initial_phase0_deg": 8.0}, {"initial_phase0_deg": -8.0}),
)
result = audit.best
```

Group membership depends on exact basis names. Multistart selects the smallest cost, not necessarily a scientifically adequate model. Inspect convergence, bounds and both residual channels where available.

Record settings in analysis output:

```python
from dataclasses import asdict
import json

print(json.dumps(asdict(config), indent=2))
```

The CLI uses ³¹P starting settings; choose the Python API for custom multinuclear configurations:

```console
lcmish prepared.nii.gz matched.BASIS --ppm-min -20 --ppm-max 10 --ref-ppm 0 --fit-domain complex --out fit
```

## 1H and voxelwise MRSI

Use a sequence/field/timing-matched proton basis with appropriate macromolecule/lipid components. The bundled phosphorus basis is not suitable. This is an API route, not systematic validation of high-resolution proton MRSI.

```python
from lcmish import read, read_basis, FitConfig, fit_spectrum

# For MATLAB, keep data loaded above instead of this read() line.
data = read("prepared_1h.nii.gz", reference_ppm=4.7)  # verified water-centred
basis = read_basis(
    "matched_1h.BASIS", transmitter_mhz=data.transmitter_mhz,
    reference_ppm=data.reference_ppm,
)
config = FitConfig(ppm_range=(0.5, 4.2), fit_domain="complex", max_nfev=500)
result = fit_spectrum(data, basis, config)
result.save_csv("proton.csv")
result.save_pdf("proton.pdf")
```

Run [fit_prepared_1h.py](fit_prepared_1h.py) with either input:

```console
python examples/fit_prepared_1h.py prepared_fid.mat matched_1h.BASIS --out fits/voxel
python examples/fit_prepared_1h.py prepared_1h.nii.gz matched_1h.BASIS --ref-ppm 4.7 --fit-domain complex --out fits/voxel
```

The script accepts NIfTI `--index x y z ...`, writes amplitude/component CSVs, a PDF and a settings audit, and rejects implicitly flattened MATLAB grids.

For a declared MATLAB array `(time, x, y)`, select a voxel first:

```python
mat = loadmat("prepared_mrsi.mat")
fids = np.asarray(mat["fids"])
if fids.ndim != 3 or not np.iscomplexobj(fids):
    raise ValueError("Expected complex fids with axes (time, x, y)")
data = SpectralData(
    fids[:, 0, 0],  # Python (0, 0) = MATLAB (:, 1, 1)
    float(mat["dwell_time_s"].item()),
    float(mat["transmitter_mhz"].item()),
    float(mat["reference_ppm"].item()),
)
```

Repeat the matched-basis fit per selected voxel, retaining geometry, indices and QC masks. `CSIData` instead uses `(row, column, time)`; moving the time axis is not an anatomical reorientation. The ³¹P redox masked-composite helper is not a generic voxelwise proton pipeline.

### Water-reference preparation

Reconstruct metabolite and water with consistent voxel/coil correspondence. Water-derived coil weights can be applied to both; a validated water-derived phase correction can address eddy currents before fitting. Do not divide metabolite samples by the complete decaying water FID as an ECC recipe.

Water-referenced concentration scaling is a separate calibration after fitting. Match or correct gain, averaging and normalization, and account for relaxation, tissue water and acquisition response as appropriate. Without calibration, report defined signal amplitudes or ratios rather than absolute concentrations.

## Redox fitting and phase estimation

Use prepared, PCr-referenced ³¹P data:

```python
from lcmish import P31RedoxConfig, fit_p31_redox

config = P31RedoxConfig(fit_domain="real")
redox = fit_p31_redox(data, config)
print(redox.success, redox.apparent_redox_ratio)
```

An apparent ratio is withheld when a NAD component is at its amplitude boundary. The single-spectrum property does not otherwise automatically gate on convergence/percentage error; inspect residuals, uncertainty and stability. CSI additionally applies configured workflow QC.

Non-NAD anchors can constrain acquisition phase before the local NAD fit:

```python
from lcmish import fit_p31_redox_anchor_informed, P31RedoxConfig

anchored = fit_p31_redox_anchor_informed(
    data, P31RedoxConfig(fit_domain="complex"),
)
print(anchored.phase.phase0_deg, anchored.phase.phase1_deg_per_ppm)
print(anchored.phase.phase_residual_rms_deg)
print(anchored.apparent_redox_ratio)
```

The anchors are Pi, PCr, gamma-ATP and beta-ATP, excluding NAD and alpha-ATP. The helper returns the phase audit and corrected signal, uses limited residual NAD phase, and optionally fits an alpha-ATP-only phase offset. Delay correction remains a preparation step. Magnitude can be selected through the same config, but does not estimate common residual phase.

A linked sugar model uses the less-overlapped partner to constrain signal under NAD:

```python
from lcmish import P31RedoxConfig, fit_p31_redox_anchor_informed

linked = P31RedoxConfig(
    ppm_range=(-10.4, -6.5), include_linked_nucleotide_sugars=True,
    fit_domain="complex", baseline_order=2,
)
anchored = fit_p31_redox_anchor_informed(data, linked)
```

One pooled component links equal-area phosphorus-phosphorus doublets near −9.8 and −8.2 ppm (J = 20.5 Hz), with one amplitude and a shared extra linewidth. It is not separation of individual UDP sugars. The linked and local unlinked sugar options are mutually exclusive. Both partner windows must be included.

For an explicit residual-phase sensitivity test:

```python
from lcmish import P31AnchorPhaseConfig

phase_config = P31AnchorPhaseConfig(
    residual_phase0_bounds_deg=(-30, 30),
    residual_phase1_bounds_deg_per_ppm=(-40, 40),
)
anchored = fit_p31_redox_anchor_informed(data, linked, phase_config)
```

These are hard bounds, not probabilistic priors. Wider limits are not automatically better. Check boundary occupancy, sugar linewidth, residual structure and NADH stability; do not choose settings for a preferred ratio. Conditional uncertainty and residual bootstrap do not test acquisition-model correctness or participant-level inference.

## Masked CSI and Siemens readers

`fit_p31_csi_redox()` expects reconstructed complex `(row, column, time)` data, an explicit anatomical mask and study-specific QC thresholds. It filters PCr SNR, aligns/phases voxels, combines them coherently and fits the selected local redox model.

```python
from lcmish import CSIData, P31CSIRedoxQCConfig, P31RedoxConfig, fit_p31_csi_redox

csi = CSIData(fids, dwell_time_s, transmitter_mhz)
qc = P31CSIRedoxQCConfig(
    pcr_snr_min=10, min_retained_voxels=3,
    local_fit_correlation_min=0.85, local_relative_residual_max=0.55,
)
result = fit_p31_csi_redox(
    csi, study_specific_mask, qc, P31RedoxConfig(fit_domain="complex"),
)
print(result.qc_pass, result.qc_reasons, result.apparent_redox_ratio)
```

The automatic sugar sensitivity compares the unlinked model; set `run_nucleotide_sugar_sensitivity=False` with a linked-sugar config. The wrapper does not automatically run anchor estimation: use `prepare_p31_csi_redox()`, then fit its `.combined` signal with the anchor-informed helper if desired. QC thresholds should be checked for the chosen fitting domain rather than assumed interchangeable.

The scanner examples use dedicated, constrained reader routes:

- [siemens_twix_csi.py](siemens_twix_csi.py): explicit `(Col, Lin, Ave, Seg)` layout and centred 2-D CSI reconstruction.
- [siemens_dicom_csi.py](siemens_dicom_csi.py): standard SpectroscopyData or an observed Siemens private payload.
- `read_twix()` is a separate generic raw-array accessor, not automatic reconstruction.

```console
python -m pip install -e '.[siemens]'
python examples/siemens_twix_csi.py --twix scan.dat --mask verified_mask.npy --output twix_results
python examples/siemens_dicom_csi.py --dicom spectroscopy.dcm --rows 8 --columns 8 --dwell 0.0005 --f0 49.892 --mask verified_mask.npy --output dicom_results
```

The mask is a Boolean NumPy array with `(rows, columns)`. An SNR-only mask may include scalp/muscle and is not an anatomical selection. Verify spatial orientation against scanner reconstruction/phantom and spectral sign using ATP/Pi relative to PCr; PCr alone cannot establish sign. Supported Siemens conventions include a specific axis reversal/conjugation, not a universal vendor rule.

[synthetic_31p.py](synthetic_31p.py) and [p31_2d_csi_redox.py](p31_2d_csi_redox.py) run without vendor data. Participant/composite statistics and treatment comparisons remain study-analysis tasks.

## Basis and acquisition assumptions

The LCModel-style reader handles scaling, shifts, carrier/grid conventions and bandwidth matching. That is file compatibility, not physical sequence adaptation or LCModel-equivalent quantification.

The bundled `LCMish_Brain_31P_Haukeland_Siemens3T_1024.BASIS` describes an experimental proton-decoupled Siemens 3 T acquisition: 1024 acquired points, 0.5-ms dwell, 49.891996-MHz phosphorus frequency, PCr = 0 ppm. Load it with `load_p31_brain_basis()`. Its internal model/export point count differs from the acquired count; the JSON sidecar records assumptions, components and references. Basis version 0.1.0 is independent of the package version.

The redox model retains phosphorus-phosphorus coupling but omits explicit proton coupling: NAD+ is a field-dependent AB quartet, NADH a singlet, and alpha-ATP a doublet. The NAD geometry follows [Lu et al.](https://doi.org/10.1002/mrm.24859). It is not a WALTZ pulse-train simulation and has no decoupling on/off transition across the acquired FID.

No metabolite-specific NOE, saturation, excitation-profile or receive-response correction is applied. NAD+ and NADH weights each represent two phosphorus nuclei; relating amplitudes to molecule amounts requires equal effective response or independent calibration. [Peeters et al.](https://doi.org/10.1002/nbm.4169) report metabolite-dependent enhancement, but those factors are not imported here.

The linked sugar model assumes comparable response of its two sites. [Ren et al.](https://doi.org/10.1002/nbm.4511) support using the separate sugar partner to inform NAD overlap, not validation of this pooled 3-T approximation. For data without decoupling, use an independently validated coupled basis rather than simply widening linewidths. A good fit or low conditional percentage error does not establish NADH identifiability.
