# LCMish examples

These examples are intentionally small and explicit. They are starting points
for validation, not scanner-independent recipes.

## Siemens 2-D 31P CSI

The two scanner-facing examples restore the reusable parts of the original
private Twix and Siemens DICOM workflows:

- `siemens_twix_csi.py` reads a Twix image stream with `pymapvbvd`, accepts
  only the explicit `(Col, Lin, Ave, Seg)` layout, coherently combines averages,
  and performs the centered 2-D CSI reconstruction.
- `siemens_dicom_csi.py` reads either standard DICOM SpectroscopyData
  `(5600,0020)` or the observed Siemens private payload `(7fe1,1010)`.

Both then use LCMish's current masked 31P-CSI workflow: PCr SNR filtering, PCr
alignment, phase correction, coherent voxel combination, the experimental
local NAD-region fit, and a whole-spectrum fit with the bundled Siemens 3 T
starter basis. Pass `--basis another.BASIS` to use a different basis. They save
the SNR and retained masks, fitted arrays and tables, diagnostic figures, and a
JSON audit summary.

Install the optional reader needed by the input format:

```console
python -m pip install 'lcmish[twix]'   # Twix
python -m pip install 'lcmish[siemens]' # Twix and Siemens DICOM spectroscopy
```

A verified anatomical mask is mandatory. It must be a NumPy Boolean array with
shape `(rows, columns)`. An SNR-only mask is not a substitute: it can include
scalp, muscle, or other unwanted voxels.

```console
python examples/siemens_twix_csi.py \
  --twix scan.dat --mask verified_mask.npy --output twix_results

python examples/siemens_dicom_csi.py \
  --dicom spectroscopy.dcm --rows 8 --columns 8 \
  --dwell 0.0005 --f0 49.892 \
  --mask verified_mask.npy --output dicom_results
```

### Validation requirements

- Twix dimension layouts vary by sequence and software version. Unsupported
  layouts fail rather than being guessed.
- The default second-axis reversal and complex conjugation reflect one observed
  Siemens 31P CSI convention. Verify spatial orientation against a scanner
  reconstruction or phantom, and verify spectral sign using ATP and Pi relative
  to PCr. PCr alone cannot establish the sign.
- DICOM exports and anonymizers may omit acquisition metadata. The DICOM example
  therefore requires rows, columns, dwell time, and transmitter frequency.
- Output amplitudes and the apparent NAD+/NADH ratio are experimental and are
  not absolute concentrations. QC passing is not biological validation.

## Other examples

- `synthetic_31p.py` demonstrates a synthetic whole-spectrum fit.
- `p31_2d_csi_redox.py` demonstrates the masked CSI route without vendor I/O.
The installed package includes
`LCMish_Brain_31P_Haukeland_Siemens3T_1024.BASIS` as an experimental starter
basis. It can also be loaded directly with `lcmish.load_p31_brain_basis()`.
