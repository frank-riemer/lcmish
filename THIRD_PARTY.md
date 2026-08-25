# Third-party software, formats and licensing

LCMish is distributed under the BSD 3-Clause License; see `LICENSE`.

## LCModel

LCMish is an independent Python implementation of linear-combination modelling ideas used in magnetic resonance spectroscopy. It is **not LCModel**, does not contain an LCModel executable, and is not presented as an official port or drop-in replacement.

LCModel was developed by Stephen Provencher. The released LCModel source code is available separately under a BSD 3-Clause licence. LCMish does not require that source code to run.

The name "LCModel" is used descriptively to explain interoperability and scientific context. No endorsement by the LCModel author or maintainers is implied.

## LCModel `.RAW` and `.BASIS` files

LCMish can read common LCModel-style `.RAW` and `.BASIS` files. Support for a file format does **not** transfer copyright or redistribution rights for the contents of a particular basis set.

**Basis sets are data/software artefacts with their own provenance.** Do not add a basis set to this repository unless you know that its licence permits redistribution. The LCMish BSD licence covers LCMish code; it does not magically re-licence somebody else's basis spectra. Sadly, licences are less susceptible to linear combination than spectra.

LCMish 0.3 includes one experimental Siemens 3 T brain 31P starter basis. It
was constructed specifically for LCMish from published chemical shifts and
coupling models rather than copied from a third-party basis file. Its adjacent
JSON sidecar records the acquisition assumptions, component list and primary
literature references. The basis is distributed under the repository's BSD
3-Clause licence, but remains an experimental acquisition-specific model rather
than a universal or independently validated brain basis.

## NIfTI-MRS

LCMish implements reading of the **NIfTI-MRS** data format as a vendor-neutral spectroscopy interchange format. The NIfTI-MRS specification and associated project are separate works led by William T. Clarke and collaborators. The specification content is published under **CC BY 4.0** and code in the standard repository under **BSD 3-Clause**. LCMish does not vendor the specification or its example datasets.

If you use NIfTI-MRS in scientific work, cite the format paper:

> Clarke WT, Bell TK, Emir UE, et al. *NIfTI-MRS: A standard data format for magnetic resonance spectroscopy.* Magnetic Resonance in Medicine. 2022;88:2358–2370. doi:10.1002/mrm.29418.

## NiBabel

Optional NIfTI-MRS file access uses **NiBabel**, which is a separate project licensed primarily under the **MIT License** (with some bundled code under BSD terms). NiBabel is installed only when the `nifti` or `all` optional dependency is requested and is not vendored into LCMish.

## pyMapVBVD / pymapvbvd

Optional Siemens Twix access uses `pymapvbvd`, which is a separate project and is not vendored into LCMish. Users who install the optional `twix` dependency are responsible for its separate licence terms.

## pydicom

Optional Siemens MR Spectroscopy DICOM access uses `pydicom`, a separate
MIT-licensed project that is not vendored into LCMish. It is installed through
the `dicom`, `siemens`, or `all` optional dependency.

## Derived-code rule

The current LCMish release is presented as an independent implementation and does not vendor LCModel source. If future contributors adapt or translate code directly from LCModel, the relevant Stephen Provencher copyright and BSD 3-Clause notice must be retained with that derived material. Compatibility with an algorithm or file format is not, by itself, a licence transfer.
