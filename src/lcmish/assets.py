"""Access to data files distributed with LCMish."""
from __future__ import annotations

from importlib.resources import as_file, files

from .io import read_basis
from .models import BasisSet


P31_BRAIN_BASIS_FILENAME = "LCMish_Brain_31P_Haukeland_Siemens3T_1024.BASIS"


def load_p31_brain_basis(*, reference_ppm: float = 0.0) -> BasisSet:
    """Load the bundled experimental Siemens 3 T brain 31P starter basis."""
    resource = files("lcmish").joinpath("data", P31_BRAIN_BASIS_FILENAME)
    with as_file(resource) as path:
        basis = read_basis(path, reference_ppm=reference_ppm)
    basis.metadata["bundled_with_lcmish"] = True
    basis.metadata["bundled_filename"] = P31_BRAIN_BASIS_FILENAME
    return basis
