"""Small compatibility helpers kept in one place rather than sprinkled everywhere."""
from __future__ import annotations

import numpy as np


def trapezoid(y, x=None, axis: int = -1):
    """Integrate with NumPy across old and new versions.

    NumPy 2.4 removed ``np.trapz``. Newer NumPy provides ``np.trapezoid``;
    older supported versions provide ``np.trapz``. Keeping this compatibility
    detail here avoids monkey-patching NumPy in downstream analysis scripts.
    """
    fn = getattr(np, "trapezoid", None)
    if fn is not None:
        return fn(y, x=x, axis=axis)
    return np.trapz(y, x=x, axis=axis)
