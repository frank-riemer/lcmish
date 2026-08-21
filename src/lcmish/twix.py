"""Optional Siemens Twix access via pymapvbvd.

This module intentionally keeps scanner-specific assumptions visible. Raw Twix
layout varies with sequence and software version; callers should inspect shapes
and metadata rather than assuming every `.dat` file is arranged identically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class TwixData:
    data: np.ndarray
    header: Any
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


def read_twix(path: str | Path, *, remove_os: bool = False, squeeze: bool = True) -> TwixData:
    """Read the image MDH stream of a Siemens Twix file with pymapvbvd.

    Returns the raw complex array rather than guessing which dimensions are
    spectroscopy voxels, coils or averages. Sequence-specific reshaping and coil
    combination should be explicit downstream.
    """
    try:
        import mapvbvd
    except ImportError as exc:
        raise ImportError(
            "Twix support is optional. Install LCMish with `pip install .[twix]` "
            "or install pymapvbvd directly."
        ) from exc

    obj = mapvbvd.mapVBVD(str(path))
    if isinstance(obj, list):
        obj = obj[-1]
    if not hasattr(obj, "image"):
        raise ValueError("Twix file has no image MDH stream recognised by pymapvbvd")
    obj.image.flagRemoveOS = bool(remove_os)
    obj.image.squeeze = bool(squeeze)
    arr = np.asarray(obj.image[""])
    return TwixData(
        data=arr,
        header=getattr(obj, "hdr", None),
        source=str(path),
        metadata={"remove_os": bool(remove_os), "squeeze": bool(squeeze), "shape": list(arr.shape)},
    )
