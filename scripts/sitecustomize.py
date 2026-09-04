"""Runtime compatibility for the pinned legacy TensorBoard build.

The reproducibility environment currently has TensorBoard 2.4.x alongside
NumPy 2.x.  TensorBoard imports a small set of NumPy 1.x aliases before any
experiment code runs, so define those aliases centrally for every repository
script without modifying the conda installation.
"""

from __future__ import annotations

import numpy as np


_NUMPY_COMPAT_ALIASES = {
    "bool8": np.bool_,
    "object": object,
    "float": float,
    "int": int,
    "complex": complex,
    "string_": np.bytes_,
    "unicode_": np.str_,
}
for _name, _value in _NUMPY_COMPAT_ALIASES.items():
    if _name not in np.__dict__:
        setattr(np, _name, _value)
