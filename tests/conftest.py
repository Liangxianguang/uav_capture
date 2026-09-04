"""Native-runtime settings required before importing PyBullet and PyTorch."""

from __future__ import annotations

import os

import numpy as np


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# TensorBoard 2.4.x still references aliases removed in NumPy 2.x.  Keep the
# compatibility shim at test bootstrap so the recorded runtime remains usable
# without mutating the installed environment.
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
