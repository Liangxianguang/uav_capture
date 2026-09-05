"""Run a repository script with compatibility shims for the frozen environment.

The pinned TensorBoard 2.4.1 release predates NumPy 2 and protobuf 4+.  This
wrapper keeps experiment code unchanged while making the compatibility choice
explicit in every command that writes or reads TensorBoard events.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_with_tensorboard_compat.py SCRIPT [ARGS ...]")
    project_root = Path(__file__).resolve().parents[1]
    script = (project_root / sys.argv[1]).resolve()
    if project_root not in script.parents:
        raise ValueError(f"Script must be inside project root: {script}")
    if not script.is_file():
        raise FileNotFoundError(script)

    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    for import_root in (project_root / "src", project_root / "scripts"):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))
    import numpy as np

    # TensorBoard 2.4.1 imports aliases removed by NumPy 2.x.  Define only
    # missing names and keep the compatibility decision local to this runner.
    aliases = {
        "bool8": np.bool_,
        "bool": bool,
        "object": object,
        "int": int,
        "float": float,
        "complex": complex,
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
        "uint8": np.uint8,
        "uint16": np.uint16,
        "uint32": np.uint32,
        "uint64": np.uint64,
        "float16": np.float16,
        "float32": np.float32,
        "float64": np.float64,
        "complex64": np.complex64,
        "complex128": np.complex128,
        "string_": np.bytes_,
        "unicode_": np.str_,
        "str": str,
        "unicode": str,
        "bytes": bytes,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)

    sys.argv = [str(script), *sys.argv[2:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
