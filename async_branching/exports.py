"""Load the repository's export validator without a generic 'tools' import.

The training container supplies a regular package named tools; the repository's
tools directory is a namespace package, so 'from tools...' can resolve there.
"""

from functools import lru_cache
import importlib.util
from pathlib import Path


@lru_cache(maxsize=1)
def validation_functions():
    source = Path(__file__).resolve().parents[1] / "tools/archive_completed_mixing_checkpoints.py"
    spec = importlib.util.spec_from_file_location("rlve_async_export_validation", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.tensor_shapes, module.validate_export
