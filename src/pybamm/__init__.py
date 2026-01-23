# Lazy loading implementation using lazy_loader package
# Essential imports only - everything else is lazily loaded via stub file
#
# The stub file (__init__.pyi) is auto-generated. To regenerate:
#   python scripts/generate_pyi_stub.py
#
# To validate all imports are correct:
#   python scripts/generate_pyi_stub.py --validate
#
# For CI checks (exits non-zero if stub is outdated):
#   python scripts/generate_pyi_stub.py --check

import importlib

import lazy_loader as lazy

from pybamm.version import __version__

# Configure JAX for float64 precision early, before any JAX-dependent code runs
# This must happen before lazy-loaded modules that use JAX are imported
if (
    importlib.util.find_spec("jax") is not None
    and importlib.util.find_spec("jaxlib") is not None
):
    import jax

    platform = jax.lib.xla_bridge.get_backend().platform.casefold()
    if platform != "metal":
        jax.config.update("jax_enable_x64", True)

# Core utilities that are lightweight and commonly needed
from .logger import logger, set_logging_level, get_new_logger
from .settings import settings
from . import config

# These need to be imported eagerly to shadow the submodule names
from .citations import Citations, citations, print_citations

# Expression tree modules are accessed thousands of times during model building
# so we eagerly import them to avoid __getattr__ overhead
from .expression_tree.symbol import *
from .expression_tree.binary_operators import *
from .expression_tree.concatenations import *
from .expression_tree.unary_operators import *
from .expression_tree.averages import *
from .expression_tree.averages import _BaseAverage
from .expression_tree.broadcasts import *
from .expression_tree.functions import *
from .expression_tree.interpolant import Interpolant  # needed before discrete_time_sum
from .expression_tree.discrete_time_sum import *
from .expression_tree.variable import *
from .expression_tree.coupled_variable import *
from .expression_tree.independent_variable import *
from .expression_tree.exceptions import *
from .expression_tree.scalar import *
from .expression_tree.state_vector import *
from .expression_tree.tensor_field import *
from .expression_tree.parameter import *
from .expression_tree.input_parameter import *
from .expression_tree.array import *
from .expression_tree.vector_field import *
from .expression_tree.matrix import *
from .expression_tree.vector import *

# Lazy loading via stub file - get the base __getattr__ and __dir__
_lazy_getattr, _lazy_dir, _stub_all = lazy.attach_stub(__name__, __file__)

# These are submodules that we want to expose at the top level (e.g., pybamm.lithium_ion)
from ._lazy_config import SUBMODULE_ALIASES as _SUBMODULE_ALIASES

# Cache for loaded submodule aliases
_loaded_submodules: dict[str, object] = {}


def __getattr__(name: str) -> object:
    """Custom __getattr__ that handles submodule aliases and falls back to lazy_loader."""
    # Fast path: check submodule cache
    if name in _loaded_submodules:
        return _loaded_submodules[name]

    # Check if it's a submodule alias
    if name in _SUBMODULE_ALIASES:
        module_path = _SUBMODULE_ALIASES[name]
        module = importlib.import_module(module_path, package="pybamm")
        _loaded_submodules[name] = module
        return module

    # Fall back to lazy_loader's __getattr__
    return _lazy_getattr(name)


def __dir__() -> list[str]:
    """Include submodule aliases in dir() output."""
    return sorted(set(_lazy_dir()) | set(_SUBMODULE_ALIASES.keys()))


# Extend __all__ to include submodule aliases
__all__ = _stub_all + list(_SUBMODULE_ALIASES.keys())


# Fix Casadi import - this needs to happen at import time
import os
import pathlib
import sysconfig

os.environ["CASADIPATH"] = str(pathlib.Path(sysconfig.get_path("purelib")) / "casadi")

config.generate()

# Eagerly load core simulation modules to optimize first-solve performance
from . import simulation  # noqa: F401, E402
from . import solvers  # noqa: F401, E402
from .parameters import parameter_values  # noqa: F401, E402
from . import meshes  # noqa: F401, E402
from . import spatial_methods  # noqa: F401, E402
from .expression_tree.operations import convert_to_casadi  # noqa: F401, E402
