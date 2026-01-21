# Lazy loading implementation using lazy_loader package
# Essential imports only - everything else is lazily loaded via stub file

import importlib

import lazy_loader as lazy

from pybamm.version import __version__

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

# Lazy loading via stub file - get the base __getattr__ and __dir__
_lazy_getattr, _lazy_dir, _stub_all = lazy.attach_stub(__name__, __file__)

# Submodule aliases that need special handling (module path mappings)
# These are submodules that we want to expose at the top level
_SUBMODULE_ALIASES: dict[str, str] = {
    # Battery models (as modules)
    "lead_acid": ".models.full_battery_models.lead_acid",
    "lithium_ion": ".models.full_battery_models.lithium_ion",
    "equivalent_circuit": ".models.full_battery_models.equivalent_circuit",
    "sodium_ion": ".models.full_battery_models.sodium_ion",
    # Submodels
    "active_material": ".models.submodels.active_material",
    "convection": ".models.submodels.convection",
    "current_collector": ".models.submodels.current_collector",
    "electrolyte_conductivity": ".models.submodels.electrolyte_conductivity",
    "electrolyte_diffusion": ".models.submodels.electrolyte_diffusion",
    "electrode": ".models.submodels.electrode",
    "external_circuit": ".models.submodels.external_circuit",
    "interface": ".models.submodels.interface",
    "oxygen_diffusion": ".models.submodels.oxygen_diffusion",
    "particle": ".models.submodels.particle",
    "porosity": ".models.submodels.porosity",
    "thermal": ".models.submodels.thermal",
    "transport_efficiency": ".models.submodels.transport_efficiency",
    "particle_mechanics": ".models.submodels.particle_mechanics",
    "equivalent_circuit_elements": ".models.submodels.equivalent_circuit_elements",
    "kinetics": ".models.submodels.interface.kinetics",
    "sei": ".models.submodels.interface.sei",
    "lithium_plating": ".models.submodels.interface.lithium_plating",
    "interface_utilisation": ".models.submodels.interface.interface_utilisation",
    "open_circuit_potential": ".models.submodels.interface.open_circuit_potential",
    # Geometry
    "standard_spatial_vars": ".geometry.standard_spatial_vars",
    # Parameters
    "constants": ".parameters.constants",
    # Experiments
    "experiment": ".experiment",
    "step": ".experiment.step",
    # Callbacks and telemetry
    "callbacks": ".callbacks",
    "telemetry": ".telemetry",
}

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
