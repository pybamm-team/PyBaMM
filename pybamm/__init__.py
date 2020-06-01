#
# Root of the pybamm module.
# Provides access to all shared functionality (models, solvers, etc.).
#
# The code in this file is adapted from Pints
# (see https://github.com/pints-team/pints)
#
import sys
import os

#
# Version info
#
def _load_version_int():
    try:
        root = os.path.abspath(os.path.dirname(__file__))
        with open(os.path.join(root, "version"), "r") as f:
            version = f.read().strip().split(",")
        major, minor, revision = [int(x) for x in version]
        return major, minor, revision
    except Exception as e:
        raise RuntimeError("Unable to read version number (" + str(e) + ").")


__version_int__ = _load_version_int()
__version__ = ".".join([str(x) for x in __version_int__])
if sys.version_info[0] < 3:
    del x  # Before Python3, list comprehension iterators leaked

#
# Expose PyBaMM version
#
def version(formatted=False):
    """
    Returns the version number, as a 3-part integer (major, minor, revision).
    If ``formatted=True``, it returns a string formatted version (for example
    "PyBaMM 1.0.0").
    """
    if formatted:
        return "PyBaMM " + __version__
    else:
        return __version_int__


#
# Constants
#
# Float format: a float can be converted to a 17 digit decimal and back without
# loss of information
FLOAT_FORMAT = "{: .17e}"
# Absolute path to the PyBaMM repo
script_path = os.path.abspath(__file__)

from .util import root_dir
ABSOLUTE_PATH = root_dir()
PARAMETER_PATH = [os.getcwd(), os.path.join(root_dir(), "pybamm", "input", "parameters")]

#
# Utility classes and methods
#
from .util import Timer, FuzzyDict
from .util import root_dir, load_function, rmse, get_infinite_nested_dict, load
from .util import get_parameters_filepath
from .logger import logger, set_logging_level
from .settings import settings
from .citations import Citations, citations, print_citations

#
# Classes for the Expression Tree
#
from .expression_tree.symbol import *
from .expression_tree.binary_operators import *
from .expression_tree.concatenations import *
from .expression_tree.array import Array, linspace, meshgrid
from .expression_tree.matrix import Matrix
from .expression_tree.unary_operators import *
from .expression_tree.functions import *
from .expression_tree.interpolant import Interpolant
from .expression_tree.input_parameter import InputParameter
from .expression_tree.parameter import Parameter, FunctionParameter
from .expression_tree.broadcasts import *
from .expression_tree.scalar import Scalar
from .expression_tree.variable import Variable, ExternalVariable, VariableDot
from .expression_tree.variable import VariableBase
from .expression_tree.independent_variable import *
from .expression_tree.independent_variable import t
from .expression_tree.vector import Vector
from .expression_tree.state_vector import StateVectorBase, StateVector, StateVectorDot

from .expression_tree.exceptions import *

# Operations
from .expression_tree.operations.simplify import (
    Simplification,
    simplify_if_constant,
    simplify_addition_subtraction,
    simplify_multiplication_division,
)
from .expression_tree.operations.evaluate import (
    find_symbols,
    id_to_python_variable,
    to_python,
    EvaluatorPython,
)
from .expression_tree.operations.jacobian import Jacobian
from .expression_tree.operations.convert_to_casadi import CasadiConverter
from .expression_tree.operations.unpack_symbols import SymbolUnpacker

#
# Model classes
#
from .models.base_model import BaseModel
from .models import standard_variables
from .models.event import Event
from .models.event import EventType

# Battery models
from .models.full_battery_models.base_battery_model import BaseBatteryModel
from .models.full_battery_models import lead_acid
from .models.full_battery_models import lithium_ion

#
# Submodel classes
#
from .models.submodels.base_submodel import BaseSubModel

from .models.submodels import (
    convection,
    current_collector,
    electrolyte_conductivity,
    electrolyte_diffusion,
    electrode,
    external_circuit,
    interface,
    oxygen_diffusion,
    particle,
    porosity,
    thermal,
    tortuosity,
)
from .models.submodels.interface import sei

#
# Geometry
#
from .geometry.geometry import Geometry
from .geometry.battery_geometry import battery_geometry

from .expression_tree.independent_variable import KNOWN_COORD_SYS
from .geometry import standard_spatial_vars

#
# Parameters class and methods
#
from .parameters.parameter_values import ParameterValues
from .parameters import constants
from .parameters import geometric_parameters
from .parameters import electrical_parameters
from .parameters import thermal_parameters
from .parameters import standard_parameters_lithium_ion, standard_parameters_lead_acid
from .parameters import sei_parameters
from .parameters import parameter_sets


#
# Mesh and Discretisation classes
#
from .discretisations.discretisation import Discretisation
from .discretisations.discretisation import has_bc_of_form
from .meshes.meshes import Mesh, SubMesh, MeshGenerator
from .meshes.zero_dimensional_submesh import SubMesh0D
from .meshes.one_dimensional_submeshes import (
    SubMesh1D,
    Uniform1DSubMesh,
    Exponential1DSubMesh,
    Chebyshev1DSubMesh,
    UserSupplied1DSubMesh,
)
from .meshes.scikit_fem_submeshes import (
    ScikitSubMesh2D,
    ScikitUniform2DSubMesh,
    ScikitExponential2DSubMesh,
    ScikitChebyshev2DSubMesh,
    UserSupplied2DSubMesh,
)

#
# Spatial Methods
#
from .spatial_methods.spatial_method import SpatialMethod
from .spatial_methods.zero_dimensional_method import ZeroDimensionalSpatialMethod
from .spatial_methods.finite_volume import FiniteVolume
from .spatial_methods.scikit_finite_element import ScikitFiniteElement

#
# Solver classes
#
from .solvers.solution import Solution, _BaseSolution
from .solvers.processed_variable import ProcessedVariable
from .solvers.processed_symbolic_variable import ProcessedSymbolicVariable
from .solvers.base_solver import BaseSolver
from .solvers.dummy_solver import DummySolver
from .solvers.algebraic_solver import AlgebraicSolver
from .solvers.casadi_solver import CasadiSolver
from .solvers.casadi_algebraic_solver import CasadiAlgebraicSolver
from .solvers.scikits_dae_solver import ScikitsDaeSolver
from .solvers.scikits_ode_solver import ScikitsOdeSolver, have_scikits_odes
from .solvers.scipy_solver import ScipySolver
from .solvers.idaklu_solver import IDAKLUSolver, have_idaklu

#
# Experiments
#
from .experiments.experiment import Experiment
from . import experiments

#
# Plotting
#
from .plotting.quick_plot import QuickPlot
from .plotting.plot import plot
from .plotting.plot2D import plot2D
from .plotting.dynamic_plot import dynamic_plot

#
# Simulation
#
from .simulation import Simulation, load_sim, is_notebook

#
# Remove any imported modules, so we don't expose them as part of pybamm
#
del sys
