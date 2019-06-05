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
# Version info: Remember to keep this in sync with setup.py!
#
VERSION_INT = 0, 0, 0
VERSION = ".".join([str(x) for x in VERSION_INT])
if sys.version_info[0] < 3:
    del (x)  # Before Python3, list comprehension iterators leaked

#
# Expose pybamm version
#


def version(formatted=False):
    if formatted:
        return "PyBaMM " + VERSION
    else:
        return VERSION_INT


#
# Constants
#
# Float format: a float can be converted to a 17 digit decimal and back without
# loss of information
FLOAT_FORMAT = "{: .17e}"
# Absolute path to the PyBaMM repo
script_path = os.path.abspath(__file__)
ABSOLUTE_PATH = os.path.join(os.path.split(script_path)[0], "..")

#
# Utility classes and methods
#
from .util import Timer
from .util import load_function
from .logger import logger, set_logging_level
from .settings import settings

#
# Classes for the Expression Tree
#
from .expression_tree.symbol import Symbol, evaluate_for_shape_using_domain
from .expression_tree.binary_operators import (
    is_scalar_zero,
    is_matrix_zero,
    BinaryOperator,
    Addition,
    Power,
    Subtraction,
    Multiplication,
    MatrixMultiplication,
    Division,
    Outer,
    outer,
)
from .expression_tree.concatenations import (
    Concatenation,
    NumpyConcatenation,
    DomainConcatenation,
    SparseStack,
)
from .expression_tree.array import Array
from .expression_tree.matrix import Matrix
from .expression_tree.unary_operators import (
    UnaryOperator,
    Negate,
    AbsoluteValue,
    Index,
    SpatialOperator,
    Gradient,
    Divergence,
    BoundaryOperator,
    BoundaryValue,
    BoundaryFlux,
    Integral,
    IndefiniteIntegral,
    grad,
    div,
    surf,
    average,
    boundary_value,
)
from .expression_tree.function import Function, Exponential, exp
from .expression_tree.parameter import Parameter, FunctionParameter
from .expression_tree.broadcasts import Broadcast
from .expression_tree.scalar import Scalar
from .expression_tree.variable import Variable
from .expression_tree.independent_variable import (
    IndependentVariable,
    Time,
    SpatialVariable,
)
from .expression_tree.independent_variable import t
from .expression_tree.vector import Vector, StateVector

from .expression_tree.exceptions import (
    DomainError,
    ModelError,
    SolverError,
    ShapeError,
    ModelWarning,
    UndefinedOperationError,
    GeometryError,
)
from .expression_tree.simplify import (
    Simplification,
    simplify_if_constant,
    simplify_addition_subtraction,
    simplify_multiplication_division,
)
from .expression_tree.evaluate import (
    find_symbols,
    id_to_python_variable,
    to_python,
    EvaluatorPython,
)

#
# Model classes
#
from .models import standard_variables
from .models.base_models import (
    BaseModel,
    StandardBatteryBaseModel,
    SubModel,
    LeadAcidBaseModel,
    LithiumIonBaseModel,
)
from .models.reaction_diffusion import ReactionDiffusionModel
from .models.simple_ode_model import SimpleODEModel
from .models import lead_acid
from .models import lithium_ion

#
# Submodel classes
#
from .models.submodels import (
    electrode,
    electrolyte_current,
    electrolyte_diffusion,
    interface,
    particle,
    porosity,
    potential,
    velocity,
    vertical,
)

#
# Parameters class and methods
#
from .parameters.parameter_values import ParameterValues
from .parameters import standard_current_functions
from .parameters import geometric_parameters
from .parameters import electrical_parameters
from .parameters import standard_parameters_lithium_ion, standard_parameters_lead_acid
from .parameters.print_parameters import print_parameters, print_evaluated_parameters

#
# Geometry
#
from .geometry.geometry import (
    Geometry,
    Geometry1DMacro,
    Geometry3DMacro,
    Geometry1p1DMacro,
    Geometry1DMicro,
    Geometry1p1DMicro,
)

from .expression_tree.independent_variable import KNOWN_SPATIAL_VARS, KNOWN_COORD_SYS
from .geometry import standard_spatial_vars

#
# Mesh and Discretisation classes
#
from .discretisations.discretisation import Discretisation
from .meshes.meshes import Mesh
from .meshes.submeshes import SubMesh1D, Uniform1DSubMesh

#
# Spatial Methods
#
from .spatial_methods.spatial_method import SpatialMethod
from .spatial_methods.finite_volume import FiniteVolume

#
# Solver classes
#
from .solvers.solution import Solution
from .solvers.base_solver import BaseSolver
from .solvers.ode_solver import OdeSolver
from .solvers.dae_solver import DaeSolver
from .solvers.scipy_solver import ScipySolver
from .solvers.scikits_dae_solver import ScikitsDaeSolver
from .solvers.scikits_ode_solver import ScikitsOdeSolver

#
# other
#
from .processed_variable import post_process_variables, ProcessedVariable
from .quick_plot import QuickPlot

#
# Remove any imported modules, so we don't expose them as part of pybamm
#
del (sys)
