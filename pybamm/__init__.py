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
ABSOLUTE_PATH = os.path.join(os.path.split(script_path)[0], "..")

#
# Utility classes and methods
#
from .util import Timer
from .util import root_dir, load_function, rmse, get_infinite_nested_dict
from .logger import logger, set_logging_level
from .settings import settings

#
# Classes for the Expression Tree
#
from .expression_tree.symbol import (
    Symbol,
    domain_size,
    create_object_of_size,
    evaluate_for_shape_using_domain,
)
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
    Inner,
    inner,
    Outer,
    Kron,
    outer,
    source,
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
    Laplacian,
    Gradient_Squared,
    Mass,
    BoundaryMass,
    BoundaryOperator,
    BoundaryValue,
    BoundaryGradient,
    Integral,
    IndefiniteIntegral,
    DefiniteIntegralVector,
    BoundaryIntegral,
    DeltaFunction,
    grad,
    div,
    laplacian,
    grad_squared,
    surf,
    x_average,
    z_average,
    yz_average,
    boundary_value,
    r_average,
)
from .expression_tree.functions import *
from .expression_tree.interpolant import Interpolant
from .expression_tree.parameter import Parameter, FunctionParameter
from .expression_tree.broadcasts import Broadcast, PrimaryBroadcast, FullBroadcast
from .expression_tree.scalar import Scalar
from .expression_tree.variable import Variable
from .expression_tree.independent_variable import (
    IndependentVariable,
    Time,
    SpatialVariable,
)
from .expression_tree.independent_variable import t
from .expression_tree.vector import Vector
from .expression_tree.state_vector import StateVector

from .expression_tree.exceptions import (
    DomainError,
    OptionError,
    ModelError,
    SolverError,
    ShapeError,
    ModelWarning,
    UndefinedOperationError,
    GeometryError,
)

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

#
# Model classes
#
from .models.base_model import BaseModel
from .models import standard_variables

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
    electrolyte,
    electrode,
    interface,
    oxygen_diffusion,
    particle,
    porosity,
    thermal,
)

#
# Parameters class and methods
#
from .parameters.parameter_values import ParameterValues
from .parameters import standard_current_functions
from .parameters import geometric_parameters
from .parameters import electrical_parameters
from .parameters import thermal_parameters
from .parameters import standard_parameters_lithium_ion, standard_parameters_lead_acid
from .parameters.print_parameters import print_parameters, print_evaluated_parameters
from .parameters import parameter_sets

#
# Geometry
#
from .geometry.geometry import (
    Geometry,
    Geometry1DMacro,
    Geometry3DMacro,
    Geometry1DMicro,
    Geometry1p1DMicro,
    Geometryxp1DMacro,
    Geometryxp0p1DMicro,
    Geometryxp1p1DMicro,
    Geometry2DCurrentCollector,
)

from .expression_tree.independent_variable import KNOWN_SPATIAL_VARS, KNOWN_COORD_SYS
from .geometry import standard_spatial_vars

#
# Mesh and Discretisation classes
#
from .discretisations.discretisation import Discretisation
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
from .spatial_methods.zero_dimensional_method import ZeroDimensionalMethod
from .spatial_methods.finite_volume import FiniteVolume
from .spatial_methods.scikit_finite_element import ScikitFiniteElement

#
# Solver classes
#
from .solvers.solution import Solution
from .solvers.base_solver import BaseSolver
from .solvers.ode_solver import OdeSolver
from .solvers.dae_solver import DaeSolver
from .solvers.algebraic_solver import AlgebraicSolver
from .solvers.casadi_solver import CasadiSolver
from .solvers.scikits_dae_solver import ScikitsDaeSolver
from .solvers.scikits_ode_solver import ScikitsOdeSolver, have_scikits_odes
from .solvers.scipy_solver import ScipySolver
from .solvers.idaklu_solver import IDAKLUSolver, have_idaklu

#
# Current profiles
#
from .parameters.standard_current_functions.base_current import GetCurrent
from .parameters.standard_current_functions.get_constant_current import (
    GetConstantCurrent,
)
from .parameters.standard_current_functions.get_user_current import GetUserCurrent
from .parameters.standard_current_functions.get_current_data import GetCurrentData

#
# other
#
from .processed_variable import post_process_variables, ProcessedVariable
from .quick_plot import QuickPlot, ax_min, ax_max

from .simulation import Simulation

#
# Remove any imported modules, so we don't expose them as part of pybamm
#
del sys
