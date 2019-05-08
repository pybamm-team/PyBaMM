#
# Root of the pybamm module.
# Provides access to all shared functionality (simulation, models, etc.).
#
# The code in this file is adapted from Pints
# (see https://github.com/pints-team/pints)
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
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
# Expose pints version
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
from .util import profile
from .util import load_function

#
# Classes for the Expression Tree
#
from .expression_tree.symbol import Symbol
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
    Function,
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
from .expression_tree.parameter import Parameter, FunctionParameter
from .expression_tree.broadcasts import Broadcast, NumpyBroadcast
from .expression_tree.scalar import Scalar
from .expression_tree.variable import Variable
from .expression_tree.independent_variable import (
    IndependentVariable,
    Time,
    SpatialVariable,
)
from .expression_tree.independent_variable import t
from .expression_tree.vector import Vector, StateVector

from .expression_tree.exceptions import DomainError, ModelError
from .expression_tree.simplify import (
    simplify,
    simplify_if_constant,
    simplify_addition_subtraction,
    simplify_multiplication_division,
)

#
# Model classes
#
from .meshes.meshes import KNOWN_DOMAINS  # need this for importing standard variables
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
)

#
# Parameters class and methods
#
from .parameters.parameter_values import ParameterValues
from .parameters import standard_current_functions
from .parameters import geometric_parameters
from .parameters import electrical_parameters
from .parameters import standard_parameters_lithium_ion, standard_parameters_lead_acid

#
# Geometry
#
from .geometry.geometry import (
    Geometry,
    Geometry1DMacro,
    Geometry1DMicro,
    Geometry1p1DMicro,
    Geometry3DMacro,
)

from .expression_tree.independent_variable import KNOWN_SPATIAL_VARS
from .geometry import standard_spatial_vars
from .geometry.standard_spatial_vars import KNOWN_COORD_SYS

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
# Simulation class
#
from .simulation import Simulation

#
# Solver classes
#
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

#
# Remove any imported modules, so we don't expose them as part of pybamm
#
del (sys)
