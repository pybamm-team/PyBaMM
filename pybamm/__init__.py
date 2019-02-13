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

#
# Classes for the Expression Tree
#
from .expression_tree.symbol import Symbol
from .expression_tree.binary_operators import (
    BinaryOperator,
    Addition,
    Power,
    Subtraction,
    Multiplication,
    Division,
)
from .expression_tree.concatenations import (
    Concatenation,
    NumpyModelConcatenation,
    DomainConcatenation,
)
from .expression_tree.array import Array
from .expression_tree.matrix import Matrix
from .expression_tree.parameter import Parameter
from .expression_tree.unary_operators import (
    UnaryOperator,
    Negate,
    AbsoluteValue,
    SpatialOperator,
    Gradient,
    Divergence,
    Broadcast,
    NumpyBroadcast,
    grad,
    div,
)
from .expression_tree.scalar import Scalar
from .expression_tree.variable import Variable
from .expression_tree.independent_variable import IndependentVariable, Time, Space
from .expression_tree.independent_variable import t
from .expression_tree.vector import Vector, StateVector

from .expression_tree.exceptions import DomainError, ModelError

#
# Model classes
#
from .models.base_model import BaseModel
from .models.reaction_diffusion import ReactionDiffusionModel
from .models.simple_ode_model import SimpleODEModel
from .models import lead_acid

#
# Submodel classes
#
from .models.submodels import electrolyte, interface

#
# Parameters class and methods
#
from .parameters.parameter_values import ParameterValues
from .parameters import functions_lead_acid
from .parameters import standard_parameters
from .parameters import standard_parameters_lead_acid  # calls standard_parameters

#
# Geometry
#
from .geometry.geometry import (
    Geometry,
    Geometry1DMacro,
    Geometry1DMicro,
    Geometry3DMacro,
)

#
# Mesh and Discretisation classes
#
from .discretisations.discretisation import Discretisation
from .meshes.meshes import KNOWN_DOMAINS
from .meshes.meshes import Mesh
from .meshes.submeshes import SubMesh1D, Uniform1DSubMesh

#
# Spatial Methods
#
from .spatial_methods.spatial_method import SpatialMethod
from .spatial_methods.finite_volume import FiniteVolume, NodeToEdge

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
# Remove any imported modules, so we don't expose them as part of pybamm
#
del (sys)
