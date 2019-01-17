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
from .expression_tree.concatenations import Concatenation, NumpyConcatenation
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
    grad,
    div,
)
from .expression_tree.scalar import Scalar
from .expression_tree.variable import Variable
from .expression_tree.independent_variable import IndependentVariable
from .expression_tree.independent_variable import t
from .expression_tree.vector import Vector, StateVector

from .expression_tree.exceptions import DomainError

#
# Model classes
#
from .models.core import BaseModel
from .models.reaction_diffusion import ReactionDiffusionModel
from .models.electrolyte_current import ElectrolyteCurrentModel

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

#
# Mesh and Discretisation classes
#
from .discretisations.base_discretisation import BaseDiscretisation
from .discretisations.finite_volume_discretisations import FiniteVolumeDiscretisation
from .discretisations.base_mesh import KNOWN_DOMAINS
from .discretisations.base_mesh import BaseMesh, BaseSubmesh
from .discretisations.finite_volume_meshes import (
    FiniteVolumeMacroMesh,
    FiniteVolumeSubmesh,
)

#
# Simulation class
#
from .simulation import Simulation

#
# Solver classes
#
from .solvers.base_solver import BaseSolver
from .solvers.scipy_solver import ScipySolver

#
# Remove any imported modules, so we don't expose them as part of pybamm
#
del (sys)
