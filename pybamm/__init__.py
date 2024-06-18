import sys

from pybamm.version import __version__

# Demote expressions to 32-bit floats/ints - option used for IDAKLU-MLIR compilation
demote_expressions_to_32bit = False

# Utility classes and methods
from .util import root_dir
from .util import Timer, TimerTime, FuzzyDict
from .util import (
    root_dir,
    load,
    is_constant_and_can_evaluate,
)
from .util import (
    get_parameters_filepath,
    have_jax,
    install_jax,
    import_optional_dependency,
    is_jax_compatible,
    get_git_commit_info,
)
from .logger import logger, set_logging_level, get_new_logger
from .settings import settings
from .citations import Citations, citations, print_citations

# Classes for the Expression Tree
from .expression_tree.symbol import *
from .expression_tree.binary_operators import *
from .expression_tree.concatenations import *
from .expression_tree.array import Array, linspace, meshgrid
from .expression_tree.matrix import Matrix
from .expression_tree.unary_operators import *
from .expression_tree.averages import *
from .expression_tree.averages import _BaseAverage
from .expression_tree.broadcasts import *
from .expression_tree.functions import *
from .expression_tree.interpolant import Interpolant
from .expression_tree.input_parameter import InputParameter
from .expression_tree.parameter import Parameter, FunctionParameter
from .expression_tree.scalar import Scalar
from .expression_tree.variable import *
from .expression_tree.independent_variable import *
from .expression_tree.independent_variable import t
from .expression_tree.vector import Vector
from .expression_tree.state_vector import StateVectorBase, StateVector, StateVectorDot

from .expression_tree.exceptions import *

# Operations
from .expression_tree.operations.evaluate_python import (
    find_symbols,
    id_to_python_variable,
    to_python,
    EvaluatorPython,
)

from .expression_tree.operations.evaluate_python import EvaluatorJax
from .expression_tree.operations.evaluate_python import JaxCooMatrix

from .expression_tree.operations.jacobian import Jacobian
from .expression_tree.operations.convert_to_casadi import CasadiConverter
from .expression_tree.operations.unpack_symbols import SymbolUnpacker

# Model classes
from .models.base_model import BaseModel
from .models.event import Event
from .models.event import EventType

# Battery models
from .models.full_battery_models.base_battery_model import (
    BaseBatteryModel,
    BatteryModelOptions,
)
from .models.full_battery_models import lead_acid
from .models.full_battery_models import lithium_ion
from .models.full_battery_models import equivalent_circuit

# Submodel classes
from .models.submodels.base_submodel import BaseSubModel

from .models.submodels import (
    active_material,
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
    transport_efficiency,
    particle_mechanics,
    equivalent_circuit_elements,
)
from .models.submodels.interface import kinetics
from .models.submodels.interface import sei
from .models.submodels.interface import lithium_plating
from .models.submodels.interface import interface_utilisation
from .models.submodels.interface import open_circuit_potential

# Geometry
from .geometry.geometry import Geometry
from .geometry.battery_geometry import battery_geometry

from .expression_tree.independent_variable import KNOWN_COORD_SYS
from .geometry import standard_spatial_vars

# Parameter classes and methods
from .parameters.parameter_values import ParameterValues
from .parameters import constants
from .parameters.geometric_parameters import geometric_parameters, GeometricParameters
from .parameters.electrical_parameters import (
    electrical_parameters,
    ElectricalParameters,
)
from .parameters.thermal_parameters import thermal_parameters, ThermalParameters
from .parameters.lithium_ion_parameters import LithiumIonParameters
from .parameters.lead_acid_parameters import LeadAcidParameters
from .parameters.ecm_parameters import EcmParameters
from .parameters.size_distribution_parameters import *
from .parameters.parameter_sets import parameter_sets

# Mesh and Discretisation classes
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
    SpectralVolume1DSubMesh,
)
from .meshes.scikit_fem_submeshes import (
    ScikitSubMesh2D,
    ScikitUniform2DSubMesh,
    ScikitExponential2DSubMesh,
    ScikitChebyshev2DSubMesh,
    UserSupplied2DSubMesh,
)

# Serialisation
from .models.base_model import load_model

# Spatial Methods
from .spatial_methods.spatial_method import SpatialMethod
from .spatial_methods.zero_dimensional_method import ZeroDimensionalSpatialMethod
from .spatial_methods.finite_volume import FiniteVolume
from .spatial_methods.spectral_volume import SpectralVolume
from .spatial_methods.scikit_finite_element import ScikitFiniteElement

# Solver classes
from .solvers.solution import Solution, EmptySolution, make_cycle_solution
from .solvers.processed_variable import ProcessedVariable
from .solvers.processed_variable_computed import ProcessedVariableComputed
from .solvers.base_solver import BaseSolver
from .solvers.dummy_solver import DummySolver
from .solvers.algebraic_solver import AlgebraicSolver
from .solvers.casadi_solver import CasadiSolver
from .solvers.casadi_algebraic_solver import CasadiAlgebraicSolver
from .solvers.scipy_solver import ScipySolver

from .solvers.jax_solver import JaxSolver
from .solvers.jax_bdf_solver import jax_bdf_integrate

from .solvers.idaklu_jax import IDAKLUJax
from .solvers.idaklu_solver import IDAKLUSolver, have_idaklu, have_iree

# Experiments
from .experiment.experiment import Experiment
from . import experiment
from .experiment import step

# Plotting
from .plotting.quick_plot import QuickPlot, close_plots, QuickPlotAxes
from .plotting.plot import plot
from .plotting.plot2D import plot2D
from .plotting.plot_voltage_components import plot_voltage_components
from .plotting.plot_thermal_components import plot_thermal_components
from .plotting.plot_summary_variables import plot_summary_variables
from .plotting.dynamic_plot import dynamic_plot

# Simulation
from .simulation import Simulation, load_sim, is_notebook

# Batch Study
from .batch_study import BatchStudy

# Callbacks
from . import callbacks

# Pybamm Data manager using pooch
from .pybamm_data import DataLoader

# Remove any imported modules, so we don't expose them as part of pybamm
del sys

__all__ = [
    "batch_study",
    "callbacks",
    "citations",
    "discretisations",
    "doc_utils",
    "experiment",
    "expression_tree",
    "geometry",
    "input",
    "logger",
    "meshes",
    "models",
    "parameters",
    "plotting",
    "settings",
    "simulation",
    "solvers",
    "spatial_methods",
    "type_definitions",
    "util",
    "version",
    "pybamm_data",
]
