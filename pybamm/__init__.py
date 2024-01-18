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
from pybamm.version import __version__
from pybamm.util import lazy_import

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
PARAMETER_PATH = [
    root_dir(),
    os.getcwd(),
    os.path.join(root_dir(), "pybamm", "input", "parameters"),
]


#
# Utility classes and methods
#
Timer = lazy_import("pybamm.util", "Timer")
TimerTime = lazy_import("pybamm.util", "TimerTime")
FuzzyDict = lazy_import("pybamm.util", "FuzzyDict")

root_dir = lazy_import("pybamm.util", "root_dir")
rmse = lazy_import("pybamm.util", "rmse")
load = lazy_import("pybamm.util", "load")
is_constant_and_can_evaluate = lazy_import("pybamm.util", "is_constant_and_can_evaluate")

get_parameters_filepath = lazy_import("pybamm.util", "get_parameters_filepath")
have_jax = lazy_import("pybamm.util", "have_jax")
install_jax = lazy_import("pybamm.util", "install_jax")
have_optional_dependency = lazy_import("pybamm.util", "have_optional_dependency")
is_jax_compatible = lazy_import("pybamm.util", "is_jax_compatible")
get_git_commit_info = lazy_import("pybamm.util", "get_git_commit_info")

logger = lazy_import("pybamm.logger", "logger")
set_logging_level = lazy_import("pybamm.logger", "set_logging_level")
get_new_logger = lazy_import("pybamm.logger", "get_new_logger")

settings = lazy_import("pybamm.settings","settings")

Citations = lazy_import("pybamm.citations", "Citations")
citations = lazy_import("pybamm.citations", "citations")
print_citations = lazy_import("pybamm.citations", "print_citations")

#
# Classes for the Expression Tree
#
from .expression_tree.symbol import *
from .expression_tree.binary_operators import *
from .expression_tree.concatenations import *
Array = lazy_import("pybamm.expression_tree.array", "Array")
linspace = lazy_import("pybamm.expression_tree.array", "linspace")
meshgrid = lazy_import("pybamm.expression_tree.array", "meshgrid")
Matrix = lazy_import("pybamm.expression_tree.matrix", "Matrix")
from .expression_tree.unary_operators import *
from .expression_tree.averages import *
_BaseAverage = lazy_import("pybamm.expression_tree.averages", "_BaseAverage")
from .expression_tree.broadcasts import *
from .expression_tree.functions import *
Interpolant = lazy_import("pybamm.expression_tree.interpolant", "Interpolant")
InputParameter = lazy_import("pybamm.expression_tree.input_parameter", "InputParameter")
Parameter = lazy_import("pybamm.expression_tree.parameter", "Parameter")
FunctionParameter = lazy_import("pybamm.expression_tree.parameter", "FunctionParameter")
Scalar = lazy_import("pybamm.expression_tree.scalar", "Scalar")
from .expression_tree.variable import *
from .expression_tree.independent_variable import *
from .expression_tree.independent_variable import t
Vector = lazy_import("pybamm.expression_tree.vector", "Vector")
StateVectorBase = lazy_import("pybamm.expression_tree.state_vector", "StateVectorBase")
StateVector = lazy_import("pybamm.expression_tree.state_vector", "StateVector")
StateVectorDot = lazy_import("pybamm.expression_tree.state_vector", "StateVectorDot")

from .expression_tree.exceptions import *

# Operations
find_symbols = lazy_import("pybamm.expression_tree.operations.evaluate_python", "find_symbols")
id_to_python_variable = lazy_import("pybamm.expression_tree.operations.evaluate_python", "id_to_python_variable")
to_python = lazy_import("pybamm.expression_tree.operations.evaluate_python", "to_python")
EvaluatorPython = lazy_import("pybamm.expression_tree.operations.evaluate_python", "EvaluatorPython")

EvaluatorJax = lazy_import("pybamm.expression_tree.operations.evaluate_python", "EvaluatorJax")
JaxCooMatrix = lazy_import("pybamm.expression_tree.operations.evaluate_python", "JaxCooMatrix")
Jacobian = lazy_import("pybamm.expression_tree.operations.jacobian", "Jacobian")
CasadiConverter = lazy_import("pybamm.expression_tree.operations.convert_to_casadi", "CasadiConverter")
SymbolUnpacker = lazy_import("pybamm.expression_tree.operations.unpack_symbols", "SymbolUnpacker")

#
# Model classes
#
BaseModel = lazy_import("pybamm.models.base_model", "BaseModel")
Event = lazy_import("pybamm.models.event", "Event")
EventType = lazy_import("pybamm.models.event", "EventType")

# Battery models
BaseBatteryModel = lazy_import("pybamm.models.full_battery_models.base_battery_model", "BaseBatteryModel")
BatteryModelOptions = lazy_import("pybamm.models.full_battery_models.base_battery_model", "BatteryModelOptions")

lead_acid = lazy_import("pybamm.models.full_battery_models.lead_acid", "lead_acid")
lithium_ion = lazy_import("pybamm.models.full_battery_models.lithium_ion", "lithium_ion")
equivalent_circuit = lazy_import("pybamm.models.full_battery_models.equivalent_circuit", "equivalent_circuit")

#
# Submodel classes
#
BaseSubModel = lazy_import("pybamm.models.submodels.base_submodel", "BaseSubModel")

active_material = lazy_import("pybamm.models.submodels.active_material", "active_material")
convection = lazy_import("pybamm.models.submodels.convection", "convection")
current_collector = lazy_import("pybamm.models.submodels.current_collector", "current_collector")
electrolyte_conductivity = lazy_import("pybamm.models.submodels.electrolyte_conductivity", "electrolyte_conductivity")
electrolyte_diffusion = lazy_import("pybamm.models.submodels.electrolyte_diffusion", "electrolyte_diffusion")
electrode = lazy_import("pybamm.models.submodels.electrode", "electrode")
external_circuit = lazy_import("pybamm.models.submodels.external_circuit", "external_circuit")
interface = lazy_import("pybamm.models.submodels.interface", "interface")
oxygen_diffusion = lazy_import("pybamm.models.submodels.oxygen_diffusion", "oxygen_diffusion")
particle = lazy_import("pybamm.models.submodels.particle", "particle")
porosity = lazy_import("pybamm.models.submodels.porosity", "porosity")
thermal = lazy_import("pybamm.models.submodels.thermal", "thermal")
transport_efficiency = lazy_import("pybamm.models.submodels.transport_efficiency", "transport_efficiency")
particle_mechanics = lazy_import("pybamm.models.submodels.particle_mechanics", "particle_mechanics")
equivalent_circuit_elements = lazy_import("pybamm.models.submodels.equivalent_circuit_elements", "equivalent_circuit_elements")

kinetics = lazy_import("pybamm.models.submodels.interface.kinetics", "kinetics")
sei = lazy_import("pybamm.models.submodels.interface.sei", "sei")
lithium_plating = lazy_import("pybamm.models.submodels.interface.lithium_plating", "lithium_plating")
interface_utilisation = lazy_import("pybamm.models.submodels.interface.interface_utilisation", "interface_utilisation")
open_circuit_potential = lazy_import("pybamm.models.submodels.interface.open_circuit_potential", "open_circuit_potential")

#
# Geometry
#
Geometry = lazy_import("pybamm.geometry.geometry", "Geometry")
battery_geometry = lazy_import("pybamm.geometry.battery_geometry", "battery_geometry")

KNOWN_COORD_SYS = lazy_import("pybamm.expression_tree.independent_variable", "KNOWN_COORD_SYS")
standard_spatial_vars = lazy_import("pybamm.geometry.standard_spatial_vars", "standard_spatial_vars")

#
# Parameter classes and methods
#
ParameterValues = lazy_import("pybamm.parameters.parameter_values", "ParameterValues")
constants = lazy_import("pybamm.parameters.constants", "constants")
geometric_parameters = lazy_import("pybamm.parameters.geometric_parameters", "geometric_parameters")
GeometricParameters = lazy_import("pybamm.parameters.geometric_parameters", "GeometricParameters")

electrical_parameters = lazy_import("pybamm.parameters.electrical_parameters", "electrical_parameters")
ElectricalParameters = lazy_import("pybamm.parameters.electrical_parameters", "ElectricalParameters")

thermal_parameters = lazy_import("pybamm.parameters.thermal_parameters", "thermal_parameters")
ThermalParameters = lazy_import("pybamm.parameters.thermal_parameters", "ThermalParameters")
LithiumIonParameters = lazy_import("pybamm.parameters.lithium_ion_parameters", "LithiumIonParameters")
LeadAcidParameters = lazy_import("pybamm.parameters.lead_acid_parameters", "LeadAcidParameters")
EcmParameters = lazy_import("pybamm.parameters.ecm_parameters", "EcmParameters")
from .parameters.size_distribution_parameters import *
parameter_sets = lazy_import("pybamm.parameters.parameter_sets", "parameter_sets")
add_parameter = lazy_import("pybamm.parameters_cli", "add_parameter")
remove_parameter = lazy_import("pybamm.parameters_cli", "remove_parameter")
edit_parameter = lazy_import("pybamm.parameters_cli", "edit_parameter")

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
    SpectralVolume1DSubMesh,
)
from .meshes.scikit_fem_submeshes import (
    ScikitSubMesh2D,
    ScikitUniform2DSubMesh,
    ScikitExponential2DSubMesh,
    ScikitChebyshev2DSubMesh,
    UserSupplied2DSubMesh,
)

#
# Serialisation
#
from .models.base_model import load_model

#
# Spatial Methods
#
from .spatial_methods.spatial_method import SpatialMethod
from .spatial_methods.zero_dimensional_method import ZeroDimensionalSpatialMethod
from .spatial_methods.finite_volume import FiniteVolume
from .spatial_methods.spectral_volume import SpectralVolume
from .spatial_methods.scikit_finite_element import ScikitFiniteElement

#
# Solver classes
#
from .solvers.solution import Solution, EmptySolution, make_cycle_solution
from .solvers.processed_variable import ProcessedVariable
from .solvers.processed_variable_computed import ProcessedVariableComputed
from .solvers.base_solver import BaseSolver
from .solvers.dummy_solver import DummySolver
from .solvers.algebraic_solver import AlgebraicSolver
from .solvers.casadi_solver import CasadiSolver
from .solvers.casadi_algebraic_solver import CasadiAlgebraicSolver
from .solvers.scikits_dae_solver import ScikitsDaeSolver
from .solvers.scikits_ode_solver import ScikitsOdeSolver, have_scikits_odes
from .solvers.scipy_solver import ScipySolver

from .solvers.jax_solver import JaxSolver
from .solvers.jax_bdf_solver import jax_bdf_integrate

IDAKLUSolver = lazy_import("pybamm.solvers.idaklu_solver","IDAKLUSolver")
have_idaklu = lazy_import("pybamm.solvers.idaklu_solver", "have_idaklu")

#
# Experiments
#
from .experiment.experiment import Experiment
from . import experiment
from .experiment import step


#
# Plotting
#
from .plotting.quick_plot import QuickPlot, close_plots, QuickPlotAxes
from .plotting.plot import plot
from .plotting.plot2D import plot2D
from .plotting.plot_voltage_components import plot_voltage_components
from .plotting.plot_summary_variables import plot_summary_variables
from .plotting.dynamic_plot import dynamic_plot

#
# Simulation
#
from .simulation import Simulation, load_sim, is_notebook

#
# Batch Study
#
from .batch_study import BatchStudy

#
# Callbacks
#
from . import callbacks

#
# Remove any imported modules, so we don't expose them as part of pybamm
#
del sys
