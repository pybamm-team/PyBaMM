# PyBaMM stub file for IDE support and type hints
# This file enables autocomplete and type checking for lazy-loaded attributes
# Note: Submodule aliases (lithium_ion, lead_acid, etc.) are handled by custom __getattr__
# Note: Variable annotations (lithium_ion: ModuleType) are not supported by lazy_loader,
#       so they are omitted from this file. IDE support for submodules comes from __dir__().

# Version
from .version import __version__ as __version__

# Logger and settings (eagerly loaded)
from .logger import logger as logger
from .logger import set_logging_level as set_logging_level
from .logger import get_new_logger as get_new_logger
from .settings import settings as settings
from . import config as config
from .citations import Citations as Citations
from .citations import citations as citations
from .citations import print_citations as print_citations

# Expression tree - eagerly loaded (accessed thousands of times during model building)
# symbol.py
from .expression_tree.symbol import Symbol as Symbol
from .expression_tree.symbol import domain_size as domain_size
from .expression_tree.symbol import create_object_of_size as create_object_of_size
from .expression_tree.symbol import evaluate_for_shape_using_domain as evaluate_for_shape_using_domain
from .expression_tree.symbol import is_constant as is_constant
from .expression_tree.symbol import is_scalar_zero as is_scalar_zero
from .expression_tree.symbol import is_scalar_one as is_scalar_one
from .expression_tree.symbol import is_scalar_minus_one as is_scalar_minus_one
from .expression_tree.symbol import is_matrix_zero as is_matrix_zero
from .expression_tree.symbol import is_matrix_one as is_matrix_one
from .expression_tree.symbol import is_matrix_minus_one as is_matrix_minus_one
from .expression_tree.symbol import simplify_if_constant as simplify_if_constant
from .expression_tree.symbol import convert_to_symbol as convert_to_symbol

# binary_operators.py
from .expression_tree.binary_operators import BinaryOperator as BinaryOperator
from .expression_tree.binary_operators import Power as Power
from .expression_tree.binary_operators import Addition as Addition
from .expression_tree.binary_operators import Subtraction as Subtraction
from .expression_tree.binary_operators import Multiplication as Multiplication
from .expression_tree.binary_operators import KroneckerProduct as KroneckerProduct
from .expression_tree.binary_operators import TensorProduct as TensorProduct
from .expression_tree.binary_operators import MatrixMultiplication as MatrixMultiplication
from .expression_tree.binary_operators import Division as Division
from .expression_tree.binary_operators import Inner as Inner
from .expression_tree.binary_operators import Equality as Equality
from .expression_tree.binary_operators import EqualHeaviside as EqualHeaviside
from .expression_tree.binary_operators import NotEqualHeaviside as NotEqualHeaviside
from .expression_tree.binary_operators import Modulo as Modulo
from .expression_tree.binary_operators import Minimum as Minimum
from .expression_tree.binary_operators import Maximum as Maximum
from .expression_tree.binary_operators import softplus as softplus
from .expression_tree.binary_operators import softminus as softminus
from .expression_tree.binary_operators import sigmoid as sigmoid
from .expression_tree.binary_operators import source as source

# concatenations.py
from .expression_tree.concatenations import Concatenation as Concatenation
from .expression_tree.concatenations import NumpyConcatenation as NumpyConcatenation
from .expression_tree.concatenations import DomainConcatenation as DomainConcatenation
from .expression_tree.concatenations import SparseStack as SparseStack
from .expression_tree.concatenations import ConcatenationVariable as ConcatenationVariable
from .expression_tree.concatenations import concatenation as concatenation
from .expression_tree.concatenations import numpy_concatenation as numpy_concatenation

# unary_operators.py
from .expression_tree.unary_operators import UnaryOperator as UnaryOperator
from .expression_tree.unary_operators import Negate as Negate
from .expression_tree.unary_operators import AbsoluteValue as AbsoluteValue
from .expression_tree.unary_operators import Transpose as Transpose
from .expression_tree.unary_operators import Sign as Sign
from .expression_tree.unary_operators import Floor as Floor
from .expression_tree.unary_operators import Ceiling as Ceiling
from .expression_tree.unary_operators import Index as Index
from .expression_tree.unary_operators import SpatialOperator as SpatialOperator
from .expression_tree.unary_operators import Gradient as Gradient
from .expression_tree.unary_operators import Divergence as Divergence
from .expression_tree.unary_operators import Laplacian as Laplacian
from .expression_tree.unary_operators import GradientSquared as GradientSquared
from .expression_tree.unary_operators import Mass as Mass
from .expression_tree.unary_operators import BoundaryMass as BoundaryMass
from .expression_tree.unary_operators import Integral as Integral
from .expression_tree.unary_operators import BaseIndefiniteIntegral as BaseIndefiniteIntegral
from .expression_tree.unary_operators import IndefiniteIntegral as IndefiniteIntegral
from .expression_tree.unary_operators import BackwardIndefiniteIntegral as BackwardIndefiniteIntegral
from .expression_tree.unary_operators import DefiniteIntegralVector as DefiniteIntegralVector
from .expression_tree.unary_operators import BoundaryIntegral as BoundaryIntegral
from .expression_tree.unary_operators import OneDimensionalIntegral as OneDimensionalIntegral
from .expression_tree.unary_operators import DeltaFunction as DeltaFunction
from .expression_tree.unary_operators import BoundaryOperator as BoundaryOperator
from .expression_tree.unary_operators import BoundaryValue as BoundaryValue
from .expression_tree.unary_operators import BoundaryMeshSize as BoundaryMeshSize
from .expression_tree.unary_operators import ExplicitTimeIntegral as ExplicitTimeIntegral
from .expression_tree.unary_operators import BoundaryGradient as BoundaryGradient
from .expression_tree.unary_operators import EvaluateAt as EvaluateAt
from .expression_tree.unary_operators import UpwindDownwind as UpwindDownwind
from .expression_tree.unary_operators import UpwindDownwind2D as UpwindDownwind2D
from .expression_tree.unary_operators import NodeToEdge2D as NodeToEdge2D
from .expression_tree.unary_operators import Magnitude as Magnitude
from .expression_tree.unary_operators import Upwind as Upwind
from .expression_tree.unary_operators import Downwind as Downwind
from .expression_tree.unary_operators import NotConstant as NotConstant
from .expression_tree.unary_operators import grad as grad
from .expression_tree.unary_operators import div as div
from .expression_tree.unary_operators import laplacian as laplacian
from .expression_tree.unary_operators import grad_squared as grad_squared
from .expression_tree.unary_operators import surf as surf
from .expression_tree.unary_operators import x_average as x_average
from .expression_tree.unary_operators import yz_average as yz_average
from .expression_tree.unary_operators import z_average as z_average
from .expression_tree.unary_operators import r_average as r_average
from .expression_tree.unary_operators import size_average as size_average
from .expression_tree.unary_operators import boundary_value as boundary_value
from .expression_tree.unary_operators import sign as sign
from .expression_tree.unary_operators import upwind as upwind
from .expression_tree.unary_operators import downwind as downwind

# averages.py
from .expression_tree.averages import _BaseAverage as _BaseAverage
from .expression_tree.averages import XAverage as XAverage
from .expression_tree.averages import YZAverage as YZAverage
from .expression_tree.averages import ZAverage as ZAverage
from .expression_tree.averages import RAverage as RAverage
from .expression_tree.averages import SizeAverage as SizeAverage

# broadcasts.py
from .expression_tree.broadcasts import Broadcast as Broadcast
from .expression_tree.broadcasts import PrimaryBroadcast as PrimaryBroadcast
from .expression_tree.broadcasts import PrimaryBroadcastToEdges as PrimaryBroadcastToEdges
from .expression_tree.broadcasts import SecondaryBroadcast as SecondaryBroadcast
from .expression_tree.broadcasts import SecondaryBroadcastToEdges as SecondaryBroadcastToEdges
from .expression_tree.broadcasts import TertiaryBroadcast as TertiaryBroadcast
from .expression_tree.broadcasts import TertiaryBroadcastToEdges as TertiaryBroadcastToEdges
from .expression_tree.broadcasts import FullBroadcast as FullBroadcast
from .expression_tree.broadcasts import FullBroadcastToEdges as FullBroadcastToEdges
from .expression_tree.broadcasts import ones_like as ones_like
from .expression_tree.broadcasts import zeros_like as zeros_like

# functions.py
from .expression_tree.functions import Function as Function
from .expression_tree.functions import SpecificFunction as SpecificFunction
from .expression_tree.functions import Arcsinh as Arcsinh
from .expression_tree.functions import Arctan as Arctan
from .expression_tree.functions import Cos as Cos
from .expression_tree.functions import Cosh as Cosh
from .expression_tree.functions import Erf as Erf
from .expression_tree.functions import Exp as Exp
from .expression_tree.functions import Log as Log
from .expression_tree.functions import Max as Max
from .expression_tree.functions import Min as Min
from .expression_tree.functions import Sin as Sin
from .expression_tree.functions import Sinh as Sinh
from .expression_tree.functions import Sqrt as Sqrt
from .expression_tree.functions import Tanh as Tanh
from .expression_tree.functions import arcsinh as arcsinh
from .expression_tree.functions import arctan as arctan
from .expression_tree.functions import cos as cos
from .expression_tree.functions import cosh as cosh
from .expression_tree.functions import erf as erf
from .expression_tree.functions import exp as exp
from .expression_tree.functions import log as log
from .expression_tree.functions import log10 as log10
from .expression_tree.functions import max as max
from .expression_tree.functions import min as min
from .expression_tree.functions import sin as sin
from .expression_tree.functions import sinh as sinh
from .expression_tree.functions import sqrt as sqrt
from .expression_tree.functions import tanh as tanh
from .expression_tree.functions import Abs as Abs

# interpolant.py
from .expression_tree.interpolant import Interpolant as Interpolant

# discrete_time_sum.py
from .expression_tree.discrete_time_sum import DiscreteTimeData as DiscreteTimeData
from .expression_tree.discrete_time_sum import DiscreteTimeSum as DiscreteTimeSum

# variable.py
from .expression_tree.variable import VariableBase as VariableBase
from .expression_tree.variable import Variable as Variable
from .expression_tree.variable import VariableDot as VariableDot

# coupled_variable.py
from .expression_tree.coupled_variable import CoupledVariable as CoupledVariable

# independent_variable.py
from .expression_tree.independent_variable import IndependentVariable as IndependentVariable
from .expression_tree.independent_variable import Time as Time
from .expression_tree.independent_variable import SpatialVariable as SpatialVariable
from .expression_tree.independent_variable import SpatialVariableEdge as SpatialVariableEdge
from .expression_tree.independent_variable import t as t
from .expression_tree.independent_variable import KNOWN_COORD_SYS as KNOWN_COORD_SYS

# exceptions.py
from .expression_tree.exceptions import DomainError as DomainError
from .expression_tree.exceptions import OptionError as OptionError
from .expression_tree.exceptions import OptionWarning as OptionWarning
from .expression_tree.exceptions import GeometryError as GeometryError
from .expression_tree.exceptions import ModelError as ModelError
from .expression_tree.exceptions import SolverError as SolverError
from .expression_tree.exceptions import SolverWarning as SolverWarning
from .expression_tree.exceptions import ShapeError as ShapeError
from .expression_tree.exceptions import ModelWarning as ModelWarning
from .expression_tree.exceptions import DiscretisationError as DiscretisationError
from .expression_tree.exceptions import InvalidModelJSONError as InvalidModelJSONError

# ============================================================================
# LAZY LOADED ATTRIBUTES (from _LAZY_IMPORTS)
# ============================================================================

# Utility classes and methods
from .util import root_dir as root_dir
from .util import Timer as Timer
from .util import TimerTime as TimerTime
from .util import FuzzyDict as FuzzyDict
from .util import load as load
from .util import is_constant_and_can_evaluate as is_constant_and_can_evaluate
from .util import get_parameters_filepath as get_parameters_filepath
from .util import has_jax as has_jax
from .util import import_optional_dependency as import_optional_dependency

# Array/Matrix
from .expression_tree.array import Array as Array
from .expression_tree.array import linspace as linspace
from .expression_tree.array import meshgrid as meshgrid
from .expression_tree.matrix import Matrix as Matrix
from .expression_tree.vector import Vector as Vector
from .expression_tree.tensor_field import TensorField as TensorField
from .expression_tree.vector_field import VectorField as VectorField

# Parameters
from .expression_tree.input_parameter import InputParameter as InputParameter
from .expression_tree.parameter import Parameter as Parameter
from .expression_tree.parameter import FunctionParameter as FunctionParameter
from .expression_tree.scalar import Scalar as Scalar
from .expression_tree.scalar import Constant as Constant
from .expression_tree.state_vector import StateVectorBase as StateVectorBase
from .expression_tree.state_vector import StateVector as StateVector
from .expression_tree.state_vector import StateVectorDot as StateVectorDot

# Operations
from .expression_tree.operations.evaluate_python import find_symbols as find_symbols
from .expression_tree.operations.evaluate_python import id_to_python_variable as id_to_python_variable
from .expression_tree.operations.evaluate_python import to_python as to_python
from .expression_tree.operations.evaluate_python import EvaluatorPython as EvaluatorPython
from .expression_tree.operations.evaluate_python import EvaluatorJax as EvaluatorJax
from .expression_tree.operations.evaluate_python import JaxCooMatrix as JaxCooMatrix
from .expression_tree.operations.jacobian import Jacobian as Jacobian
from .expression_tree.operations.convert_to_casadi import CasadiConverter as CasadiConverter
from .expression_tree.operations.unpack_symbols import SymbolUnpacker as SymbolUnpacker
from .expression_tree.operations.serialise import Serialise as Serialise
from .expression_tree.operations.serialise import ExpressionFunctionParameter as ExpressionFunctionParameter

# Model classes
from .models.base_model import BaseModel as BaseModel
from .models.base_model import ModelSolutionObservability as ModelSolutionObservability
from .models.base_model import load_model as load_model
from .models.symbol_processor import SymbolProcessor as SymbolProcessor
from .models.event import Event as Event
from .models.event import EventType as EventType

# Battery models (classes only - modules handled by custom __getattr__)
from .models.full_battery_models.base_battery_model import BaseBatteryModel as BaseBatteryModel
from .models.full_battery_models.base_battery_model import BatteryModelOptions as BatteryModelOptions

# Submodels (class only - modules handled by custom __getattr__)
from .models.submodels.base_submodel import BaseSubModel as BaseSubModel

# Geometry
from .geometry.geometry import Geometry as Geometry
from .geometry.battery_geometry import battery_geometry as battery_geometry

# Parameters
from .parameters.parameter_values import ParameterValues as ParameterValues
from .parameters.parameter_values import scalarize_dict as scalarize_dict
from .parameters.parameter_values import arrayize_dict as arrayize_dict
from .parameters.geometric_parameters import geometric_parameters as geometric_parameters
from .parameters.geometric_parameters import GeometricParameters as GeometricParameters
from .parameters.electrical_parameters import electrical_parameters as electrical_parameters
from .parameters.electrical_parameters import ElectricalParameters as ElectricalParameters
from .parameters.thermal_parameters import thermal_parameters as thermal_parameters
from .parameters.thermal_parameters import ThermalParameters as ThermalParameters
from .parameters.lithium_ion_parameters import LithiumIonParameters as LithiumIonParameters
from .parameters.lead_acid_parameters import LeadAcidParameters as LeadAcidParameters
from .parameters.ecm_parameters import EcmParameters as EcmParameters

# size_distribution_parameters
from .parameters.size_distribution_parameters import get_size_distribution_parameters as get_size_distribution_parameters
from .parameters.size_distribution_parameters import lognormal as lognormal

# Mesh and Discretisation
from .discretisations.discretisation import Discretisation as Discretisation
from .discretisations.discretisation import has_bc_of_form as has_bc_of_form
from .meshes.meshes import Mesh as Mesh
from .meshes.meshes import SubMesh as SubMesh
from .meshes.meshes import MeshGenerator as MeshGenerator
from .meshes.zero_dimensional_submesh import SubMesh0D as SubMesh0D
from .meshes.one_dimensional_submeshes import SubMesh1D as SubMesh1D
from .meshes.one_dimensional_submeshes import Uniform1DSubMesh as Uniform1DSubMesh
from .meshes.one_dimensional_submeshes import Exponential1DSubMesh as Exponential1DSubMesh
from .meshes.one_dimensional_submeshes import Chebyshev1DSubMesh as Chebyshev1DSubMesh
from .meshes.one_dimensional_submeshes import UserSupplied1DSubMesh as UserSupplied1DSubMesh
from .meshes.one_dimensional_submeshes import SpectralVolume1DSubMesh as SpectralVolume1DSubMesh
from .meshes.one_dimensional_submeshes import SymbolicUniform1DSubMesh as SymbolicUniform1DSubMesh
from .meshes.two_dimensional_submeshes import SubMesh2D as SubMesh2D
from .meshes.two_dimensional_submeshes import Uniform2DSubMesh as Uniform2DSubMesh
from .meshes.scikit_fem_submeshes import ScikitSubMesh2D as ScikitSubMesh2D
from .meshes.scikit_fem_submeshes import ScikitUniform2DSubMesh as ScikitUniform2DSubMesh
from .meshes.scikit_fem_submeshes import ScikitExponential2DSubMesh as ScikitExponential2DSubMesh
from .meshes.scikit_fem_submeshes import ScikitChebyshev2DSubMesh as ScikitChebyshev2DSubMesh
from .meshes.scikit_fem_submeshes import UserSupplied2DSubMesh as UserSupplied2DSubMesh
from .meshes.scikit_fem_submeshes_3d import ScikitFemSubMesh3D as ScikitFemSubMesh3D
from .meshes.scikit_fem_submeshes_3d import ScikitFemGenerator3D as ScikitFemGenerator3D
from .meshes.scikit_fem_submeshes_3d import UserSuppliedSubmesh3D as UserSuppliedSubmesh3D

# Spatial Methods
from .spatial_methods.spatial_method import SpatialMethod as SpatialMethod
from .spatial_methods.zero_dimensional_method import ZeroDimensionalSpatialMethod as ZeroDimensionalSpatialMethod
from .spatial_methods.finite_volume import FiniteVolume as FiniteVolume
from .spatial_methods.finite_volume_2d import FiniteVolume2D as FiniteVolume2D
from .spatial_methods.spectral_volume import SpectralVolume as SpectralVolume
from .spatial_methods.scikit_finite_element import ScikitFiniteElement as ScikitFiniteElement
from .spatial_methods.scikit_finite_element_3d import ScikitFiniteElement3D as ScikitFiniteElement3D

# Solvers
from .solvers.solution import Solution as Solution
from .solvers.solution import EmptySolution as EmptySolution
from .solvers.solution import make_cycle_solution as make_cycle_solution
from .solvers.processed_variable_time_integral import ProcessedVariableTimeIntegral as ProcessedVariableTimeIntegral
from .solvers.processed_variable import ProcessedVariable as ProcessedVariable
from .solvers.processed_variable import ProcessedVariable2DFVM as ProcessedVariable2DFVM
from .solvers.processed_variable import process_variable as process_variable
from .solvers.processed_variable import ProcessedVariableUnstructured as ProcessedVariableUnstructured
from .solvers.processed_variable_computed import ProcessedVariableComputed as ProcessedVariableComputed
from .solvers.summary_variable import SummaryVariables as SummaryVariables
from .solvers.base_solver import BaseSolver as BaseSolver
from .solvers.dummy_solver import DummySolver as DummySolver
from .solvers.algebraic_solver import AlgebraicSolver as AlgebraicSolver
from .solvers.casadi_solver import CasadiSolver as CasadiSolver
from .solvers.casadi_algebraic_solver import CasadiAlgebraicSolver as CasadiAlgebraicSolver
from .solvers.scipy_solver import ScipySolver as ScipySolver
from .solvers.jax_solver import JaxSolver as JaxSolver
from .solvers.jax_bdf_solver import jax_bdf_integrate as jax_bdf_integrate
from .solvers.idaklu_jax import IDAKLUJax as IDAKLUJax
from .solvers.idaklu_solver import IDAKLUSolver as IDAKLUSolver

# Experiments
from .experiment.experiment import Experiment as Experiment

# Plotting
from .plotting.quick_plot import QuickPlot as QuickPlot
from .plotting.quick_plot import close_plots as close_plots
from .plotting.quick_plot import QuickPlotAxes as QuickPlotAxes
from .plotting.plot import plot as plot
from .plotting.plot2D import plot2D as plot2D
from .plotting.plot_voltage_components import plot_voltage_components as plot_voltage_components
from .plotting.plot_thermal_components import plot_thermal_components as plot_thermal_components
from .plotting.plot_summary_variables import plot_summary_variables as plot_summary_variables
from .plotting.dynamic_plot import dynamic_plot as dynamic_plot
from .plotting.plot_3d_cross_section import plot_3d_cross_section as plot_3d_cross_section
from .plotting.plot_3d_heatmap import plot_3d_heatmap as plot_3d_heatmap

# Simulation
from .simulation import Simulation as Simulation
from .simulation import load_sim as load_sim
from .simulation import is_notebook as is_notebook

# Batch Study
from .batch_study import BatchStudy as BatchStudy

# Pybamm Data
from .pybamm_data import DataLoader as DataLoader

# Dispatch
from .dispatch import parameter_sets as parameter_sets
from .dispatch import Model as Model
