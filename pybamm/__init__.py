import sys

from pybamm.version import __version__

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
from .solvers.idaklu_solver import IDAKLUSolver, have_idaklu

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

# Remove any imported modules, so we don't expose them as part of pybamm
del sys

__all__ = ['AbsoluteValue', 'Addition', 'Ai2020', 'AlgebraicSolver',
           'AlternativeEffectiveResistance2D', 'Arcsinh', 'Arctan', 'Array',
           'AsymmetricButlerVolmer', 'BackwardIndefiniteIntegral',
           'BaseBatteryModel', 'BaseEffectiveResistance', 'BaseElectrode',
           'BaseElectrolyteConductivity', 'BaseElectrolyteDiffusion',
           'BaseIndefiniteIntegral', 'BaseInterface', 'BaseKinetics',
           'BaseLeadingOrderSurfaceForm', 'BaseMechanics', 'BaseModel',
           'BaseOpenCircuitPotential', 'BaseParameters', 'BaseParticle',
           'BasePlating', 'BasePotentialPair', 'BaseSolver', 'BaseStep',
           'BaseStepExplicit', 'BaseStepImplicit', 'BaseSubModel',
           'BaseTermination', 'BaseThermal', 'BaseThroughCellModel',
           'BaseTransverseModel', 'BasicDFN', 'BasicDFNComposite',
           'BasicDFNHalfCell', 'BasicFull', 'BasicSPM', 'BatchStudy',
           'BatteryModelDomainOptions', 'BatteryModelOptions',
           'BatteryModelPhaseOptions', 'BinaryOperator',
           'BoundaryConditionsDict', 'BoundaryGradient', 'BoundaryIntegral',
           'BoundaryMass', 'BoundaryOperator', 'BoundaryValue', 'Broadcast',
           'Bruggeman', 'CCCVFunctionControl', 'CRate', 'Callback',
           'CallbackList', 'CasadiAlgebraicSolver', 'CasadiConverter',
           'CasadiSolver', 'Ceiling', 'Chebyshev1DSubMesh', 'Chen2020',
           'Chen2020_composite', 'Citations', 'Composite',
           'CompositeAlgebraic', 'CompositeDifferential', 'Concatenation',
           'ConcatenationVariable', 'Constant', 'ConstantConcentration',
           'ConstantSEI', 'Cos', 'Cosh', 'CrackPropagation',
           'CrateTermination', 'Current', 'CurrentCollector1D',
           'CurrentCollector2D', 'CurrentDriven',
           'CurrentForInverseButlerVolmer',
           'CurrentForInverseButlerVolmerLithiumMetal',
           'CurrentSigmoidOpenCircuitPotential', 'CurrentTermination',
           'CustomPrint', 'CustomStepExplicit', 'CustomStepImplicit',
           'CustomTermination', 'DFN', 'DOMAIN_LEVELS',
           'DefiniteIntegralVector', 'DeltaFunction', 'DiffusionLimited',
           'Discretisation', 'DiscretisationError', 'Divergence', 'Division',
           'Domain', 'DomainConcatenation', 'DomainError',
           'DomainGeometricParameters', 'DomainLeadAcidParameters',
           'DomainLithiumIonParameters', 'DomainThermalParameters', 'Downwind',
           'DummySolver', 'E', 'Ecker2015', 'Ecker2015_graphite_halfcell',
           'EcmParameters', 'EffectiveResistance', 'ElectricalParameters',
           'ElectrodeSOHHalfCell', 'ElectrodeSOHSolver', 'EmptySolution',
           'EqualHeaviside', 'Equality', 'EquationDict', 'Erf', 'EvaluateAt',
           'EvaluatorJax', 'EvaluatorJaxJacobian', 'EvaluatorJaxSensitivities',
           'EvaluatorPython', 'Event', 'EventType', 'Exp', 'Experiment',
           'Explicit', 'ExplicitCurrentControl', 'ExplicitPowerControl',
           'ExplicitResistanceControl', 'ExplicitTimeIntegral',
           'Exponential1DSubMesh', 'F', 'FORMAT', 'FickianDiffusion',
           'FiniteVolume', 'Floor', 'ForwardTafel', 'Full', 'FullAlgebraic',
           'FullBroadcast', 'FullBroadcastToEdges', 'FullDifferential',
           'Function', 'FunctionControl', 'FunctionParameter', 'FuzzyDict',
           'GREEK_LETTERS', 'GeometricParameters', 'Geometry', 'GeometryError',
           'Gradient', 'GradientSquared', 'IDAKLUJax', 'IDAKLUSolver',
           'IndefiniteIntegral', 'IndependentVariable', 'Index', 'Inner',
           'InputParameter', 'Integral', 'Integrated', 'Interpolant',
           'InverseButlerVolmer', 'Isothermal', 'JAXLIB_VERSION',
           'JAX_VERSION', 'Jacobian', 'JaxCooMatrix', 'JaxSolver',
           'KNOWN_COORD_SYS',
           'LFP_electrolyte_exchange_current_density_kashkooli2017',
           'LFP_ocp_Afshar2017', 'LOG_FORMATTER', 'LOQS', 'LRUDict',
           'Laplacian', 'Latexify', 'LeadAcidParameters', 'LeadingOrder',
           'LeadingOrderAlgebraic', 'LeadingOrderDifferential', 'Linear',
           'LithiumIonParameters', 'LithiumMetalBaseModel',
           'LithiumMetalExplicit', 'LithiumMetalSurfaceForm', 'Log',
           'LoggingCallback', 'LoopList', 'LossActiveMaterial', 'Lumped',
           'MPM', 'MSMR', 'MSMRButlerVolmer', 'MSMRDiffusion',
           'MSMROpenCircuitPotential', 'MSMR_example_set', 'Marcus',
           'MarcusHushChidsey', 'Marquis2019', 'Mass', 'Matrix',
           'MatrixMultiplication', 'Max', 'Maximum', 'Mesh', 'MeshGenerator',
           'Min', 'Minimum', 'ModelError', 'ModelWarning', 'Modulo',
           'Mohtat2020', 'Multiplication', 'NCA_Kim2011',
           'NMC_diffusivity_PeymanMPM',
           'NMC_electrolyte_exchange_current_density_PeymanMPM',
           'NMC_entropic_change_PeymanMPM', 'NMC_ocp_PeymanMPM',
           'NaturalNumberOption', 'Negate', 'NewmanTobias', 'NoConvection',
           'NoMechanics', 'NoOxygen', 'NoPlating', 'NoReaction', 'NoSEI',
           'NotConstant', 'NotEqualHeaviside', 'NullParameters',
           'NumpyConcatenation', 'NumpyEncoder', 'OCVElement', 'OKane2022',
           'OKane2022_graphite_SiOx_halfcell', 'ORegan2022', 'OneDimensionalX',
           'OperatingModes', 'OptionError', 'OptionWarning', 'PHASE_NAMES',
           'PRINT_NAME_OVERRIDES', 'Parameter', 'ParameterSets',
           'ParameterValues', 'ParticleGeometricParameters',
           'ParticleLithiumIonParameters', 'PhaseLeadAcidParameters',
           'Plating', 'PolynomialProfile', 'PotentialPair1plus1D',
           'PotentialPair2plus1D', 'Power', 'PowerFunctionControl',
           'Prada2013', 'PrimaryBroadcast', 'PrimaryBroadcastToEdges',
           'ProcessedVariable', 'ProcessedVariableComputed', 'QuickPlot',
           'QuickPlotAxes', 'R', 'RAverage', 'RCElement', 'R_n', 'R_n_edge',
           'R_p', 'R_p_edge', 'Ramadass2004', 'ReactionDriven',
           'ReactionDrivenODE', 'Resistance', 'ResistanceFunctionControl',
           'ResistorElement', 'SEIGrowth',
           'SEI_limited_dead_lithium_OKane2022', 'SF', 'SPM', 'SPMe', 'Scalar',
           'ScikitChebyshev2DSubMesh', 'ScikitExponential2DSubMesh',
           'ScikitFiniteElement', 'ScikitSubMesh2D', 'ScikitUniform2DSubMesh',
           'ScipySolver', 'SecondaryBroadcast', 'SecondaryBroadcastToEdges',
           'Serialise', 'Settings', 'ShapeError', 'Sign', 'Simulation', 'Sin',
           'SingleOpenCircuitPotential', 'Sinh', 'SizeAverage', 'Solution',
           'SolverError', 'SolverWarning', 'SparseStack', 'SpatialMethod',
           'SpatialOperator', 'SpatialVariable', 'SpatialVariableEdge',
           'SpecificFunction', 'SpectralVolume', 'SpectralVolume1DSubMesh',
           'Sqrt', 'StateVector', 'StateVectorBase', 'StateVectorDot',
           'SubMesh', 'SubMesh0D', 'SubMesh1D', 'Subtraction', 'Sulzer2019',
           'SurfaceForm', 'SwellingOnly', 'Symbol', 'SymbolUnpacker',
           'SymmetricButlerVolmer', 'Tanh', 'TertiaryBroadcast',
           'TertiaryBroadcastToEdges', 'ThermalParameters', 'ThermalSubModel',
           'Thevenin', 'Time', 'Timer', 'TimerTime', 'Total',
           'TotalConcentration', 'TotalInterfacialCurrent',
           'TotalMainKinetics', 'TotalSEI', 'UnaryOperator', 'Uniform',
           'Uniform1DSubMesh', 'Upwind', 'UpwindDownwind',
           'UserSupplied1DSubMesh', 'UserSupplied2DSubMesh', 'Variable',
           'VariableBase', 'VariableDot', 'Vector', 'Voltage',
           'VoltageFunctionControl', 'VoltageModel', 'VoltageTermination',
           'XAverage', 'XAveragedPolynomialProfile', 'Xu2019', 'YZAverage',
           'Yang2017', 'ZAverage', 'ZeroDimensionalSpatialMethod',
           'active_material', 'add', 'algebraic_solver',
           'aluminium_heat_capacity_CRC', 'arcsinh', 'arctan', 'array',
           'averages', 'ax_max', 'ax_min', 'base_active_material',
           'base_battery_model', 'base_convection', 'base_current_collector',
           'base_electrode', 'base_electrolyte_conductivity',
           'base_electrolyte_diffusion', 'base_external_circuit',
           'base_interface', 'base_kinetics', 'base_lead_acid_model',
           'base_lithium_ion_model', 'base_mechanics', 'base_model',
           'base_ocp', 'base_ohm', 'base_oxygen_diffusion', 'base_parameters',
           'base_particle', 'base_plating', 'base_porosity', 'base_sei',
           'base_solver', 'base_step', 'base_submodel', 'base_thermal',
           'base_through_cell_convection', 'base_transport_efficiency',
           'base_transverse_convection', 'base_utilisation', 'basic_dfn',
           'basic_dfn_composite', 'basic_dfn_half_cell', 'basic_full',
           'basic_spm', 'batch_study', 'battery_geometry', 'binary_operators',
           'boundary_gradient', 'boundary_value', 'bpx', 'broadcasts',
           'bruggeman_transport_efficiency', 'butler_volmer', 'c1', 'c1_data',
           'c_rate', 'c_solvers', 'calculate_theoretical_energy',
           'callback_loop_decorator', 'callbacks', 'casadi_algebraic_solver',
           'casadi_solver', 'cell', 'citations', 'close_plots',
           'composite_conductivity', 'composite_ohm',
           'composite_surface_form_conductivity', 'concatenation',
           'concatenations', 'conductivity_Gu1997', 'constant_active_material',
           'constant_concentration', 'constant_porosity', 'constant_sei',
           'constant_utilisation', 'constants', 'convection',
           'convert_to_casadi', 'copper_heat_capacity_CRC',
           'copper_thermal_conductivity_CRC', 'copy_parameter_doc_from_parent',
           'cos', 'cosh', 'crack_propagation', 'cracking_rate_Ai2020',
           'create_jax_coo_matrix', 'create_object_of_size', 'current',
           'current_collector', 'current_driven_utilisation',
           'current_sigmoid_ocp', 'custom_print_func', 'dUdT', 'dUdT_data',
           'darken_thermodynamic_factor_Chapman1968', 'dfn',
           'diffusion_limited', 'diffusivity_Gu1997', 'discretisation',
           'discretisations', 'div', 'divide', 'dlnf_dlnc_Ai2020',
           'doc_extend_parent', 'doc_utils', 'domain_concatenation',
           'domain_size', 'downwind', 'dummy_solver', 'dynamic_plot', 'ecm',
           'ecm_model_options', 'ecm_parameters',
           'effective_resistance_current_collector', 'electrical_parameters',
           'electrode', 'electrode_soh', 'electrode_soh_half_cell',
           'electrolyte', 'electrolyte_TDF_EC_EMC_3_7_Landesfeind2019',
           'electrolyte_TDF_base_Landesfeind2019', 'electrolyte_conductivity',
           'electrolyte_conductivity_Ai2020',
           'electrolyte_conductivity_Capiglia1999',
           'electrolyte_conductivity_EC_EMC_3_7_Landesfeind2019',
           'electrolyte_conductivity_Ecker2015',
           'electrolyte_conductivity_Kim2011',
           'electrolyte_conductivity_Nyman2008',
           'electrolyte_conductivity_Nyman2008_arrhenius',
           'electrolyte_conductivity_PeymanMPM',
           'electrolyte_conductivity_Prada2013',
           'electrolyte_conductivity_Ramadass2004',
           'electrolyte_conductivity_Valoen2005',
           'electrolyte_conductivity_base_Landesfeind2019',
           'electrolyte_diffusion', 'electrolyte_diffusivity_Ai2020',
           'electrolyte_diffusivity_Capiglia1999',
           'electrolyte_diffusivity_EC_EMC_3_7_Landesfeind2019',
           'electrolyte_diffusivity_Ecker2015',
           'electrolyte_diffusivity_Kim2011',
           'electrolyte_diffusivity_Nyman2008',
           'electrolyte_diffusivity_Nyman2008_arrhenius',
           'electrolyte_diffusivity_PeymanMPM',
           'electrolyte_diffusivity_Ramadass2004',
           'electrolyte_diffusivity_Valoen2005',
           'electrolyte_diffusivity_base_Landesfeind2019',
           'electrolyte_transference_number_EC_EMC_3_7_Landesfeind2019',
           'electrolyte_transference_number_base_Landesfeind2019',
           'equivalent_circuit', 'equivalent_circuit_elements', 'erf', 'erfc',
           'evaluate_for_shape_using_domain', 'evaluate_python', 'event',
           'example_set', 'exceptions', 'exp', 'experiment',
           'explicit_control_external_circuit', 'explicit_convection',
           'explicit_surface_form_conductivity', 'expression_tree',
           'external_circuit', 'fickian_diffusion', 'find_symbol_in_dict',
           'find_symbol_in_model', 'find_symbol_in_tree', 'find_symbols',
           'finite_volume', 'full', 'full_battery_models', 'full_conductivity',
           'full_convection', 'full_diffusion', 'full_like', 'full_ohm',
           'full_oxygen_diffusion', 'full_surface_form_conductivity',
           'full_utilisation', 'function_control_external_circuit',
           'functions', 'geometric_parameters', 'geometry',
           'get_git_commit_info', 'get_initial_ocps',
           'get_initial_stoichiometries',
           'get_initial_stoichiometry_half_cell', 'get_log_level_func',
           'get_min_max_ocps', 'get_min_max_stoichiometries', 'get_new_logger',
           'get_parameter_values', 'get_parameters_filepath',
           'get_rng_min_max_name', 'get_size_distribution_parameters', 'grad',
           'grad_squared', 'graphite_LGM50_diffusivity_Chen2020',
           'graphite_LGM50_diffusivity_ORegan2022',
           'graphite_LGM50_electrolyte_exchange_current_density_Chen2020',
           'graphite_LGM50_electrolyte_exchange_current_density_ORegan2022',
           'graphite_LGM50_entropic_change_ORegan2022',
           'graphite_LGM50_heat_capacity_ORegan2022',
           'graphite_LGM50_ocp_Chen2020', 'graphite_LGM50_ocp_Chen2020_data',
           'graphite_LGM50_thermal_conductivity_ORegan2022',
           'graphite_cracking_rate_Ai2020',
           'graphite_diffusivity_Dualfoil1998',
           'graphite_diffusivity_Ecker2015', 'graphite_diffusivity_Kim2011',
           'graphite_diffusivity_PeymanMPM',
           'graphite_electrolyte_exchange_current_density_Dualfoil1998',
           'graphite_electrolyte_exchange_current_density_Ecker2015',
           'graphite_electrolyte_exchange_current_density_Kim2011',
           'graphite_electrolyte_exchange_current_density_PeymanMPM',
           'graphite_electrolyte_exchange_current_density_Ramadass2004',
           'graphite_entropic_change_Moura2016',
           'graphite_entropic_change_PeymanMPM',
           'graphite_entropy_Enertech_Ai2020_function',
           'graphite_mcmb2528_diffusivity_Dualfoil1998',
           'graphite_mcmb2528_ocp_Dualfoil1998', 'graphite_ocp_Ecker2015',
           'graphite_ocp_Enertech_Ai2020', 'graphite_ocp_Enertech_Ai2020_data',
           'graphite_ocp_Kim2011', 'graphite_ocp_PeymanMPM',
           'graphite_ocp_Ramadass2004', 'graphite_volume_change_Ai2020',
           'has_bc_of_form', 'have_idaklu', 'have_jax',
           'homogeneous_current_collector', 'id_to_python_variable',
           'idaklu_jax', 'idaklu_solver', 'idaklu_spec',
           'import_optional_dependency', 'independent_variable', 'inner',
           'input', 'input_parameter', 'install_jax',
           'integrated_conductivity', 'interface', 'interface_utilisation',
           'interpolant', 'intersect', 'inverse_butler_volmer',
           'inverse_kinetics', 'is_constant', 'is_constant_and_can_evaluate',
           'is_jax_compatible', 'is_matrix_minus_one', 'is_matrix_one',
           'is_matrix_x', 'is_matrix_zero', 'is_notebook', 'is_scalar',
           'is_scalar_minus_one', 'is_scalar_one', 'is_scalar_x',
           'is_scalar_zero', 'isothermal', 'jacobian', 'jax_bdf_integrate',
           'jax_bdf_solver', 'jax_solver', 'k_b', 'kinetics', 'laplacian',
           'latexify', 'lead_acid', 'lead_acid_parameters',
           'lead_dioxide_exchange_current_density_Sulzer2019',
           'lead_dioxide_ocp_Bode1977',
           'lead_exchange_current_density_Sulzer2019', 'lead_ocp_Bode1977',
           'leading_ohm', 'leading_order_conductivity',
           'leading_order_diffusion', 'leading_oxygen_diffusion',
           'leading_surface_form_conductivity', 'li_metal',
           'li_metal_electrolyte_exchange_current_density_Xu2019',
           'lico2_cracking_rate_Ai2020', 'lico2_diffusivity_Dualfoil1998',
           'lico2_diffusivity_Ramadass2004',
           'lico2_electrolyte_exchange_current_density_Dualfoil1998',
           'lico2_electrolyte_exchange_current_density_Ramadass2004',
           'lico2_entropic_change_Ai2020_function',
           'lico2_entropic_change_Moura2016', 'lico2_ocp_Ai2020',
           'lico2_ocp_Ai2020_data', 'lico2_ocp_Dualfoil1998',
           'lico2_ocp_Ramadass2004', 'lico2_volume_change_Ai2020', 'linear',
           'linspace', 'lithium_ion', 'lithium_ion_parameters',
           'lithium_plating', 'load', 'load_model', 'load_sim', 'log', 'log10',
           'logger', 'lognormal', 'loqs', 'loss_active_material', 'lrudict',
           'lumped', 'make_cycle_solution', 'marcus', 'matmul', 'matrix',
           'maximum', 'meshes', 'meshgrid', 'minimum', 'models', 'mpm', 'msmr',
           'msmr_butler_volmer', 'msmr_diffusion', 'msmr_ocp', 'multiply',
           'nca_diffusivity_Kim2011',
           'nca_electrolyte_exchange_current_density_Kim2011',
           'nca_ocp_Kim2011', 'nco_diffusivity_Ecker2015',
           'nco_electrolyte_exchange_current_density_Ecker2015',
           'nco_ocp_Ecker2015', 'negative_current_collector',
           'negative_electrode', 'negative_particle', 'new_levels',
           'newman_tobias', 'nmc_LGM50_diffusivity_Chen2020',
           'nmc_LGM50_diffusivity_ORegan2022',
           'nmc_LGM50_electrolyte_exchange_current_density_Chen2020',
           'nmc_LGM50_electrolyte_exchange_current_density_ORegan2022',
           'nmc_LGM50_electronic_conductivity_ORegan2022',
           'nmc_LGM50_entropic_change_ORegan2022',
           'nmc_LGM50_heat_capacity_ORegan2022', 'nmc_LGM50_ocp_Chen2020',
           'nmc_LGM50_thermal_conductivity_ORegan2022',
           'nmc_electrolyte_exchange_current_density_Xu2019', 'nmc_ocp_Xu2019',
           'no_convection', 'no_mechanics', 'no_oxygen', 'no_plating',
           'no_reaction', 'no_sei', 'normal_cdf', 'normal_pdf',
           'numpy_concatenation', 'ocv', 'ocv_data', 'ocv_element', 'ohm',
           'one_dimensional_submeshes', 'ones_like', 'open_circuit_potential',
           'operations', 'oxygen_diffusion',
           'oxygen_exchange_current_density_Sulzer2019', 'parameter',
           'parameter_sets', 'parameter_values', 'parameters', 'particle',
           'particle_mechanics', 'plating',
           'plating_exchange_current_density_OKane2020', 'plot', 'plot2D',
           'plot_summary_variables', 'plot_thermal_components',
           'plot_voltage_components', 'plotting', 'polynomial_profile',
           'porosity', 'positive_current_collector', 'positive_electrode',
           'positive_particle', 'potential_pair', 'pouch_cell',
           'pouch_cell_1D_current_collectors',
           'pouch_cell_2D_current_collectors', 'power', 'preamble',
           'prettify_print_name', 'print_citations', 'print_name', 'printing',
           'process', 'process_1D_data', 'process_2D_data',
           'process_2D_data_csv', 'process_3D_data_csv',
           'process_float_function_table', 'process_parameter_data',
           'processed_variable', 'processed_variable_computed', 'q_e',
           'quick_plot', 'r0', 'r0_data', 'r1', 'r1_data', 'r_average',
           'r_macro', 'r_macro_edge', 'r_n', 'r_n_edge', 'r_n_prim', 'r_n_sec',
           'r_p', 'r_p_edge', 'r_p_prim', 'r_p_sec', 'rc_element',
           'reaction_driven_porosity', 'reaction_driven_porosity_ode',
           'represents_positive_integer', 'resistance', 'resistor_element',
           'rest', 'root_dir', 'scalar', 'scikit_fem_submeshes',
           'scikit_finite_element', 'scipy_solver', 'sech', 'sei',
           'sei_growth', 'separator',
           'separator_LGM50_heat_capacity_ORegan2022', 'serialise',
           'set_logging_level', 'settings', 'setup_callbacks', 'sigmoid',
           'sign',
           'silicon_LGM50_electrolyte_exchange_current_density_Chen2020',
           'silicon_ocp_delithiation_Mark2016',
           'silicon_ocp_lithiation_Mark2016', 'simplified_concatenation',
           'simplified_domain_concatenation', 'simplified_function',
           'simplified_numpy_concatenation', 'simplified_power',
           'simplify_if_constant', 'simulation', 'sin', 'single_ocp', 'sinh',
           'size_average', 'size_distribution_parameters',
           'smooth_absolute_value', 'smooth_max', 'smooth_min', 'softminus',
           'softplus', 'solution', 'solvers', 'source', 'spatial_method',
           'spatial_methods', 'spectral_volume', 'split_long_string', 'spm',
           'spme', 'sqrt', 'standard_spatial_vars', 'state_vector', 'step',
           'step_termination', 'steps', 'string',
           'stripping_exchange_current_density_OKane2020', 'submodels',
           'substrings', 'subtract', 'surf', 'surface_form_ohm',
           'surface_potential_form', 'swelling_only', 'symbol',
           'sympy_overrides', 't', 'tafel', 'tanh',
           'theoretical_energy_integral', 'thermal', 'thermal_parameters',
           'thevenin', 'through_cell', 'to_python', 'total_active_material',
           'total_interfacial_current', 'total_main_kinetics',
           'total_particle_concentration', 'total_sei', 'transport_efficiency',
           'transverse', 'type_definitions', 'unary_operators',
           'uniform_convection', 'unpack_symbols', 'upwind', 'util',
           'value_based_charge_or_discharge', 'variable', 'vector', 'version',
           'voltage', 'voltage_model', 'volume_change_Ai2020', 'whole_cell',
           'x', 'x_average', 'x_averaged_polynomial_profile', 'x_edge',
           'x_full', 'x_n', 'x_n_edge', 'x_p', 'x_p_edge', 'x_s', 'x_s_edge',
           'xyz_average', 'y', 'y_edge', 'yz_average', 'z', 'z_average',
           'z_edge', 'zero_dimensional_method', 'zero_dimensional_submesh',
           'zeros_like']
