# Lazy loading implementation for faster import times
# Essential imports only - everything else is lazily loaded via __getattr__

from pybamm.version import __version__

# Core utilities that are lightweight and commonly needed
from .logger import logger, set_logging_level, get_new_logger
from .settings import settings
from . import config

# These need to be imported eagerly to shadow the submodule names
from .citations import Citations, citations, print_citations

# Expression tree modules are accessed thousands of times during model building
# so we eagerly import them to avoid __getattr__ overhead
from .expression_tree.symbol import *
from .expression_tree.binary_operators import *
from .expression_tree.concatenations import *
from .expression_tree.unary_operators import *
from .expression_tree.averages import *
from .expression_tree.averages import _BaseAverage
from .expression_tree.broadcasts import *
from .expression_tree.functions import *
from .expression_tree.interpolant import Interpolant  # needed before discrete_time_sum
from .expression_tree.discrete_time_sum import *
from .expression_tree.variable import *
from .expression_tree.coupled_variable import *
from .expression_tree.independent_variable import *
from .expression_tree.exceptions import *

# Lazy loading infrastructure
import importlib

# Modules to search for unknown attributes (fallback for lazy loading)
_WILDCARD_MODULES = [
    ".parameters.size_distribution_parameters",
]

# Explicit mapping for non-wildcard imports
# Format: "name": ("module_path", "attribute_name") or "name": "module_path" for modules
_LAZY_IMPORTS: dict[str, tuple[str, str] | str] = {
    # Utility classes and methods
    "root_dir": (".util", "root_dir"),
    "Timer": (".util", "Timer"),
    "TimerTime": (".util", "TimerTime"),
    "FuzzyDict": (".util", "FuzzyDict"),
    "load": (".util", "load"),
    "is_constant_and_can_evaluate": (".util", "is_constant_and_can_evaluate"),
    "get_parameters_filepath": (".util", "get_parameters_filepath"),
    "has_jax": (".util", "has_jax"),
    "import_optional_dependency": (".util", "import_optional_dependency"),
    "Abs": (".expression_tree.functions", "Abs"),  # Missing from functions
    # Array/Matrix
    "Array": (".expression_tree.array", "Array"),
    "linspace": (".expression_tree.array", "linspace"),
    "meshgrid": (".expression_tree.array", "meshgrid"),
    "Matrix": (".expression_tree.matrix", "Matrix"),
    "Vector": (".expression_tree.vector", "Vector"),
    "TensorField": (".expression_tree.tensor_field", "TensorField"),
    "VectorField": (".expression_tree.vector_field", "VectorField"),
    # Interpolant
    "Interpolant": (".expression_tree.interpolant", "Interpolant"),
    # Parameters
    "InputParameter": (".expression_tree.input_parameter", "InputParameter"),
    "Parameter": (".expression_tree.parameter", "Parameter"),
    "FunctionParameter": (".expression_tree.parameter", "FunctionParameter"),
    "Scalar": (".expression_tree.scalar", "Scalar"),
    "Constant": (".expression_tree.scalar", "Constant"),
    "StateVectorBase": (".expression_tree.state_vector", "StateVectorBase"),
    "StateVector": (".expression_tree.state_vector", "StateVector"),
    "StateVectorDot": (".expression_tree.state_vector", "StateVectorDot"),
    # Operations
    "find_symbols": (".expression_tree.operations.evaluate_python", "find_symbols"),
    "id_to_python_variable": (".expression_tree.operations.evaluate_python", "id_to_python_variable"),
    "to_python": (".expression_tree.operations.evaluate_python", "to_python"),
    "EvaluatorPython": (".expression_tree.operations.evaluate_python", "EvaluatorPython"),
    "EvaluatorJax": (".expression_tree.operations.evaluate_python", "EvaluatorJax"),
    "JaxCooMatrix": (".expression_tree.operations.evaluate_python", "JaxCooMatrix"),
    "Jacobian": (".expression_tree.operations.jacobian", "Jacobian"),
    "CasadiConverter": (".expression_tree.operations.convert_to_casadi", "CasadiConverter"),
    "SymbolUnpacker": (".expression_tree.operations.unpack_symbols", "SymbolUnpacker"),
    "Serialise": (".expression_tree.operations.serialise", "Serialise"),
    "ExpressionFunctionParameter": (".expression_tree.operations.serialise", "ExpressionFunctionParameter"),
    # Model classes
    "BaseModel": (".models.base_model", "BaseModel"),
    "ModelSolutionObservability": (".models.base_model", "ModelSolutionObservability"),
    "SymbolProcessor": (".models.symbol_processor", "SymbolProcessor"),
    "Event": (".models.event", "Event"),
    "EventType": (".models.event", "EventType"),
    "load_model": (".models.base_model", "load_model"),
    # Battery models (as modules)
    "BaseBatteryModel": (".models.full_battery_models.base_battery_model", "BaseBatteryModel"),
    "BatteryModelOptions": (".models.full_battery_models.base_battery_model", "BatteryModelOptions"),
    "lead_acid": ".models.full_battery_models.lead_acid",
    "lithium_ion": ".models.full_battery_models.lithium_ion",
    "equivalent_circuit": ".models.full_battery_models.equivalent_circuit",
    "sodium_ion": ".models.full_battery_models.sodium_ion",
    # Submodels
    "BaseSubModel": (".models.submodels.base_submodel", "BaseSubModel"),
    "active_material": ".models.submodels.active_material",
    "convection": ".models.submodels.convection",
    "current_collector": ".models.submodels.current_collector",
    "electrolyte_conductivity": ".models.submodels.electrolyte_conductivity",
    "electrolyte_diffusion": ".models.submodels.electrolyte_diffusion",
    "electrode": ".models.submodels.electrode",
    "external_circuit": ".models.submodels.external_circuit",
    "interface": ".models.submodels.interface",
    "oxygen_diffusion": ".models.submodels.oxygen_diffusion",
    "particle": ".models.submodels.particle",
    "porosity": ".models.submodels.porosity",
    "thermal": ".models.submodels.thermal",
    "transport_efficiency": ".models.submodels.transport_efficiency",
    "particle_mechanics": ".models.submodels.particle_mechanics",
    "equivalent_circuit_elements": ".models.submodels.equivalent_circuit_elements",
    "kinetics": ".models.submodels.interface.kinetics",
    "sei": ".models.submodels.interface.sei",
    "lithium_plating": ".models.submodels.interface.lithium_plating",
    "interface_utilisation": ".models.submodels.interface.interface_utilisation",
    "open_circuit_potential": ".models.submodels.interface.open_circuit_potential",
    # Geometry
    "Geometry": (".geometry.geometry", "Geometry"),
    "battery_geometry": (".geometry.battery_geometry", "battery_geometry"),
    "standard_spatial_vars": ".geometry.standard_spatial_vars",
    # Parameters
    "ParameterValues": (".parameters.parameter_values", "ParameterValues"),
    "scalarize_dict": (".parameters.parameter_values", "scalarize_dict"),
    "arrayize_dict": (".parameters.parameter_values", "arrayize_dict"),
    "constants": ".parameters.constants",
    "geometric_parameters": (".parameters.geometric_parameters", "geometric_parameters"),
    "GeometricParameters": (".parameters.geometric_parameters", "GeometricParameters"),
    "electrical_parameters": (".parameters.electrical_parameters", "electrical_parameters"),
    "ElectricalParameters": (".parameters.electrical_parameters", "ElectricalParameters"),
    "thermal_parameters": (".parameters.thermal_parameters", "thermal_parameters"),
    "ThermalParameters": (".parameters.thermal_parameters", "ThermalParameters"),
    "LithiumIonParameters": (".parameters.lithium_ion_parameters", "LithiumIonParameters"),
    "LeadAcidParameters": (".parameters.lead_acid_parameters", "LeadAcidParameters"),
    "EcmParameters": (".parameters.ecm_parameters", "EcmParameters"),
    # Mesh and Discretisation
    "Discretisation": (".discretisations.discretisation", "Discretisation"),
    "has_bc_of_form": (".discretisations.discretisation", "has_bc_of_form"),
    "Mesh": (".meshes.meshes", "Mesh"),
    "SubMesh": (".meshes.meshes", "SubMesh"),
    "MeshGenerator": (".meshes.meshes", "MeshGenerator"),
    "SubMesh0D": (".meshes.zero_dimensional_submesh", "SubMesh0D"),
    "SubMesh1D": (".meshes.one_dimensional_submeshes", "SubMesh1D"),
    "Uniform1DSubMesh": (".meshes.one_dimensional_submeshes", "Uniform1DSubMesh"),
    "Exponential1DSubMesh": (".meshes.one_dimensional_submeshes", "Exponential1DSubMesh"),
    "Chebyshev1DSubMesh": (".meshes.one_dimensional_submeshes", "Chebyshev1DSubMesh"),
    "UserSupplied1DSubMesh": (".meshes.one_dimensional_submeshes", "UserSupplied1DSubMesh"),
    "SpectralVolume1DSubMesh": (".meshes.one_dimensional_submeshes", "SpectralVolume1DSubMesh"),
    "SymbolicUniform1DSubMesh": (".meshes.one_dimensional_submeshes", "SymbolicUniform1DSubMesh"),
    "SubMesh2D": (".meshes.two_dimensional_submeshes", "SubMesh2D"),
    "Uniform2DSubMesh": (".meshes.two_dimensional_submeshes", "Uniform2DSubMesh"),
    "ScikitSubMesh2D": (".meshes.scikit_fem_submeshes", "ScikitSubMesh2D"),
    "ScikitUniform2DSubMesh": (".meshes.scikit_fem_submeshes", "ScikitUniform2DSubMesh"),
    "ScikitExponential2DSubMesh": (".meshes.scikit_fem_submeshes", "ScikitExponential2DSubMesh"),
    "ScikitChebyshev2DSubMesh": (".meshes.scikit_fem_submeshes", "ScikitChebyshev2DSubMesh"),
    "UserSupplied2DSubMesh": (".meshes.scikit_fem_submeshes", "UserSupplied2DSubMesh"),
    "ScikitFemSubMesh3D": (".meshes.scikit_fem_submeshes_3d", "ScikitFemSubMesh3D"),
    "ScikitFemGenerator3D": (".meshes.scikit_fem_submeshes_3d", "ScikitFemGenerator3D"),
    "UserSuppliedSubmesh3D": (".meshes.scikit_fem_submeshes_3d", "UserSuppliedSubmesh3D"),
    # Spatial Methods
    "SpatialMethod": (".spatial_methods.spatial_method", "SpatialMethod"),
    "ZeroDimensionalSpatialMethod": (".spatial_methods.zero_dimensional_method", "ZeroDimensionalSpatialMethod"),
    "FiniteVolume": (".spatial_methods.finite_volume", "FiniteVolume"),
    "FiniteVolume2D": (".spatial_methods.finite_volume_2d", "FiniteVolume2D"),
    "SpectralVolume": (".spatial_methods.spectral_volume", "SpectralVolume"),
    "ScikitFiniteElement": (".spatial_methods.scikit_finite_element", "ScikitFiniteElement"),
    "ScikitFiniteElement3D": (".spatial_methods.scikit_finite_element_3d", "ScikitFiniteElement3D"),
    # Solvers
    "Solution": (".solvers.solution", "Solution"),
    "EmptySolution": (".solvers.solution", "EmptySolution"),
    "make_cycle_solution": (".solvers.solution", "make_cycle_solution"),
    "ProcessedVariableTimeIntegral": (".solvers.processed_variable_time_integral", "ProcessedVariableTimeIntegral"),
    "ProcessedVariable": (".solvers.processed_variable", "ProcessedVariable"),
    "ProcessedVariable2DFVM": (".solvers.processed_variable", "ProcessedVariable2DFVM"),
    "process_variable": (".solvers.processed_variable", "process_variable"),
    "ProcessedVariableComputed": (".solvers.processed_variable_computed", "ProcessedVariableComputed"),
    "ProcessedVariableUnstructured": (".solvers.processed_variable", "ProcessedVariableUnstructured"),
    "SummaryVariables": (".solvers.summary_variable", "SummaryVariables"),
    "BaseSolver": (".solvers.base_solver", "BaseSolver"),
    "DummySolver": (".solvers.dummy_solver", "DummySolver"),
    "AlgebraicSolver": (".solvers.algebraic_solver", "AlgebraicSolver"),
    "CasadiSolver": (".solvers.casadi_solver", "CasadiSolver"),
    "CasadiAlgebraicSolver": (".solvers.casadi_algebraic_solver", "CasadiAlgebraicSolver"),
    "ScipySolver": (".solvers.scipy_solver", "ScipySolver"),
    "CompositeSolver": (".solvers.composite_solver", "CompositeSolver"),
    "JaxSolver": (".solvers.jax_solver", "JaxSolver"),
    "jax_bdf_integrate": (".solvers.jax_bdf_solver", "jax_bdf_integrate"),
    "IDAKLUJax": (".solvers.idaklu_jax", "IDAKLUJax"),
    "IDAKLUSolver": (".solvers.idaklu_solver", "IDAKLUSolver"),
    # Experiments
    "Experiment": (".experiment.experiment", "Experiment"),
    "experiment": ".experiment",
    "step": ".experiment.step",
    # Plotting
    "QuickPlot": (".plotting.quick_plot", "QuickPlot"),
    "close_plots": (".plotting.quick_plot", "close_plots"),
    "QuickPlotAxes": (".plotting.quick_plot", "QuickPlotAxes"),
    "plot": (".plotting.plot", "plot"),
    "plot2D": (".plotting.plot2D", "plot2D"),
    "plot_voltage_components": (".plotting.plot_voltage_components", "plot_voltage_components"),
    "plot_thermal_components": (".plotting.plot_thermal_components", "plot_thermal_components"),
    "plot_summary_variables": (".plotting.plot_summary_variables", "plot_summary_variables"),
    "dynamic_plot": (".plotting.dynamic_plot", "dynamic_plot"),
    "plot_3d_cross_section": (".plotting.plot_3d_cross_section", "plot_3d_cross_section"),
    "plot_3d_heatmap": (".plotting.plot_3d_heatmap", "plot_3d_heatmap"),
    # Simulation
    "Simulation": (".simulation", "Simulation"),
    "load_sim": (".simulation", "load_sim"),
    "is_notebook": (".simulation", "is_notebook"),
    # Batch Study
    "BatchStudy": (".batch_study", "BatchStudy"),
    # Callbacks and telemetry
    "callbacks": ".callbacks",
    "telemetry": ".telemetry",
    # Pybamm Data
    "DataLoader": (".pybamm_data", "DataLoader"),
    # Dispatch
    "parameter_sets": (".dispatch", "parameter_sets"),
    "Model": (".dispatch", "Model"),
}

# Cache for already-loaded lazy imports
_loaded_attrs: dict[str, object] = {}
# Cache for loaded wildcard modules
_loaded_wildcard_modules: dict[str, object] = {}


def __getattr__(name: str) -> object:
    """Lazily import attributes on first access."""
    # Check cache first
    if name in _loaded_attrs:
        return _loaded_attrs[name]

    # Check explicit lazy imports
    if name in _LAZY_IMPORTS:
        import_info = _LAZY_IMPORTS[name]

        if isinstance(import_info, str):
            # It's a module import
            module = importlib.import_module(import_info, package="pybamm")
            _loaded_attrs[name] = module
            return module
        else:
            # It's an attribute from a module
            module_path, attr_name = import_info
            module = importlib.import_module(module_path, package="pybamm")
            attr = getattr(module, attr_name)
            _loaded_attrs[name] = attr
            return attr

    # Fall back to searching wildcard modules
    for module_path in _WILDCARD_MODULES:
        if module_path not in _loaded_wildcard_modules:
            try:
                _loaded_wildcard_modules[module_path] = importlib.import_module(
                    module_path, package="pybamm"
                )
            except ImportError:
                _loaded_wildcard_modules[module_path] = None
                continue

        module = _loaded_wildcard_modules[module_path]
        if module is not None and hasattr(module, name):
            attr = getattr(module, name)
            _loaded_attrs[name] = attr
            return attr

    raise AttributeError(f"module 'pybamm' has no attribute {name!r}")


def __dir__() -> list[str]:
    """List all available attributes including lazy ones."""
    module_attrs = list(globals().keys())
    lazy_attrs = list(_LAZY_IMPORTS.keys())

    # Also get attrs from wildcard modules
    wildcard_attrs = []
    for module_path in _WILDCARD_MODULES:
        if module_path not in _loaded_wildcard_modules:
            try:
                _loaded_wildcard_modules[module_path] = importlib.import_module(
                    module_path, package="pybamm"
                )
            except ImportError:
                _loaded_wildcard_modules[module_path] = None
                continue

        module = _loaded_wildcard_modules[module_path]
        if module is not None:
            wildcard_attrs.extend(
                n for n in dir(module) if not n.startswith("_")
            )

    return sorted(set(module_attrs) | set(lazy_attrs) | set(wildcard_attrs))


# Fix Casadi import - this needs to happen at import time
import os
import pathlib
import sysconfig

os.environ["CASADIPATH"] = str(pathlib.Path(sysconfig.get_path("purelib")) / "casadi")

__all__ = [
    "batch_study",
    "callbacks",
    "citations",
    "config",
    "discretisations",
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
    "telemetry",
    "type_definitions",
    "util",
    "version",
    "pybamm_data",
    "dispatch",
]

config.generate()
