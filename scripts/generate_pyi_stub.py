#!/usr/bin/env python
"""
Generate the __init__.pyi stub file for PyBaMM.

This script generates the stub file that enables:
1. IDE autocomplete for lazy-loaded attributes
2. Type checking support
3. lazy_loader integration for lazy imports

Usage:
    python scripts/generate_pyi_stub.py [--check]

Options:
    --check     Don't write the file, just check if it would change (for CI)
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from pathlib import Path
from typing import NamedTuple

# Add src to path for imports
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


class ImportEntry(NamedTuple):
    """Represents a single import in the stub file."""

    module_path: str  # e.g., ".util" or ".solvers.casadi_solver"
    attr_name: str  # e.g., "Timer" or "CasadiSolver"
    alias: str | None = None  # If different from attr_name


# ============================================================================
# CONFIGURATION: Define what gets exported from pybamm
# ============================================================================

# Eagerly loaded items (imported at module load time)
# These are imported via wildcard (*) or explicit imports in __init__.py
EAGER_IMPORTS: dict[str, list[str]] = {
    # Logger and settings
    ".logger": ["logger", "set_logging_level", "get_new_logger"],
    ".settings": ["settings"],
    ".citations": ["Citations", "citations", "print_citations"],
    # Expression tree modules (wildcard imports)
    ".expression_tree.symbol": [
        "Symbol",
        "domain_size",
        "create_object_of_size",
        "evaluate_for_shape_using_domain",
        "is_constant",
        "is_scalar_zero",
        "is_scalar_one",
        "is_scalar_minus_one",
        "is_matrix_zero",
        "is_matrix_one",
        "is_matrix_minus_one",
        "simplify_if_constant",
        "convert_to_symbol",
    ],
    ".expression_tree.binary_operators": [
        "BinaryOperator",
        "Power",
        "Addition",
        "Subtraction",
        "Multiplication",
        "KroneckerProduct",
        "TensorProduct",
        "MatrixMultiplication",
        "Division",
        "Inner",
        "Equality",
        "EqualHeaviside",
        "NotEqualHeaviside",
        "Modulo",
        "Minimum",
        "Maximum",
        "softplus",
        "softminus",
        "sigmoid",
        "source",
    ],
    ".expression_tree.concatenations": [
        "Concatenation",
        "NumpyConcatenation",
        "DomainConcatenation",
        "SparseStack",
        "ConcatenationVariable",
        "concatenation",
        "numpy_concatenation",
    ],
    ".expression_tree.unary_operators": [
        "UnaryOperator",
        "Negate",
        "AbsoluteValue",
        "Transpose",
        "Sign",
        "Floor",
        "Ceiling",
        "Index",
        "SpatialOperator",
        "Gradient",
        "Divergence",
        "Laplacian",
        "GradientSquared",
        "Mass",
        "BoundaryMass",
        "Integral",
        "BaseIndefiniteIntegral",
        "IndefiniteIntegral",
        "BackwardIndefiniteIntegral",
        "DefiniteIntegralVector",
        "BoundaryIntegral",
        "OneDimensionalIntegral",
        "DeltaFunction",
        "BoundaryOperator",
        "BoundaryValue",
        "BoundaryMeshSize",
        "ExplicitTimeIntegral",
        "BoundaryGradient",
        "EvaluateAt",
        "UpwindDownwind",
        "UpwindDownwind2D",
        "NodeToEdge2D",
        "Magnitude",
        "Upwind",
        "Downwind",
        "NotConstant",
        "grad",
        "div",
        "laplacian",
        "grad_squared",
        "surf",
        "boundary_value",
        "sign",
        "upwind",
        "downwind",
    ],
    ".expression_tree.averages": [
        "_BaseAverage",
        "XAverage",
        "YZAverage",
        "ZAverage",
        "RAverage",
        "SizeAverage",
        "x_average",
        "yz_average",
        "z_average",
        "r_average",
        "size_average",
    ],
    ".expression_tree.broadcasts": [
        "Broadcast",
        "PrimaryBroadcast",
        "PrimaryBroadcastToEdges",
        "SecondaryBroadcast",
        "SecondaryBroadcastToEdges",
        "TertiaryBroadcast",
        "TertiaryBroadcastToEdges",
        "FullBroadcast",
        "FullBroadcastToEdges",
        "ones_like",
        "zeros_like",
    ],
    ".expression_tree.functions": [
        "Function",
        "SpecificFunction",
        "Arcsinh",
        "Arctan",
        "Cos",
        "Cosh",
        "Erf",
        "Exp",
        "Log",
        "Max",
        "Min",
        "Sin",
        "Sinh",
        "Sqrt",
        "Tanh",
        "arcsinh",
        "arctan",
        "cos",
        "cosh",
        "erf",
        "exp",
        "log",
        "log10",
        "max",
        "min",
        "sin",
        "sinh",
        "sqrt",
        "tanh",
    ],
    ".expression_tree.interpolant": ["Interpolant"],
    ".expression_tree.discrete_time_sum": ["DiscreteTimeData", "DiscreteTimeSum"],
    ".expression_tree.variable": ["VariableBase", "Variable", "VariableDot"],
    ".expression_tree.coupled_variable": ["CoupledVariable"],
    ".expression_tree.independent_variable": [
        "IndependentVariable",
        "Time",
        "SpatialVariable",
        "SpatialVariableEdge",
        "t",
        "KNOWN_COORD_SYS",
    ],
    ".expression_tree.exceptions": [
        "DomainError",
        "OptionError",
        "OptionWarning",
        "GeometryError",
        "ModelError",
        "SolverError",
        "SolverWarning",
        "ShapeError",
        "ModelWarning",
        "DiscretisationError",
        "InvalidModelJSONError",
    ],
}

# Lazily loaded attributes (loaded on first access via lazy_loader)
LAZY_IMPORTS: dict[str, list[str]] = {
    # Utility classes and methods
    ".util": [
        "root_dir",
        "Timer",
        "TimerTime",
        "FuzzyDict",
        "load",
        "is_constant_and_can_evaluate",
        "get_parameters_filepath",
        "has_jax",
        "import_optional_dependency",
    ],
    # Array/Matrix
    ".expression_tree.array": ["Array", "linspace", "meshgrid"],
    ".expression_tree.matrix": ["Matrix"],
    ".expression_tree.vector": ["Vector"],
    ".expression_tree.tensor_field": ["TensorField"],
    ".expression_tree.vector_field": ["VectorField"],
    # Parameters (expression tree)
    ".expression_tree.input_parameter": ["InputParameter"],
    ".expression_tree.parameter": ["Parameter", "FunctionParameter"],
    ".expression_tree.scalar": ["Scalar", "Constant"],
    ".expression_tree.state_vector": ["StateVectorBase", "StateVector", "StateVectorDot"],
    # Operations
    ".expression_tree.operations.evaluate_python": [
        "find_symbols",
        "id_to_python_variable",
        "to_python",
        "EvaluatorPython",
        "EvaluatorJax",
        "JaxCooMatrix",
    ],
    ".expression_tree.operations.jacobian": ["Jacobian"],
    ".expression_tree.operations.convert_to_casadi": ["CasadiConverter"],
    ".expression_tree.operations.unpack_symbols": ["SymbolUnpacker"],
    ".expression_tree.operations.serialise": ["Serialise", "ExpressionFunctionParameter"],
    # Model classes
    ".models.base_model": ["BaseModel", "ModelSolutionObservability", "load_model"],
    ".models.symbol_processor": ["SymbolProcessor"],
    ".models.event": ["Event", "EventType"],
    ".models.full_battery_models.base_battery_model": [
        "BaseBatteryModel",
        "BatteryModelOptions",
    ],
    ".models.submodels.base_submodel": ["BaseSubModel"],
    # Geometry
    ".geometry.geometry": ["Geometry"],
    ".geometry.battery_geometry": ["battery_geometry"],
    # Parameters
    ".parameters.parameter_values": ["ParameterValues", "scalarize_dict", "arrayize_dict"],
    ".parameters.geometric_parameters": ["geometric_parameters", "GeometricParameters"],
    ".parameters.electrical_parameters": ["electrical_parameters", "ElectricalParameters"],
    ".parameters.thermal_parameters": ["thermal_parameters", "ThermalParameters"],
    ".parameters.lithium_ion_parameters": ["LithiumIonParameters"],
    ".parameters.lead_acid_parameters": ["LeadAcidParameters"],
    ".parameters.ecm_parameters": ["EcmParameters"],
    ".parameters.size_distribution_parameters": [
        "get_size_distribution_parameters",
        "lognormal",
    ],
    # Mesh and Discretisation
    ".discretisations.discretisation": ["Discretisation", "has_bc_of_form"],
    ".meshes.meshes": ["Mesh", "SubMesh", "MeshGenerator"],
    ".meshes.zero_dimensional_submesh": ["SubMesh0D"],
    ".meshes.one_dimensional_submeshes": [
        "SubMesh1D",
        "Uniform1DSubMesh",
        "Exponential1DSubMesh",
        "Chebyshev1DSubMesh",
        "UserSupplied1DSubMesh",
        "SpectralVolume1DSubMesh",
        "SymbolicUniform1DSubMesh",
    ],
    ".meshes.two_dimensional_submeshes": ["SubMesh2D", "Uniform2DSubMesh"],
    ".meshes.scikit_fem_submeshes": [
        "ScikitSubMesh2D",
        "ScikitUniform2DSubMesh",
        "ScikitExponential2DSubMesh",
        "ScikitChebyshev2DSubMesh",
        "UserSupplied2DSubMesh",
    ],
    ".meshes.scikit_fem_submeshes_3d": [
        "ScikitFemSubMesh3D",
        "ScikitFemGenerator3D",
        "UserSuppliedSubmesh3D",
    ],
    # Spatial Methods
    ".spatial_methods.spatial_method": ["SpatialMethod"],
    ".spatial_methods.zero_dimensional_method": ["ZeroDimensionalSpatialMethod"],
    ".spatial_methods.finite_volume": ["FiniteVolume"],
    ".spatial_methods.finite_volume_2d": ["FiniteVolume2D"],
    ".spatial_methods.spectral_volume": ["SpectralVolume"],
    ".spatial_methods.scikit_finite_element": ["ScikitFiniteElement"],
    ".spatial_methods.scikit_finite_element_3d": ["ScikitFiniteElement3D"],
    # Solvers
    ".solvers.solution": ["Solution", "EmptySolution", "make_cycle_solution"],
    ".solvers.processed_variable_time_integral": ["ProcessedVariableTimeIntegral"],
    ".solvers.processed_variable": [
        "ProcessedVariable",
        "ProcessedVariable2DFVM",
        "process_variable",
        "ProcessedVariableUnstructured",
    ],
    ".solvers.processed_variable_computed": ["ProcessedVariableComputed"],
    ".solvers.summary_variable": ["SummaryVariables"],
    ".solvers.base_solver": ["BaseSolver"],
    ".solvers.dummy_solver": ["DummySolver"],
    ".solvers.algebraic_solver": ["AlgebraicSolver"],
    ".solvers.casadi_solver": ["CasadiSolver"],
    ".solvers.casadi_algebraic_solver": ["CasadiAlgebraicSolver"],
    ".solvers.scipy_solver": ["ScipySolver"],
    ".solvers.composite_solver": ["CompositeSolver"],
    ".solvers.jax_solver": ["JaxSolver"],
    ".solvers.jax_bdf_solver": ["jax_bdf_integrate"],
    ".solvers.idaklu_jax": ["IDAKLUJax"],
    ".solvers.idaklu_solver": ["IDAKLUSolver"],
    # Experiments
    ".experiment.experiment": ["Experiment"],
    # Plotting
    ".plotting.quick_plot": ["QuickPlot", "close_plots", "QuickPlotAxes"],
    ".plotting.plot": ["plot"],
    ".plotting.plot2D": ["plot2D"],
    ".plotting.plot_voltage_components": ["plot_voltage_components"],
    ".plotting.plot_thermal_components": ["plot_thermal_components"],
    ".plotting.plot_summary_variables": ["plot_summary_variables"],
    ".plotting.dynamic_plot": ["dynamic_plot"],
    ".plotting.plot_3d_cross_section": ["plot_3d_cross_section"],
    ".plotting.plot_3d_heatmap": ["plot_3d_heatmap"],
    # Simulation
    ".simulation": ["Simulation", "load_sim", "is_notebook"],
    # Batch Study
    ".batch_study": ["BatchStudy"],
    # Pybamm Data
    ".pybamm_data": ["DataLoader"],
    # Dispatch
    ".dispatch": ["parameter_sets", "Model"],
}

# Submodule aliases (handled by custom __getattr__, not lazy_loader)
# These are declared in the stub for documentation but lazy_loader won't process them
SUBMODULE_ALIASES: dict[str, str] = {
    # Battery models (as modules)
    "lead_acid": ".models.full_battery_models.lead_acid",
    "lithium_ion": ".models.full_battery_models.lithium_ion",
    "equivalent_circuit": ".models.full_battery_models.equivalent_circuit",
    "sodium_ion": ".models.full_battery_models.sodium_ion",
    # Submodels
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
    "standard_spatial_vars": ".geometry.standard_spatial_vars",
    # Parameters
    "constants": ".parameters.constants",
    # Experiments
    "experiment": ".experiment",
    "step": ".experiment.step",
    # Callbacks and telemetry
    "callbacks": ".callbacks",
    "telemetry": ".telemetry",
}


# ============================================================================
# STUB GENERATION
# ============================================================================


def generate_import_line(module_path: str, attr: str) -> str:
    """Generate a single import line for the stub file."""
    return f"from {module_path} import {attr} as {attr}"


def generate_stub_content() -> str:
    """Generate the complete stub file content."""
    lines = [
        "# PyBaMM stub file for IDE support and type hints",
        "# This file is auto-generated by scripts/generate_pyi_stub.py",
        "# Do not edit manually - regenerate with: python scripts/generate_pyi_stub.py",
        "#",
        "# This file enables:",
        "# - IDE autocomplete for lazy-loaded attributes",
        "# - Type checking support via mypy/pyright",
        "# - lazy_loader integration for lazy imports",
        "",
        "# Version",
        "from .version import __version__ as __version__",
        "",
    ]

    # Add eagerly loaded imports
    lines.append("# ============================================================================")
    lines.append("# EAGERLY LOADED (imported at module load time)")
    lines.append("# ============================================================================")
    lines.append("")

    # Group by module for readability
    for module_path, attrs in EAGER_IMPORTS.items():
        module_name = module_path.split(".")[-1]
        lines.append(f"# {module_name}")
        for attr in attrs:
            lines.append(generate_import_line(module_path, attr))
        lines.append("")

    # Add config import
    lines.append("# Config module")
    lines.append("from . import config as config")
    lines.append("")

    # Add lazily loaded imports
    lines.append("# ============================================================================")
    lines.append("# LAZILY LOADED (via lazy_loader stub mechanism)")
    lines.append("# ============================================================================")
    lines.append("")

    for module_path, attrs in LAZY_IMPORTS.items():
        module_name = module_path.split(".")[-1]
        lines.append(f"# {module_name}")
        for attr in attrs:
            lines.append(generate_import_line(module_path, attr))
        lines.append("")

    # Note about submodule aliases
    lines.append("# ============================================================================")
    lines.append("# SUBMODULE ALIASES")
    lines.append("# These are handled by custom __getattr__ in __init__.py, not lazy_loader.")
    lines.append("# They are documented here for reference but not included in the stub imports")
    lines.append("# because lazy_loader cannot handle aliasing nested submodules to top level.")
    lines.append("#")
    lines.append("# Available submodule aliases:")
    for alias, path in sorted(SUBMODULE_ALIASES.items()):
        lines.append(f"#   pybamm.{alias} -> pybamm{path}")
    lines.append("# ============================================================================")

    return "\n".join(lines) + "\n"


def validate_imports() -> list[str]:
    """Validate that all configured imports actually exist."""
    errors = []

    # Check eager imports
    for module_path, attrs in EAGER_IMPORTS.items():
        try:
            full_path = f"pybamm{module_path}"
            module = importlib.import_module(full_path)
            for attr in attrs:
                if not hasattr(module, attr):
                    errors.append(f"EAGER: {module_path}.{attr} not found")
        except ImportError as e:
            errors.append(f"EAGER: Cannot import {module_path}: {e}")

    # Check lazy imports
    for module_path, attrs in LAZY_IMPORTS.items():
        try:
            full_path = f"pybamm{module_path}"
            module = importlib.import_module(full_path)
            for attr in attrs:
                if not hasattr(module, attr):
                    errors.append(f"LAZY: {module_path}.{attr} not found")
        except ImportError as e:
            errors.append(f"LAZY: Cannot import {module_path}: {e}")

    # Check submodule aliases
    for alias, module_path in SUBMODULE_ALIASES.items():
        try:
            full_path = f"pybamm{module_path}"
            importlib.import_module(full_path)
        except ImportError as e:
            errors.append(f"SUBMODULE: {alias} -> {module_path}: {e}")

    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Generate __init__.pyi stub file for PyBaMM"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if stub would change (for CI), exit 1 if different",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate that all configured imports exist",
    )
    args = parser.parse_args()

    stub_path = REPO_ROOT / "src" / "pybamm" / "__init__.pyi"

    if args.validate:
        print("Validating imports...")
        errors = validate_imports()
        if errors:
            print("Validation errors:")
            for error in errors:
                print(f"  - {error}")
            sys.exit(1)
        print("All imports validated successfully!")
        return

    content = generate_stub_content()

    if args.check:
        if stub_path.exists():
            existing = stub_path.read_text()
            if existing == content:
                print("Stub file is up to date.")
                sys.exit(0)
            else:
                print("Stub file is out of date. Run 'python scripts/generate_pyi_stub.py' to update.")
                sys.exit(1)
        else:
            print("Stub file does not exist.")
            sys.exit(1)

    # Write the stub file
    stub_path.write_text(content)
    print(f"Generated {stub_path}")

    # Count exports
    eager_count = sum(len(attrs) for attrs in EAGER_IMPORTS.values())
    lazy_count = sum(len(attrs) for attrs in LAZY_IMPORTS.values())
    submodule_count = len(SUBMODULE_ALIASES)
    print(f"  - {eager_count} eagerly loaded exports")
    print(f"  - {lazy_count} lazily loaded exports")
    print(f"  - {submodule_count} submodule aliases")
    print(f"  - Total: {eager_count + lazy_count + submodule_count} exports")


if __name__ == "__main__":
    main()
