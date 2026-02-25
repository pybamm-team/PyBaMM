# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

PyBaMM (Python Battery Mathematical Modelling) is an open-source battery simulation package written in Python. The project consists of three main components:
1. A framework for writing and solving systems of differential equations
2. A library of battery models and parameters
3. Specialized tools for simulating battery-specific experiments and visualizing results

## Architecture

### Core Battery Models
PyBaMM implements several lithium-ion battery models with increasing complexity:

- **SPM (Single Particle Model)**: Simplest model with single particles in each electrode
- **SPMe (Single Particle Model with Electrolyte)**: SPM with electrolyte dynamics
- **DFN (Doyle-Fuller-Newman)**: Full-order model solving for many particles and full electrolyte transport
- **MPM (Many Particle Model)**: Extension with particle size distributions
- **MSMR (Multi-Species Multi-Reaction)**: Advanced model with multiple reaction mechanisms

### Key Components Architecture

#### Expression Tree (`src/pybamm/expression_tree/`)
- Symbolic mathematical expression system
- All model equations are built as expression trees before discretization
- Core classes: `Symbol`, `Variable`, `Parameter`, `BinaryOperator`, `UnaryOperator`
- Operations include printing, evaluation, and conversion to solvers

#### Models (`src/pybamm/models/`)
- Base model classes and full battery model implementations
- Submodels for different physical phenomena (electrolyte, particles, thermal, etc.)
- Model options system for selecting physics and numerical methods
- Entry points system for registering models and parameter sets

#### Discretisation & Solvers (`src/pybamm/discretisations/`, `src/pybamm/solvers/`)
- Spatial discretization using finite volume, finite element methods
- Temporal solvers: IDAKLU (custom), CasADi, JAX-based solvers
- Mesh generation and spatial method selection

#### Simulation (`src/pybamm/simulation.py`)
- High-level interface combining models, parameters, geometry, and solvers
- Experiment protocol handling
- Solution post-processing and visualization

#### Parameters (`src/pybamm/parameters/`, `src/pybamm/input/parameters/`)
- Parameter value management with units and validation
- Pre-defined parameter sets for different battery chemistries
- Support for functions, interpolants, and temperature-dependent parameters

## Development Commands

### Environment Setup
```bash
# Install in development mode with all extras
nox -s dev

# Or manually with pip
pip install -e ".[all,dev,jax]"

# Install pre-commit hooks
pre-commit install
```

### Testing
```bash
# Run unit tests only
nox -s unit

# Run both unit and integration tests
nox -s tests

# Run specific test file
pytest tests/unit/path/to/test_file.py

# Run specific test method
pytest tests/unit/test_plotting/test_quick_plot.py::TestQuickPlot::test_simple_ode_model

# Run tests with coverage
nox -s coverage

# Test example notebooks
nox -s examples

# Test example scripts
nox -s scripts

# Test documentation builds and doctests
nox -s doctests
```

### Code Quality
```bash
# Run pre-commit checks on all files
nox -s pre-commit

# Run pre-commit on staged files only
pre-commit run

# Run ruff linter/formatter
pre-commit run ruff
```

### Documentation
```bash
# Build documentation locally with auto-reload
nox -s docs

# Build documentation for CI (treats warnings as errors)
nox -s docs
```

### Building and Installation
```bash
# Build source distribution
python -m build --sdist

# Build wheel
python -m build --wheel

# Install from source
pip install -e .
```

### Working with Models
Models are defined through a submodel system. Each full battery model (SPM, DFN, etc.) inherits from `BaseModel` and selects appropriate submodels for:
- Particle diffusion (Fickian, polynomial profiles, MSMR)
- Electrolyte transport (constant, full diffusion/migration)
- Electrode potentials (leading order, composite, surface forms)
- Interface kinetics (Butler-Volmer variants)
- Thermal effects (isothermal, lumped, full 3D)
- Side reactions (SEI growth, lithium plating, gas evolution)

### Parameter Sets and Entry Points
The project uses Python entry points to register models and parameter sets:
- Models: `pybamm_models` entry point group
- Parameter sets: `pybamm_parameter_sets` entry point group

New parameter sets should be added to `pyproject.toml` under the appropriate entry point group.

### Solver Selection Guidelines
- **IDAKLUSolver**: Fastest for most problems, custom SUNDIALS wrapper
- **CasadiSolver**: Good for optimization problems, automatic differentiation
- **JaxSolver**: Experimental, good for differentiation through solutions
- **ScipySolver**: Fallback, pure Python solvers

### Common Development Patterns

#### Running Simulations
```python
import pybamm

# Simple discharge
model = pybamm.lithium_ion.DFN()
sim = pybamm.Simulation(model)
sim.solve([0, 3600])
sim.plot()

# With experiment protocol
experiment = pybamm.Experiment(
    [
        "Discharge at 1C until 3.0V",
        "Rest for 1 hour",
        "Charge at C/10 until 4.2V",
    ]
)
sim = pybamm.Simulation(model, experiment=experiment)
sim.solve()
```

#### Testing Models
When adding new models or submodels, ensure they:
1. Have appropriate tests in `tests/unit/test_models/`
2. Work with standard parameter sets
3. Handle edge cases and option combinations
4. Include docstrings with examples
5. Register appropriate citations

#### Managing Dependencies
Optional dependencies should be imported within functions using `pybamm.import_optional_dependency()`, never at module level. This allows core PyBaMM to work with minimal dependencies.

## Python Version Support
- Requires Python 3.10-3.12
- Uses modern Python features (match/case, type hints, etc.)
- CI tests against all supported Python versions

## Key Configuration Files
- `pyproject.toml`: Main configuration, dependencies, build settings, tool configuration
- `noxfile.py`: Testing and development session definitions
- `.pre-commit-config.yaml`: Code quality checks and formatting
- `conftest.py`: Pytest configuration and shared fixtures
