"""Pytest configuration and fixtures for pybammsolvers tests."""

from __future__ import annotations

import pytest
import os
import numpy as np
import sys
import platform


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


@pytest.fixture(scope="session")
def idaklu_module():
    """Fixture to provide the idaklu module."""
    try:
        import pybammsolvers

        return pybammsolvers.idaklu
    except ImportError as e:
        pytest.skip(f"Could not import pybammsolvers.idaklu: {e}")


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    # Ensure consistent backend for any plotting
    os.environ["MPLBACKEND"] = "Agg"

    # Set encoding for consistent behavior
    os.environ["PYTHONIOENCODING"] = "utf-8"

    yield


@pytest.fixture(scope="session")
def exponential_decay_model():
    """
    Fixture providing an exponential decay ODE model.

    The model is: dy/dt = -k*y, with exact solution y(t) = y0 * exp(-k*t)

    Returns a dictionary containing:
    - 'k': decay constant
    - 'y0': initial condition
    - 't_eval': evaluation times
    - 'exact_solution': function to compute exact solution at any time
    """

    # Model parameters
    k = 0.5  # decay constant
    y0 = 1.0  # initial condition
    t_eval = np.linspace(0, 5, 50)

    def exact_solution(t):
        """Compute exact solution at time t."""
        return y0 * np.exp(-k * t)

    return {
        "k": k,
        "y0": y0,
        "t_eval": t_eval,
        "exact_solution": exact_solution,
    }


@pytest.fixture(scope="function")
def exponential_decay_solver(idaklu_module, exponential_decay_model):
    """
    Fixture providing a configured solver for exponential decay model.

    Sets up a complete IDAKLU solver instance with the exponential decay
    ODE system ready to solve.
    """
    # Skip tests using this fixture on macOS Intel
    if sys.platform == "darwin" and platform.machine() != "arm64":
        pytest.skip("Skipping exponential_decay_solver tests on macOS Intel")

    casadi = pytest.importorskip("casadi")

    # Model parameters
    k = exponential_decay_model["k"]
    y0 = exponential_decay_model["y0"]

    # Problem dimensions
    n_states = 1
    n_inputs = 1
    n_sensitivity_params = 0

    # Create symbolic variables
    t_sym = casadi.MX.sym("t")
    y_sym = casadi.MX.sym("y", n_states)
    p_sym = casadi.MX.sym("p", n_inputs)

    # RHS function: For ODE dy/dt = -k*y
    rhs = -1.0 * p_sym * y_sym

    # Create RHS function: t, y, inputs
    rhs_alg = casadi.Function("rhs_alg", [t_sym, y_sym, p_sym], [rhs])

    # Jacobian: PyBaMM computes jac_rhs - cj * mass_matrix
    # For rhs = -k*y: jac_rhs = d(rhs)/dy = -k
    # mass_matrix = 1 (identity for ODE)
    # So: jac_times_cjmass = -k - cj * 1 = -k - cj
    cj_sym = casadi.MX.sym("cj")
    jac_result = casadi.vertcat(-p_sym - cj_sym)
    jac_times_cjmass = casadi.Function(
        "jac_times_cjmass", [t_sym, y_sym, p_sym, cj_sym], [jac_result]
    )

    # Sparse matrix structure (single element for 1x1 system)
    jac_times_cjmass_colptrs = np.array([0, 1], dtype=np.int64)
    jac_times_cjmass_rowvals = np.array([0], dtype=np.int64)
    jac_times_cjmass_nnz = 1

    # Define vector symbol for matrix-vector products
    v_sym = casadi.MX.sym("v", n_states)

    # Mass matrix action: M @ v (identity for ODE, so returns v)
    mass_action = casadi.Function("mass", [v_sym], [v_sym])

    # Jacobian action (for matrix-free methods): d(rhs)/dy @ v
    # For rhs = -k*y: d(rhs)/dy = -k, so jac_action = -k * v
    jac_action_result = -1.0 * p_sym * v_sym
    jac_action = casadi.Function(
        "jac_action", [t_sym, y_sym, p_sym, v_sym], [jac_action_result]
    )

    # Sensitivity equations
    if n_sensitivity_params > 0:
        sens = casadi.Function(
            "sens",
            [t_sym, y_sym, p_sym],
            [casadi.MX.zeros(n_states, n_sensitivity_params)],
        )
    else:
        # Empty function for no sensitivities
        sens = casadi.Function("sens", [], [])

    # No events - signature is [t, y, inputs]
    events = casadi.Function("events", [t_sym, y_sym, p_sym], [casadi.MX(0)])
    n_events = 0

    # DAE identifier (1 = differential, 0 = algebraic)
    # For ODE, all states are differential
    rhs_alg_id = np.array([1.0], dtype=np.float64)

    # Tolerances
    atol = np.array([1e-8], dtype=np.float64)
    rtol = 1e-8

    # Output variables (just return the state itself as a vector)
    var_fcn = casadi.Function("var", [t_sym, y_sym, p_sym], [y_sym])
    var_fcns = [idaklu_module.generate_function(var_fcn.serialize())]

    # Sensitivities of output wrt states and params
    if n_sensitivity_params > 0:
        dvar_dy_fcn = casadi.Function(
            "dvar_dy", [t_sym, y_sym, p_sym], [casadi.MX.ones(1, n_states)]
        )
        dvar_dy_fcns = [idaklu_module.generate_function(dvar_dy_fcn.serialize())]

        dvar_dp_fcn = casadi.Function(
            "dvar_dp", [t_sym, y_sym, p_sym], [casadi.MX.zeros(1, n_sensitivity_params)]
        )
        dvar_dp_fcns = [idaklu_module.generate_function(dvar_dp_fcn.serialize())]
    else:
        dvar_dy_fcns = []
        dvar_dp_fcns = []

    # Convert CasADi functions to idaklu Function objects
    rhs_alg_func = idaklu_module.generate_function(rhs_alg.serialize())
    jac_times_cjmass_func = idaklu_module.generate_function(
        jac_times_cjmass.serialize()
    )
    jac_action_func = idaklu_module.generate_function(jac_action.serialize())
    mass_action_func = idaklu_module.generate_function(mass_action.serialize())
    sens_func = idaklu_module.generate_function(sens.serialize())
    events_func = idaklu_module.generate_function(events.serialize())

    # Solver options
    options = {
        # SetupOptions
        "jacobian": "sparse",
        "preconditioner": "none",
        "precon_half_bandwidth": 5,
        "precon_half_bandwidth_keep": 5,
        "num_threads": 1,
        "num_solvers": 1,
        "linear_solver": "SUNLinSol_KLU",
        "linsol_max_iterations": 5,
        # SolverOptions
        "print_stats": False,
        "silence_sundials_errors": True,
        "max_order_bdf": 5,
        "max_num_steps": 10000,
        "dt_init": 0.01,
        "dt_min": 0.0,
        "dt_max": 0.1,
        "max_error_test_failures": 100,
        "max_nonlinear_iterations": 10,
        "max_convergence_failures": 10,
        "nonlinear_convergence_coefficient": 0.33,
        "nonlinear_convergence_coefficient_ic": 0.0033,
        "suppress_algebraic_error": False,
        "hermite_interpolation": True,
        "calc_ic": False,  # We provide consistent initial conditions
        "init_all_y_ic": False,
        "max_num_steps_ic": 5,
        "max_num_jacobians_ic": 4,
        "max_num_iterations_ic": 10,
        "max_linesearch_backtracks_ic": 100,
        "linesearch_off_ic": False,
        "linear_solution_scaling": True,
        "epsilon_linear_tolerance": 0.05,
        "increment_factor": 1.0,
    }

    # Bandwidths (for banded solvers, not used with KLU)
    jac_bandwidth_lower = 0
    jac_bandwidth_upper = 0

    # Create solver group
    solver = idaklu_module.create_casadi_solver_group(
        n_states,
        n_sensitivity_params,
        rhs_alg_func,
        jac_times_cjmass_func,
        jac_times_cjmass_colptrs,
        jac_times_cjmass_rowvals,
        jac_times_cjmass_nnz,
        jac_bandwidth_lower,
        jac_bandwidth_upper,
        jac_action_func,
        mass_action_func,
        sens_func,
        events_func,
        n_events,
        rhs_alg_id,
        atol,
        rtol,
        n_inputs,
        var_fcns,
        dvar_dy_fcns,
        dvar_dp_fcns,
        options,
    )

    # Initial conditions need to be 2D: [number_of_groups, n_coeffs]
    # where n_coeffs = number_of_states * (1 + number_of_sensitivity_parameters)
    # When n_sensitivity_params = 0: n_coeffs = n_states
    # When n_sensitivity_params > 0: [state_0, d(state_0)/dp_0, d(state_0)/dp_1, ...]
    n_coeffs = n_states * (1 + n_sensitivity_params)
    y0_2d = np.zeros((1, n_coeffs), dtype=np.float64)
    y0_2d[0, 0] = y0  # state value

    yp0_2d = np.zeros((1, n_coeffs), dtype=np.float64)
    yp0_2d[0, 0] = -k * y0  # state derivative

    inputs_2d = np.array([[k]], dtype=np.float64)

    return {
        "solver": solver,
        "y0": y0_2d,
        "yp0": yp0_2d,
        "inputs": inputs_2d,
        "model": exponential_decay_model,
    }
