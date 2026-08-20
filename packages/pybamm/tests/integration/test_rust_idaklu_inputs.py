"""Parity test: Rust IDAKLU with InputParameter models vs CasADi."""

import numpy as np
import pytest

import pybamm

pytest.importorskip("casadi")


def _build_input_param_model():
    """ODE: dy/dt = -k * y, with k as InputParameter. Pure ODE, identity mass."""
    model = pybamm.BaseModel()
    y = pybamm.Variable("y")
    k = pybamm.InputParameter("k")
    model.rhs = {y: -k * y}
    model.initial_conditions = {y: 1.0}
    return model


@pytest.mark.parametrize("k_val", [0.1, 1.0, 5.0])
def test_input_parameter_solve_parity(k_val):
    """Solve dy/dt = -k*y with both backends, compare trajectories."""
    t_eval = np.linspace(0, 5, 50)
    inputs = {"k": k_val}

    model_casadi = _build_input_param_model()
    model_casadi.convert_to_format = "casadi"
    sol_casadi = pybamm.IDAKLUSolver().solve(model_casadi, t_eval, inputs=inputs)

    model_rust = _build_input_param_model()
    model_rust.convert_to_format = "rust"
    sol_rust = pybamm.IDAKLUSolver().solve(model_rust, t_eval, inputs=inputs)

    np.testing.assert_allclose(sol_rust.y, sol_casadi.y, rtol=1e-5, atol=1e-8)
