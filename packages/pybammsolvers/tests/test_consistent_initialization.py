"""Regression tests for IDAKLU consistent initialization."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pybamm", reason="PyBaMM not installed")
import pybamm


def _jump_dae():
    """``y' = -a y + z``, ``0 = z - b y - c [t > 0.5]`` from the guess ``z = 0``."""
    model = pybamm.BaseModel()
    y = pybamm.Variable("y")
    z = pybamm.Variable("z")
    a, b, c = (pybamm.InputParameter(name) for name in "abc")
    model.rhs = {y: -a * y + z}
    model.algebraic = {z: z - b * y - c * (pybamm.t > 0.5)}
    model.initial_conditions = {y: 1, z: 0}
    model.variables = {"z": z}
    pybamm.Discretisation().process_model(model)
    return model


@pytest.mark.parametrize("options", [{}, {"calc_ic": False}, {"init_all_y_ic": True}])
def test_algebraic_sensitivities_at_t0_are_consistent(options):
    """z(0) = b y(0) gives dz/dp(0) = b dy(0)/dp + y(0) db/dp, not the zero of
    the guess z = 0.

    IDA_Y_INIT fits y(0) = a / (a - b) to y'(0) = -a, so y(0) depends on a, b.
    """
    a, b = 2.0, 0.5
    solver = pybamm.IDAKLUSolver(rtol=1e-10, atol=1e-12, options=options)
    sol = solver.solve(
        _jump_dae(),
        [0, 1],
        inputs={"a": a, "b": b, "c": 1.0},
        calculate_sensitivities=True,
    )

    if options.get("init_all_y_ic"):
        y0, dy0_da, dy0_db = a / (a - b), -b / (a - b) ** 2, a / (a - b) ** 2
    else:
        y0, dy0_da, dy0_db = 1.0, 0.0, 0.0
    sensitivities = sol["z"].sensitivities
    np.testing.assert_allclose(sensitivities["a"][0], b * dy0_da, atol=1e-10)
    np.testing.assert_allclose(sensitivities["b"][0], y0 + b * dy0_db, rtol=1e-9)
    np.testing.assert_allclose(sensitivities["c"][0], 0, atol=1e-12)


def test_restart_after_discontinuity_stores_the_right_limit():
    """The right point of the t_eval pair bracketing t = 0.5 holds the right
    limit, where z and dz/dc have jumped by c and 1."""
    sol = pybamm.IDAKLUSolver(rtol=1e-10, atol=1e-12).solve(
        _jump_dae(),
        [0, 1],
        inputs={"a": 2.0, "b": 0.5, "c": 1.0},
        calculate_sensitivities=True,
    )

    left, right = np.flatnonzero(sol.t > 0.5)[0] - 1, np.flatnonzero(sol.t > 0.5)[0]
    z = sol["z"].entries
    np.testing.assert_allclose(z[right] - z[left], 1.0, rtol=1e-9)
    np.testing.assert_allclose(sol["z"].sensitivities["c"][right], 1.0, rtol=1e-9)


def test_internal_ic_keeps_converged_state_at_tiny_atol():
    """Internal IC must survive a tiny atol that makes the WRMS step-norm test
    unsatisfiable. Previously aborted at t=0 with IDA_BAD_K when the
    consistent-init Newton's state was discarded for the raw guess."""
    model = pybamm.lithium_ion.DFN()
    solver = pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-20)
    sim = pybamm.Simulation(model, solver=solver)

    sol = sim.solve([0, 3600])

    assert sol["Voltage [V]"].entries.size > 1
