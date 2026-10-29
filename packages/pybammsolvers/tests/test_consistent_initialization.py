"""Regression tests for IDAKLU consistent initialization."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pybamm", reason="PyBaMM not installed")
import pybamm


def test_internal_ic_keeps_converged_state_at_tiny_atol():
    """Internal IC must survive a tiny atol that makes the WRMS step-norm test
    unsatisfiable. Previously aborted at t=0 with IDA_BAD_K when the
    consistent-init Newton's state was discarded for the raw guess."""
    model = pybamm.lithium_ion.DFN()
    solver = pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-20)
    sim = pybamm.Simulation(model, solver=solver)

    sol = sim.solve([0, 3600])

    assert sol["Voltage [V]"].entries.size > 1


class TestSensitivityInitialConditions:
    """Initial sensitivities of algebraic states, at tolerances near roundoff.

    IDACalcIC's sensitivity correction used to start from the zero algebraic
    guess, failing at some inputs and leaving those sensitivities zero at t0.
    """

    INPUT = "Current function [A]"

    @staticmethod
    def _model():
        parameter_values = pybamm.ParameterValues("Chen2020")
        parameter_values.update({"Current function [A]": "[input]"})
        simulation = pybamm.Simulation(
            pybamm.lithium_ion.DFN({"contact resistance": "true"}),
            parameter_values=parameter_values,
            var_pts={"x_n": 6, "x_s": 5, "x_p": 6, "r_n": 8, "r_p": 8},
        )
        simulation.build()
        model = simulation.built_model
        model.convert_to_format = "casadi"
        return model

    @pytest.mark.parametrize("newton_mode", ["auto", "full"])
    def test_algebraic_sensitivities_at_t0_match_finite_differences(self, newton_mode):
        model = self._model()
        solver = pybamm.IDAKLUSolver(
            rtol=1e-11, atol=1e-13, options={"newton_mode": newton_mode}
        )
        n_states = model.len_rhs_and_alg
        algebraic = slice(model.len_rhs, n_states)
        for current in (4.5, 4.875, 5.0, 5.125, 5.375, 5.5):
            solution = solver.solve(
                model,
                [0, 1],
                inputs={self.INPUT: current},
                calculate_sensitivities=True,
            )
            # Stored time-major, so the first rows are the states at t0
            sensitivity = solution.sensitivities[self.INPUT][:n_states, 0]

            step = 1e-3 * current
            plus, minus = (
                solver.solve(model, [0, 1], inputs={self.INPUT: current + sign * step})
                for sign in (1, -1)
            )
            expected = (plus.y[algebraic, 0] - minus.y[algebraic, 0]) / (2 * step)
            np.testing.assert_allclose(
                sensitivity[algebraic],
                expected,
                rtol=0,
                atol=1e-5 * np.abs(expected).max(),
                err_msg=f"current {current}",
            )
