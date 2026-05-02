"""
Regression tests for historical sensitivity calculation bug fixes.

These tests guard against the reintroduction of bugs that were fixed but
did not have regression tests added at the time of the fix.
"""

import numpy as np

import pybamm


class TestSensitivityEventFixes:
    """Guards for sensitivity calculation bug fixes with events."""

    def test_sensitivity_with_event_termination(self):
        """
        Guards against: PR #1765 - I1727 sensitivity event bug

        The bug was that when a simulation with calculate_sensitivities=True
        terminated due to an event (e.g., voltage cutoff), the sensitivity
        extraction would fail or produce incorrect results.

        The fix added `extract_sensitivities_in_solution=False` for the dense
        integration step used to find the exact event time.
        """
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")

        param.update({"Positive electrode diffusivity [m2.s-1]": "[input]"})

        solver = pybamm.IDAKLUSolver()
        sim = pybamm.Simulation(
            model,
            parameter_values=param,
            solver=solver,
        )

        sol = sim.solve(
            t_eval=[0, 7200],
            inputs={"Positive electrode diffusivity [m2.s-1]": 4e-15},
            calculate_sensitivities=True,
        )

        assert "event" in sol.termination.lower()

        assert "Positive electrode diffusivity [m2.s-1]" in sol.sensitivities

        sens = sol.sensitivities["Positive electrode diffusivity [m2.s-1]"]
        assert not np.any(np.isnan(sens))
        assert not np.any(np.isinf(sens))
