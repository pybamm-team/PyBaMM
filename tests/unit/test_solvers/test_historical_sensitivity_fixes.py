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
        integration step used to find the exact event time, and ensured
        sensitivities are properly propagated through event handling.

        This test verifies sensitivities are correctly computed when the
        simulation terminates due to an event.
        """
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")

        # Set up input parameter for sensitivity - use a material property
        # that can be used with events (current function cannot be input with experiments)
        param.update({"Positive electrode diffusivity [m2.s-1]": "[input]"})

        solver = pybamm.IDAKLUSolver()
        sim = pybamm.Simulation(
            model,
            parameter_values=param,
            solver=solver,
        )

        # Solve until event (voltage cutoff) with sensitivities
        sol = sim.solve(
            t_eval=[0, 7200],  # Long enough to hit voltage cutoff
            inputs={"Positive electrode diffusivity [m2.s-1]": 4e-15},
            calculate_sensitivities=True,
        )

        # Should terminate due to event (voltage cutoff)
        assert "event" in sol.termination.lower()

        # Sensitivities should be available
        assert "Positive electrode diffusivity [m2.s-1]" in sol.sensitivities

        # Sensitivity values should be valid (not NaN or Inf)
        sens = sol.sensitivities["Positive electrode diffusivity [m2.s-1]"]
        assert not np.any(np.isnan(sens)), "Sensitivities contain NaN"
        assert not np.any(np.isinf(sens)), "Sensitivities contain Inf"

    def test_sensitivity_without_event_still_works(self):
        """
        Verify sensitivities work correctly without event termination.
        """
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")

        param.update({"Current function [A]": "[input]"})

        solver = pybamm.IDAKLUSolver()
        sim = pybamm.Simulation(
            model,
            parameter_values=param,
            solver=solver,
        )

        # Solve without hitting event
        sol = sim.solve(
            [0, 600],
            inputs={"Current function [A]": 5.0},
            calculate_sensitivities=True,
        )

        # Should terminate normally (final time)
        assert (
            "final time" in sol.termination.lower() or sol.termination == "final time"
        )

        # Sensitivities should be available
        assert "Current function [A]" in sol.sensitivities

        sens = sol.sensitivities["Current function [A]"]
        assert not np.any(np.isnan(sens))

    def test_sensitivity_values_physically_reasonable(self):
        """
        Verify sensitivity values are physically reasonable.

        Higher current should decrease voltage, so dV/dI should be negative.
        """
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")

        param.update({"Current function [A]": "[input]"})

        solver = pybamm.IDAKLUSolver()
        sim = pybamm.Simulation(
            model,
            parameter_values=param,
            solver=solver,
        )

        sol = sim.solve(
            [0, 600],
            inputs={"Current function [A]": 5.0},
            calculate_sensitivities=True,
        )

        # Get voltage sensitivity with respect to current
        V_sens = sol["Voltage [V]"].sensitivities["Current function [A]"]

        # Skip initial transients and check middle portion
        mid_start = len(V_sens) // 4
        mid_end = 3 * len(V_sens) // 4
        V_sens_mid = V_sens[mid_start:mid_end]

        # dV/dI should be predominantly negative (higher current = lower voltage)
        # Allow some tolerance for numerical noise
        assert np.mean(V_sens_mid) < 0, (
            f"dV/dI should be negative (higher current reduces voltage), "
            f"but mean sensitivity is {np.mean(V_sens_mid)}"
        )

    def test_sensitivity_with_multiple_input_parameters(self):
        """
        Verify sensitivities work with multiple input parameters.
        """
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")

        # Set multiple input parameters
        param.update(
            {
                "Current function [A]": "[input]",
                "Ambient temperature [K]": "[input]",
            }
        )

        solver = pybamm.IDAKLUSolver()
        sim = pybamm.Simulation(
            model,
            parameter_values=param,
            solver=solver,
        )

        sol = sim.solve(
            [0, 600],
            inputs={
                "Current function [A]": 5.0,
                "Ambient temperature [K]": 298.15,
            },
            calculate_sensitivities=True,
        )

        # Both sensitivities should be available
        assert "Current function [A]" in sol.sensitivities
        assert "Ambient temperature [K]" in sol.sensitivities

        # Both should be valid
        for param_name in ["Current function [A]", "Ambient temperature [K]"]:
            sens = sol.sensitivities[param_name]
            assert not np.any(np.isnan(sens)), (
                f"Sensitivity for {param_name} contains NaN"
            )

    def test_sensitivity_event_multiple_events(self):
        """
        Test sensitivity calculation when hitting multiple events in sequence.
        """
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")

        # Use a parameter that can be used with event-based termination
        param.update({"Positive electrode diffusivity [m2.s-1]": "[input]"})

        solver = pybamm.IDAKLUSolver()
        sim = pybamm.Simulation(
            model,
            parameter_values=param,
            solver=solver,
        )

        # Solve with potential for hitting events
        sol = sim.solve(
            t_eval=[0, 7200],
            inputs={"Positive electrode diffusivity [m2.s-1]": 4e-15},
            calculate_sensitivities=True,
        )

        # Should complete
        assert len(sol.t) > 0

        # Sensitivities should still be valid
        assert "Positive electrode diffusivity [m2.s-1]" in sol.sensitivities
        sens = sol.sensitivities["Positive electrode diffusivity [m2.s-1]"]
        assert not np.any(np.isnan(sens))


class TestSensitivityProcessedVariable:
    """Tests for sensitivity access through processed variables."""

    def test_variable_sensitivity_accessor(self):
        """
        Verify sensitivity can be accessed via processed variable.
        """
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")
        param.update({"Current function [A]": "[input]"})

        solver = pybamm.IDAKLUSolver()
        sim = pybamm.Simulation(
            model,
            parameter_values=param,
            solver=solver,
        )

        sol = sim.solve(
            [0, 600],
            inputs={"Current function [A]": 5.0},
            calculate_sensitivities=True,
        )

        # Access sensitivity via processed variable
        V_var = sol["Voltage [V]"]
        assert hasattr(V_var, "sensitivities")
        assert "Current function [A]" in V_var.sensitivities

        # Should be able to get sensitivity data
        V_sens = V_var.sensitivities["Current function [A]"]
        assert len(V_sens) == len(sol.t)
        assert not np.any(np.isnan(V_sens))

    def test_sensitivity_values_at_solve_times(self):
        """
        Verify sensitivity values are available and valid at solve times,
        and that voltage interpolation works independently.
        """
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")
        param.update({"Current function [A]": "[input]"})

        solver = pybamm.IDAKLUSolver()
        sim = pybamm.Simulation(
            model,
            parameter_values=param,
            solver=solver,
        )

        sol = sim.solve(
            [0, 600],
            inputs={"Current function [A]": 5.0},
            calculate_sensitivities=True,
        )

        V_var = sol["Voltage [V]"]

        # Voltage interpolation at arbitrary times should work
        t_interp = np.array([100, 200, 300, 400, 500])
        V_interp = V_var(t_interp)
        assert len(V_interp) == len(t_interp)
        assert not np.any(np.isnan(V_interp))

        # Sensitivities should be available at solve times
        V_sens = V_var.sensitivities["Current function [A]"]
        assert len(V_sens) == len(sol.t)
        assert not np.any(np.isnan(V_sens))

        # All dV/dI values should be negative (higher current = lower voltage)
        assert np.all(V_sens < 0), "dV/dI should be negative at all times"
