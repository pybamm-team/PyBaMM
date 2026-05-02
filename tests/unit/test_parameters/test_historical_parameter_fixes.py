"""
Regression tests for historical parameter handling bug fixes.

These tests guard against the reintroduction of bugs that were fixed but
did not have regression tests added at the time of the fix.
"""

import numpy as np
import pytest

import pybamm


class TestParameterValuesFixes:
    """Guards for parameter values handling bug fixes."""

    def test_depreciated_parameter_raises_error(self):
        """
        Guards against: 388d1366f - Bug Fix (parameter deprecation message)

        The bug was an incorrect condition "name in self.keys() == ..." which
        always evaluated to False, meaning the deprecation check never triggered.
        The fix changed it to "name == ..." so the error is actually raised.

        This test verifies that using the deprecated parameter name raises
        an error rather than silently failing.
        """
        param = pybamm.ParameterValues("Chen2020")

        # The depreciated parameter "1 + dlnf/dlnc" should raise ValueError
        # (it was renamed to "Thermodynamic factor")
        # Before the fix, this would silently pass instead of raising
        with pytest.raises(ValueError):
            param.update({"1 + dlnf/dlnc": 1.5})

    def test_parameter_update_existing_works(self):
        """
        Verify parameter update correctly updates existing parameters.
        """
        param = pybamm.ParameterValues("Chen2020")

        # Should be able to update existing parameter
        param.update({"Upper voltage cut-off [V]": 4.3})
        assert param["Upper voltage cut-off [V]"] == 4.3


class TestParameterCacheFixes:
    """Guards for parameter cache handling bug fixes."""

    def test_simulation_works_after_parameter_update(self):
        """
        Guards against: e4d58ca80 - fix: cache for combined_processor, correct
        __getitem__ KeyError try-except

        The bug involved cache invalidation issues when parameters were updated.
        This test verifies that updating parameters and then running a simulation
        works correctly (no stale cache issues).
        """
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")

        # Run simulation with original parameters
        sim1 = pybamm.Simulation(model, parameter_values=param)
        sol1 = sim1.solve([0, 100])

        # Update a parameter
        param.update({"Initial temperature [K]": 310})

        # Run simulation with updated parameters
        sim2 = pybamm.Simulation(model, parameter_values=param)
        sol2 = sim2.solve([0, 100])

        # Both should succeed
        assert len(sol1.t) > 0
        assert len(sol2.t) > 0


class TestParameterNameFixes:
    """Guards for parameter name bug fixes."""

    def test_key_parameters_accessible(self):
        """
        Guards against: aac7bbf31 and 60790355c - fix parameter name issues

        These bugs involved incorrect parameter names. This test verifies
        key parameters exist and are accessible with correct names.
        """
        param = pybamm.ParameterValues("Chen2020")

        # These parameters should exist with correct names
        # (bugs involved typos or incorrect names)
        assert param["Positive electrode thickness [m]"] > 0
        assert param["Negative electrode thickness [m]"] > 0
        assert param["Separator thickness [m]"] > 0
        assert param["Initial temperature [K]"] > 0

    def test_model_runs_with_standard_parameters(self):
        """
        Verify model runs correctly with standard parameters.
        """
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")
        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 100])

        # Check that simulation succeeded
        assert len(sol.t) > 0
        V = sol["Voltage [V]"].data
        assert np.all(V > 2.5)
        assert np.all(V < 4.5)


class TestInputParameterFixes:
    """Guards for input parameter handling fixes."""

    def test_input_parameters_affect_solution(self):
        """
        Verify input parameters correctly affect the simulation results.
        """
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")

        # Set a parameter as input
        param.update({"Current function [A]": "[input]"})

        sim = pybamm.Simulation(model, parameter_values=param)

        # Solve with two different current values
        sol_1A = sim.solve([0, 100], inputs={"Current function [A]": 1.0})
        sol_2A = sim.solve([0, 100], inputs={"Current function [A]": 2.0})

        assert len(sol_1A.t) > 0
        assert len(sol_2A.t) > 0

        # Higher current should result in lower voltage at end
        V_1A_final = sol_1A["Voltage [V]"].data[-1]
        V_2A_final = sol_2A["Voltage [V]"].data[-1]
        assert V_2A_final < V_1A_final
