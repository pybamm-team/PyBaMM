"""
Regression tests for historical thermal submodel bug fixes.

These tests guard against the reintroduction of bugs that were fixed but
did not have regression tests added at the time of the fix.
"""

import pytest

import pybamm


class TestThermalInitializationFixes:
    """Guards for temperature initialization bug fixes."""

    def test_lumped_thermal_initialization_correct(self):
        """
        Guards against: aa78954c4 - Fix: correct temperature initialisation in
        lumped and x_full thermal (#5248)

        The bug was that T_init wasn't wrapped in x_average, causing issues
        when T_init could have spatial dependence.
        """
        model = pybamm.lithium_ion.SPM({"thermal": "lumped"})
        param = pybamm.ParameterValues("Chen2020")

        T_init = 310.0
        param.update({"Initial temperature [K]": T_init})

        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 100])

        T_initial = sol["Volume-averaged cell temperature [K]"].data[0]
        assert T_initial == pytest.approx(T_init, rel=1e-5)

        T_final = sol["Volume-averaged cell temperature [K]"].data[-1]
        assert T_final != pytest.approx(T_init, abs=0.01)

    def test_x_full_thermal_cc_temps_match_boundaries(self):
        """
        Guards against: aa78954c4 - Fix: correct temperature initialisation in
        lumped and x_full thermal (#5248)

        The bug was that current collector temperatures weren't initialized
        using boundary_value, causing mismatches at domain boundaries.
        """
        model = pybamm.lithium_ion.DFN({"thermal": "x-full"})
        param = pybamm.ParameterValues("Marquis2019")

        T_init = 305.0
        param.update({"Initial temperature [K]": T_init})

        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 10])

        T_cn_init = sol["Negative current collector temperature [K]"].data[0]
        T_cp_init = sol["Positive current collector temperature [K]"].data[0]

        assert T_cn_init == pytest.approx(T_init, rel=1e-4)
        assert T_cp_init == pytest.approx(T_init, rel=1e-4)
