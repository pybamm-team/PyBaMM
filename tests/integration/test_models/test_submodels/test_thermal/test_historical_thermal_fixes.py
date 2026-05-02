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
        when T_init could have spatial dependence. The fix ensures that
        x_average is applied to T_init for the lumped model.

        This test verifies the lumped thermal model correctly initializes
        temperature and that the fix is working (model runs without errors
        and temperature is properly set).
        """
        model = pybamm.lithium_ion.SPM({"thermal": "lumped"})
        param = pybamm.ParameterValues("Chen2020")

        # Set non-default initial temperature
        T_init = 310.0
        param.update({"Initial temperature [K]": T_init})

        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 100])

        # Temperature should start at the specified T_init
        T_initial = sol["Volume-averaged cell temperature [K]"].data[0]
        assert T_initial == pytest.approx(T_init, rel=1e-5), (
            f"Expected T_init={T_init}, got {T_initial}"
        )

        # Temperature should evolve during discharge (heating vs cooling)
        T_final = sol["Volume-averaged cell temperature [K]"].data[-1]
        assert T_final != pytest.approx(T_init, abs=0.01), (
            "Temperature should evolve during discharge"
        )

    def test_x_full_thermal_cc_temps_match_boundaries(self):
        """
        Guards against: aa78954c4 - Fix: correct temperature initialisation in
        lumped and x_full thermal (#5248)

        The bug was that current collector temperatures weren't initialized
        using boundary_value, causing mismatches at domain boundaries.
        This test verifies that the current collector temperatures properly
        match the electrode boundary temperatures at t=0.
        """
        model = pybamm.lithium_ion.DFN({"thermal": "x-full"})
        # Marquis2019 has the required thermal parameters for x-full
        param = pybamm.ParameterValues("Marquis2019")

        # Use a non-default initial temperature
        T_init = 305.0
        param.update({"Initial temperature [K]": T_init})

        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 10])

        # At t=0, current collector temps should match boundary temps
        T_cn_init = sol["Negative current collector temperature [K]"].data[0]
        T_cp_init = sol["Positive current collector temperature [K]"].data[0]

        # Both current collectors should start at T_init
        assert T_cn_init == pytest.approx(T_init, rel=1e-4), (
            f"Negative CC temp {T_cn_init} should match T_init {T_init}"
        )
        assert T_cp_init == pytest.approx(T_init, rel=1e-4), (
            f"Positive CC temp {T_cp_init} should match T_init {T_init}"
        )
