"""
Regression tests for historical half-cell model bug fixes.

These tests guard against the reintroduction of bugs that were fixed but
did not have regression tests added at the time of the fix.
"""

import numpy as np

import pybamm


class TestHalfCellVoltageContributions:
    """Guards for half-cell voltage contribution bug fixes."""

    def test_half_cell_electrolyte_potential_uses_interface(self):
        """
        Guards against: d421e3a55 - Fix half-cell voltage contributions (#5139)

        The bug was that for half-cell (planar negative electrode), the
        electrolyte potential at the negative electrode was set to 0 instead
        of using the lithium metal interface electrolyte potential.
        """
        # Test with half-cell model (positive working electrode)
        # Use Xu2019 which supports half-cell configurations
        model = pybamm.lithium_ion.SPMe({"working electrode": "positive"})
        param = pybamm.ParameterValues("Xu2019")
        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 600])

        # Verify the model runs and produces reasonable voltage
        voltage = sol["Voltage [V]"].data
        assert len(voltage) > 0
        assert np.all(voltage > 2.0)  # Reasonable voltage range for half-cell
        assert np.all(voltage < 5.0)

        # Verify electrolyte potential at interface is not always zero
        # (this was the bug - it was hardcoded to 0)
        phi_e_interface = sol["Lithium metal interface electrolyte potential [V]"].data
        # Should not be identically zero throughout discharge
        assert not np.allclose(phi_e_interface, 0, atol=1e-6)

    def test_half_cell_electrolyte_ohmic_losses_nonzero(self):
        """
        Guards against: d421e3a55 - Fix half-cell voltage contributions (#5139)

        The bug was that for half-cells with planar negative electrode, the
        electrolyte Ohmic losses were incorrectly calculated as zero. This test
        verifies that electrolyte resistance contributes to voltage losses.
        """
        model = pybamm.lithium_ion.SPMe({"working electrode": "positive"})
        param = pybamm.ParameterValues("Xu2019")
        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 600])

        # Get electrolyte Ohmic losses - should be non-zero during discharge
        eta_e = sol["X-averaged electrolyte ohmic losses [V]"].data

        # During active discharge, there should be non-zero Ohmic losses
        # (before the fix, these were incorrectly zero for half-cells)
        mid_idx = len(eta_e) // 2
        assert np.abs(eta_e[mid_idx]) > 1e-6, (
            "Electrolyte Ohmic losses should be non-zero during discharge"
        )


class TestHalfCellBulkOCP:
    """Guards for half-cell bulk OCP bug fixes."""

    def test_lithium_metal_bulk_ocp_is_scalar_valued(self):
        """
        Guards against: 0d02d1f63 - fix bulk ocp for a half cell

        The bug was that ocp_bulk for lithium metal plating was defined as
        "0 * T" which gives a temperature-sized array instead of a scalar.
        This test verifies the bulk OCP is correctly a scalar (zero) for the
        lithium metal reaction.

        The fix changed `ocp_bulk = 0 * T` to `ocp_bulk = pybamm.Scalar(0)`.

        The original bug caused shape mismatches during model building. We
        verify:
        1. The model builds and solves successfully (no shape errors)
        2. The bulk OCP is identically zero as expected for lithium metal
        3. The bulk OCP has no spatial variation (is scalar-valued)
        """
        model = pybamm.lithium_ion.DFN({"working electrode": "positive"})
        param = pybamm.ParameterValues("Xu2019")

        sim = pybamm.Simulation(model, parameter_values=param)
        # If the bug existed (0*T instead of Scalar(0)), this could raise
        # a ShapeError during model building/solving
        sol = sim.solve([0, 600])

        # Model should run without shape errors
        assert len(sol.t) > 0

        # Negative electrode bulk OCP should be zero for lithium metal
        ocp_n_bulk = sol["Negative electrode bulk open-circuit potential [V]"].data

        # Should be exactly zero (lithium metal reference)
        # If the bug existed, this could have non-zero values from T variation
        assert np.allclose(ocp_n_bulk, 0.0, atol=1e-10), (
            f"Lithium metal bulk OCP should be 0, got {ocp_n_bulk}"
        )

        # Verify it's truly scalar-valued (constant across all time points)
        # If 0*T were used, there could be slight variation
        assert np.std(ocp_n_bulk) < 1e-15, (
            "Lithium metal bulk OCP should have no variation"
        )

        # Voltage should be reasonable
        V = sol["Voltage [V]"].data
        assert np.all(V > 2.0)
        assert np.all(V < 5.0)

    def test_half_cell_simulation_runs_correctly(self):
        """
        Verify half-cell simulation produces reasonable results.
        """
        model = pybamm.lithium_ion.DFN({"working electrode": "positive"})
        param = pybamm.ParameterValues("Xu2019")
        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 1000])

        # Check solution is valid
        assert len(sol.t) > 0

        # Check voltage is reasonable
        V = sol["Voltage [V]"].data
        assert np.all(V > 2.0)
        assert np.all(V < 5.0)

        # Check capacity is reasonable
        Q = sol["Discharge capacity [A.h]"].data
        assert Q[-1] > 0  # Some capacity should be discharged


class TestHalfCellDomainParameter:
    """Guards for half-cell domain parameter bug fixes."""

    def test_half_cell_domain_parameters(self):
        """
        Guards against: 43b516f77 - fix domain parameter bug (#3936)

        Verify half-cell models have correct domain handling.
        """
        model = pybamm.lithium_ion.SPM({"working electrode": "positive"})
        param = pybamm.ParameterValues("Xu2019")

        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 600])

        # Model should run without domain errors
        assert len(sol.t) > 0

        # Verify positive electrode variables exist and are valid
        c_s_p = sol["Positive particle concentration [mol.m-3]"].data
        assert not np.any(np.isnan(c_s_p))
        assert np.all(c_s_p > 0)


class TestHalfCellInterfaceSubmodel:
    """Guards for half-cell interface submodel bug fixes."""

    def test_half_cell_interface_variables(self):
        """
        Guards against: 5302e86b8 - fix half-cell bug

        Verify half-cell models have correct interface variable handling.
        """
        model = pybamm.lithium_ion.SPMe({"working electrode": "positive"})
        param = pybamm.ParameterValues("Xu2019")

        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 600])

        # Model should run and have interface variables
        assert len(sol.t) > 0

        # Check that exchange current density is valid
        j = sol["Positive electrode exchange current density [A.m-2]"].data
        assert not np.any(np.isnan(j))
        assert np.all(j > 0)  # Should be positive during normal operation

        # Check that reaction overpotential is valid
        eta = sol["X-averaged positive electrode reaction overpotential [V]"].data
        assert not np.any(np.isnan(eta))
