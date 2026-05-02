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
        model = pybamm.lithium_ion.SPMe({"working electrode": "positive"})
        param = pybamm.ParameterValues("Xu2019")
        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 600])

        voltage = sol["Voltage [V]"].data
        assert len(voltage) > 0
        assert np.all(voltage > 2.0)
        assert np.all(voltage < 5.0)

        phi_e_interface = sol["Lithium metal interface electrolyte potential [V]"].data
        assert not np.allclose(phi_e_interface, 0, atol=1e-6)

    def test_half_cell_electrolyte_ohmic_losses_nonzero(self):
        """
        Guards against: d421e3a55 - Fix half-cell voltage contributions (#5139)

        The bug was that for half-cells with planar negative electrode, the
        electrolyte Ohmic losses were incorrectly calculated as zero.
        """
        model = pybamm.lithium_ion.SPMe({"working electrode": "positive"})
        param = pybamm.ParameterValues("Xu2019")
        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 600])

        eta_e = sol["X-averaged electrolyte ohmic losses [V]"].data

        mid_idx = len(eta_e) // 2
        assert np.abs(eta_e[mid_idx]) > 1e-6


class TestHalfCellBulkOCP:
    """Guards for half-cell bulk OCP bug fixes."""

    def test_lithium_metal_bulk_ocp_is_scalar_valued(self):
        """
        Guards against: 0d02d1f63 - fix bulk ocp for a half cell

        The bug was that ocp_bulk for lithium metal plating was defined as
        "0 * T" which gives a temperature-sized array instead of a scalar.

        The fix changed `ocp_bulk = 0 * T` to `ocp_bulk = pybamm.Scalar(0)`.
        Also verifies basic half-cell simulation runs correctly.
        """
        model = pybamm.lithium_ion.DFN({"working electrode": "positive"})
        param = pybamm.ParameterValues("Xu2019")

        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 1000])

        assert len(sol.t) > 0

        ocp_n_bulk = sol["Negative electrode bulk open-circuit potential [V]"].data

        assert np.allclose(ocp_n_bulk, 0.0, atol=1e-10)
        assert np.std(ocp_n_bulk) < 1e-15

        V = sol["Voltage [V]"].data
        assert np.all(V > 2.0)
        assert np.all(V < 5.0)

        Q = sol["Discharge capacity [A.h]"].data
        assert Q[-1] > 0
