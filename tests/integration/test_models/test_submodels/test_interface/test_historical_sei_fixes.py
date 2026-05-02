"""
Regression tests for historical SEI and degradation model bug fixes.

These tests guard against the reintroduction of bugs that were fixed but
did not have regression tests added at the time of the fix.
"""

import numpy as np
import pytest

import pybamm


class TestSEIThicknessFixes:
    """Guards for SEI thickness calculation bug fixes."""

    @pytest.mark.parametrize(
        "sei_option",
        ["electron-migration limited", "solvent-diffusion limited"],
    )
    def test_sei_thickness_never_decreases(self, sei_option):
        """
        Guards against: PR #3622 - Fixed a bug where the SEI thickness decreased
        at some intervals when using the 'electron-migration limited' model.

        The bug was that the SEI current density (j_sei) could become positive
        under certain conditions. The fix applied a Heaviside function to ensure
        only negative contributions affect j_sei.
        """
        model = pybamm.lithium_ion.DFN({"SEI": sei_option})
        param = pybamm.ParameterValues("OKane2022")
        sim = pybamm.Simulation(model, parameter_values=param)

        sol = sim.solve([0, 3600])

        L_sei = sol["X-averaged negative SEI thickness [m]"].data

        dL_sei = np.diff(L_sei)
        assert np.all(dL_sei >= -1e-15), (
            f"SEI thickness decreased with {sei_option}. Min change: {dL_sei.min()}"
        )
        assert L_sei[-1] > L_sei[0], "SEI should grow during cycling"
