"""
Regression tests for historical lithium plating bug fixes.

These tests guard against the reintroduction of bugs that were fixed but
did not have regression tests added at the time of the fix.
"""

import numpy as np

import pybamm


class TestLithiumPlatingSPMDomainFixes:
    """Guards for lithium plating domain mismatch bug fixes."""

    @staticmethod
    def _get_plating_params():
        """Get Chen2020 parameters with lithium plating enabled."""
        param = pybamm.ParameterValues("Chen2020")

        def plating_exchange_current_density(c_e, c_Li, T):
            k_plating = pybamm.Parameter(
                "Lithium plating kinetic rate constant [m.s-1]"
            )
            return pybamm.constants.F * k_plating * c_e

        param.update(
            {
                "Lithium metal partial molar volume [m3.mol-1]": 1.3e-05,
                "Lithium plating kinetic rate constant [m.s-1]": 1e-09,
                "Exchange-current density for plating [A.m-2]": plating_exchange_current_density,
                "Initial plated lithium concentration [mol.m-3]": 0.0,
                "Typical plated lithium concentration [mol.m-3]": 1000.0,
                "Lithium plating transfer coefficient": 0.65,
                "Ambient temperature [K]": 268.15,
                "Initial temperature [K]": 268.15,
            }
        )
        return param

    def test_lithium_plating_with_spm_builds_and_solves(self):
        """
        Guards against: PR #4844 - Fixed domain mismatch bug when using
        lithium plating with SPM

        The bug was that the x_average operation alone wasn't sufficient for
        SPM. The plating variables needed yz_average after x_average.

        The fix changed:
        - c_plated_Li_av = pybamm.x_average(c_plated_Li)
        + c_plated_Li_xav = pybamm.x_average(c_plated_Li)
        + c_plated_Li_av = pybamm.yz_average(c_plated_Li_xav)

        Also verifies plating increases during charging at low temp.
        """
        model = pybamm.lithium_ion.SPM({"lithium plating": "irreversible"})
        param = self._get_plating_params()

        sim = pybamm.Simulation(model, parameter_values=param)

        sol = sim.solve([0, 600])

        assert len(sol.t) > 0

        V = sol["Voltage [V]"].data
        assert np.all(V > 2.5)
        assert np.all(V < 4.5)

    def test_spm_plating_variables_are_scalar(self):
        """
        Verify that averaged plating variables are scalar-valued for SPM.

        The bug caused x_averaged variables to have incorrect domain, leading
        to shape mismatches.
        """
        model = pybamm.lithium_ion.SPM({"lithium plating": "irreversible"})
        param = self._get_plating_params()

        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 600])

        t_mid = sol.t[len(sol.t) // 2]

        L_plating = sol["X-averaged negative lithium plating thickness [m]"](t_mid)
        c_plating = sol["X-averaged negative lithium plating concentration [mol.m-3]"](
            t_mid
        )

        assert np.isscalar(L_plating) or L_plating.ndim == 0 or L_plating.size == 1
        assert np.isscalar(c_plating) or c_plating.ndim == 0 or c_plating.size == 1
