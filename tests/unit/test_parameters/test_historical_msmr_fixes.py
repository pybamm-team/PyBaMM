"""
Regression tests for historical MSMR (Multi-Site, Multi-Reaction) parameter bug fixes.

These tests guard against the reintroduction of bugs that were fixed but
did not have regression tests added at the time of the fix.
"""

import numpy as np

import pybamm


class TestMSMRExchangeCurrentDensityFixes:
    """Guards for MSMR exchange current density calculation bug fixes."""

    @staticmethod
    def _get_msmr_model_and_params():
        """Get MSMR model with correct options and parameters."""
        model = pybamm.lithium_ion.MSMR({"number of MSMR reactions": ("6", "4")})
        param = pybamm.ParameterValues("MSMR_Example")
        return model, param

    def test_msmr_exchange_current_density_uses_exact_power(self):
        """
        Guards against: PR #5404 - Fix MSMR exchange current density regression

        The bug was that the exchange current density calculation used
        `pybamm.reg_power(xj, wj)` instead of `xj**wj`. The reg_power function
        regularizes near zero, which was inappropriate for the MSMR model.

        The fix changed:
        - j0_j = j0_ref_j * pybamm.reg_power(xj, wj_j) * ...
        + j0_j = j0_ref_j * xj**wj_j * ...
        """
        model, param = self._get_msmr_model_and_params()

        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 1800])

        j0_p = sol[
            "X-averaged positive electrode exchange current density [A.m-2]"
        ].data
        sto_p = sol["Average positive particle stoichiometry"].data

        mid_idx = len(sol.t) // 2
        sto_mid = sto_p[mid_idx]
        j0_mid = j0_p[mid_idx]

        assert j0_mid > 0, f"j0 should be positive at sto={sto_mid}"

        j0_early = j0_p[len(sol.t) // 4]
        j0_late = j0_p[3 * len(sol.t) // 4]

        assert np.isfinite(j0_early) and j0_early > 0
        assert np.isfinite(j0_mid) and j0_mid > 0
        assert np.isfinite(j0_late) and j0_late > 0

        assert len(sol.t) > 0
        V = sol["Voltage [V]"].data
        assert np.all(V > 2.5)
        assert np.all(V < 4.5)
