"""
Regression tests for historical OCP and hysteresis model bug fixes.

These tests guard against the reintroduction of bugs that were fixed but
did not have regression tests added at the time of the fix.
"""

import numpy as np
import pytest

import pybamm


class TestHysteresisOCPFixes:
    """Guards for hysteresis OCP calculation bug fixes."""

    @staticmethod
    def _get_hysteresis_params():
        """Get Chen2020 params with hysteresis OCP functions for negative electrode."""
        param = pybamm.ParameterValues("Chen2020")

        U_eq_neg = param["Negative electrode OCP [V]"]

        param.update(
            {
                "Negative electrode lithiation OCP [V]": lambda sto: (
                    U_eq_neg(sto) - 0.05
                ),
                "Negative electrode delithiation OCP [V]": lambda sto: (
                    U_eq_neg(sto) + 0.05
                ),
                "Negative particle lithiation hysteresis decay rate": 10,
                "Negative particle delithiation hysteresis decay rate": 10,
                "Initial hysteresis state in negative electrode": 0.0,
            }
        )
        return param

    def test_bulk_ocp_uses_bulk_stoichiometry_in_hysteresis(self):
        """
        Guards against: PR #5169 - fix calculation of bulk ocp in hysteresis
        models

        The bug was that bulk OCP terms used surface stoichiometry instead of
        bulk stoichiometry.
        """
        model = pybamm.lithium_ion.DFN(
            {"open-circuit potential": ("one-state hysteresis", "single")}
        )
        param = self._get_hysteresis_params()

        param.update({"Current function [A]": 10})

        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 300])

        V = sol["Voltage [V]"].data
        assert len(V) > 0
        assert np.all(V > 2.5)
        assert np.all(V < 4.5)

        ocp_bulk = sol["Negative electrode bulk open-circuit potential [V]"].data
        ocp_surf = sol["Negative electrode open-circuit potential [V]"].data

        assert not np.any(np.isnan(ocp_bulk))
        assert not np.any(np.isnan(ocp_surf))

        mid_start = len(ocp_bulk) // 3
        mid_end = 2 * len(ocp_bulk) // 3
        ocp_bulk_mid = np.mean(ocp_bulk[mid_start:mid_end])
        ocp_surf_mid = np.mean(ocp_surf[mid_start:mid_end])

        assert ocp_bulk_mid != pytest.approx(ocp_surf_mid, abs=1e-10)

    def test_bulk_ocp_uses_lithiation_curve_during_lithiation(self):
        """
        Guards against: PR #5280 - fix bug with bulk ocp lithiation

        The bug was that U_lith_bulk was computed without the "lithiation"
        argument, so it used the equilibrium OCP instead of the lithiation OCP.

        The fix changed:
        - U_lith_bulk = self.phase_param.U(sto_bulk, T_bulk)
        + U_lith_bulk = self.phase_param.U(sto_bulk, T_bulk, "lithiation")
        """
        model = pybamm.lithium_ion.DFN(
            {"open-circuit potential": ("one-state hysteresis", "single")}
        )
        param = self._get_hysteresis_params()
        U_eq_neg = param["Negative electrode OCP [V]"]

        param.update({"Current function [A]": -5})

        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 300], initial_soc=0.2)

        ocp_bulk = sol[
            "Battery negative electrode bulk open-circuit potential [V]"
        ].data
        sto_bulk_2d = sol["X-averaged negative particle stoichiometry"].data
        if sto_bulk_2d.ndim > 1:
            sto_bulk = np.mean(sto_bulk_2d, axis=0)
        else:
            sto_bulk = sto_bulk_2d

        mid_idx = len(ocp_bulk) // 2

        U_eq_at_sto = float(U_eq_neg(sto_bulk[mid_idx]))

        U_lith_expected = U_eq_at_sto - 0.05
        U_delith_expected = U_eq_at_sto + 0.05

        h_state_raw = sol["X-averaged negative electrode hysteresis state"].data
        if h_state_raw.ndim > 1:
            h_state = np.mean(h_state_raw, axis=0)
        else:
            h_state = h_state_raw
        assert h_state[mid_idx] < h_state[0]

        ocp_bulk_mid = ocp_bulk[mid_idx]
        assert U_lith_expected <= ocp_bulk_mid <= U_delith_expected

        dist_to_lith = abs(ocp_bulk_mid - U_lith_expected)
        dist_to_delith = abs(ocp_bulk_mid - U_delith_expected)
        if h_state[mid_idx] < -0.1:
            assert dist_to_lith < dist_to_delith
