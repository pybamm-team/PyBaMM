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

        # Define hysteresis OCP functions for negative electrode
        U_eq_neg = param["Negative electrode OCP [V]"]

        param.update(
            {
                # Negative electrode lithiation/delithiation OCPs
                "Negative electrode lithiation OCP [V]": lambda sto: (
                    U_eq_neg(sto) - 0.05
                ),
                "Negative electrode delithiation OCP [V]": lambda sto: (
                    U_eq_neg(sto) + 0.05
                ),
                # Hysteresis decay rates
                "Negative particle lithiation hysteresis decay rate": 10,
                "Negative particle delithiation hysteresis decay rate": 10,
                # Initial hysteresis state
                "Initial hysteresis state in negative electrode": 0.0,
            }
        )
        return param

    def test_bulk_ocp_uses_bulk_stoichiometry_in_hysteresis(self):
        """
        Guards against: PR #5169 - fix calculation of bulk ocp in hysteresis
        models

        The bug was that bulk OCP terms used surface stoichiometry instead of
        bulk stoichiometry. This test uses a hysteresis OCP model with DFN
        (which has particle concentration gradients) to verify bulk OCP is
        computed from bulk concentration.
        """
        model = pybamm.lithium_ion.DFN(
            {"open-circuit potential": ("one-state hysteresis", "single")}
        )
        param = self._get_hysteresis_params()

        # Use higher C-rate to create concentration gradients in particles
        param.update({"Current function [A]": 10})  # ~2C rate

        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 300])

        # The model should run and produce reasonable results
        V = sol["Voltage [V]"].data
        assert len(V) > 0
        assert np.all(V > 2.5)
        assert np.all(V < 4.5)

        # Both bulk and surface OCP should exist and be valid
        ocp_bulk = sol["Negative electrode bulk open-circuit potential [V]"].data
        ocp_surf = sol["Negative electrode open-circuit potential [V]"].data

        assert not np.any(np.isnan(ocp_bulk))
        assert not np.any(np.isnan(ocp_surf))

        # At higher C-rate, bulk and surface OCP should differ due to
        # concentration gradients - bulk uses volume-averaged concentration
        # Use middle portion of discharge to avoid initial transients
        mid_start = len(ocp_bulk) // 3
        mid_end = 2 * len(ocp_bulk) // 3
        ocp_bulk_mid = np.mean(ocp_bulk[mid_start:mid_end])
        ocp_surf_mid = np.mean(ocp_surf[mid_start:mid_end])

        # They should differ because bulk uses sto_bulk, surface uses sto_surf
        assert ocp_bulk_mid != pytest.approx(ocp_surf_mid, abs=1e-10)

    def test_hysteresis_state_evolves(self):
        """
        Verify hysteresis state variable evolves during discharge.
        """
        model = pybamm.lithium_ion.SPM(
            {"open-circuit potential": ("one-state hysteresis", "single")}
        )
        param = self._get_hysteresis_params()
        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 3600])

        # Hysteresis state should exist and evolve
        h = sol["X-averaged negative electrode hysteresis state"].data
        assert not np.any(np.isnan(h))
        # State should change during discharge
        assert h[-1] != pytest.approx(h[0], abs=1e-6)


class TestBulkOCPLithiationFixes:
    """Guards for bulk OCP lithiation calculation fixes."""

    @staticmethod
    def _get_hysteresis_params():
        """Get Chen2020 params with distinct lithiation/delithiation OCP."""
        param = pybamm.ParameterValues("Chen2020")
        U_eq_neg = param["Negative electrode OCP [V]"]

        # Create distinct lithiation and delithiation OCPs
        # The bug was that U_lith_bulk used equilibrium OCP instead of lithiation OCP
        param.update(
            {
                "Negative electrode lithiation OCP [V]": lambda sto: (
                    U_eq_neg(sto) - 0.05  # 50mV lower for lithiation
                ),
                "Negative electrode delithiation OCP [V]": lambda sto: (
                    U_eq_neg(sto) + 0.05  # 50mV higher for delithiation
                ),
                "Negative particle lithiation hysteresis decay rate": 10,
                "Negative particle delithiation hysteresis decay rate": 10,
                "Initial hysteresis state in negative electrode": 0.0,
            }
        )
        return param

    def test_bulk_ocp_uses_lithiation_curve_during_lithiation(self):
        """
        Guards against: PR #5280 - fix bug with bulk ocp lithiation

        The bug was that U_lith_bulk was computed without the "lithiation"
        argument, so it used the equilibrium OCP instead of the lithiation OCP.
        This test verifies that during charge (lithiation), the bulk OCP
        correctly uses the lithiation OCP curve.

        The fix changed:
        - U_lith_bulk = self.phase_param.U(sto_bulk, T_bulk)
        + U_lith_bulk = self.phase_param.U(sto_bulk, T_bulk, "lithiation")
        """
        model = pybamm.lithium_ion.DFN(
            {"open-circuit potential": ("one-state hysteresis", "single")}
        )
        param = self._get_hysteresis_params()
        U_eq_neg = param["Negative electrode OCP [V]"]

        # Use charging (negative current = lithiation)
        param.update({"Current function [A]": -5})

        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 300], initial_soc=0.2)

        # Get bulk OCP (x-averaged, 1D over time) and stoichiometry
        ocp_bulk = sol[
            "Battery negative electrode bulk open-circuit potential [V]"
        ].data
        # sto_bulk is 2D (space x time), average over space to get 1D
        sto_bulk_2d = sol["X-averaged negative particle stoichiometry"].data
        if sto_bulk_2d.ndim > 1:
            sto_bulk = np.mean(sto_bulk_2d, axis=0)
        else:
            sto_bulk = sto_bulk_2d

        # Use middle of simulation to avoid initial transients
        mid_idx = len(ocp_bulk) // 2

        # Get equilibrium OCP at the bulk stoichiometry
        U_eq_at_sto = float(U_eq_neg(sto_bulk[mid_idx]))

        # Expected lithiation OCP = U_eq - 0.05 (from our parameter definition)
        U_lith_expected = U_eq_at_sto - 0.05
        # Expected delithiation OCP = U_eq + 0.05
        U_delith_expected = U_eq_at_sto + 0.05

        # During lithiation (h trending toward -1), bulk OCP should be closer
        # to the lithiation curve than to equilibrium or delithiation
        # OCP_bulk = (1+h)/2 * U_delith + (1-h)/2 * U_lith, where h ∈ [-1, 1]
        # At h=-1 (full lithiation): OCP_bulk = U_lith
        # At h=0 (equilibrium): OCP_bulk = (U_delith + U_lith)/2 = U_eq
        # At h=1 (full delithiation): OCP_bulk = U_delith

        # The hysteresis state h should be trending toward -1 during lithiation
        h_state_raw = sol["X-averaged negative electrode hysteresis state"].data
        if h_state_raw.ndim > 1:
            h_state = np.mean(h_state_raw, axis=0)
        else:
            h_state = h_state_raw
        assert h_state[mid_idx] < h_state[0], (
            "During lithiation, h should decrease (trend toward -1)"
        )

        # Verify bulk OCP is between lithiation and delithiation curves
        ocp_bulk_mid = ocp_bulk[mid_idx]
        assert U_lith_expected <= ocp_bulk_mid <= U_delith_expected, (
            f"Bulk OCP ({ocp_bulk_mid:.4f}V) should be between "
            f"lithiation ({U_lith_expected:.4f}V) and delithiation ({U_delith_expected:.4f}V)"
        )

        # With h < 0 (lithiation direction), bulk OCP should be closer to lithiation
        # curve than to delithiation curve
        dist_to_lith = abs(ocp_bulk_mid - U_lith_expected)
        dist_to_delith = abs(ocp_bulk_mid - U_delith_expected)
        if h_state[mid_idx] < -0.1:  # Only check if h is significantly negative
            assert dist_to_lith < dist_to_delith, (
                f"With h={h_state[mid_idx]:.3f}, bulk OCP should be closer to "
                f"lithiation curve, but dist_to_lith={dist_to_lith:.4f} >= "
                f"dist_to_delith={dist_to_delith:.4f}"
            )
