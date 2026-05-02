"""
Regression tests for historical electrode submodel bug fixes.

These tests guard against the reintroduction of bugs that were fixed but
did not have regression tests added at the time of the fix.
"""

import numpy as np

import pybamm


class TestPositiveElectrodeOhmicLossesFixes:
    """Guards for positive electrode ohmic losses bug fixes."""

    def test_positive_electrode_ohmic_losses_formula_correct(self):
        """
        Guards against: PR #1407 - #1406 fixed bug with positive electrode ohmic losses

        The bug was a sign error in the calculation of delta_phi_s for the
        positive electrode. The code had:
            delta_phi_s = phi_s - v  (WRONG)
        but should have been:
            delta_phi_s = v - phi_s  (CORRECT)

        where v = boundary_value(phi_s, "right") is the potential at the
        current collector.
        """
        model = pybamm.lithium_ion.DFN()
        param = pybamm.ParameterValues("Chen2020")

        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 600])

        t = 300

        phi_s_p = sol["Positive electrode potential [V]"](t=t, x=None)

        phi_s_p_boundary_right = phi_s_p[-1]

        phi_s_p_avg = np.mean(phi_s_p)
        expected_ohmic_losses = phi_s_p_boundary_right - phi_s_p_avg

        computed_ohmic_losses = float(
            sol["X-averaged positive electrode ohmic losses [V]"](t)
        )

        np.testing.assert_allclose(
            computed_ohmic_losses,
            expected_ohmic_losses,
            rtol=0.1,
            err_msg=(
                f"Ohmic losses formula mismatch: computed={computed_ohmic_losses}, "
                f"expected (v - avg(phi_s))={expected_ohmic_losses}"
            ),
        )
