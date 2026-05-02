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

        This test directly verifies the formula by:
        1. Getting phi_s (positive electrode potential) as a function of x
        2. Computing boundary_value(phi_s, "right") - x_average(phi_s) manually
        3. Comparing with the "X-averaged positive electrode ohmic losses [V]"
        """
        model = pybamm.lithium_ion.DFN()
        param = pybamm.ParameterValues("Chen2020")

        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 600])

        # Pick a time point during discharge
        t = 300

        # Get the positive electrode potential (spatially varying)
        phi_s_p = sol["Positive electrode potential [V]"](t=t, x=None)

        # The boundary value at "right" is the last spatial point
        # in the positive electrode domain
        phi_s_p_boundary_right = phi_s_p[-1]

        # x-average of (v - phi_s) = v - x_average(phi_s)
        # where v = boundary_value at right
        phi_s_p_avg = np.mean(phi_s_p)
        expected_ohmic_losses = phi_s_p_boundary_right - phi_s_p_avg

        # Get the computed ohmic losses from the model
        computed_ohmic_losses = float(
            sol["X-averaged positive electrode ohmic losses [V]"](t)
        )

        # They should match (allowing for discretization differences)
        np.testing.assert_allclose(
            computed_ohmic_losses,
            expected_ohmic_losses,
            rtol=0.1,  # Allow 10% tolerance for discretization differences
            err_msg=(
                f"Ohmic losses formula mismatch: computed={computed_ohmic_losses}, "
                f"expected (v - avg(phi_s))={expected_ohmic_losses}"
            ),
        )

    def test_negative_electrode_ohmic_losses_formula_correct(self):
        """
        Verify negative electrode ohmic losses use the correct formula:
        delta_phi_s = boundary_value(phi_s, "left") - phi_s
        """
        model = pybamm.lithium_ion.DFN()
        param = pybamm.ParameterValues("Chen2020")

        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 600])

        t = 300

        # Get the negative electrode potential (spatially varying)
        phi_s_n = sol["Negative electrode potential [V]"](t=t, x=None)

        # The boundary value at "left" is the first spatial point
        phi_s_n_boundary_left = phi_s_n[0]

        # x-average of (v - phi_s) = v - x_average(phi_s)
        phi_s_n_avg = np.mean(phi_s_n)
        expected_ohmic_losses = phi_s_n_boundary_left - phi_s_n_avg

        # Get the computed ohmic losses
        computed_ohmic_losses = float(
            sol["X-averaged negative electrode ohmic losses [V]"](t)
        )

        np.testing.assert_allclose(
            computed_ohmic_losses,
            expected_ohmic_losses,
            rtol=0.1,
            err_msg=(
                f"Negative electrode ohmic losses formula mismatch: "
                f"computed={computed_ohmic_losses}, expected={expected_ohmic_losses}"
            ),
        )

    def test_ohmic_losses_opposite_sign_for_opposite_current(self):
        """
        Verify ohmic losses change sign when current direction reverses.

        During discharge: current flows in positive direction
        During charge: current flows in negative direction

        The ohmic losses should have opposite signs.
        """
        model = pybamm.lithium_ion.DFN()
        param_discharge = pybamm.ParameterValues("Chen2020")
        param_charge = pybamm.ParameterValues("Chen2020")

        # Discharge at 1C
        param_discharge.update({"Current function [A]": 5.0})
        # Charge at 1C (negative current)
        param_charge.update({"Current function [A]": -5.0})

        sim_discharge = pybamm.Simulation(model, parameter_values=param_discharge)
        sim_charge = pybamm.Simulation(model, parameter_values=param_charge)

        sol_discharge = sim_discharge.solve([0, 300])
        sol_charge = sim_charge.solve([0, 300], initial_soc=0.5)

        t = 150

        losses_p_discharge = float(
            sol_discharge["X-averaged positive electrode ohmic losses [V]"](t)
        )
        losses_p_charge = float(
            sol_charge["X-averaged positive electrode ohmic losses [V]"](t)
        )

        # Signs should be opposite (or one near zero if at rest)
        # The product of discharge and charge losses should be negative
        assert losses_p_discharge * losses_p_charge < 0, (
            f"Ohmic losses should have opposite sign for opposite current: "
            f"discharge={losses_p_discharge}, charge={losses_p_charge}"
        )
