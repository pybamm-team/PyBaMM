#
# Tests for the electrolyte submodels
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
from pybamm.solvers.scikits_ode_solver import scikits_odes_spec
import tests
import numpy as np
import numbers

import unittest


@unittest.skipIf(scikits_odes_spec is None, "scikits.odes not installed")
class TestOhm(unittest.TestCase):
    def test_basic_processing(self):
        # Parameters
        param = pybamm.standard_parameters_lithium_ion

        # Variables and reactions
        phi_s_n = pybamm.standard_variables.phi_s_n
        phi_s_p = pybamm.standard_variables.phi_s_p
        onen = pybamm.Broadcast(1, ["negative electrode"])
        onep = pybamm.Broadcast(1, ["positive electrode"])
        reactions = {
            "main": {"neg": {"s_plus": 1, "aj": onen}, "pos": {"s_plus": 1, "aj": onep}}
        }

        # Set up model and test
        # Negative only
        model_n = pybamm.electrode.Ohm(param)
        model_n.set_algebraic_system(phi_s_n, reactions)
        model_n_test = tests.StandardModelTest(model_n)
        model_n_test.test_all()

        # Positive only
        model_p = pybamm.electrode.Ohm(param)
        model_p.set_algebraic_system(phi_s_p, reactions)
        # overwrite boundary conditions for purposes of the test
        i_s_p = model_p.variables["Positive electrode current density"]
        model_p.boundary_conditions = {phi_s_p: {"right": 0}, i_s_p: {"left": 0}}
        model_p_test = tests.StandardModelTest(model_p)
        model_p_test.test_all()

        # Both
        model_n = pybamm.electrode.Ohm(param)
        model_n.set_algebraic_system(phi_s_n, reactions)
        model_p = pybamm.electrode.Ohm(param)
        model_p.set_algebraic_system(phi_s_p, reactions)
        model_n.update(model_p)
        model_whole = model_n
        # overwrite boundary conditions for purposes of the test
        i_s_n = model_whole.variables["Negative electrode current density"]
        i_s_p = model_whole.variables["Positive electrode current density"]
        model_whole.boundary_conditions = {
            phi_s_n: {"left": 0},
            i_s_n: {"right": 0},
            phi_s_p: {"right": 0},
            i_s_p: {"left": 0},
        }
        model_whole_test = tests.StandardModelTest(model_whole)
        model_whole_test.test_all()

    def test_explicit(self):
        param = pybamm.standard_parameters_lithium_ion

        # Set up
        phi_e_n = pybamm.Broadcast(1, domain=["negative electrode"])
        phi_e_s = pybamm.Broadcast(1, domain=["separator"])
        phi_e_p = pybamm.Broadcast(1, domain=["positive electrode"])
        phi_e = pybamm.Concatenation(phi_e_n, phi_e_s, phi_e_p)

        ocp_p = pybamm.Scalar(0)
        eta_r_p = pybamm.Scalar(0)

        # Model
        model = pybamm.electrode.Ohm(param)
        leading_order_vars = model.get_explicit_leading_order(ocp_p, eta_r_p, phi_e)
        phi_s_n = model.get_neg_pot_explicit_combined()
        combined_vars = model.get_explicit_combined(phi_s_n, phi_e, ocp_p, eta_r_p)

        # Get disc
        modeltest = tests.StandardModelTest(model)

        for order, out_vars in [
            ("leading", leading_order_vars),
            ("combined", combined_vars),
        ]:
            # Process parameters
            for name, var in out_vars.items():
                out_vars[name] = modeltest.parameter_values.process_symbol(var)

            # Unpack
            phi_s = out_vars["Electrode potential"]
            i_s = out_vars["Electrode current density"]
            delta_phi_s_av = out_vars["Average solid phase ohmic losses"]

            self.assertIsInstance(phi_s, pybamm.Concatenation)
            self.assertIsInstance(i_s, pybamm.Concatenation)
            self.assertIsInstance(delta_phi_s_av, pybamm.Symbol)

            phi_s_disc = modeltest.disc.process_symbol(phi_s)
            phi_s_eval = phi_s_disc.evaluate(0, None)

            i_s_disc = modeltest.disc.process_symbol(i_s)
            i_s_eval = i_s_disc.evaluate(0, None)

            delta_phi_s_disc = modeltest.disc.process_symbol(delta_phi_s_av)
            delta_phi_s_eval = delta_phi_s_disc.evaluate(0, None)

            self.assertTrue(type(phi_s_eval) is np.ndarray)
            self.assertTrue(type(i_s_eval) is np.ndarray)
            self.assertIsInstance(delta_phi_s_eval, numbers.Number)

            np.testing.assert_array_less(-0.001, i_s_eval)
            np.testing.assert_array_less(i_s_eval, 1.001)

            if order == "leading":
                self.assertEqual(delta_phi_s_eval, 0)
            elif order == "combined":
                self.assertLess(delta_phi_s_eval, 0)

            # check that left boundary of phi_s is approx 0
            phi_s_left = pybamm.BoundaryValue(phi_s, "left")
            phi_s_left_disc = modeltest.disc.process_symbol(phi_s_left)
            phi_s_left_eval = phi_s_left_disc.evaluate(0, None)

            # check that right boundary of phi_s is approx 1 (phi_e = 1 aswell)
            phi_s_right = pybamm.BoundaryValue(phi_s, "right")
            phi_s_right_disc = modeltest.disc.process_symbol(phi_s_right)
            phi_s_right_eval = phi_s_right_disc.evaluate(0, None)

            np.testing.assert_almost_equal(phi_s_left_eval, 0, 3)  # extrapolation error

            if order == "leading":
                np.testing.assert_almost_equal(phi_s_right_eval, 1, 3)
            if order == "combined":
                # have ocp_p_av = eta_p_av = 0 and phi_e_p_av = 1
                # so phi_s_p_av = 1

                i_cell = param.current_with_time
                _, _, eps_p = [e.orphans[0] for e in param.epsilon.orphans]
                sigma_p_eff = param.sigma_p * (1 - eps_p)
                voltage = 1 - i_cell * param.l_p / 3 / sigma_p_eff

                voltage_param = modeltest.parameter_values.process_symbol(voltage)
                voltage_disc = modeltest.disc.process_symbol(voltage_param)
                voltage_eval = voltage_disc.evaluate(0, None)

                np.testing.assert_almost_equal(
                    phi_s_right_eval, voltage_eval, 3
                )  # extrapolation error

    def test_failure(self):
        param = pybamm.standard_parameters_lithium_ion
        model = pybamm.electrode.Ohm(param)
        phi_s = pybamm.Symbol("sym", domain="test")
        with self.assertRaises(pybamm.DomainError):
            model.set_algebraic_system(phi_s, None)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
