#
# Tests for the electrolyte submodels
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import tests
import numpy as np
import numbers

import unittest


class TestOhm(unittest.TestCase):
    def test_basic_processing(self):
        # Parameters
        param = pybamm.standard_parameters_lithium_ion

        # Variables
        phi_s_n = pybamm.Variable(
            "Negative electrode potential", domain=["negative electrode"]
        )
        phi_s_s = pybamm.Broadcast(pybamm.Scalar(0), ["separator"])
        phi_s_p = pybamm.Variable(
            "Positive electrode potential", domain=["positive electrode"]
        )
        phi_s = pybamm.Concatenation(phi_s_n, phi_s_s, phi_s_p)

        # Current density
        j = pybamm.interface.homogeneous_reaction(
            ["negative electrode", "separator", "positive electrode"]
        )
        j_n, j_s, j_p = j.orphans

        # Set up model and test
        model_n = pybamm.electrode.Ohm(phi_s_n, j_n, param)
        model_n_test = tests.StandardModelTest(model_n)
        model_n_test.test_all()

        model_p = pybamm.electrode.Ohm(phi_s_p, j_p, param)
        # overwrite boundary conditions for purposes of the test
        i_s_p = model_p.variables["Positive electrode current density"]
        model_p.boundary_conditions = {phi_s_p: {"right": 0}, i_s_p: {"left": 0}}
        model_p_test = tests.StandardModelTest(model_p)
        model_p_test.test_all()

        model_whole = pybamm.electrode.Ohm(phi_s, j, param)
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


class TestExplicitOhm(unittest.TestCase):
    def test_explicit_combined_ohm(self):

        param = pybamm.standard_parameters_lithium_ion

        base_model = pybamm.LithiumIonBaseModel()
        mtest = tests.StandardModelTest(base_model)

        phi_e_n = pybamm.Broadcast(1, domain=["negative electrode"])
        phi_e_s = pybamm.Broadcast(1, domain=["separator"])
        phi_e_p = pybamm.Broadcast(1, domain=["positive electrode"])
        phi_e = pybamm.Concatenation(phi_e_n, phi_e_s, phi_e_p)

        ocp_p = pybamm.Broadcast(0, domain=["negative electrode"])
        eta_r_p = pybamm.Broadcast(0, domain=["negative electrode"])

        eco = pybamm.electrode.explicit_combined_ohm
        phi_s, i_s, Delta_Phi_s_av = eco(param, phi_e, ocp_p, eta_r_p)

        self.assertIsInstance(phi_s, pybamm.Concatenation)
        self.assertIsInstance(i_s, pybamm.Concatenation)
        self.assertIsInstance(Delta_Phi_s_av, pybamm.Symbol)

        phi_s_param = mtest.parameter_values.process_symbol(phi_s)
        phi_s_disc = mtest.disc.process_symbol(phi_s_param)
        phi_s_eval = phi_s_disc.evaluate(0, None)

        i_s_param = mtest.parameter_values.process_symbol(i_s)
        i_s_disc = mtest.disc.process_symbol(i_s_param)
        i_s_eval = i_s_disc.evaluate(0, None)

        Delta_Phi_s_param = mtest.parameter_values.process_symbol(Delta_Phi_s_av)
        Delta_Phi_s_disc = mtest.disc.process_symbol(Delta_Phi_s_param)
        Delta_Phi_s_eval = Delta_Phi_s_disc.evaluate(0, None)

        self.assertTrue(type(phi_s_eval) is np.ndarray)
        self.assertTrue(type(i_s_eval) is np.ndarray)
        self.assertIsInstance(Delta_Phi_s_eval, numbers.Number)

        np.testing.assert_array_less(-0.001, i_s_eval)
        np.testing.assert_array_less(i_s_eval, 1.001)

        self.assertLess(Delta_Phi_s_eval, 0)

        # check that left boundary of phi_s is approx 0
        phi_s_left = pybamm.BoundaryValue(phi_s, "left")
        phi_s_left_param = mtest.parameter_values.process_symbol(phi_s_left)
        phi_s_left_disc = mtest.disc.process_symbol(phi_s_left_param)
        phi_s_left_eval = phi_s_left_disc.evaluate(0, None)

        # check that right boundary of phi_s is approx 1 (phi_e = 1 aswell)
        phi_s_right = pybamm.BoundaryValue(phi_s, "right")
        phi_s_right_param = mtest.parameter_values.process_symbol(phi_s_right)
        phi_s_right_disc = mtest.disc.process_symbol(phi_s_right_param)
        phi_s_right_eval = phi_s_right_disc.evaluate(0, None)

        np.testing.assert_almost_equal(phi_s_left_eval, 0, 3)  # extrapolation error
        np.testing.assert_almost_equal(phi_s_right_eval, 1, 3)  # extrapolation error

    def test_explicit_leading_order_ohm(self):

        param = pybamm.standard_parameters_lithium_ion

        base_model = pybamm.LithiumIonBaseModel()
        mtest = tests.StandardModelTest(base_model)

        phi_e_n = pybamm.Broadcast(1, domain=["negative electrode"])
        phi_e_s = pybamm.Broadcast(1, domain=["separator"])
        phi_e_p = pybamm.Broadcast(1, domain=["positive electrode"])
        phi_e = pybamm.Concatenation(phi_e_n, phi_e_s, phi_e_p)

        ocp_p = pybamm.Broadcast(0, domain=["negative electrode"])
        eta_r_p = pybamm.Broadcast(0, domain=["negative electrode"])

        eloo = pybamm.electrode.explicit_leading_order_ohm
        phi_s, i_s, Delta_Phi_s_av = eloo(param, phi_e, ocp_p, eta_r_p)

        self.assertIsInstance(phi_s, pybamm.Concatenation)
        self.assertIsInstance(i_s, pybamm.Concatenation)
        self.assertIsInstance(Delta_Phi_s_av, pybamm.Symbol)

        phi_s_param = mtest.parameter_values.process_symbol(phi_s)
        phi_s_disc = mtest.disc.process_symbol(phi_s_param)
        phi_s_eval = phi_s_disc.evaluate(0, None)

        i_s_param = mtest.parameter_values.process_symbol(i_s)
        i_s_disc = mtest.disc.process_symbol(i_s_param)
        i_s_eval = i_s_disc.evaluate(0, None)

        Delta_Phi_s_param = mtest.parameter_values.process_symbol(Delta_Phi_s_av)
        Delta_Phi_s_disc = mtest.disc.process_symbol(Delta_Phi_s_param)
        Delta_Phi_s_eval = Delta_Phi_s_disc.evaluate(0, None)

        self.assertTrue(type(phi_s_eval) is np.ndarray)
        self.assertTrue(type(i_s_eval) is np.ndarray)
        self.assertIsInstance(Delta_Phi_s_eval, numbers.Number)

        np.testing.assert_array_less(-0.001, i_s_eval)
        np.testing.assert_array_less(i_s_eval, 1.001)

        self.assertEqual(Delta_Phi_s_eval, 0)

        # check that left boundary of phi_s is approx 0
        phi_s_left = pybamm.BoundaryValue(phi_s, "left")
        phi_s_left_param = mtest.parameter_values.process_symbol(phi_s_left)
        phi_s_left_disc = mtest.disc.process_symbol(phi_s_left_param)
        phi_s_left_eval = phi_s_left_disc.evaluate(0, None)

        # check that right boundary of phi_s is approx 1 (phi_e = 1 aswell)
        phi_s_right = pybamm.BoundaryValue(phi_s, "right")
        phi_s_right_param = mtest.parameter_values.process_symbol(phi_s_right)
        phi_s_right_disc = mtest.disc.process_symbol(phi_s_right_param)
        phi_s_right_eval = phi_s_right_disc.evaluate(0, None)

        np.testing.assert_almost_equal(phi_s_left_eval, 0, 3)  # extrapolation error
        np.testing.assert_almost_equal(phi_s_right_eval, 1, 3)  # extrapolation error


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
