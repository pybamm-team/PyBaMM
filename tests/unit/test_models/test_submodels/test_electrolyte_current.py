#
# Tests for the electrolyte submodels
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
from pybamm.solvers.scikits_ode_solver import scikits_odes_spec
import tests
import numbers
import numpy as np

import unittest


@unittest.skipIf(scikits_odes_spec is None, "scikits.odes not installed")
class TestMacInnesStefanMaxwell(unittest.TestCase):
    def test_basic_processing(self):
        # Parameters
        param = pybamm.standard_parameters_lithium_ion

        # Variables
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        phi_e = pybamm.Variable("potential", whole_cell)

        # Other
        c_e = pybamm.Variable("concentration", whole_cell)
        eps = pybamm.Broadcast(pybamm.Scalar(1), whole_cell)
        j = pybamm.interface.homogeneous_reaction(whole_cell)

        # Set up model
        model = pybamm.electrolyte_current.MacInnesStefanMaxwell(
            c_e, phi_e, j, param, eps=eps
        )
        # some small changes so that tests pass
        i_e = model.variables["Electrolyte current density"]
        model.algebraic.update({c_e: c_e - pybamm.Scalar(1)})
        model.initial_conditions.update({c_e: pybamm.Scalar(1)})
        model.boundary_conditions = {
            c_e: {"left": 1},
            phi_e: {"left": 0},
            i_e: {"right": 0},
        }

        # Test
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()


class TestFirstOrderPotential(unittest.TestCase):
    def test_basic_processing(self):
        loqs_model = pybamm.lead_acid.LOQS()
        c_e_n = pybamm.Broadcast(pybamm.Scalar(1), ["negative electrode"])
        c_e_s = pybamm.Broadcast(pybamm.Scalar(1), ["separator"])
        c_e_p = pybamm.Broadcast(pybamm.Scalar(1), ["positive electrode"])
        c_e = pybamm.Concatenation(c_e_n, c_e_s, c_e_p)

        param = pybamm.standard_parameters_lead_acid

        model = pybamm.electrolyte_current.StefanMaxwellFirstOrderPotential(
            loqs_model, c_e, param
        )

        parameter_values = loqs_model.default_parameter_values
        parameter_values.process_model(model)


class TestExplicitStefanMaxwell(unittest.TestCase):
    def test_explicit_combined_stefan_maxwell(self):

        param = pybamm.standard_parameters_lithium_ion

        base_model = pybamm.LithiumIonBaseModel()
        mtest = tests.StandardModelTest(base_model)

        c_e_n = pybamm.Broadcast(1, domain=["negative electrode"])
        c_e_s = pybamm.Broadcast(1, domain=["separator"])
        c_e_p = pybamm.Broadcast(1, domain=["positive electrode"])
        c_e = pybamm.Concatenation(c_e_n, c_e_s, c_e_p)

        ocp_n = pybamm.Broadcast(0, domain=["negative electrode"])
        eta_r_n = pybamm.Broadcast(0, domain=["negative electrode"])

        ecsm = pybamm.electrolyte_current.explicit_combined_stefan_maxwell
        phi_e, i_e, Delta_Phi_e_av, eta_c_av = ecsm(param, c_e, ocp_n, eta_r_n)

        self.assertIsInstance(phi_e, pybamm.Concatenation)
        self.assertIsInstance(i_e, pybamm.Concatenation)
        self.assertIsInstance(Delta_Phi_e_av, pybamm.Symbol)
        self.assertIsInstance(eta_c_av, pybamm.Symbol)

        phi_e_param = mtest.parameter_values.process_symbol(phi_e)
        phi_e_disc = mtest.disc.process_symbol(phi_e_param)
        phi_e_eval = phi_e_disc.evaluate(0, None)

        i_e_param = mtest.parameter_values.process_symbol(i_e)
        i_e_disc = mtest.disc.process_symbol(i_e_param)
        i_e_eval = i_e_disc.evaluate(0, None)

        Delta_Phi_e_param = mtest.parameter_values.process_symbol(Delta_Phi_e_av)
        Delta_Phi_e_disc = mtest.disc.process_symbol(Delta_Phi_e_param)
        Delta_Phi_e_eval = Delta_Phi_e_disc.evaluate(0, None)

        self.assertTrue(type(phi_e_eval) is np.ndarray)
        self.assertTrue(type(i_e_eval) is np.ndarray)
        self.assertIsInstance(Delta_Phi_e_eval, numbers.Number)

        np.testing.assert_array_less(0, i_e_eval)
        np.testing.assert_array_less(i_e_eval, 1.01)

        self.assertLess(Delta_Phi_e_eval, 0)

        # check that left boundary of phi_e is approx 0
        phi_e_left = pybamm.BoundaryValue(phi_e, "left")
        phi_e_left_param = mtest.parameter_values.process_symbol(phi_e_left)
        phi_e_left_disc = mtest.disc.process_symbol(phi_e_left_param)
        phi_e_left_eval = phi_e_left_disc.evaluate(0, None)

        np.testing.assert_almost_equal(phi_e_left_eval, 0, 3)  # extrapolation error

    def test_explicit_leading_order_stefan_maxwell(self):

        param = pybamm.standard_parameters_lithium_ion

        base_model = pybamm.LithiumIonBaseModel()
        mtest = tests.StandardModelTest(base_model)

        c_e_n = pybamm.Broadcast(1, domain=["negative electrode"])
        c_e_s = pybamm.Broadcast(1, domain=["separator"])
        c_e_p = pybamm.Broadcast(1, domain=["positive electrode"])
        c_e = pybamm.Concatenation(c_e_n, c_e_s, c_e_p)

        ocp_n = pybamm.Broadcast(0, domain=["negative electrode"])
        eta_r_n = pybamm.Broadcast(0, domain=["negative electrode"])

        elosm = pybamm.electrolyte_current.explicit_leading_order_stefan_maxwell
        phi_e, i_e, Delta_Phi_e_av, eta_c_av = elosm(param, c_e, ocp_n, eta_r_n)

        self.assertIsInstance(phi_e, pybamm.Concatenation)
        self.assertIsInstance(i_e, pybamm.Concatenation)
        self.assertIsInstance(Delta_Phi_e_av, pybamm.Symbol)
        self.assertIsInstance(eta_c_av, pybamm.Symbol)

        phi_e_param = mtest.parameter_values.process_symbol(phi_e)
        phi_e_disc = mtest.disc.process_symbol(phi_e_param)
        phi_e_eval = phi_e_disc.evaluate(0, None)

        i_e_param = mtest.parameter_values.process_symbol(i_e)
        i_e_disc = mtest.disc.process_symbol(i_e_param)
        i_e_eval = i_e_disc.evaluate(0, None)

        Delta_Phi_e_param = mtest.parameter_values.process_symbol(Delta_Phi_e_av)
        Delta_Phi_e_disc = mtest.disc.process_symbol(Delta_Phi_e_param)
        Delta_Phi_e_eval = Delta_Phi_e_disc.evaluate(0, None)

        self.assertTrue(type(phi_e_eval) is np.ndarray)
        self.assertTrue(type(i_e_eval) is np.ndarray)
        self.assertIsInstance(Delta_Phi_e_eval, numbers.Number)

        np.testing.assert_array_less(0, i_e_eval)
        np.testing.assert_array_less(i_e_eval, 1.01)

        self.assertEqual(Delta_Phi_e_eval, 0)

        # check that left boundary of phi_e is approx 0
        phi_e_left = pybamm.BoundaryValue(phi_e, "left")
        phi_e_left_param = mtest.parameter_values.process_symbol(phi_e_left)
        phi_e_left_disc = mtest.disc.process_symbol(phi_e_left_param)
        phi_e_left_eval = phi_e_left_disc.evaluate(0, None)

        np.testing.assert_almost_equal(phi_e_left_eval, 0, 3)  # extrapolation error


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
