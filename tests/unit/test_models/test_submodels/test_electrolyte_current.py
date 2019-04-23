#
# Tests for the electrolyte submodels
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import tests
import numbers
import numpy as np

import unittest


@unittest.skip("")
class TestMacInnesStefanMaxwell(unittest.TestCase):
    def test_basic_processing(self):
        # Parameters
        param = pybamm.standard_parameters_lithium_ion

        # Variables
        phi_e = pybamm.standard_variables.phi_e
        c_e = pybamm.standard_variables.c_e

        # Interfacial current density
        int_curr_model = pybamm.interface.InterfacialCurrent(param)
        variables = int_curr_model.get_homogeneous_interfacial_current()
        variables.update({"Electrolyte concentration": c_e})

        # Set up model
        model = pybamm.electrolyte_current.MacInnesStefanMaxwell(param)
        model.set_algebraic_system(phi_e, variables)

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

    def test_explicit(self):
        param = pybamm.standard_parameters_lithium_ion

        # Set up
        c_e_n = pybamm.Broadcast(1, domain=["negative electrode"])
        c_e_s = pybamm.Broadcast(1, domain=["separator"])
        c_e_p = pybamm.Broadcast(1, domain=["positive electrode"])
        c_e = pybamm.Concatenation(c_e_n, c_e_s, c_e_p)

        ocp_n = pybamm.Broadcast(0, domain=["negative electrode"])
        eta_r_n = pybamm.Broadcast(0, domain=["negative electrode"])

        in_vars = {
            "Electrolyte concentration": c_e,
            "Negative electrode open circuit potential": ocp_n,
            "Negative reaction overpotential": eta_r_n,
        }

        # Model
        model = pybamm.electrolyte_current.MacInnesStefanMaxwell(param)
        leading_order_vars = model.get_explicit_leading_order(in_vars)
        combined_vars = model.get_explicit_combined(in_vars)

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
            phi_e = out_vars["Electrolyte potential"]
            i_e = out_vars["Electrolyte current density"]
            delta_phi_e_av = out_vars["Average electrolyte ohmic losses"]
            eta_c_av = out_vars["Average concentration overpotential"]

            # Test
            self.assertIsInstance(phi_e, pybamm.Concatenation)
            self.assertIsInstance(i_e, pybamm.Concatenation)
            self.assertIsInstance(delta_phi_e_av, pybamm.Symbol)
            self.assertIsInstance(eta_c_av, pybamm.Symbol)

            phi_e_disc = modeltest.disc.process_symbol(phi_e)
            phi_e_eval = phi_e_disc.evaluate(0, None)

            i_e_disc = modeltest.disc.process_symbol(i_e)
            i_e_eval = i_e_disc.evaluate(0, None)

            delta_phi_e_disc = modeltest.disc.process_symbol(delta_phi_e_av)
            delta_phi_e_eval = delta_phi_e_disc.evaluate(0, None)

            self.assertTrue(type(phi_e_eval) is np.ndarray)
            self.assertTrue(type(i_e_eval) is np.ndarray)
            self.assertIsInstance(delta_phi_e_eval, numbers.Number)

            np.testing.assert_array_less(0, i_e_eval)
            np.testing.assert_array_less(i_e_eval, 1.01)

            if order == "leading":
                self.assertEqual(delta_phi_e_eval, 0)
            if order == "combined":
                self.assertLess(delta_phi_e_eval, 0)

            # check that left boundary of phi_e is approx 0
            phi_e_left = pybamm.BoundaryValue(phi_e, "left")
            phi_e_left_disc = modeltest.disc.process_symbol(phi_e_left)
            phi_e_left_eval = phi_e_left_disc.evaluate(0, None)

            np.testing.assert_almost_equal(phi_e_left_eval, 0, 3)  # extrapolation error


class TestMacInnesCapacitance(unittest.TestCase):
    def test_basic_processing(self):
        # Parameters
        param = pybamm.standard_parameters_lithium_ion

        # Variables
        delta_phi_n = pybamm.standard_variables.delta_phi_n
        delta_phi_p = pybamm.standard_variables.delta_phi_p
        c_e_n = pybamm.standard_variables.c_e_n
        c_e_p = pybamm.standard_variables.c_e_p

        # Interfacial current density
        int_curr_model = pybamm.interface.InterfacialCurrent(param)
        variables = int_curr_model.get_homogeneous_interfacial_current()
        variables.update(
            {
                "Negative electrolyte concentration": c_e_n,
                "Positive electrolyte concentration": c_e_p,
            }
        )

        # Negative electrode
        model_n = pybamm.electrolyte_current.MacInnesCapacitance(param)
        model_n.set_differential_system(delta_phi_n, variables)
        # Update model for tests
        model_n.rhs.update({c_e_n: pybamm.Scalar(0)})
        model_n.initial_conditions.update({c_e_n: pybamm.Scalar(1)})
        # Test
        modeltest_n = tests.StandardModelTest(model_n)
        modeltest_n.test_all()

        # Positive electrode
        model_p = pybamm.electrolyte_current.MacInnesCapacitance(param)
        model_p.set_differential_system(delta_phi_p, variables)
        # Update model for tests
        model_p.rhs.update({c_e_p: pybamm.Scalar(0)})
        model_p.initial_conditions.update({c_e_p: pybamm.Scalar(1)})
        # Test
        modeltest_p = tests.StandardModelTest(model_p)
        modeltest_p.test_all()

        # Both
        model_n = pybamm.electrolyte_current.MacInnesCapacitance(param)
        model_n.set_differential_system(delta_phi_n, variables)
        model_p = pybamm.electrolyte_current.MacInnesCapacitance(param)
        model_p.set_differential_system(delta_phi_p, variables)
        model_n.update(model_p)
        model_whole = model_n
        model_whole.rhs.update({c_e_n: pybamm.Scalar(0), c_e_p: pybamm.Scalar(0)})
        model_whole.initial_conditions.update(
            {c_e_n: pybamm.Scalar(1), c_e_p: pybamm.Scalar(1)}
        )
        model_whole_test = tests.StandardModelTest(model_whole)
        model_whole_test.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
