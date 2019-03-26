#
# Tests for updating parameter values
#
import pybamm

import unittest
import numpy as np
import tests


class TestUpdateParameters(unittest.TestCase):
    def test_update_parameters_eqn(self):
        a = pybamm.Scalar(1)
        b = pybamm.Scalar(2, name="test parameter")
        c = pybamm.Scalar(3)
        eqn = a + b * c
        self.assertEqual(eqn.evaluate(), 7)

        parameter_values = pybamm.ParameterValues({"test parameter": 3})
        eqn_changed = parameter_values.process_symbol(eqn)
        self.assertEqual(eqn_changed.evaluate(), 10)

    def test_set_and_update_parameters(self):
        a = pybamm.Scalar(1)
        b = pybamm.Parameter(name="test parameter")
        c = pybamm.Scalar(3)
        eqn = a + b * c

        parameter_values = pybamm.ParameterValues({"test parameter": 2})
        eqn_changed = parameter_values.process_symbol(eqn)
        self.assertEqual(eqn_changed.evaluate(), 7)

        parameter_values = pybamm.ParameterValues({"test parameter": 3})
        eqn_changed_again = parameter_values.process_symbol(eqn_changed)
        self.assertEqual(eqn_changed_again.evaluate(), 10)

    def test_update_model(self):
        # standard model
        model1 = pybamm.ReactionDiffusionModel()
        modeltest1 = tests.StandardModelTest(model1)
        modeltest1.test_all()
        T1, Y1 = modeltest1.solver.t, modeltest1.solver.y

        # double initial conditions
        model2 = pybamm.ReactionDiffusionModel()
        modeltest2 = tests.StandardModelTest(model2)
        modeltest2.test_all()
        parameter_values_update = pybamm.ParameterValues(
            {
                "Initial concentration in electrolyte": 2
                * model2.default_parameter_values[
                    "Initial concentration in electrolyte"
                ]
            }
        )
        parameter_values_update.process_model(model2)
        modeltest2.test_all()
        T2, Y2 = modeltest2.solver.t, modeltest2.solver.y
        np.testing.assert_array_equal(T1, T2)
        np.testing.assert_array_equal(Y1, Y2)


if __name__ == "__main__":
    import sys

    print("Add -v for more debug output")

    if "-v" in sys.argv:
        debug = True
    unittest.main()
