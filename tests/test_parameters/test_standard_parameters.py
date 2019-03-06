#
# Tests for the standard parameters
#
import pybamm

import os
import unittest


class TestStandardParameters(unittest.TestCase):
    def test_geometric_parameters(self):
        L_n = pybamm.standard_parameters.L_n
        L_s = pybamm.standard_parameters.L_s
        L_p = pybamm.standard_parameters.L_p
        L_x = pybamm.standard_parameters.L_x
        l_n = pybamm.standard_parameters.l_n
        l_s = pybamm.standard_parameters.l_s
        l_p = pybamm.standard_parameters.l_p

        parameter_values = pybamm.ParameterValues(
            base_parameters={
                "Negative electrode width": 0.05,
                "Separator width": 0.02,
                "Positive electrode width": 0.21,
            }
        )
        L_n_eval = parameter_values.process_symbol(L_n)
        L_s_eval = parameter_values.process_symbol(L_s)
        L_p_eval = parameter_values.process_symbol(L_p)
        L_x_eval = parameter_values.process_symbol(L_x)
        self.assertEqual(
            (L_n_eval + L_s_eval + L_p_eval).evaluate(), L_x_eval.evaluate()
        )
        self.assertEqual((L_n_eval + L_s_eval + L_p_eval).id, L_x_eval.id)
        l_n_eval = parameter_values.process_symbol(l_n)
        l_s_eval = parameter_values.process_symbol(l_s)
        l_p_eval = parameter_values.process_symbol(l_p)
        self.assertAlmostEqual((l_n_eval + l_s_eval + l_p_eval).evaluate(), 1)

    def test_current_functions(self):
        # create current functions
        dimensional_current = pybamm.standard_parameters.dimensional_current_with_time
        dimensionless_current = pybamm.standard_parameters.current_with_time

        # process
        parameter_values = pybamm.ParameterValues(
            {
                "Electrode height": 0.1,
                "Electrode depth": 0.1,
                "Number of electrodes connected in parallel to make a cell": 8,
                "Typical current density": 2,
                "Current function": os.path.join(
                    os.getcwd(),
                    "pybamm",
                    "parameters",
                    "standard_current_functions",
                    "constant_current.py",
                ),
            }
        )
        dimensional_current_eval = parameter_values.process_symbol(dimensional_current)
        dimensionless_current_eval = parameter_values.process_symbol(
            dimensionless_current
        )
        self.assertAlmostEqual(
            dimensional_current_eval.evaluate(t=3), 2 / (8 * 0.1 * 0.1)
        )
        self.assertEqual(dimensionless_current_eval.evaluate(t=3), 1)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
