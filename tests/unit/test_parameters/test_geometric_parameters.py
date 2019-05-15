#
# Tests for the standard parameters
#
import pybamm

import unittest


class TestGeometricParameters(unittest.TestCase):
    def test_macroscale_parameters(self):
        L_n = pybamm.geometric_parameters.L_n
        L_s = pybamm.geometric_parameters.L_s
        L_p = pybamm.geometric_parameters.L_p
        L_x = pybamm.geometric_parameters.L_x
        l_n = pybamm.geometric_parameters.l_n
        l_s = pybamm.geometric_parameters.l_s
        l_p = pybamm.geometric_parameters.l_p

        parameter_values = pybamm.ParameterValues(
            base_parameters={
                "Negative electrode width [m]": 0.05,
                "Separator width [m]": 0.02,
                "Positive electrode width [m]": 0.21,
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


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
