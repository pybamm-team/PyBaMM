#
# Tests for the standard parameters
#
import pybamm

import unittest


class TestGeometricParameters(unittest.TestCase):
    def test_macroscale_parameters(self):
        geo = pybamm.GeometricParameters()
        L_n = geo.L_n
        L_s = geo.L_s
        L_p = geo.L_p
        L_x = geo.L_x
        l_n = geo.l_n
        l_s = geo.l_s
        l_p = geo.l_p

        parameter_values = pybamm.ParameterValues(
            values={
                "Negative electrode thickness [m]": 0.05,
                "Separator thickness [m]": 0.02,
                "Positive electrode thickness [m]": 0.21,
            }
        )
        L_n_eval = parameter_values.process_symbol(L_n)
        L_s_eval = parameter_values.process_symbol(L_s)
        L_p_eval = parameter_values.process_symbol(L_p)
        L_x_eval = parameter_values.process_symbol(L_x)

        self.assertEqual(
            (L_n_eval + L_s_eval + L_p_eval).evaluate(), L_x_eval.evaluate()
        )
        l_n_eval = parameter_values.process_symbol(l_n)
        l_s_eval = parameter_values.process_symbol(l_s)
        l_p_eval = parameter_values.process_symbol(l_p)
        self.assertAlmostEqual((l_n_eval + l_s_eval + l_p_eval).evaluate(), 1)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
