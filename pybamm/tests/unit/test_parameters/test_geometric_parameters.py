#
# Tests for the standard parameters
#
from tests import TestCase
import pybamm
import unittest


class TestGeometricParameters(TestCase):
    def test_macroscale_parameters(self):
        geo = pybamm.geometric_parameters
        L_n = geo.n.L
        L_s = geo.s.L
        L_p = geo.p.L
        L_x = geo.L_x

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


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
