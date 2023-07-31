#
# Tests for O'Kane (2022) parameter set
#
from tests import TestCase
import pybamm
import unittest


class TestOKane2022_graphite_halfcell(TestCase):
    def test_functions(self):
        param = pybamm.ParameterValues("OKane2022_graphite_halfcell")
        sto = pybamm.Scalar(0.9)
        T = pybamm.Scalar(298.15)

        fun_test = {
            # Lithium plating
            "Exchange-current density for plating [A.m-2]": ([1e3, 1e4, T], 9.6485e-2),
            "Exchange-current density for stripping [A.m-2]": (
                [1e3, 1e4, T],
                9.6485e-1,
            ),
            "Dead lithium decay rate [s-1]": ([1e-8], 5e-7),
            # Positive electrode
            "Positive electrode diffusivity [m2.s-1]": ([sto, T], 3.3e-14),
            "Positive electrode exchange-current density [A.m-2]": (
                [1000, 16566.5, 33133, T],
                0.33947,
            ),
            "Positive electrode cracking rate": ([T], 3.9e-20),
            "Positive electrode volume change": ([sto, 33133], 0.0897),
        }

        for name, value in fun_test.items():
            self.assertAlmostEqual(
                param.evaluate(param[name](*value[0])), value[1], places=4
            )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
