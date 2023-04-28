#
# Tests for Ai (2020) Enertech parameter set loads
#
from tests import TestCase
import pybamm
import unittest


class TestRamadass2004(TestCase):
    def test_functions(self):
        param = pybamm.ParameterValues("Ramadass2004")
        sto = pybamm.Scalar(0.5)
        T = pybamm.Scalar(298.15)

        c_p_max = param["Maximum concentration in positive electrode [mol.m-3]"]
        c_n_max = param["Maximum concentration in negative electrode [mol.m-3]"]
        fun_test = {
            # Positive electrode
            "Positive electrode diffusivity [m2.s-1]": ([sto, T], 1e-14),
            "Positive electrode exchange-current density [A.m-2]": (
                [1e3, 1e4, c_p_max, T],
                1.4517,
            ),
            "Positive electrode OCP entropic change [V.K-1]": (
                [sto, c_p_max],
                -3.4664e-5,
            ),
            "Positive electrode OCP [V]": ([sto], 4.1249),
            # Negative electrode
            "Negative electrode diffusivity [m2.s-1]": ([sto, T], 3.9e-14),
            "Negative electrode exchange-current density [A.m-2]": (
                [1e3, 1e4, c_n_max, T],
                2.2007,
            ),
            "Negative electrode OCP entropic change [V.K-1]": (
                [sto, c_n_max],
                -1.5079e-5,
            ),
            "Negative electrode OCP [V]": ([sto], 0.1215),
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
