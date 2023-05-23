#
# Tests for Ai (2020) Enertech parameter set loads
#
from tests import TestCase
import pybamm
import unittest


class TestAi2020(TestCase):
    def test_functions(self):
        param = pybamm.ParameterValues("Ai2020")
        sto = pybamm.Scalar(0.5)
        T = pybamm.Scalar(298.15)

        c_p_max = param["Maximum concentration in positive electrode [mol.m-3]"]
        c_n_max = param["Maximum concentration in negative electrode [mol.m-3]"]
        fun_test = {
            # Positive electrode
            "Positive electrode cracking rate": ([T], 3.9e-20),
            "Positive electrode diffusivity [m2.s-1]": ([sto, T], 5.387e-15),
            "Positive electrode exchange-current density [A.m-2]": (
                [1e3, 1e4, c_p_max, T],
                0.6098,
            ),
            "Positive electrode OCP entropic change [V.K-1]": (
                [sto, c_p_max],
                -2.1373e-4,
            ),
            "Positive electrode volume change": ([sto, c_p_max], -1.8179e-2),
            # Negative electrode
            "Negative electrode cracking rate": ([T], 3.9e-20),
            "Negative electrode diffusivity [m2.s-1]": ([sto, T], 3.9e-14),
            "Negative electrode exchange-current density [A.m-2]": (
                [1e3, 1e4, c_n_max, T],
                0.4172,
            ),
            "Negative electrode OCP entropic change [V.K-1]": (
                [sto, c_n_max],
                -1.1033e-4,
            ),
            "Negative electrode volume change": ([sto, c_n_max], 5.1921e-2),
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
