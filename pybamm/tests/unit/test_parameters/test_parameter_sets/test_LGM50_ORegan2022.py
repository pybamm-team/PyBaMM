#
# Tests for LG M50 parameter set loads
#
from tests import TestCase
import pybamm
import unittest


class TestORegan2022(TestCase):
    def test_functions(self):
        param = pybamm.ParameterValues("ORegan2022")
        T = pybamm.Scalar(298.15)

        c_p_max = param["Maximum concentration in positive electrode [mol.m-3]"]
        c_n_max = param["Maximum concentration in negative electrode [mol.m-3]"]
        fun_test = {
            # Positive electrode
            "Positive electrode OCP entropic change [V.K-1]": (
                [0.5, c_p_max],
                -9.7940e-07,
            ),
            "Positive electrode specific heat capacity [J.kg-1.K-1]": (
                [298.15],
                902.6502,
            ),
            "Positive electrode diffusivity [m2.s-1]": ([0.5, 298.15], 7.2627e-15),
            "Positive electrode exchange-current density [A.m-2]": (
                [1e3, 1e4, c_p_max, 298.15],
                2.1939,
            ),
            "Positive electrode OCP [V]": ([0.5], 3.9720),
            "Positive electrode conductivity [S.m-1]": ([298.15], 0.8473),
            "Positive electrode thermal conductivity [W.m-1.K-1]": ([T], 0.8047),
            # Negative electrode
            "Negative electrode OCP entropic change [V.K-1]": (
                [0.5, c_n_max],
                -2.6460e-07,
            ),
            "Negative electrode specific heat capacity [J.kg-1.K-1]": (
                [298.15],
                847.7155,
            ),
            "Negative electrode diffusivity [m2.s-1]": ([0.5, 298.15], 2.8655e-16),
            "Negative electrode exchange-current density [A.m-2]": (
                [1e3, 1e4, c_n_max, 298.15],
                1.0372,
            ),
            "Negative electrode OCP [V]": ([0.5], 0.1331),
            "Negative electrode thermal conductivity [W.m-1.K-1]": ([T], 3.7695),
            # Cells
            "Positive current collector specific heat capacity [J.kg-1.K-1]": (
                [T],
                897.1585,
            ),
            "Negative current collector specific heat capacity [J.kg-1.K-1]": (
                [T],
                388.5190,
            ),
            "Negative current collector thermal conductivity [W.m-1.K-1]": (
                [T],
                400.8491,
            ),
            # Separator
            "Separator specific heat capacity [J.kg-1.K-1]": (
                [298.15],
                1130.9656,
            ),
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
