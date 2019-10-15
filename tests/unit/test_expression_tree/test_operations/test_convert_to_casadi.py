#
# Test for the Simplify class
#
import casadi
import math
import numpy as np
import pybamm
import unittest
from tests import get_discretisation_for_testing


class TestCasadiConverter(unittest.TestCase):
    def test_convert_scalar_symbols(self):
        a = pybamm.Scalar(0)
        b = pybamm.Scalar(1)
        c = pybamm.Scalar(-1)
        d = pybamm.Scalar(2)

        self.assertEqual(a.to_casadi(), casadi.SX(0))
        self.assertEqual(d.to_casadi(), casadi.SX(2))

        # negate
        self.assertEqual((-b).to_casadi(), casadi.SX(-1))
        # absolute value
        self.assertEqual(abs(c).to_casadi(), casadi.SX(1))

        # function
        def sin(x):
            return np.sin(x)

        f = pybamm.Function(sin, b)
        self.assertEqual((f).to_casadi(), casadi.SX(np.sin(1)))

        def myfunction(x, y):
            return x + y

        f = pybamm.Function(myfunction, b, d)
        self.assertEqual((f).to_casadi(), casadi.SX(3))

        # addition
        self.assertEqual((a + b).to_casadi(), casadi.SX(1))
        # subtraction
        self.assertEqual((c - d).to_casadi(), casadi.SX(-3))
        # multiplication
        self.assertEqual((c * d).to_casadi(), casadi.SX(-2))
        # power
        self.assertEqual((c ** d).to_casadi(), casadi.SX(1))
        # division
        self.assertEqual((b / d).to_casadi(), casadi.SX(1 / 2))

    def test_convert_array_symbols(self):
        pass


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
