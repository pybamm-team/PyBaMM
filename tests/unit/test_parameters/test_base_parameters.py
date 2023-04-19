"""
Tests for the base_parameters.py
"""
from tests import TestCase
import pybamm
import unittest


class TestBaseParameters(TestCase):
    def test_getattr__(self):
        param = pybamm.LithiumIonParameters()
        # ending in _n / _s / _p
        with self.assertRaisesRegex(AttributeError, "param.n.L"):
            getattr(param, "L_n")
        with self.assertRaisesRegex(AttributeError, "param.s.L"):
            getattr(param, "L_s")
        with self.assertRaisesRegex(AttributeError, "param.p.L"):
            getattr(param, "L_p")
        # _n_ in the name
        with self.assertRaisesRegex(AttributeError, "param.n.prim.c_max"):
            getattr(param, "c_n_max")

        # _n_ or _p_ not in name
        with self.assertRaisesRegex(
            AttributeError, "has no attribute 'c_n_not_a_parameter"
        ):
            getattr(param, "c_n_not_a_parameter")

        with self.assertRaisesRegex(AttributeError, "has no attribute 'c_s_test"):
            getattr(pybamm.electrical_parameters, "c_s_test")

        self.assertEqual(param.n.cap_init, param.n.Q_init)
        self.assertEqual(param.p.prim.cap_init, param.p.prim.Q_init)

    def test__setattr__(self):
        # domain gets added as a subscript
        param = pybamm.GeometricParameters()
        self.assertEqual(param.n.L.print_name, r"L_{\mathrm{n}}")


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
