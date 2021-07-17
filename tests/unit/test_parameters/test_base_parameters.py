"""
Tests for the base_parameters.py
"""
import pybamm
import unittest


class TestBaseParameters(unittest.TestCase):
    def test__setattr__(self):
        param = pybamm.ElectricalParameters()
        self.assertEqual(param.I_typ.print_name, r"I^{typ}")


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
