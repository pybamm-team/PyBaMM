"""
Tests for the base_parameters.py
"""
import pybamm
import unittest


class TestBaseParameters(unittest.TestCase):
    def test__setattr__(self):
        model = pybamm.lithium_ion.SPM()
        var, eqn = list(model.rhs.items())[0]
        p_name = eqn.children[0].children[0].print_name
        self.assertEqual(p_name, "dimensional_current_with_time")


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
