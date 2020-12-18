#
# Tests for the NotConstantOne class
#
import numbers
import pybamm
import unittest


class TestNotConstantOne(unittest.TestCase):
    def test_init(self):
        a = pybamm.NotConstantOne()
        self.assertEqual(a.name, "not_constant_one")
        self.assertEqual(a.domain, [])
        self.assertEqual(a.evaluate(), 1)
        self.assertEqual(a.jac(pybamm.StateVector(slice(0, 1))).evaluate(), 0)
        self.assertFalse(a.is_constant())
        self.assertFalse((2 * a).is_constant())


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
