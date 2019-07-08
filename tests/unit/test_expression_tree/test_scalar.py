#
# Tests for the Scalar class
#
import pybamm

import unittest


class TestScalar(unittest.TestCase):
    def test_scalar_eval(self):
        a = pybamm.Scalar(5)
        self.assertEqual(a.value, 5)
        self.assertEqual(a.evaluate(), 5)

        # with known_evals
        self.assertEqual(a.evaluate(known_evals={a.id: 5})[0], 5)
        # it's possible to provide a conflicting value of known_evals
        self.assertEqual(a.evaluate(known_evals={a.id: 15})[0], 15)

    def test_scalar_operations(self):
        a = pybamm.Scalar(5)
        b = pybamm.Scalar(6)
        self.assertEqual((a + b).evaluate(), 11)
        self.assertEqual((a - b).evaluate(), -1)
        self.assertEqual((a * b).evaluate(), 30)
        self.assertEqual((a / b).evaluate(), 5 / 6)

    def test_scalar_id(self):
        a1 = pybamm.Scalar(4)
        a2 = pybamm.Scalar(4)
        self.assertEqual(a1.id, a2.id)
        a3 = pybamm.Scalar(5)
        self.assertNotEqual(a1.id, a3.id)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
