import pybamm
import unittest


class TestOdeSolver(unittest.TestCase):
    def test_exceptions(self):
        solver = pybamm.OdeSolver()
        with self.assertRaises(NotImplementedError):
            solver.integrate(None, None, None)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
