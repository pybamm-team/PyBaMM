#
# Tests for the base battery model class
#
import pybamm
import unittest


class TestBaseBatteryModel(unittest.TestCase):
    def test_process_parameters_and_discretise(self):
        model = pybamm.SimpleODEModel()
        c = pybamm.Parameter("Negative electrode width [m]") * pybamm.Variable("c")
        processed_c = model.process_parameters_and_discretise(c)
        self.assertIsInstance(processed_c, pybamm.Multiplication)
        self.assertIsInstance(processed_c.left, pybamm.Scalar)
        self.assertIsInstance(processed_c.right, pybamm.StateVector)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
