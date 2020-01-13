#
# Test an experiment with various constant current steps
#
import pybamm
import unittest


class TestConstantCurrentSteps(unittest.TestCase):
    def test_discharge_rest_charge(self):
        # Make sure not to hit limits
        experiment = pybamm.Experiment(
            ["0.5 C for 1 hour", "0 C for 1 hour", "-0.5 C for 1 hour"]
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
