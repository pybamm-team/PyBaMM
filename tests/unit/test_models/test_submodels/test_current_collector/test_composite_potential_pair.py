#
# Test base current collector submodel
#

import pybamm
import tests
import unittest


class TestBaseModel(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.LithiumIonParameters()
        submodel = pybamm.current_collector.CompositePotentialPair1plus1D(param)
        variables = {
            "Positive current collector potential": pybamm.PrimaryBroadcast(
                0, "current collector"
            ),
            "Total current density": 0,
        }
        std_tests = tests.StandardSubModelTests(submodel, variables)

        std_tests.test_all()
        submodel = pybamm.current_collector.CompositePotentialPair2plus1D(param)
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
