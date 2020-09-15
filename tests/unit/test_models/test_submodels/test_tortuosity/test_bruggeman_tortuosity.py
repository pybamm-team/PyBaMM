#
# Test Bruggeman tortuosity submodel
#

import pybamm
import tests
import unittest


class TestBruggeman(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.LithiumIonParameters()
        a = pybamm.Concatenation(
            pybamm.FullBroadcast(0, ["negative electrode"], "current collector"),
            pybamm.FullBroadcast(0, ["separator"], "current collector"),
            pybamm.FullBroadcast(0, ["positive electrode"], "current collector"),
        )
        variables = {"Porosity": a, "Active material volume fraction": a}
        submodel = pybamm.tortuosity.Bruggeman(param, "Electrolyte")
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()

        submodel = pybamm.tortuosity.Bruggeman(param, "Electrode")
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
