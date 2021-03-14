#
# Test Bruggeman tortuosity submodel
#

import pybamm
import tests
import unittest


class TestBruggeman(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.LithiumIonParameters()
        a_n = pybamm.FullBroadcast(0, ["negative electrode"], "current collector")
        a_s = pybamm.FullBroadcast(0, ["separator"], "current collector")
        a_p = pybamm.FullBroadcast(0, ["positive electrode"], "current collector")

        variables = {
            "Negative electrode porosity": a_n,
            "Separator porosity": a_s,
            "Positive electrode porosity": a_p,
            "Negative electrode active material volume fraction": a_n,
            "Positive electrode active material volume fraction": a_p,
        }
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
