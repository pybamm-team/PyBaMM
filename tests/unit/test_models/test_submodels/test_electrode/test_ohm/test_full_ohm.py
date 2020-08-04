#
# Test full ohm submodel
#

import pybamm
import tests
import unittest


class TestFull(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.LeadAcidParameters()

        a = pybamm.Scalar(0)
        variables = {
            "Current collector current density": a,
            "Negative electrode porosity": a,
            "Positive electrode porosity": a,
            "Negative electrode interfacial current density": a,
            "Positive electrode interfacial current density": a,
        }
        submodel = pybamm.electrode.ohm.Full(param, "Negative")
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()

        submodel = pybamm.electrode.ohm.Full(param, "Positive")
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
