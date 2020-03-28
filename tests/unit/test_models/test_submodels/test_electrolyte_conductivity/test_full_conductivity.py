#
# Test full stefan maxwell electrolyte conductivity submodel
#

import pybamm
import tests
import unittest


class TestFull(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.standard_parameters_lithium_ion
        a = pybamm.Scalar(0)
        variables = {
            "Electrolyte tortuosity": a,
            "Electrolyte concentration": a,
            "Negative electrode interfacial current density": pybamm.FullBroadcast(
                a, "negative electrode", "current collector"
            ),
            "Positive electrode interfacial current density": pybamm.FullBroadcast(
                a, "positive electrode", "current collector"
            ),
            "Cell temperature": a,
        }
        icd = " interfacial current density"
        reactions = {
            "main": {
                "Negative": {"s": 1, "aj": "Negative electrode" + icd},
                "Positive": {"s": 1, "aj": "Positive electrode" + icd},
            }
        }
        submodel = pybamm.electrolyte_conductivity.Full(param, reactions)
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
