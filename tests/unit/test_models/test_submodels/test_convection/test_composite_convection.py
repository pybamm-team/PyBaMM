#
# Test combined convection submodel
#

import pybamm
import tests
import unittest


class TestComposite(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.standard_parameters_lead_acid

        a = pybamm.Scalar(0)
        variables = {
            "Current collector current density": a,
            "Negative electrode interfacial current density": pybamm.Broadcast(
                a, ["negative electrode"]
            ),
            "Positive electrode interfacial current density": pybamm.Broadcast(
                a, ["positive electrode"]
            ),
        }
        submodel = pybamm.convection.Composite(param)
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
