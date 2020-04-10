#
# Test base inverse butler volmer submodel
#

import pybamm
import tests
import unittest


class TestBaseModel(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.standard_parameters_lithium_ion

        a = pybamm.Scalar(0)
        variables = {
            "Negative electrode open circuit potential": a,
            "Negative particle surface concentration": a,
            "Negative electrode temperature": a,
            "Negative electrolyte concentration": a,
            "Current collector current density": a,
        }
        submodel = pybamm.interface.inverse_kinetics.InverseButlerVolmer(
            param, "Negative", "lithium-ion main"
        )
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()

        variables = {
            "Positive electrode open circuit potential": a,
            "Positive particle surface concentration": a,
            "Positive electrode temperature": a,
            "Positive electrolyte concentration": a,
            "Current collector current density": a,
        }
        submodel = pybamm.interface.inverse_kinetics.InverseButlerVolmer(
            param, "Positive", "lithium-ion main"
        )
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
