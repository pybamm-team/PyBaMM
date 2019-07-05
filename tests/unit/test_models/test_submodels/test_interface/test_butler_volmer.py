#
# Test base butler volmer submodel
#

import pybamm
import tests
import unittest


class TestBaseModel(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.standard_parameters_lithium_ion

        a = pybamm.Scalar(0)
        variables = {
            "Current collector current density": a,
            "Negative electrode potential": a,
            "Negative electrolyte potential": a,
            "Negative electrode open circuit potential": a,
        }
        submodel = pybamm.interface.butler_volmer.BaseModel(param, "Negative")
        std_tests = tests.StandardSubModelTests(submodel, variables)

        with self.assertRaises(NotImplementedError):
            std_tests.test_all()

        variables = {
            "Current collector current density": a,
            "Positive electrode potential": a,
            "Positive electrolyte potential": a,
            "Positive electrode open circuit potential": a,
        }
        submodel = pybamm.interface.butler_volmer.BaseModel(param, "Positive")
        std_tests = tests.StandardSubModelTests(submodel, variables)
        with self.assertRaises(NotImplementedError):
            std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
