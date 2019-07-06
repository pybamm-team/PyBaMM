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
        variables = {"Negative electrode open circuit potential": a}
        submodel = pybamm.interface.inverse_kinetics.BaseInverseButlerVolmer(
            param, "Negative"
        )
        std_tests = tests.StandardSubModelTests(submodel, variables)

        with self.assertRaises(NotImplementedError):
            std_tests.test_all()

        variables = {"Positive electrode open circuit potential": a}
        submodel = pybamm.interface.inverse_kinetics.BaseInverseButlerVolmer(
            param, "Positive"
        )
        std_tests = tests.StandardSubModelTests(submodel, variables)
        with self.assertRaises(NotImplementedError):
            std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
