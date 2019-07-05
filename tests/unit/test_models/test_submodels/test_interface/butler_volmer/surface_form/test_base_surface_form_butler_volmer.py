#
# Test base surface form butler volmer submodel
#

import pybamm
import tests
import unittest


class TestBaseModel(unittest.TestCase):
    def test_public_functions(self):

        a = pybamm.Scalar(0)
        variables = {
            "Current collector current density": a,
            "Negative electrode surface potential difference": a,
            "Negative electrode open circuit potential": a,
        }
        submodel = pybamm.interface.butler_volmer.surface_form.BaseModel(
            None, "Negative"
        )
        std_tests = tests.StandardSubModelTests(submodel, variables)

        with self.assertRaises(NotImplementedError):
            std_tests.test_all()

        variables = {
            "Current collector current density": a,
            "Positive electrode surface potential difference": a,
            "Positive electrode open circuit potential": a,
        }
        submodel = pybamm.interface.butler_volmer.surface_form.BaseModel(
            None, "Positive"
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
