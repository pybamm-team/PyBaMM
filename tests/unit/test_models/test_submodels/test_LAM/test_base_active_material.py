#
# Test base porosity submodel
#

import pybamm
import tests
import unittest


class TestBaseModel(unittest.TestCase):
    def test_public_functions(self):
        a = pybamm.Scalar(0)
        am = "active material volume fraction"
        variables = {f"Negative electrode {am}": a, f"Positive electrode {am}": a}
        submodel = pybamm.loss_of_active_materials.BaseLAM(None)
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
