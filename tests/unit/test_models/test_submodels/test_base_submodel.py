#
# Test base submodel
#

import pybamm
import tests
import unittest


class TestBaseSubModel(unittest.TestCase):
    def test_public_functions(self):
        submodel = pybamm.BaseSubModel(None)
        std_tests = tests.StandardSubModelTests(submodel)
        std_tests.test_all()

    def test_add_internal_boundary_conditions(self):
        submodel = pybamm.BaseSubModel(None)

        c_e = pybamm.standard_variables.c_e
        lbc = (pybamm.Scalar(0), "Neumann")
        rbc = (pybamm.Scalar(0), "Neumann")
        submodel.boundary_conditions = {c_e: {"left": lbc, "right": rbc}}

        submodel.set_internal_boundary_conditions()

        for child in c_e.children:
            self.assertTrue(child in submodel.boundary_conditions.keys())


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
