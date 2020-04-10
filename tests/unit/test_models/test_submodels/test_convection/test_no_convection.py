#
# Test no convection submodel
#

import pybamm
import tests
import unittest


class TestNoConvectionModel(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.standard_parameters_lead_acid

        a = pybamm.Scalar(0)
        a_s = pybamm.PrimaryBroadcast(a, "separator")
        variables = {"Current collector current density": a, "Separator pressure": a_s}
        submodel = pybamm.convection.through_cell.NoConvection(param)
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
