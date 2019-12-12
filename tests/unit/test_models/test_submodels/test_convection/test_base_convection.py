#
# Test base convection submodel
#

import pybamm
import tests
import unittest


class TestBaseModel(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.standard_parameters_lead_acid
        submodel = pybamm.convection.through_cell.BaseThroughCellModel(param)
        a = pybamm.PrimaryBroadcast(0, "current collector")
        a_n = pybamm.PrimaryBroadcast(0, "negative electrode")
        a_s = pybamm.PrimaryBroadcast(0, "separator")
        a_p = pybamm.PrimaryBroadcast(0, "positive electrode")
        variables = {
            "X-averaged separator transverse volume-averaged acceleration": a,
            "Current collector current density": a,
            "Negative electrode volume-averaged velocity": a_n,
            "Positive electrode volume-averaged velocity": a_p,
            "Negative electrode volume-averaged acceleration": a_n,
            "Positive electrode volume-averaged acceleration": a_p,
            "Negative electrode pressure": a_n,
            "Separator pressure": a_s,
            "Positive electrode pressure": a_p,
        }
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
