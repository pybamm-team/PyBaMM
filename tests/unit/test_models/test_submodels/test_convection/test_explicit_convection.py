#
# Test combined convection submodel
#

import pybamm
import tests
import unittest


class TestExplicit(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.standard_parameters_lead_acid

        a = pybamm.PrimaryBroadcast(0, "current collector")
        a_n = pybamm.PrimaryBroadcast(a, ["negative electrode"])
        a_s = pybamm.PrimaryBroadcast(a, ["separator"])
        a_p = pybamm.PrimaryBroadcast(a, ["positive electrode"])
        variables = {
            "Current collector current density": a,
            "Negative electrode interfacial current density": a_n,
            "X-averaged negative electrode interfacial current density": a,
            "Positive electrode interfacial current density": a_p,
            "X-averaged positive electrode interfacial current density": a,
            "X-averaged separator pressure": a,
            "X-averaged separator transverse volume-averaged acceleration": a,
            "Separator pressure": a_s,
        }
        submodel = pybamm.convection.through_cell.Explicit(param)
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
