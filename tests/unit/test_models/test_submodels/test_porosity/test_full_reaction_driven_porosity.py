#
# Test full porosity submodel
#

import pybamm
import tests
import unittest


class TestFull(unittest.TestCase):
    def test_public_functions(self):
        # param = pybamm.LeadAcidParameters()
        param = pybamm.LithiumIonParameters()
        a_n = pybamm.PrimaryBroadcast(pybamm.Scalar(0), ["negative electrode"])

        variables = {
            "Total negative electrode SEI thickness [m]": a_n,
            "Negative electrode lithium plating thickness [m]": a_n,
            "Negative electrode surface area to volume ratio [m-1]": a_n,

        }
        options = {
            "SEI": "ec reaction limited",
            "SEI film resistance": "distributed",
            "SEI porosity change": "true",
            "lithium plating": "irreversible",
            "lithium plating porosity change": "true",
        }
        submodel = pybamm.porosity.ReactionDriven(param, options, False)
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
