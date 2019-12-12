#
# Test full convection submodel
#

import pybamm
import tests
import unittest


class TestFull(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.standard_parameters_lead_acid

        a = pybamm.Scalar(0)
        variables = {
            "Current collector current density": pybamm.PrimaryBroadcast(
                a, "current collector"
            ),
            "Interfacial current density": a,
            "Negative electrode interfacial current density": a,
            "Positive electrode interfacial current density": a,
            "X-averaged separator transverse volume-averaged acceleration": a,
            "X-averaged separator pressure": a,
            "Separator pressure": pybamm.FullBroadcast(
                a, "separator", "current collector"
            ),
        }
        submodel = pybamm.convection.through_cell.Full(param)
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
