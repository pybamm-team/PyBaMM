#
# Test irreversible lithium plating submodel
#

import pybamm
import tests
import unittest


class TestIrreversiblePlating(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.LithiumIonParameters()

        a_n = pybamm.FullBroadcast(
            pybamm.Scalar(0), ["negative electrode"], "current collector"
        )
        variables = {
            "Negative electrode potential": a_n,
            "Negative electrolyte potential": a_n,
            "Negative electrolyte concentration": a_n,
            "Negative electrode SEI film overpotential": a_n,
        }
        submodel = pybamm.lithium_plating.IrreversiblePlating(param, "Negative")
        std_tests = tests.StandardSubModelTests(submodel, variables)

        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
