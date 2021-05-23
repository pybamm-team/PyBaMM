#
# Test reversible lithium plating submodel
#

import pybamm
import tests
import unittest


class TestReversiblePlating(unittest.TestCase):
    def test_not_implemented(self):
        with self.assertRaisesRegex(
            NotImplementedError,
            "Lithium plating models are not implemented for the positive electrode",
        ):
            pybamm.lithium_plating.ReversiblePlating(None, "Positive", True)

    def test_public_functions(self):
        param = pybamm.LithiumIonParameters()

        a_n = pybamm.FullBroadcast(
            pybamm.Scalar(0), ["negative electrode"], "current collector"
        )
        variables = {
            "Negative electrode surface potential difference": a_n,
            "Negative electrode temperature": a_n,
            "Negative electrolyte concentration": a_n,
            "Negative electrode SEI film overpotential": a_n,
        }

        submodel = pybamm.lithium_plating.ReversiblePlating(param, "Negative", True)
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()

        submodel = pybamm.lithium_plating.ReversiblePlating(param, "Negative", False)
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
