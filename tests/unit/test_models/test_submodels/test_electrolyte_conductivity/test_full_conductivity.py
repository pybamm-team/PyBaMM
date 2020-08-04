#
# Test full stefan maxwell electrolyte conductivity submodel
#

import pybamm
import tests
import unittest


class TestFull(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.LithiumIonParameters()
        a = pybamm.Scalar(0)
        surf = "surface area per unit volume distribution in x"
        variables = {
            "Electrolyte tortuosity": a,
            "Electrolyte concentration": pybamm.FullBroadcast(
                a,
                ["negative electrode", "separator", "positive electrode"],
                "current collector",
            ),
            "Negative "
            + surf: pybamm.FullBroadcast(a, "negative electrode", "current collector"),
            "Positive "
            + surf: pybamm.FullBroadcast(a, "positive electrode", "current collector"),
            "Sum of interfacial current densities": pybamm.FullBroadcast(
                a,
                ["negative electrode", "separator", "positive electrode"],
                "current collector",
            ),
            "Cell temperature": a,
        }
        submodel = pybamm.electrolyte_conductivity.Full(param)
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
