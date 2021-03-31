#
# Test full stefan maxwell electrolyte conductivity submodel
#

import pybamm
import tests
import unittest


class TestFull(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.LithiumIonParameters()
        a = pybamm.Scalar(1)
        surf = "electrode surface area to volume ratio"
        a_n = pybamm.FullBroadcast(a, "negative electrode", "current collector")
        a_s = pybamm.FullBroadcast(a, "separator", "current collector")
        a_p = pybamm.FullBroadcast(a, "positive electrode", "current collector")

        variables = {
            "Electrolyte tortuosity": a,
            "Electrolyte concentration": pybamm.concatenation(a_n, a_s, a_p),
            "Negative electrolyte concentration": a_n,
            "Separator electrolyte concentration": a_s,
            "Positive electrolyte concentration": a_p,
            "Negative electrode temperature": a_n,
            "Separator temperature": a_s,
            "Positive electrode temperature": a_p,
            "Negative "
            + surf: pybamm.FullBroadcast(a, "negative electrode", "current collector"),
            "Positive "
            + surf: pybamm.FullBroadcast(a, "positive electrode", "current collector"),
            "Sum of interfacial current densities": pybamm.FullBroadcast(
                a,
                ["negative electrode", "separator", "positive electrode"],
                "current collector",
            ),
            "Cell temperature": pybamm.concatenation(a_n, a_s, a_p),
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
