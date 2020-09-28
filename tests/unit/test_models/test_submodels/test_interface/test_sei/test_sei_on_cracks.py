#
# Test base particle submodel
#

import pybamm
import tests
import unittest


class TestSEIonCracks(unittest.TestCase):
    def test_public_functions(self):
        a = pybamm.Scalar(0)
        full = pybamm.FullBroadcast(
            a,
            ["negative electrode", "separator", "positive electrode"],
            "current collector",
        )
        variables = {
            "Sum of electrolyte reaction source terms": full,
            "Sum of negative electrode electrolyte reaction source terms": a,
            "Sum of positive electrode electrolyte reaction source terms": a,
            "Sum of x-averaged negative electrode electrolyte reaction source terms": a,
            "Sum of x-averaged positive electrode electrolyte reaction source terms": a,
            "Sum of interfacial current densities": a,
            "Sum of negative electrode interfacial current densities": a,
            "Sum of positive electrode interfacial current densities": a,
            "Sum of x-averaged negative electrode interfacial current densities": a,
            "Sum of x-averaged positive electrode interfacial current densities": a,
            "Negative particle crack length": a,
            "Negative particle cracking rate": a,
            "Negative electrode roughness ratio": a,
        }
        param = pybamm.LithiumIonParameters()
        submodel = pybamm.sei.SEIonCracks(param, "Negative")
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
