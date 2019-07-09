#
# Test lumped submodel
#

import pybamm
import tests
import unittest


class TestLumped(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.standard_parameters_lithium_ion
        a_n = pybamm.Broadcast(pybamm.Scalar(0), ["negative electrode"])
        a_p = pybamm.Broadcast(pybamm.Scalar(0), ["positive electrode"])
        variables = {
            "Negative electrode interfacial current density": a_n,
            "Positive electrode interfacial current density": a_p,
            "Negative electrode reaction overpotential": a_n,
            "Positive electrode reaction overpotential": a_p,
            "Negative electrode entropic change": a_n,
            "Positive electrode entropic change": a_p,
        }
        submodel = pybamm.thermal.Lumped(param)
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
