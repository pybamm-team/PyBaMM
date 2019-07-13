#
# Test full porosity submodel
#

import pybamm
import tests
import unittest


class TestFull(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.standard_parameters_lead_acid
        a_n = pybamm.Broadcast(pybamm.Scalar(0), ["negative electrode"])
        a_p = pybamm.Broadcast(pybamm.Scalar(0), ["positive electrode"])
        variables = {
            "Negative electrode interfacial current density": a_n,
            "Positive electrode interfacial current density": a_p,
        }
        submodel = pybamm.porosity.Full(param)
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
