#
# Test leading-order porosity submodel
#

import pybamm
import tests
import unittest


class TestLeadingOrder(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.standard_parameters_lead_acid
        a = pybamm.Broadcast(pybamm.Scalar(0), "test")
        variables = {
            "Negative electrode interfacial current density": a,
            "Positive electrode interfacial current density": a,
        }
        submodel = pybamm.porosity.LeadingOrder(param)
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
