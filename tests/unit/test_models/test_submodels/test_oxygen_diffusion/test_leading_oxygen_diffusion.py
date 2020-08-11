#
# Test leading-order concentration submodel
#

import pybamm
import tests
import unittest


class TestLeadingOrder(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.LeadAcidParameters()
        a = pybamm.Scalar(0)
        variables = {
            "X-averaged negative electrode porosity": a,
            "X-averaged separator porosity": a,
            "X-averaged positive electrode porosity": a,
            "X-averaged negative electrode porosity change": a,
            "X-averaged separator porosity change": a,
            "X-averaged positive electrode porosity change": a,
            "X-averaged negative electrode interfacial current density": a,
            "X-averaged positive electrode interfacial current density": a,
            "X-averaged negative electrode oxygen interfacial current density": a,
            "X-averaged positive electrode oxygen interfacial current density": a,
        }
        submodel = pybamm.oxygen_diffusion.LeadingOrder(param)
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
