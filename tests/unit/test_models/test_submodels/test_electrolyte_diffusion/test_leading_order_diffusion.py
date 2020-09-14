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
            "Porosity": a,
            "X-averaged negative electrode porosity": a,
            "X-averaged separator porosity": a,
            "X-averaged positive electrode porosity": a,
            "X-averaged negative electrode porosity change": a,
            "X-averaged separator porosity change": a,
            "X-averaged positive electrode porosity change": a,
            "Sum of x-averaged negative electrode interfacial current densities": a,
            "Sum of x-averaged positive electrode interfacial current densities": a,
            "Sum of x-averaged negative electrode electrolyte reaction source terms": a,
            "Sum of x-averaged positive electrode electrolyte reaction source terms": a,
            "X-averaged separator transverse volume-averaged acceleration": a,
        }
        submodel = pybamm.electrolyte_diffusion.LeadingOrder(param)
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
