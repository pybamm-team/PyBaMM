#
# Test leading-order porosity submodel
#

import pybamm
import tests
import unittest


class TestLeadingOrder(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.LithiumIonParameters()
        a = pybamm.PrimaryBroadcast(pybamm.Scalar(0), "current collector")
        variables = {
            "X-averaged negative particle surface tangential stress": a,
            "X-averaged positive particle surface tangential stress": a,
            "X-averaged negative particle surface radial stress": a,
            "X-averaged positive particle surface radial stress": a,
        }
        submodel = pybamm.loss_of_active_materials.LeadingOrder(param)
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
