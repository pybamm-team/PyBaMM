#
# Test full porosity submodel
#

import pybamm
import tests
import unittest


class TestFull(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.LithiumIonParameters()
        a_n = pybamm.PrimaryBroadcast(pybamm.Scalar(0), ["negative electrode"])
        a_p = pybamm.PrimaryBroadcast(pybamm.Scalar(0), ["positive electrode"])
        variables = {
            "Negative particle surface tangential stress": a_n,
            "Positive particle surface tangential stress": a_p,
            "Negative particle surface radial stress": a_n,
            "Positive particle surface radial stress": a_p,
        }
        submodel = pybamm.loss_of_active_materials.Full(param)
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
