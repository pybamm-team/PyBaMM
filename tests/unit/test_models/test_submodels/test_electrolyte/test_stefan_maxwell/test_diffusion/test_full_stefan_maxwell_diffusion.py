#
# Test full concentration submodel
#

import pybamm
import tests
import unittest


class TestFull(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.standard_parameters_lithium_ion
        a = pybamm.Scalar(0)
        variables = {
            "Porosity": a,
            "Porosity change": a,
            "Volume-averaged velocity": a,
            "Electrolyte concentration": a,
            "Negative electrode interfacial current density": pybamm.Broadcast(
                a, "negative electrode"
            ),
            "Positive electrode interfacial current density": pybamm.Broadcast(
                a, "positive electrode"
            ),
            "Cell temperature": pybamm.Broadcast(
                a, ["negative electrode", "separator", "positive electrode"]
            ),
        }
        icd = " interfacial current density"
        reactions = {
            "main": {
                "Negative": {"s": 1, "aj": "Negative electrode" + icd},
                "Positive": {"s": 1, "aj": "Positive electrode" + icd},
            }
        }
        submodel = pybamm.electrolyte.stefan_maxwell.diffusion.Full(param, reactions)
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
