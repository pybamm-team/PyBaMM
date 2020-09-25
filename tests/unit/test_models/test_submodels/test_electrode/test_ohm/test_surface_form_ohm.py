#
# Test surface form ohm submodel
#

import pybamm
import tests
import unittest


class TestSurfaceForm(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.LeadAcidParameters()

        a = pybamm.Scalar(0)
        variables = {
            "Current collector current density": a,
            "Negative electrolyte current density": pybamm.PrimaryBroadcast(
                a, ["negative electrode"]
            ),
            "Positive electrolyte current density": pybamm.PrimaryBroadcast(
                a, ["positive electrode"]
            ),
            "Negative electrode porosity": a,
            "Positive electrode porosity": a,
            "Separator electrolyte potential": a,
            "Positive electrode surface potential difference": a,
        }
        submodel = pybamm.electrode.ohm.SurfaceForm(param, "Negative")
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()

        submodel = pybamm.electrode.ohm.SurfaceForm(param, "Positive")
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
