#
# Test base current collector submodel
#

import pybamm
import tests
import unittest
import pybamm.models.submodels.current_collector as cc


class TestSetPotetetialSPM1plus1DModel(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.standard_parameters_lithium_ion
        submodel = cc.SetPotentialSingleParticle1plus1D(param)
        val = pybamm.PrimaryBroadcast(0.0, "current collector")
        variables = {
            "X-averaged positive electrode open circuit potential": val,
            "X-averaged negative electrode open circuit potential": val,
            "X-averaged positive electrode reaction overpotential": val,
            "X-averaged negative electrode reaction overpotential": val,
            "X-averaged electrolyte overpotential": val,
            "X-averaged positive electrode ohmic losses": val,
            "X-averaged negative electrode ohmic losses": val
        }
        std_tests = tests.StandardSubModelTests(submodel, variables)

        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
