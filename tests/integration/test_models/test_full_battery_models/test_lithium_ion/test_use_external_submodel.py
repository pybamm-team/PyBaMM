#
# Tests for external submodels
#
import pybamm
import unittest


# only works for statevector inputs
class TestExternalSubmodel(unittest.TestCase):
    def test_external_temperature(self):

        model_options = {"thermal": "x-full", "external submodels": ["thermal"]}
        model = pybamm.lithium_ion.SPMe(model_options)

        sim = pybamm.Simulation(model)
        sim.build()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
