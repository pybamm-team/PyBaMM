import pybamm
import unittest


class TestSimulation(unittest.TestCase):
    def test_run_with_spm(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)
        sim.solve()

    def test_update_parameters(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)

        sim.parameterize_model()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
