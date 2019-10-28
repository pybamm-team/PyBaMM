import pybamm
import unittest


class TestSimulation(unittest.TestCase):
    def test_set_model(self):

        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)

        self.assertEqual(sim.model, model)

    def test_reset_model(self):

        sim = pybamm.Simulation(pybamm.SPM())

        sim.discretize_model()

        sim.reset_model()

        # check can now re-parameterize model
