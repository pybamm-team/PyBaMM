import pybamm
import unittest


class TestSimulation(unittest.TestCase):
    def test_set_model(self):

        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)

        self.assertEqual(sim.model, model)

