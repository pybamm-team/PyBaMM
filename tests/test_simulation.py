from pybamm.simulation import *
import unittest

class TestSolution(unittest.TestCase):

    def test_simulation_init(self):
        name = 'name'
        sim = Simulation(None, None, None, name=name)
        self.assertEqual(sim.name, name)

    def test_simulation_physics(self):
        pass
        # integral of c is known
        # integral of j is known
        # check convergence to steady state when current is zero
        # concentration and porosity limits

if __name__ == '__main__':
    unittest.main()
