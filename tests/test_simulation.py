#
# Test the simulation class
#
import pybamm

import numpy as np

import unittest


class TestSimulation(unittest.TestCase):
    """Test the simulation class."""

    def test_simulation_name(self):
        sim = pybamm.Simulation(None, None, None, "test name")
        self.assertEqual(str(sim), "test name")

    def test_simulation_initialisation(self):
        model = pybamm.ReactionDiffusionModel()
        param = pybamm.Parameters()
        mesh = pybamm.Mesh(param, 50)
        solver = pybamm.Solver()
        sim = pybamm.Simulation(model, param, mesh)
        sim.solver = solver
        sim.initialise()
        self.assertEqual(sim.param.s.shape, mesh.xc.shape)
        self.assertTrue(
            np.all(sim.operators["xc"].div(mesh.x) == np.ones_like(mesh.xc))
        )
        self.assertEqual(param, sim.model.param)
        self.assertEqual(mesh, sim.model.mesh)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
