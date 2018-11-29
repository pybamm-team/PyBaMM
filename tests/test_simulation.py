#
# Test the simulation class
#
import pybamm

import numpy as np

import unittest


class TestSimulation(unittest.TestCase):
    """Test the simulation class."""

    def test_simulation_name(self):
        model = pybamm.ReactionDiffusionModel()
        param = pybamm.Parameters()
        mesh = pybamm.Mesh(param, target_npts=50)
        solver = pybamm.Solver()
        sim = pybamm.Simulation(
            model, param=param, mesh=mesh, solver=solver, name="test name"
        )

        self.assertEqual(sim.param.electrolyte.s.shape, sim.mesh.x.centres.shape)
        np.testing.assert_array_equal(
            sim.operators.x.div(mesh.x.edges), np.ones_like(mesh.x.centres)
        )
        self.assertEqual(param, sim.model.param)
        self.assertEqual(mesh, sim.model.mesh)

        self.assertEqual(str(sim), "test name")


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
