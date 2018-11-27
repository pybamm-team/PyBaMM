#
# Test for the reaction-diffusion model
#
import pybamm
from tests.shared import pdes_io

import unittest
import numpy as np


class TestElectrolyteCurrent(unittest.TestCase):
    def setUp(self):
        self.model = pybamm.ElectrolyteCurrentModel()
        self.param = pybamm.Parameters()
        target_npts = 3
        tsteps = 10
        tend = 1
        self.mesh = pybamm.Mesh(self.param, target_npts, tsteps=tsteps, tend=tend)
        self.param.set_mesh_dependent_parameters(self.mesh)

    def tearDown(self):
        del self.model
        del self.param
        del self.mesh

    def test_model_shape(self):
        for spatial_discretisation in pybamm.KNOWN_SPATIAL_DISCRETISATIONS:
            operators = {
                domain: pybamm.Operators(spatial_discretisation, domain, self.mesh)
                for domain in self.model.domains()
            }
            self.model.set_simulation(self.param, operators, self.mesh)
            y, dydt = pdes_io(self.model)
            self.assertEqual(y.shape, dydt.shape)

    def test_model_physics(self):
        sim = pybamm.Simulation(self.model, self.param, self.mesh)
        solver = pybamm.Solver(
            integrator="BDF", spatial_discretisation="Finite Volumes"
        )

        sim.run(solver)

        interface = pybamm.Interface()
        interface.set_simulation(self.param, self.mesh)
        cn = np.ones((len(self.mesh.xcn), len(sim.vars.t)))
        cp = np.ones((len(self.mesh.xcp), len(sim.vars.t)))
        sim.vars.jn = interface.butler_volmer("xcn", cn, sim.vars.en)
        sim.vars.jp = interface.butler_volmer("xcp", cp, sim.vars.ep)

        sim.average()

        # integral of en is known
        jn_avg_expected = self.param.icell(sim.vars.t) / self.param.ln
        jp_avg_expected = -self.param.icell(sim.vars.t) / self.param.lp

        self.assertTrue(
            np.allclose(sim.vars.jn_avg[1:], jn_avg_expected[1:], atol=1e-15)
        )
        self.assertTrue(
            np.allclose(sim.vars.jp_avg[1:], jp_avg_expected[1:], atol=1e-15)
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
