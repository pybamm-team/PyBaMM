#
# Test for the reaction-diffusion model
#
import pybamm
from tests.shared import pdes_io

import unittest
import numpy as np
from numpy.linalg import norm
import scipy.integrate as it


class TestReactionDiffusion(unittest.TestCase):
    def setUp(self):
        self.model = pybamm.ReactionDiffusionModel()
        self.param = pybamm.Parameters()
        target_npts = 10
        tsteps = 10
        tend = 1
        self.mesh = pybamm.Mesh(self.param, target_npts, tsteps=tsteps, tend=tend)
        self.param.set_mesh(self.mesh)

    def tearDown(self):
        del self.model
        del self.param
        del self.mesh

    def test_model_shape(self):
        for spatial_discretisation in pybamm.KNOWN_SPATIAL_DISCRETISATIONS:
            solver = pybamm.Solver(spatial_discretisation=spatial_discretisation)
            sim = pybamm.Simulation(self.model, solver=solver)
            y, dydt = pdes_io(sim.model)
            self.assertEqual(y.shape, dydt.shape)

    def test_model_physics(self):
        """Check that the average concentration is as expected"""
        sim = pybamm.Simulation(self.model)
        sim.run()
        sim.average()

        c_avg_expected = self.param.electrolyte.c0 + (
            self.param.electrolyte.sn - self.param.electrolyte.sp
        ) * it.cumtrapz(self.param.icell(sim.vars.t), sim.vars.t, initial=0.0)

        np.testing.assert_allclose(sim.vars.c_avg, c_avg_expected, atol=4e-16)

    def test_model_convergence(self):
        """
        Exact solution: c = exp(-4*pi**2*t) * cos(2*pi*x)
        Initial conditions: c0 = cos(2*pi*x)
        Boundary conditions: Zero flux
        Source terms: 0

        Can achieve "convergence" in time by changing the integrator tolerance
        Can't get h**2 convergence in space
        """
        param = pybamm.Parameters(tests="convergence")
        param.set_mesh(self.mesh)

        def c_exact(t):
            return np.exp(-4 * np.pi ** 2 * t) * np.cos(2 * np.pi * self.mesh.x.centres)

        inits = {"concentration": c_exact(0)}

        def bcs(t):
            return {"concentration": (np.array([0]), np.array([0]))}

        def sources(t):
            return {"concentration": 0}

        tests = {"inits": inits, "bcs": bcs, "sources": sources}

        model = pybamm.ReactionDiffusionModel(tests=tests)

        ns = [1, 2, 3]
        errs = [0] * len(ns)
        for i, n in enumerate(ns):
            solver = pybamm.Solver(
                integrator="BDF",
                spatial_discretisation="Finite Volumes",
                tol=10 ** (-n),
            )
            sim = pybamm.Simulation(model, param=param, mesh=self.mesh, solver=solver)
            sim.run()
            errs[i] = norm(
                sim.vars.c.T - c_exact(self.mesh.time[:, np.newaxis])
            ) / norm(c_exact(self.mesh.time[:, np.newaxis]))
        [self.assertLess(errs[i + 1] / errs[i], 0.14) for i in range(len(errs) - 1)]


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
