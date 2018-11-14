import pybamm

import unittest
import numpy as np
from numpy.linalg import norm
import scipy.integrate as it


class TestSolution(unittest.TestCase):
    def test_simulation_physics(self):
        param = pybamm.Parameters()
        tsteps = 100
        tend = 1
        target_npts = 10
        mesh = pybamm.Mesh(param, target_npts, tsteps=tsteps, tend=tend)

        model = pybamm.Model("Electrolyte diffusion")
        simulation = pybamm.Simulation(
            model, param, mesh, name="Electrolyte diffusion"
        )
        solver = pybamm.Solver(
            integrator="BDF", spatial_discretisation="Finite Volumes"
        )

        simulation.run(solver)

        simulation.average()
        # integral of c is known
        c_avg_expected = 1 + (param.sn - param.sp) * it.cumtrapz(
            param.icell(simulation.vars.t), simulation.vars.t, initial=0.0
        )

        self.assertTrue(
            np.allclose(simulation.vars.c_avg, c_avg_expected, atol=4e-16)
        )
        # integral of j is known
        # check convergence to steady state when current is zero
        # concentration and porosity limits

    def test_electrolyte_diffusion_convergence(self):
        """
        Exact solution: c = exp(-4*pi**2*t * cos(2*pi*x))
        Initial conditions: c0 = cos(2*pi*x)
        Boundary conditions: Zero flux
        Source terms: 0

        Can achieve "convergence" in time by changing the integrator tolerance
        Can't get h**2 convergence in space
        """
        param = pybamm.Parameters()
        tsteps = 100
        tend = 1
        mesh = pybamm.Mesh(param, 50, tsteps=tsteps, tend=tend)

        def c_exact(t):
            return np.exp(-4 * np.pi ** 2 * t) * np.cos(2 * np.pi * mesh.xc)

        inits = c_exact(0)

        def bcs(t):
            return {"concentration": (np.array([0]), np.array([0]))}

        def sources(t):
            return {"concentration": 0}

        tests = {"inits": inits, "bcs": bcs, "sources": sources}

        model = pybamm.Model("Electrolyte diffusion", tests=tests)
        simulation = pybamm.Simulation(model, param, mesh)

        ns = [1, 2, 3]
        errs = [0] * len(ns)
        for i, n in enumerate(ns):
            solver = pybamm.Solver(
                integrator="BDF",
                spatial_discretisation="Finite Volumes",
                tol=10 ** (-n),
            )
            simulation.run(solver)
            errs[i] = norm(
                simulation.vars.c.T - c_exact(mesh.time[:, np.newaxis])
            ) / norm(c_exact(mesh.time[:, np.newaxis]))
        [
            self.assertLess(errs[i + 1] / errs[i], 0.14)
            for i in range(len(errs) - 1)
        ]


if __name__ == "__main__":
    unittest.main()
