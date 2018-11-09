from pybamm.parameters import Parameters
from pybamm.mesh import Mesh
from pybamm.models.model_class import Model
from pybamm.simulation import Simulation
from pybamm.solver import Solver
import unittest

import numpy as np
import scipy.integrate as it

class TestSolution(unittest.TestCase):

    def test_simulation_physics(self):
        param = Parameters()
        tsteps = 100
        tend = 1
        target_npts = 10
        mesh = Mesh(param, target_npts, tsteps=tsteps, tend=tend)

        model = Model("Simple Diffusion")
        simulation = Simulation(model, param, mesh, name="Simple Diffusion")
        solver = Solver(integrator="BDF",
                        spatial_discretisation="Finite Volumes")

        simulation.run(solver)

        simulation.average()
        # integral of c is known
        c_avg_expected = (1 + (param.sn - param.sp)
                          * it.cumtrapz(param.icell(simulation.vars.t),
                                        simulation.vars.t,
                                        initial=0.0))

        self.assertTrue(np.allclose(simulation.vars.c_avg,
                                    c_avg_expected,
                                    atol=4e-16))
        # integral of j is known
        # check convergence to steady state when current is zero
        # concentration and porosity limits

if __name__ == '__main__':
    unittest.main()
