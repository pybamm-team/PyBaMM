# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

import pybamm
import numpy as np
import sys

class TimeSolveSPM:
    params = [True, False]

    def setup(self, solve_first):
        self.model = pybamm.lithium_ion.SPM()
        geometry = self.model.default_geometry

        # load parameter values and process model and geometry
        chemistry = pybamm.parameter_sets.Marquis2019
        param = pybamm.ParameterValues(chemistry=chemistry)
        param.process_model(self.model)
        param.process_geometry(geometry)

        # set mesh
        var = pybamm.standard_spatial_vars
        var_pts = {
            var.x_n: 20,
            var.x_s: 20,
            var.x_p: 20,
            var.r_n: 30,
            var.r_p: 30,
            var.y: 10,
            var.z: 10,
        }
        mesh = pybamm.Mesh(geometry, self.model.default_submesh_types, var_pts)

        # discretise model
        disc = pybamm.Discretisation(mesh,self.model.default_spatial_methods)
        disc.process_model(self.model)

        c_rate = 1
        self.tmax = 4000/c_rate
        self.nb_points = 500
        if solve_first:
            solver = pybamm.CasadiSolver()
            t_eval = np.linspace(0, self.tmax, self.nb_points)
            solver.solve(self.model, t_eval=t_eval)


    def time_solve_model(self, solve_first):
        solver = pybamm.CasadiSolver()
        t_eval = np.linspace(0, self.tmax, self.nb_points)
        solver.solve(self.model, t_eval=t_eval)

    
