# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

import pybamm
import numpy as np


def prepare_model(model):
    geometry = model.default_geometry

    # load parameter values and process model and geometry
    chemistry = pybamm.parameter_sets.Marquis2019
    param = pybamm.ParameterValues(chemistry=chemistry)
    param.process_model(model)
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
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

    # discretise model
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)


def solve_model_once(model, solver, t_eval):
    solver.solve(model, t_eval=t_eval)


class TimeSolveSPM:
    params = [True, False]
    solver = pybamm.CasadiSolver()

    def setup(self, solve_first):
        self.model = pybamm.lithium_ion.SPM()
        c_rate = 1
        tmax = 4000 / c_rate
        nb_points = 500
        self.t_eval = np.linspace(0, tmax, nb_points)
        prepare_model(self.model)
        if solve_first:
            solve_model_once(self.model, TimeSolveSPM.solver, self.t_eval)

    def time_solve_model(self, solve_first):
        TimeSolveSPM.solver.solve(self.model, t_eval=self.t_eval)


class TimeSolveSPMe:
    params = [True, False]
    solver = pybamm.CasadiSolver()

    def setup(self, solve_first):
        self.model = pybamm.lithium_ion.SPMe()
        c_rate = 1
        tmax = 4000 / c_rate
        nb_points = 500
        self.t_eval = np.linspace(0, tmax, nb_points)
        prepare_model(self.model)
        if solve_first:
            solve_model_once(self.model, TimeSolveSPMe.solver, self.t_eval)

    def time_solve_model(self, solve_first):
        TimeSolveSPMe.solver.solve(self.model, t_eval=self.t_eval)


class TimeSolveDFN:
    params = [True, False]
    solver = pybamm.CasadiSolver()

    def setup(self, solve_first):
        self.model = pybamm.lithium_ion.DFN()
        c_rate = 1
        tmax = 4000 / c_rate
        nb_points = 500
        self.t_eval = np.linspace(0, tmax, nb_points)
        prepare_model(self.model)
        if solve_first:
            solve_model_once(self.model, TimeSolveDFN.solver, self.t_eval)

    def time_solve_model(self, solve_first):
        TimeSolveDFN.solver.solve(self.model, t_eval=self.t_eval)
