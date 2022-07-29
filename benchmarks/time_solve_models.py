# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

import pybamm
import numpy as np


def solve_model_once(model, solver, t_eval):
    solver.solve(model, t_eval=t_eval)


class TimeSolveSPM:
    param_names = ["solve first", "parameter"]
    params = (
        [False, True],
        [
            "Marquis2019",
            "ORegan2022",
            "NCA_Kim2011",
            "Prada2013",
            # "Ai2020",
            "Ramadass2004",
            "Mohtat2020",
            "Chen2020",
            "Chen2020_plating",
            "Ecker2015",
        ],
    )
    solver = pybamm.CasadiSolver()

    def setup(self, solve_first, parameters):
        self.model = pybamm.lithium_ion.SPM()
        c_rate = 1
        tmax = 4000 / c_rate
        nb_points = 500
        self.t_eval = np.linspace(0, tmax, nb_points)
        geometry = self.model.default_geometry

        # load parameter values and process model and geometry
        param = pybamm.ParameterValues(parameters)
        param.process_model(self.model)
        param.process_geometry(geometry)

        # set mesh
        var_pts = {
            "x_n": 20,
            "x_s": 20,
            "x_p": 20,
            "r_n": 30,
            "r_p": 30,
            "y": 10,
            "z": 10,
        }
        mesh = pybamm.Mesh(geometry, self.model.default_submesh_types, var_pts)

        # discretise model
        disc = pybamm.Discretisation(mesh, self.model.default_spatial_methods)
        disc.process_model(self.model)
        if solve_first:
            solve_model_once(self.model, TimeSolveSPM.solver, self.t_eval)

    def time_solve_model(self, solve_first, parameters):
        TimeSolveSPM.solver.solve(self.model, t_eval=self.t_eval)


class TimeSolveSPMe:
    param_names = ["solve first", "parameter"]
    params = (
        [False, True],
        [
            "Marquis2019",
            "ORegan2022",
            "NCA_Kim2011",
            "Prada2013",
            # "Ai2020",
            "Ramadass2004",
            "Mohtat2020",
            "Chen2020",
            "Chen2020_plating",
            "Ecker2015",
        ],
    )
    solver = pybamm.CasadiSolver()

    def setup(self, solve_first, parameters):
        self.model = pybamm.lithium_ion.SPMe()
        c_rate = 1
        tmax = 4000 / c_rate
        nb_points = 500
        self.t_eval = np.linspace(0, tmax, nb_points)
        geometry = self.model.default_geometry

        # load parameter values and process model and geometry
        param = pybamm.ParameterValues(parameters)
        param.process_model(self.model)
        param.process_geometry(geometry)

        # set mesh
        var_pts = {
            "x_n": 20,
            "x_s": 20,
            "x_p": 20,
            "r_n": 30,
            "r_p": 30,
            "y": 10,
            "z": 10,
        }
        mesh = pybamm.Mesh(geometry, self.model.default_submesh_types, var_pts)

        # discretise model
        disc = pybamm.Discretisation(mesh, self.model.default_spatial_methods)
        disc.process_model(self.model)
        if solve_first:
            solve_model_once(self.model, TimeSolveSPMe.solver, self.t_eval)

    def time_solve_model(self, solve_first, parameters):
        TimeSolveSPMe.solver.solve(self.model, t_eval=self.t_eval)


class TimeSolveDFN:
    param_names = ["solve first", "parameter"]
    params = (
        [False, True],
        [
            "Marquis2019",
            "ORegan2022",
            # "NCA_Kim2011",
            "Prada2013",
            "Ai2020",
            "Ramadass2004",
            # "Mohtat2020",
            "Chen2020",
            "Chen2020_plating",
            "Ecker2015",
        ],
    )
    solver = pybamm.CasadiSolver()

    def setup(self, solve_first, parameters):
        self.model = pybamm.lithium_ion.DFN()
        c_rate = 1
        tmax = 4000 / c_rate
        nb_points = 500
        self.t_eval = np.linspace(0, tmax, nb_points)
        geometry = self.model.default_geometry

        # load parameter values and process model and geometry
        param = pybamm.ParameterValues(parameters)
        param.process_model(self.model)
        param.process_geometry(geometry)

        # set mesh
        var_pts = {
            "x_n": 20,
            "x_s": 20,
            "x_p": 20,
            "r_n": 30,
            "r_p": 30,
            "y": 10,
            "z": 10,
        }
        mesh = pybamm.Mesh(geometry, self.model.default_submesh_types, var_pts)

        # discretise model
        disc = pybamm.Discretisation(mesh, self.model.default_spatial_methods)
        disc.process_model(self.model)
        if solve_first:
            solve_model_once(self.model, TimeSolveDFN.solver, self.t_eval)

    def time_solve_model(self, solve_first, parameters):
        TimeSolveDFN.solver.solve(self.model, t_eval=self.t_eval)
