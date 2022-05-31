# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

import pybamm
import numpy as np


def prepare_model_Marquis2019(model):
    geometry = model.default_geometry

    # load parameter values and process model and geometry
    param = pybamm.ParameterValues("Marquis2019")
    param.process_model(model)
    param.process_geometry(geometry)

    # set mesh
    var_pts = {"x_n": 20, "x_s": 20, "x_p": 20, "r_n": 30, "r_p": 30, "y": 10, "z": 10}
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

    # discretise model
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)


def prepare_model_ORegan2021(model):
    geometry = model.default_geometry

    # load parameter values and process model and geometry
    param = pybamm.ParameterValues("ORegan2021")
    param.process_model(model)
    param.process_geometry(geometry)

    # set mesh
    var_pts = {"x_n": 20, "x_s": 20, "x_p": 20, "r_n": 30, "r_p": 30, "y": 10, "z": 10}
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

    # discretise model
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)


def prepare_model_NCA_Kim2011(model):
    geometry = model.default_geometry

    # load parameter values and process model and geometry
    param = pybamm.ParameterValues("NCA_Kim2011")
    param.process_model(model)
    param.process_geometry(geometry)

    # set mesh
    var_pts = {"x_n": 20, "x_s": 20, "x_p": 20, "r_n": 30, "r_p": 30, "y": 10, "z": 10}
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

    # discretise model
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)


def prepare_model_Prada2013(model):
    geometry = model.default_geometry

    # load parameter values and process model and geometry
    param = pybamm.ParameterValues("Prada2013")
    param.process_model(model)
    param.process_geometry(geometry)

    # set mesh
    var_pts = {"x_n": 20, "x_s": 20, "x_p": 20, "r_n": 30, "r_p": 30, "y": 10, "z": 10}
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

    # discretise model
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)


def prepare_model_Ai2020(model):
    geometry = model.default_geometry

    # load parameter values and process model and geometry
    param = pybamm.ParameterValues("Ai2020")
    param.process_model(model)
    param.process_geometry(geometry)

    # set mesh
    var_pts = {"x_n": 20, "x_s": 20, "x_p": 20, "r_n": 30, "r_p": 30, "y": 10, "z": 10}
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

    # discretise model
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)


def prepare_model_Ramadass2004(model):
    geometry = model.default_geometry

    # load parameter values and process model and geometry
    param = pybamm.ParameterValues("Ramadass2004")
    param.process_model(model)
    param.process_geometry(geometry)

    # set mesh
    var_pts = {"x_n": 20, "x_s": 20, "x_p": 20, "r_n": 30, "r_p": 30, "y": 10, "z": 10}
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

    # discretise model
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)


def prepare_model_Mohtat2020(model):
    geometry = model.default_geometry

    # load parameter values and process model and geometry
    param = pybamm.ParameterValues("Mohtat2020")
    param.process_model(model)
    param.process_geometry(geometry)

    # set mesh
    var_pts = {"x_n": 20, "x_s": 20, "x_p": 20, "r_n": 30, "r_p": 30, "y": 10, "z": 10}
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

    # discretise model
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)


def prepare_model_Chen2020(model):
    geometry = model.default_geometry

    # load parameter values and process model and geometry
    param = pybamm.ParameterValues("Chen2020")
    param.process_model(model)
    param.process_geometry(geometry)

    # set mesh
    var_pts = {"x_n": 20, "x_s": 20, "x_p": 20, "r_n": 30, "r_p": 30, "y": 10, "z": 10}
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

    # discretise model
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)


def prepare_model_Chen2020_plating(model):
    geometry = model.default_geometry

    # load parameter values and process model and geometry
    param = pybamm.ParameterValues("Chen2020_plating")
    param.process_model(model)
    param.process_geometry(geometry)

    # set mesh
    var_pts = {"x_n": 20, "x_s": 20, "x_p": 20, "r_n": 30, "r_p": 30, "y": 10, "z": 10}
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

    # discretise model
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)


def prepare_model_Ecker2015(model):
    geometry = model.default_geometry

    # load parameter values and process model and geometry
    param = pybamm.ParameterValues("Ecker2015")
    param.process_model(model)
    param.process_geometry(geometry)

    # set mesh
    var_pts = {"x_n": 20, "x_s": 20, "x_p": 20, "r_n": 30, "r_p": 30, "y": 10, "z": 10}
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

    # discretise model
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)


def solve_model_once(model, solver, t_eval):
    solver.solve(model, t_eval=t_eval)


class TimeSolveSPMMarquis2019:
    params = [True, False]
    solver = pybamm.CasadiSolver()

    def setup(self, solve_first):
        self.model = pybamm.lithium_ion.SPM()
        c_rate = 1
        tmax = 4000 / c_rate
        nb_points = 500
        self.t_eval = np.linspace(0, tmax, nb_points)
        prepare_model_Marquis2019(self.model)
        if solve_first:
            solve_model_once(self.model, TimeSolveSPMMarquis2019.solver, self.t_eval)

    def time_solve_model(self, solve_first):
        TimeSolveSPMMarquis2019.solver.solve(self.model, t_eval=self.t_eval)


class TimeSolveSPMORegan2021:
    params = [True, False]
    solver = pybamm.CasadiSolver()

    def setup(self, solve_first):
        self.model = pybamm.lithium_ion.SPM()
        c_rate = 1
        tmax = 4000 / c_rate
        nb_points = 500
        self.t_eval = np.linspace(0, tmax, nb_points)
        prepare_model_ORegan2021(self.model)
        if solve_first:
            solve_model_once(self.model, TimeSolveSPMORegan2021.solver, self.t_eval)

    def time_solve_model(self, solve_first):
        TimeSolveSPMORegan2021.solver.solve(self.model, t_eval=self.t_eval)


class TimeSolveSPMNCA_Kim2011:
    params = [True, False]
    solver = pybamm.CasadiSolver()

    def setup(self, solve_first):
        self.model = pybamm.lithium_ion.SPM()
        c_rate = 1
        tmax = 4000 / c_rate
        nb_points = 500
        self.t_eval = np.linspace(0, tmax, nb_points)
        prepare_model_NCA_Kim2011(self.model)
        if solve_first:
            solve_model_once(self.model, TimeSolveSPMNCA_Kim2011.solver, self.t_eval)

    def time_solve_model(self, solve_first):
        TimeSolveSPMNCA_Kim2011.solver.solve(self.model, t_eval=self.t_eval)


class TimeSolveSPMPrada2013:
    params = [True, False]
    solver = pybamm.CasadiSolver()

    def setup(self, solve_first):
        self.model = pybamm.lithium_ion.SPM()
        c_rate = 1
        tmax = 4000 / c_rate
        nb_points = 500
        self.t_eval = np.linspace(0, tmax, nb_points)
        prepare_model_Prada2013(self.model)
        if solve_first:
            solve_model_once(self.model, TimeSolveSPMPrada2013.solver, self.t_eval)

    def time_solve_model(self, solve_first):
        TimeSolveSPMPrada2013.solver.solve(self.model, t_eval=self.t_eval)


class TimeSolveSPMAi2020:
    params = [True, False]
    solver = pybamm.ScikitsDaeSolver()

    def setup(self, solve_first):
        self.model = pybamm.lithium_ion.SPM()
        c_rate = 1
        tmax = 4000 / c_rate
        nb_points = 500
        self.t_eval = np.linspace(0, tmax, nb_points)
        prepare_model_Ai2020(self.model)
        if solve_first:
            solve_model_once(self.model, TimeSolveSPMAi2020.solver, self.t_eval)

    def time_solve_model(self, solve_first):
        TimeSolveSPMAi2020.solver.solve(self.model, t_eval=self.t_eval)


class TimeSolveSPMRamadass2004:
    params = [True, False]
    solver = pybamm.CasadiSolver()

    def setup(self, solve_first):
        self.model = pybamm.lithium_ion.SPM()
        c_rate = 1
        tmax = 4000 / c_rate
        nb_points = 500
        self.t_eval = np.linspace(0, tmax, nb_points)
        prepare_model_Ramadass2004(self.model)
        if solve_first:
            solve_model_once(self.model, TimeSolveSPMRamadass2004.solver, self.t_eval)

    def time_solve_model(self, solve_first):
        TimeSolveSPMRamadass2004.solver.solve(self.model, t_eval=self.t_eval)


class TimeSolveSPMMohtat2020:
    params = [True, False]
    solver = pybamm.CasadiSolver()

    def setup(self, solve_first):
        self.model = pybamm.lithium_ion.SPM()
        c_rate = 1
        tmax = 4000 / c_rate
        nb_points = 500
        self.t_eval = np.linspace(0, tmax, nb_points)
        prepare_model_Mohtat2020(self.model)
        if solve_first:
            solve_model_once(self.model, TimeSolveSPMMohtat2020.solver, self.t_eval)

    def time_solve_model(self, solve_first):
        TimeSolveSPMMohtat2020.solver.solve(self.model, t_eval=self.t_eval)


class TimeSolveSPMChen2020:
    params = [True, False]
    solver = pybamm.CasadiSolver()

    def setup(self, solve_first):
        self.model = pybamm.lithium_ion.SPM()
        c_rate = 1
        tmax = 4000 / c_rate
        nb_points = 500
        self.t_eval = np.linspace(0, tmax, nb_points)
        prepare_model_Chen2020(self.model)
        if solve_first:
            solve_model_once(self.model, TimeSolveSPMChen2020.solver, self.t_eval)

    def time_solve_model(self, solve_first):
        TimeSolveSPMChen2020.solver.solve(self.model, t_eval=self.t_eval)


class TimeSolveSPMChen2020_plating:
    params = [True, False]
    solver = pybamm.CasadiSolver()

    def setup(self, solve_first):
        self.model = pybamm.lithium_ion.SPM()
        c_rate = 1
        tmax = 4000 / c_rate
        nb_points = 500
        self.t_eval = np.linspace(0, tmax, nb_points)
        prepare_model_Chen2020_plating(self.model)
        if solve_first:
            solve_model_once(
                self.model, TimeSolveSPMChen2020_plating.solver, self.t_eval
            )

    def time_solve_model(self, solve_first):
        TimeSolveSPMChen2020_plating.solver.solve(self.model, t_eval=self.t_eval)


class TimeSolveSPMEcker2015:
    params = [True, False]
    solver = pybamm.CasadiSolver()

    def setup(self, solve_first):
        self.model = pybamm.lithium_ion.SPM()
        c_rate = 1
        tmax = 4000 / c_rate
        nb_points = 500
        self.t_eval = np.linspace(0, tmax, nb_points)
        prepare_model_Ecker2015(self.model)
        if solve_first:
            solve_model_once(self.model, TimeSolveSPMEcker2015.solver, self.t_eval)

    def time_solve_model(self, solve_first):
        TimeSolveSPMEcker2015.solver.solve(self.model, t_eval=self.t_eval)


class TimeSolveSPMeMarquis2019:
    params = [True, False]
    solver = pybamm.CasadiSolver()

    def setup(self, solve_first):
        self.model = pybamm.lithium_ion.SPMe()
        c_rate = 1
        tmax = 4000 / c_rate
        nb_points = 500
        self.t_eval = np.linspace(0, tmax, nb_points)
        prepare_model_Marquis2019(self.model)
        if solve_first:
            solve_model_once(self.model, TimeSolveSPMeMarquis2019.solver, self.t_eval)

    def time_solve_model(self, solve_first):
        TimeSolveSPMeMarquis2019.solver.solve(self.model, t_eval=self.t_eval)


class TimeSolveSPMeORegan2021:
    params = [True, False]
    solver = pybamm.CasadiSolver()

    def setup(self, solve_first):
        self.model = pybamm.lithium_ion.SPMe()
        c_rate = 1
        tmax = 4000 / c_rate
        nb_points = 500
        self.t_eval = np.linspace(0, tmax, nb_points)
        prepare_model_ORegan2021(self.model)
        if solve_first:
            solve_model_once(self.model, TimeSolveSPMeORegan2021.solver, self.t_eval)

    def time_solve_model(self, solve_first):
        TimeSolveSPMeORegan2021.solver.solve(self.model, t_eval=self.t_eval)


class TimeSolveSPMeNCA_Kim2011:
    params = [True, False]
    solver = pybamm.CasadiSolver()

    def setup(self, solve_first):
        self.model = pybamm.lithium_ion.SPMe()
        c_rate = 1
        tmax = 4000 / c_rate
        nb_points = 500
        self.t_eval = np.linspace(0, tmax, nb_points)
        prepare_model_NCA_Kim2011(self.model)
        if solve_first:
            solve_model_once(self.model, TimeSolveSPMeNCA_Kim2011.solver, self.t_eval)

    def time_solve_model(self, solve_first):
        TimeSolveSPMeNCA_Kim2011.solver.solve(self.model, t_eval=self.t_eval)


class TimeSolveSPMePrada2013:
    params = [True, False]
    solver = pybamm.CasadiSolver()

    def setup(self, solve_first):
        self.model = pybamm.lithium_ion.SPMe()
        c_rate = 1
        tmax = 4000 / c_rate
        nb_points = 500
        self.t_eval = np.linspace(0, tmax, nb_points)
        prepare_model_Prada2013(self.model)
        if solve_first:
            solve_model_once(self.model, TimeSolveSPMePrada2013.solver, self.t_eval)

    def time_solve_model(self, solve_first):
        TimeSolveSPMePrada2013.solver.solve(self.model, t_eval=self.t_eval)


class TimeSolveSPMeAi2020:
    params = [True, False]
    solver = pybamm.ScikitsDaeSolver()

    def setup(self, solve_first):
        self.model = pybamm.lithium_ion.SPMe()
        c_rate = 1
        tmax = 4000 / c_rate
        nb_points = 500
        self.t_eval = np.linspace(0, tmax, nb_points)
        prepare_model_Ai2020(self.model)
        if solve_first:
            solve_model_once(self.model, TimeSolveSPMeAi2020.solver, self.t_eval)

    def time_solve_model(self, solve_first):
        TimeSolveSPMeAi2020.solver.solve(
            self.model, t_eval=self.t_eval, calc_esoh=False
        )


class TimeSolveSPMeRamadass2004:
    params = [True, False]
    solver = pybamm.CasadiSolver()

    def setup(self, solve_first):
        self.model = pybamm.lithium_ion.SPMe()
        c_rate = 1
        tmax = 4000 / c_rate
        nb_points = 500
        self.t_eval = np.linspace(0, tmax, nb_points)
        prepare_model_Ramadass2004(self.model)
        if solve_first:
            solve_model_once(self.model, TimeSolveSPMeRamadass2004.solver, self.t_eval)

    def time_solve_model(self, solve_first):
        TimeSolveSPMeRamadass2004.solver.solve(self.model, t_eval=self.t_eval)


class TimeSolveSPMeMohtat2020:
    params = [True, False]
    solver = pybamm.CasadiSolver()

    def setup(self, solve_first):
        self.model = pybamm.lithium_ion.SPMe()
        c_rate = 1
        tmax = 4000 / c_rate
        nb_points = 500
        self.t_eval = np.linspace(0, tmax, nb_points)
        prepare_model_Mohtat2020(self.model)
        if solve_first:
            solve_model_once(self.model, TimeSolveSPMeMohtat2020.solver, self.t_eval)

    def time_solve_model(self, solve_first):
        TimeSolveSPMeMohtat2020.solver.solve(self.model, t_eval=self.t_eval)


class TimeSolveSPMeChen2020:
    params = [True, False]
    solver = pybamm.CasadiSolver()

    def setup(self, solve_first):
        self.model = pybamm.lithium_ion.SPMe()
        c_rate = 1
        tmax = 4000 / c_rate
        nb_points = 500
        self.t_eval = np.linspace(0, tmax, nb_points)
        prepare_model_Chen2020(self.model)
        if solve_first:
            solve_model_once(self.model, TimeSolveSPMeChen2020.solver, self.t_eval)

    def time_solve_model(self, solve_first):
        TimeSolveSPMeChen2020.solver.solve(self.model, t_eval=self.t_eval)


class TimeSolveSPMeChen2020_plating:
    params = [True, False]
    solver = pybamm.CasadiSolver()

    def setup(self, solve_first):
        self.model = pybamm.lithium_ion.SPMe()
        c_rate = 1
        tmax = 4000 / c_rate
        nb_points = 500
        self.t_eval = np.linspace(0, tmax, nb_points)
        prepare_model_Chen2020_plating(self.model)
        if solve_first:
            solve_model_once(
                self.model, TimeSolveSPMeChen2020_plating.solver, self.t_eval
            )

    def time_solve_model(self, solve_first):
        TimeSolveSPMeChen2020_plating.solver.solve(self.model, t_eval=self.t_eval)


class TimeSolveSPMeEcker2015:
    params = [True, False]
    solver = pybamm.CasadiSolver()

    def setup(self, solve_first):
        self.model = pybamm.lithium_ion.SPMe()
        c_rate = 1
        tmax = 4000 / c_rate
        nb_points = 500
        self.t_eval = np.linspace(0, tmax, nb_points)
        prepare_model_Ecker2015(self.model)
        if solve_first:
            solve_model_once(self.model, TimeSolveSPMeEcker2015.solver, self.t_eval)

    def time_solve_model(self, solve_first):
        TimeSolveSPMeEcker2015.solver.solve(self.model, t_eval=self.t_eval)


class TimeSolveDFNMarquis2019:
    params = [True, False]
    solver = pybamm.CasadiSolver()

    def setup(self, solve_first):
        self.model = pybamm.lithium_ion.DFN()
        c_rate = 1
        tmax = 4000 / c_rate
        nb_points = 500
        self.t_eval = np.linspace(0, tmax, nb_points)
        prepare_model_Marquis2019(self.model)
        if solve_first:
            solve_model_once(self.model, TimeSolveDFNMarquis2019.solver, self.t_eval)

    def time_solve_model(self, solve_first):
        TimeSolveDFNMarquis2019.solver.solve(self.model, t_eval=self.t_eval)


class TimeSolveDFNORegan2021:
    params = [True, False]
    solver = pybamm.CasadiSolver()

    def setup(self, solve_first):
        self.model = pybamm.lithium_ion.DFN()
        c_rate = 1
        tmax = 4000 / c_rate
        nb_points = 500
        self.t_eval = np.linspace(0, tmax, nb_points)
        prepare_model_ORegan2021(self.model)
        if solve_first:
            solve_model_once(self.model, TimeSolveDFNORegan2021.solver, self.t_eval)

    def time_solve_model(self, solve_first):
        TimeSolveDFNORegan2021.solver.solve(self.model, t_eval=self.t_eval)


class TimeSolveDFNNCA_Kim2011:
    params = [True, False]
    solver = pybamm.ScikitsDaeSolver()

    def setup(self, solve_first):
        self.model = pybamm.lithium_ion.DFN()
        c_rate = 1
        tmax = 4000 / c_rate
        nb_points = 500
        self.t_eval = np.linspace(0, tmax, nb_points)
        prepare_model_NCA_Kim2011(self.model)
        if solve_first:
            solve_model_once(self.model, TimeSolveDFNNCA_Kim2011.solver, self.t_eval)

    def time_solve_model(self, solve_first):
        TimeSolveDFNNCA_Kim2011.solver.solve(self.model, t_eval=self.t_eval)


class TimeSolveDFNPrada2013:
    params = [True, False]
    solver = pybamm.CasadiSolver()

    def setup(self, solve_first):
        self.model = pybamm.lithium_ion.DFN()
        c_rate = 1
        tmax = 4000 / c_rate
        nb_points = 500
        self.t_eval = np.linspace(0, tmax, nb_points)
        prepare_model_Prada2013(self.model)
        if solve_first:
            solve_model_once(self.model, TimeSolveDFNPrada2013.solver, self.t_eval)

    def time_solve_model(self, solve_first):
        TimeSolveDFNPrada2013.solver.solve(self.model, t_eval=self.t_eval)


class TimeSolveDFNAi2020:
    params = [True, False]
    solver = pybamm.CasadiSolver()

    def setup(self, solve_first):
        self.model = pybamm.lithium_ion.DFN()
        c_rate = 1
        tmax = 4000 / c_rate
        nb_points = 500
        self.t_eval = np.linspace(0, tmax, nb_points)
        prepare_model_Ai2020(self.model)
        if solve_first:
            solve_model_once(self.model, TimeSolveDFNAi2020.solver, self.t_eval)

    def time_solve_model(self, solve_first):
        TimeSolveDFNAi2020.solver.solve(self.model, t_eval=self.t_eval)


class TimeSolveDFNRamadass2004:
    params = [True, False]
    solver = pybamm.CasadiSolver()

    def setup(self, solve_first):
        self.model = pybamm.lithium_ion.DFN()
        c_rate = 1
        tmax = 4000 / c_rate
        nb_points = 500
        self.t_eval = np.linspace(0, tmax, nb_points)
        prepare_model_Ramadass2004(self.model)
        if solve_first:
            solve_model_once(self.model, TimeSolveDFNRamadass2004.solver, self.t_eval)

    def time_solve_model(self, solve_first):
        TimeSolveDFNRamadass2004.solver.solve(self.model, t_eval=self.t_eval)


# class TimeSolveDFNMohtat2020:
#     params = [True, False]
#     solver = pybamm.CasadiSolver()

#     def setup(self, solve_first):
#         self.model = pybamm.lithium_ion.DFN()
#         c_rate = 1
#         tmax = 4000 / c_rate
#         nb_points = 500
#         self.t_eval = np.linspace(0, tmax, nb_points)
#         prepare_model_Mohtat2020(self.model)
#         if solve_first:
#             solve_model_once(self.model, TimeSolveDFNMohtat2020.solver, self.t_eval)

#     def time_solve_model(self, solve_first):
#         TimeSolveDFNMohtat2020.solver.solve(self.model, t_eval=self.t_eval)


class TimeSolveDFNChen2020:
    params = [True, False]
    solver = pybamm.CasadiSolver()

    def setup(self, solve_first):
        self.model = pybamm.lithium_ion.DFN()
        c_rate = 1
        tmax = 4000 / c_rate
        nb_points = 500
        self.t_eval = np.linspace(0, tmax, nb_points)
        prepare_model_Chen2020(self.model)
        if solve_first:
            solve_model_once(self.model, TimeSolveDFNChen2020.solver, self.t_eval)

    def time_solve_model(self, solve_first):
        TimeSolveDFNChen2020.solver.solve(self.model, t_eval=self.t_eval)


class TimeSolveDFNChen2020_plating:
    params = [True, False]
    solver = pybamm.CasadiSolver()

    def setup(self, solve_first):
        self.model = pybamm.lithium_ion.DFN()
        c_rate = 1
        tmax = 4000 / c_rate
        nb_points = 500
        self.t_eval = np.linspace(0, tmax, nb_points)
        prepare_model_Chen2020_plating(self.model)
        if solve_first:
            solve_model_once(
                self.model, TimeSolveDFNChen2020_plating.solver, self.t_eval
            )

    def time_solve_model(self, solve_first):
        TimeSolveDFNChen2020_plating.solver.solve(self.model, t_eval=self.t_eval)


class TimeSolveDFNEcker2015:
    params = [True, False]
    solver = pybamm.CasadiSolver()

    def setup(self, solve_first):
        self.model = pybamm.lithium_ion.DFN()
        c_rate = 1
        tmax = 4000 / c_rate
        nb_points = 500
        self.t_eval = np.linspace(0, tmax, nb_points)
        prepare_model_Ecker2015(self.model)
        if solve_first:
            solve_model_once(self.model, TimeSolveDFNEcker2015.solver, self.t_eval)

    def time_solve_model(self, solve_first):
        TimeSolveDFNEcker2015.solver.solve(self.model, t_eval=self.t_eval)
