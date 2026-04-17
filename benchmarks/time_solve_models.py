# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

import numpy as np
import numpy.typing as npt

import pybamm
from benchmarks.benchmark_utils import set_random_seed


def solve_model_once(model, solver, t_eval, t_interp):
    solver.solve(model, t_eval=t_eval, t_interp=t_interp)


class TimeSolveSPM:
    param_names = ["solve first", "parameter", "solver_class"]
    params = (
        [False, True],
        [
            "Marquis2019",
            "ORegan2022",
            "NCA_Kim2011",
            "Prada2013",
            "Ramadass2004",
            "Chen2020",
            "Ecker2015",
        ],
        [
            pybamm.CasadiSolver,
            pybamm.IDAKLUSolver,
        ],
    )
    model: pybamm.BaseModel
    solver: pybamm.BaseSolver
    t_eval: npt.NDArray[np.float64]
    t_interp: npt.NDArray[np.float64] | None

    def setup(self, solve_first, parameters, solver_class):
        set_random_seed()
        self.solver = solver_class()
        self.model = pybamm.lithium_ion.SPM()
        c_rate = 1
        tmax = 4000 / c_rate
        if self.solver.supports_interp:
            self.t_eval = np.array([0, tmax])
            self.t_interp = None
        else:
            nb_points = 500
            self.t_eval = np.linspace(0, tmax, nb_points)
            self.t_interp = None

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
            solve_model_once(self.model, self.solver, self.t_eval, self.t_interp)

    def time_solve_model(self, _solve_first, _parameters, _solver_class):
        self.solver.solve(self.model, t_eval=self.t_eval, t_interp=self.t_interp)


class TimeSolveSPMe:
    param_names = ["solve first", "parameter", "solver_class"]
    params = (
        [False, True],
        [
            "Marquis2019",
            "ORegan2022",
            "NCA_Kim2011",
            "Prada2013",
            "Ramadass2004",
            "Chen2020",
            "Ecker2015",
        ],
        [
            pybamm.CasadiSolver,
            pybamm.IDAKLUSolver,
        ],
    )
    model: pybamm.BaseModel
    solver: pybamm.BaseSolver
    t_eval: npt.NDArray[np.float64]

    def setup(self, solve_first, parameters, solver_class):
        set_random_seed()
        self.solver = solver_class()
        self.model = pybamm.lithium_ion.SPMe()
        c_rate = 1
        tmax = 4000 / c_rate
        if self.solver.supports_interp:
            self.t_eval = np.array([0, tmax])
            self.t_interp = None
        else:
            nb_points = 500
            self.t_eval = np.linspace(0, tmax, nb_points)
            self.t_interp = None
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
            solve_model_once(self.model, self.solver, self.t_eval, self.t_interp)

    def time_solve_model(self, _solve_first, _parameters, _solver_class):
        self.solver.solve(self.model, t_eval=self.t_eval, t_interp=self.t_interp)


class TimeRepeatedSolveAndObserveVoltage:
    param_names = ["model", "compilation"]
    params = (
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.SPMe, pybamm.lithium_ion.DFN],
        ["vm", "aot"],
    )
    sim: pybamm.Simulation
    sol: pybamm.Solution
    t_eval: list[float]
    t_interp: npt.NDArray[np.float64]

    def setup(self, model_class, compilation):
        set_random_seed()
        self.sim = pybamm.Simulation(
            model_class(),
            solver=pybamm.IDAKLUSolver(options={"compilation": compilation}),
        )
        self.t_eval = [0.0, 3600.0]
        self.t_interp = np.linspace(self.t_eval[0], self.t_eval[-1], 10000)
        # Warm the casadi/AOT caches and the voltage observer before timing.
        self.sol = self.sim.solve(self.t_eval, t_interp=self.t_interp)
        _ = self.sol["Voltage [V]"].data

    def time_repeated_solve(self, _model_class, _compilation):
        self.sim.solve(self.t_eval, t_interp=self.t_interp)

    def time_voltage_observe(self, _model_class, _compilation):
        self.sol._variables.clear()
        _ = self.sol["Voltage [V]"].data


class TimeSolveDFN:
    param_names = ["solve first", "parameter", "solver_class"]
    params = (
        [False, True],
        [
            "Marquis2019",
            "ORegan2022",
            "Prada2013",
            "Ai2020",
            "Ramadass2004",
            "Chen2020",
            "Ecker2015",
        ],
        [
            pybamm.CasadiSolver,
            pybamm.IDAKLUSolver,
        ],
    )
    model: pybamm.BaseModel
    solver: pybamm.BaseSolver
    t_eval: npt.NDArray[np.float64]

    def setup(self, solve_first, parameters, solver_class):
        set_random_seed()
        if (parameters, solver_class) == (
            "ORegan2022",
            pybamm.CasadiSolver,
        ):
            raise NotImplementedError
        self.solver = solver_class()
        self.model = pybamm.lithium_ion.DFN()
        c_rate = 1
        tmax = 4000 / c_rate
        if self.solver.supports_interp:
            self.t_eval = np.array([0, tmax])
            self.t_interp = None
        else:
            nb_points = 500
            self.t_eval = np.linspace(0, tmax, nb_points)
            self.t_interp = None
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
            solve_model_once(self.model, self.solver, self.t_eval, self.t_interp)

    def time_solve_model(self, _solve_first, _parameters, _solver_class):
        self.solver.solve(self.model, t_eval=self.t_eval, t_interp=self.t_interp)
