# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

import pybamm
import numpy as np
import sys

class TimeSolveStandardModels:
    models = [pybamm.lithium_ion.SPM(), pybamm.lithium_ion.SPMe(), pybamm.lithium_ion.DFN()]
    solve_first = [True, False]
    params = (models, solve_first)

    def setup(self, model, solve_first):
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


    def time_solve_model(self, model, solve_first):
        solver = pybamm.CasadiSolver()
        t_eval = np.linspace(0, self.tmax, self.nb_points)
        solver.solve(self.model, t_eval=t_eval)

class TimeSimulationDFN:
    params = [0.1, 0.5, 1, 2]
    timeout = 240.0
    def setup(self, C_rate):
        model = pybamm.lithium_ion.DFN()
        # experiment = pybamm.Experiment(
        #     ["Discharge at {}C until 3.105 V".format(C_rate)]
        # )
        experiment = pybamm.Experiment(
            [
                "Discharge at {}C until 3.105 V".format(C_rate),
                "Rest for 2 hours",
                "Charge at C/3 until 4.4 V",
                "Hold at 4.4 V until C/10",
                # "Rest for 2 hours"
            ]
        )
        self.sim = pybamm.Simulation(model, experiment=experiment)

    def time_solve_CC(self, C_rate):
        self.sim.solve()

class TimeBuild: 
    params = [pybamm.lithium_ion.SPM(), pybamm.lithium_ion.SPMe(), pybamm.lithium_ion.DFN()]
    def setup(self, model):
        chemistry = pybamm.parameter_sets.Marquis2019
        self.param = pybamm.ParameterValues(chemistry=chemistry)
        self.model = model

    def time_setup_model(self, model):
        geometry = self.model.default_geometry

        # process model and geometry
        self.param.process_model(self.model)
        self.param.process_geometry(geometry)

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
        disc = pybamm.Discretisation(mesh, self.model.default_spatial_methods)
        disc.process_model(self.model)

    def time_setup_model_simulation(self, model):
        pybamm.Simulation(self.model, parameter_values=self.param, C_rate=1)

    def time_setup_model_simulation_with_experiment(self, model):
        C_rate = 0.1
        experiment = pybamm.Experiment(
            [
                "Discharge at {}C until 3.105 V".format(C_rate),
            ]
        )
        pybamm.Simulation(self.model, parameter_values=self.param, experiment=experiment)
