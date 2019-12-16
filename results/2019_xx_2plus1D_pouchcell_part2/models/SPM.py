import pybamm
import numpy as np


class SPM:
    def __init__(self, thermal=False, param=None):
        options = {}
        if thermal is True:
            options.update({"thermal": "x-lumped"})

        self.model = pybamm.lithium_ion.SPM(options)
        self.param = self.model.default_parameter_values

        if param:
            self.param.update(param)

    def solve(self, var_pts, C_rate=1, t_eval=None):

        # discharge timescale
        if t_eval is None:
            tau = self.param.evaluate(
                pybamm.standard_parameters_lithium_ion.tau_discharge
            )
            t_end = 900 / tau
            t_eval = np.linspace(0, t_end, 120)

        self.sim = pybamm.Simulation(
            self.model, parameter_values=self.param, var_pts=var_pts, C_rate=C_rate
        )
        self.sim.solve(t_eval=t_eval)

        self.t = self.sim.solution.t
        self.y = self.sim.solution.y

    def processed_variables(self, variables):
        built_vars = {var: self.sim.built_model.variables[var] for var in variables}
        processed_vars = pybamm.post_process_variables(
            built_vars, self.t, self.y, mesh=self.sim.mesh
        )
        return processed_vars
