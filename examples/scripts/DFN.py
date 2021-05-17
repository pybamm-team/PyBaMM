#
# Example showing how to load and solve the DFN
#

import pybamm
import numpy as np

parameter_values = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Mohtat2020)
parameter_values.update(
    {
        "SEI kinetic rate constant [m.s-1]": 1e-15,
        #         "SEI resistivity [Ohm.m]": 0,
    },
)
spm = pybamm.lithium_ion.SPM({"SEI": "ec reaction limited"})
esoh_model = pybamm.lithium_ion.ElectrodeSOH()
esoh_sim = pybamm.Simulation(esoh_model, parameter_values=parameter_values)
param = spm.param

Vmin = 3.0
Vmax = 4.2
Cn = parameter_values.evaluate(param.C_n_init)
Cp = parameter_values.evaluate(param.C_p_init)
n_Li_init = parameter_values.evaluate(param.n_Li_particles_init)
c_n_max = parameter_values.evaluate(param.c_n_max)
c_p_max = parameter_values.evaluate(param.c_p_max)

esoh_sol = esoh_sim.solve(
    [0],
    inputs={"V_min": Vmin, "V_max": Vmax, "C_n": Cn, "C_p": Cp, "n_Li": n_Li_init},
)
print(esoh_sol["x_100"].data[0])
print(esoh_sol["y_100"].data[0])
pybamm.set_logging_level("NOTICE")
experiment = pybamm.Experiment(
    [
        (
            f"Discharge at 1C until {Vmin}V",
            "Rest for 1 hour",
            f"Charge at 1C until {Vmax}V",
            f"Hold at {Vmax}V until C/50",
        )
    ]
    * 500,
    termination="80% capacity",
)
sim_100 = pybamm.Simulation(
    spm,
    experiment=experiment,
    parameter_values=parameter_values,
    solver=pybamm.CasadiSolver("safe"),
)
spm_sol_100 = sim_100.solve(
    starting_solution=pybamm.load("examples/notebooks/bad_sol.pkl")
)
