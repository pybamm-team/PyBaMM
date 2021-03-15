#
# Example showing how to load and solve the SPMe
#

import pybamm
import numpy as np

pybamm.set_logging_level("INFO")

# load model
model = pybamm.lithium_ion.SPM(
    {
        "SEI": "ec reaction limited",
        "SEI film resistance": "none",
        "lithium plating": "reversible",
    }
)
model.convert_to_format = "python"

# create geometry
geometry = model.default_geometry

# load parameter values and process model and geometry
parameter_values = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Mohtat2020)
parameter_values.update(
    {
        "Lithium plating kinetic rate constant [m.s-1]": 1e-10,
        "Initial plated lithium concentration [mol.m-3]": 0,
        "Lithium metal partial molar volume [m3.mol-1]": 1.3e-5,
    },
    check_already_exists=False,
)
param = model.param

Vmin = 2.5
Vmax = 4.2
Cn = parameter_values.evaluate(param.C_n_init)
Cp = parameter_values.evaluate(param.C_p_init)
n_Li_init = parameter_values.evaluate(param.n_Li_particles_init)
c_n_max = parameter_values.evaluate(param.c_n_max)
c_p_max = parameter_values.evaluate(param.c_p_max)

esoh_model = pybamm.lithium_ion.ElectrodeSOH()
esoh_sim = pybamm.Simulation(esoh_model, parameter_values=parameter_values)
esoh_sol = esoh_sim.solve(
    [0],
    inputs={"V_min": Vmin, "V_max": Vmax, "C_n": Cn, "C_p": Cp, "n_Li": n_Li_init},
)

parameter_values.update(
    {
        "Initial concentration in negative electrode [mol.m-3]": esoh_sol["x_100"].data[
            0
        ]
        * c_n_max,
        "Initial concentration in positive electrode [mol.m-3]": esoh_sol["y_100"].data[
            0
        ]
        * c_p_max,
        "Lower voltage cut-off [V]": Vmin,
        "Upper voltage cut-off [V]": Vmax,
    }
)
# parameter_values["Current function [A]"] *= -1
parameter_values.process_model(model)
parameter_values.process_geometry(geometry)

# set mesh
mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# solve model for 1 hour
t_eval = np.linspace(0, 3300, 100)
solution = model.default_solver.solve(model, t_eval)

# plot
plot = pybamm.QuickPlot(
    solution,
    [
        "Negative electrode SOC",
        "Electrolyte concentration [mol.m-3]",
        "Positive electrode SOC",
        "Current [A]",
        "Negative electrode potential [V]",
        "Electrolyte potential [V]",
        "Positive electrode potential [V]",
        ["Measured open circuit voltage [V]", "Terminal voltage [V]"],
    ],
    time_unit="seconds",
    spatial_unit="um",
)
plot.dynamic_plot()
