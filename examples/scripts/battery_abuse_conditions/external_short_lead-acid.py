#
# Example showing how to load and solve the full model
#

import pybamm
import numpy as np

pybamm.set_logging_level("INFO")

# definitions for battery operation
class ExternalCircuitFunction:
    num_switches = 0

    def __call__(self, variables):
        I = variables["Current [A]"]
        V = variables["Terminal voltage [V]"]
        return V / I - pybamm.FunctionParameter("Function", pybamm.t)


def ambient_temperature(t):
    T = 0  # degree temperature change per hour
    return 300 + t * T / 3600


# load model
options = {
    "thermal": "x-full",
    "thermal current collector": False,
    "operating mode": ExternalCircuitFunction(),
}
model = pybamm.lead_acid.Full(options)
model.events = {}

# create geometry
geometry = model.default_geometry

# load parameter values and process model and geometry
param = model.default_parameter_values
param.update(
    {"Ambient temperature [K]": ambient_temperature, "Function": 0.05},
    check_already_exists=False,
)
param.process_model(model)
param.process_geometry(geometry)

# set mesh
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 30, var.x_s: 30, var.x_p: 30, var.r_n: 10, var.r_p: 10}
mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# solve model
t_eval = np.linspace(0, 0.46 * 3600, 100)

# solver = model.default_solver
solver = pybamm.CasadiSolver(mode="fast")
solver.rtol = 1e-6
solver.atol = 1e-6
solution = solver.solve(model, t_eval)

# plot
output_variables = [
    "Electrolyte concentration",
    "Electrolyte potential [V]",
    "X-averaged cell temperature [K]",
    "Interfacial current density",
    "Negative electrode potential [V]",
    "Ambient temperature [K]",
    "Current [A]",
    "Terminal voltage [V]",
    "Cell temperature [K]",
]

plot = pybamm.QuickPlot(solution, output_variables)
plot.dynamic_plot()
