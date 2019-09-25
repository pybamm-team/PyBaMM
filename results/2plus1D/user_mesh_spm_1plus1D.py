import pybamm
import numpy as np
import sys

# set logging level
pybamm.set_logging_level("INFO")

# load (1+1D) SPMe model
options = {
    "current collector": "jelly roll",
    "dimensionality": 1,
    "thermal": "lumped",
}
model = pybamm.lithium_ion.SPM(options)

# create geometry
geometry = model.default_geometry

# load parameter values and process model and geometry
param = model.default_parameter_values
C_rate = 1
current_1C = 24 * param.process_symbol(pybamm.geometric_parameters.A_cc).evaluate()
param.update(
    {
        "Typical current [A]": C_rate * current_1C,
        "Initial temperature [K]": 298.15,
        "Negative current collector conductivity [S.m-1]": 1e5,
        "Positive current collector conductivity [S.m-1]": 1e5,
        "Heat transfer coefficient [W.m-2.K-1]": 1,
    }
)
param.process_model(model)
param.process_geometry(geometry)

# set mesh
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 5, var.x_s: 5, var.x_p: 5, var.r_n: 10, var.r_p: 10, var.z: 15}
# depnding on number of points in y-z plane may need to increase recursion depth...
sys.setrecursionlimit(10000)
submesh_types = model.default_submesh_types
z = np.array([0, 0.1, 0.3, 0.5, 0.8, 1])
submesh_types["current collector"] = pybamm.GetUserSupplied1DSubMesh(z)
mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.defualt_spatial_methods)
disc.process_model(model)

# solve model -- simulate one hour discharge
tau = param.process_symbol(pybamm.standard_parameters_lithium_ion.tau_discharge)
t_end = 3600 / tau.evaluate(0)
t_eval = np.linspace(0, t_end, 120)
solution = model.default_solver.solve(model, t_eval)

# plot
output_variables = [
    "X-averaged negative particle surface concentration [mol.m-3]",
    "X-averaged positive particle surface concentration [mol.m-3]",
    #"X-averaged cell temperature [K]",
    "Local potenital difference [V]",
    "Current collector current density [A.m-2]",
    "Terminal voltage [V]",
    "Volume-averaged cell temperature [K]",
]
plot = pybamm.QuickPlot(model, mesh, solution, output_variables)
plot.dynamic_plot()
